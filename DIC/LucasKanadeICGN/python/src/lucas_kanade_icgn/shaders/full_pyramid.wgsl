// GPU-resident image reduction and coarse-to-fine affine seed transfer.
//
// The IC-GN equations remain in icgn.wgsl.  These kernels only implement the
// pyramid orchestration that is specific to the full-pyramid backend.

struct Float4Buffer {
    values: array<vec4<f32>>,
}

struct UintBuffer {
    values: array<u32>,
}

struct AtomicBuffer {
    values: array<atomic<u32>>,
}

struct TemporalConfig {
    count: u32,
    max_age: u32,
    use_filtered: u32,
    padding: u32,
}

struct SeedEstimate {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    weight: f32,
}

struct SeedConfig {
    source_grid_width: u32,
    source_grid_height: u32,
    destination_grid_width: u32,
    destination_grid_count: u32,
    source_scale: f32,
    destination_scale: f32,
    padding_0: f32,
    padding_1: f32,
}

override SUBSET_SIZE: f32 = 16.0;
override SUBSET_STRIDE: f32 = 16.0;
// Matches icgn.wgsl: 0 affine, 1 translation, 2 quadratic.
override MOTION_MODEL: u32 = 0u;
override TRACK_SEED_AVAILABILITY: bool = false;

@group(0) @binding(0) var reduce_x_source: texture_2d<f32>;
@group(0) @binding(1) var reduce_x_destination: texture_storage_2d<r32float, write>;

@group(1) @binding(0) var reduce_y_source: texture_2d<f32>;
@group(1) @binding(1) var reduce_y_destination: texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(0) var<storage, read> source_state: Float4Buffer;
@group(2) @binding(1) var<storage, read_write> destination_seed: Float4Buffer;
@group(2) @binding(2) var<storage, read_write> destination_active: UintBuffer;
@group(2) @binding(3) var<uniform> seed_config: SeedConfig;
// Only the native ROI entry point requires this static link table.
@group(2) @binding(4) var<storage, read> roi_seed_links: UintBuffer;
@group(2) @binding(5) var<storage, read_write> projection_counts: AtomicBuffer;
@group(2) @binding(6) var<storage, read> fine_history: Float4Buffer;
@group(2) @binding(7) var<storage, read> fine_history_ages: UintBuffer;
@group(2) @binding(8) var<uniform> fine_history_config: TemporalConfig;
@group(2) @binding(9) var<storage, read> coarse_history_ages: UintBuffer;

@group(3) @binding(0) var<storage, read_write> temporal_state: Float4Buffer;
@group(3) @binding(1) var<storage, read_write> temporal_ages: UintBuffer;
@group(3) @binding(2) var<storage, read> raw_state: Float4Buffer;
@group(3) @binding(3) var<storage, read> output_state: Float4Buffer;
@group(3) @binding(4) var<uniform> temporal_config: TemporalConfig;
@group(3) @binding(5) var<storage, read_write> temporal_counts: AtomicBuffer;

fn mirror_coordinate(coordinate: i32, size: i32) -> i32 {
    if coordinate < 0 {
        return -coordinate;
    }
    if coordinate >= size {
        return 2 * size - 2 - coordinate;
    }
    return coordinate;
}

fn reduction_weight(tap: i32) -> f32 {
    return array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0)[tap + 2] / 16.0;
}

@compute @workgroup_size(16, 16)
fn reduce_x(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let destination_size = textureDimensions(reduce_x_destination);
    if global_id.x >= destination_size.x || global_id.y >= destination_size.y {
        return;
    }
    let source_size = textureDimensions(reduce_x_source);
    let source_x = i32(2u * global_id.x);
    let source_y = i32(global_id.y);
    var value = 0.0;
    for (var tap = -2; tap <= 2; tap = tap + 1) {
        let sample_x = mirror_coordinate(source_x + tap, i32(source_size.x));
        // Recover the source byte before filtering. In grayscale units the
        // five-tap sums are exact binary fractions, retained by r32float.
        let sample = round(
            textureLoad(reduce_x_source, vec2<i32>(sample_x, source_y), 0).x * 255.0,
        );
        value += reduction_weight(tap) * sample;
    }
    textureStore(
        reduce_x_destination,
        vec2<i32>(global_id.xy),
        vec4<f32>(value, 0.0, 0.0, 1.0),
    );
}

@compute @workgroup_size(16, 16)
fn reduce_y(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let destination_size = textureDimensions(reduce_y_destination);
    if global_id.x >= destination_size.x || global_id.y >= destination_size.y {
        return;
    }
    let source_size = textureDimensions(reduce_y_source);
    let source_x = i32(global_id.x);
    let source_y = i32(2u * global_id.y);
    var value = 0.0;
    for (var tap = -2; tap <= 2; tap = tap + 1) {
        let sample_y = mirror_coordinate(source_y + tap, i32(source_size.y));
        value += reduction_weight(tap)
            * textureLoad(reduce_y_source, vec2<i32>(source_x, sample_y), 0).x;
    }
    // Both passes use grayscale units [0, 255], matching the CPU reduction.
    // Truncate only the final sum, then normalize for rgba8unorm storage.
    let quantized = floor(clamp(value, 0.0, 255.0)) / 255.0;
    textureStore(
        reduce_y_destination,
        vec2<i32>(global_id.xy),
        vec4<f32>(quantized, 0.0, 0.0, 1.0),
    );
}

var<workgroup> reduction_pixels: array<u32, 1296>;

// Exact five-tap separable reduction, evaluated as one integer 5x5 sum.
// A 16x16 output tile shares its reflected 36x36 byte-valued footprint.
// Integer sums stay below 65281, so truncation matches both original passes.
@compute @workgroup_size(16, 16)
fn reduce_fused(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let source_size = vec2<i32>(textureDimensions(reduce_y_source));
    let origin = vec2<i32>(group.xy) * 32 - vec2<i32>(2);
    for (var index = lane; index < 1296u; index += 256u) {
        let pixel = origin + vec2<i32>(i32(index % 36u), i32(index / 36u));
        let sample = clamp(vec2<i32>(
            mirror_coordinate(pixel.x, source_size.x),
            mirror_coordinate(pixel.y, source_size.y),
        ), vec2<i32>(0), source_size - vec2<i32>(1));
        reduction_pixels[index] = u32(round(textureLoad(reduce_y_source, sample, 0).x * 255.0));
    }
    workgroupBarrier();
    let destination = group.xy * 16u + local.xy;
    if any(destination >= textureDimensions(reduce_y_destination)) { return; }
    let taps = array<u32, 5>(1u, 4u, 6u, 4u, 1u);
    var value = 0u;
    for (var y = 0u; y < 5u; y += 1u) {
        for (var x = 0u; x < 5u; x += 1u) {
            let index = (2u * local.y + y) * 36u + 2u * local.x + x;
            value += taps[y] * taps[x] * reduction_pixels[index];
        }
    }
    textureStore(reduce_y_destination, vec2<i32>(destination),
        vec4<f32>(f32(value / 256u) / 255.0, 0.0, 0.0, 1.0));
}

fn accumulate_seed(
    source_index: u32,
    weight: f32,
    source_position: vec2<f32>,
    destination_position: vec2<f32>,
    translation_scale: f32,
    accumulated_a: ptr<function, vec4<f32>>,
    accumulated_b: ptr<function, vec4<f32>>,
    accumulated_c: ptr<function, vec4<f32>>,
    weight_sum: ptr<function, f32>,
) {
    let stride = select(2u, 4u, MOTION_MODEL == 2u);
    let state_index = source_index * stride;
    let source_a = source_state.values[state_index];
    let source_b = source_state.values[state_index + 1u];
    var source_c = vec4<f32>(0.0);
    var status = source_b.z;
    if MOTION_MODEL == 2u {
        source_c = source_state.values[state_index + 2u];
        status = source_state.values[state_index + 3u].x;
    }
    if status < -0.5 {
        return;
    }
    let delta = destination_position - source_position;
    let weighted = weight;
    var translated_u = source_a.x + source_a.y * delta.x + source_a.z * delta.y;
    var translated_v = source_a.w + source_b.x * delta.x + source_b.y * delta.y;
    var ux = source_a.y;
    var uy = source_a.z;
    var vx = source_b.x;
    var vy = source_b.y;
    if MOTION_MODEL == 2u {
        translated_u += 0.5 * source_b.z * delta.x * delta.x
            + source_b.w * delta.x * delta.y
            + 0.5 * source_c.x * delta.y * delta.y;
        translated_v += 0.5 * source_c.y * delta.x * delta.x
            + source_c.z * delta.x * delta.y
            + 0.5 * source_c.w * delta.y * delta.y;
        ux += source_b.z * delta.x + source_b.w * delta.y;
        uy += source_b.w * delta.x + source_c.x * delta.y;
        vx += source_c.y * delta.x + source_c.z * delta.y;
        vy += source_c.z * delta.x + source_c.w * delta.y;
    }
    (*accumulated_a).x += weighted * translated_u * translation_scale;
    (*accumulated_a).y += weighted * ux;
    (*accumulated_a).z += weighted * uy;
    (*accumulated_a).w += weighted * translated_v * translation_scale;
    (*accumulated_b).x += weighted * vx;
    (*accumulated_b).y += weighted * vy;
    if MOTION_MODEL == 2u {
        let curvature_scale = 1.0 / translation_scale;
        (*accumulated_b).z += weighted * source_b.z * curvature_scale;
        (*accumulated_b).w += weighted * source_b.w * curvature_scale;
        *accumulated_c += weighted * source_c * curvature_scale;
    }
    *weight_sum += weighted;
}

fn interpolate_subset_seed(destination_index: u32, permissions: u32) -> SeedEstimate {
    if destination_index >= seed_config.destination_grid_count {
        return SeedEstimate(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), 0.0);
    }
    let destination_x = destination_index % seed_config.destination_grid_width;
    let destination_y = destination_index / seed_config.destination_grid_width;
    let source_offset = (1.0 - seed_config.source_scale) * 0.5;
    let destination_offset = (1.0 - seed_config.destination_scale) * 0.5;
    let destination_center = vec2<f32>(
        0.5 * SUBSET_SIZE + SUBSET_STRIDE * f32(destination_x),
        0.5 * SUBSET_SIZE + SUBSET_STRIDE * f32(destination_y),
    );
    let source_coordinate = (
        seed_config.destination_scale * destination_center
        + vec2<f32>(destination_offset - source_offset)
    ) / seed_config.source_scale;
    let source_grid_coordinate = (
        source_coordinate - vec2<f32>(0.5 * SUBSET_SIZE)
    ) / SUBSET_STRIDE;
    let maximum_lower = vec2<i32>(
        max(0, i32(seed_config.source_grid_width) - 2),
        max(0, i32(seed_config.source_grid_height) - 2),
    );
    let lower = clamp(vec2<i32>(floor(source_grid_coordinate)), vec2<i32>(0), maximum_lower);
    let fraction = clamp(source_grid_coordinate - vec2<f32>(lower), vec2<f32>(0.0), vec2<f32>(1.0));
    let translation_scale = seed_config.source_scale / seed_config.destination_scale;
    var accumulated_a = vec4<f32>(0.0);
    var accumulated_b = vec4<f32>(0.0);
    var accumulated_c = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var corner = 0; corner < 4; corner = corner + 1) {
        if (permissions & (1u << u32(corner))) == 0u {
            continue;
        }
        let offset = vec2<i32>(corner & 1, corner >> 1);
        let source_grid = lower + offset;
        if source_grid.x >= i32(seed_config.source_grid_width)
            || source_grid.y >= i32(seed_config.source_grid_height) {
            continue;
        }
        let source_index = u32(source_grid.y) * seed_config.source_grid_width
            + u32(source_grid.x);
        let weight_x = select(1.0 - fraction.x, fraction.x, offset.x == 1);
        let weight_y = select(1.0 - fraction.y, fraction.y, offset.y == 1);
        let source_center = vec2<f32>(
            0.5 * SUBSET_SIZE + SUBSET_STRIDE * f32(source_grid.x),
            0.5 * SUBSET_SIZE + SUBSET_STRIDE * f32(source_grid.y),
        );
        accumulate_seed(
            source_index,
            weight_x * weight_y,
            source_center,
            source_coordinate,
            translation_scale,
            &accumulated_a,
            &accumulated_b,
            &accumulated_c,
            &weight_sum,
        );
    }
    let has_seed = weight_sum > 0.0;
    if has_seed {
        accumulated_a /= weight_sum;
        accumulated_b /= weight_sum;
        accumulated_c /= weight_sum;
    }
    return SeedEstimate(accumulated_a, accumulated_b, accumulated_c, weight_sum);
}

fn transfer_subset_seed(destination_index: u32, permissions: u32) -> bool {
    if destination_index >= seed_config.destination_grid_count { return false; }
    let estimate = interpolate_subset_seed(destination_index, permissions);
    let stride = select(2u, 4u, MOTION_MODEL == 2u);
    let destination_state_index = destination_index * stride;
    destination_seed.values[destination_state_index] = estimate.a;
    destination_seed.values[destination_state_index + 1u] = estimate.b;
    if MOTION_MODEL == 2u {
        destination_seed.values[destination_state_index + 2u] = estimate.c;
        destination_seed.values[destination_state_index + 3u] = vec4<f32>(0.0);
    }
    if TRACK_SEED_AVAILABILITY {
        let marker = select(-1.0, 0.0, estimate.weight >= 0.99999);
        if MOTION_MODEL == 2u {
            destination_seed.values[destination_state_index + 3u].x = marker;
        } else {
            destination_seed.values[destination_state_index + 1u].z = marker;
        }
    }
    // A missing coarse estimate is not a reason to suppress a potentially
    // valid finer subset. Run it from the zero seed instead.
    destination_active.values[destination_index] = (permissions >> 4u) & 1u;
    return estimate.weight > 0.0 && (permissions & 16u) != 0u;
}

@compute @workgroup_size(256)
fn transfer_seed(@builtin(global_invocation_id) global_id: vec3<u32>) {
    transfer_subset_seed(global_id.x, 31u);
}

@compute @workgroup_size(256)
fn transfer_seed_roi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= seed_config.destination_grid_count {
        return;
    }
    transfer_subset_seed(global_id.x, roi_seed_links.values[global_id.x]);
}

// Project the last accepted fine-grid field onto the *coarsest* grid. Every
// target still traverses every pyramid level; this is not a selective retry.
@compute @workgroup_size(256)
fn project_temporal_seed(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= seed_config.destination_grid_count { return; }
    if transfer_subset_seed(global_id.x, roi_seed_links.values[global_id.x]) {
        atomicAdd(&projection_counts.values[0], 1u);
    } else if TRACK_SEED_AVAILABILITY {
        let stride = select(2u, 4u, MOTION_MODEL == 2u);
        if MOTION_MODEL == 2u {
            destination_seed.values[global_id.x * stride + 3u].x = -1.0;
        } else {
            destination_seed.values[global_id.x * stride + 1u].z = -1.0;
        }
    }
}

// Preserve previously resolved fine detail, adding only the correction found
// by the current coarse solve. Every frame still solves coarse and fine GN.
@compute @workgroup_size(256)
fn refine_temporal_seed(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= seed_config.destination_grid_count { return; }
    if destination_active.values[index] == 0u { return; }
    let age = fine_history_ages.values[index];
    if age == 0xffffffffu || age > fine_history_config.max_age { return; }
    let stride = select(2u, 4u, MOTION_MODEL == 2u);
    let base = index * stride;
    let previous_a = fine_history.values[base];
    let previous_b = fine_history.values[base + 1u];
    var absolute_status = destination_seed.values[base + 1u].z;
    if MOTION_MODEL == 2u { absolute_status = destination_seed.values[base + 3u].x; }
    if absolute_status < -0.5 { return; }
    // Match the last-accepted frame of every source to that of this fine
    // point. Different ages would mix temporal increments from different times.
    let point = vec2<f32>(f32(index % seed_config.destination_grid_width),
                         f32(index / seed_config.destination_grid_width));
    let coordinate = (seed_config.destination_scale * (vec2<f32>(0.5 * SUBSET_SIZE)
        + point * SUBSET_STRIDE) + vec2<f32>(
            (seed_config.source_scale - seed_config.destination_scale) * 0.5))
        / seed_config.source_scale;
    let maximum = vec2<i32>(max(0, i32(seed_config.source_grid_width)-2),
                            max(0, i32(seed_config.source_grid_height)-2));
    let lower = clamp(vec2<i32>(floor((coordinate - vec2<f32>(0.5 * SUBSET_SIZE))
                                    / SUBSET_STRIDE)), vec2<i32>(0), maximum);
    let permissions = roi_seed_links.values[index];
    for (var corner = 0u; corner < 4u; corner += 1u) {
        if (permissions & (1u << corner)) == 0u { continue; }
        let cell = lower + vec2<i32>(i32(corner & 1u), i32(corner >> 1u));
        if cell.x >= i32(seed_config.source_grid_width)
            || cell.y >= i32(seed_config.source_grid_height) { continue; }
        let source_index = u32(cell.y) * seed_config.source_grid_width + u32(cell.x);
        if coarse_history_ages.values[source_index] != age { return; }
    }
    let predicted = interpolate_subset_seed(index, roi_seed_links.values[index]);
    // Without a complete admissible predicted stencil, retain the ordinary
    // absolute pyramid seed. Do not fabricate history across missing links.
    if predicted.weight < 0.99999 { return; }
    let absolute_a = destination_seed.values[base];
    let absolute_b = destination_seed.values[base + 1u];
    destination_seed.values[base] = previous_a + (absolute_a - predicted.a);
    var corrected_b = absolute_b;
    corrected_b.x = previous_b.x + (absolute_b.x - predicted.b.x);
    corrected_b.y = previous_b.y + (absolute_b.y - predicted.b.y);
    if MOTION_MODEL == 2u {
        corrected_b.z = previous_b.z + (absolute_b.z - predicted.b.z);
        corrected_b.w = previous_b.w + (absolute_b.w - predicted.b.w);
        destination_seed.values[base + 2u] = fine_history.values[base + 2u]
            + (destination_seed.values[base + 2u] - predicted.c);
    }
    destination_seed.values[base + 1u] = corrected_b;
}

@compute @workgroup_size(256)
fn update_temporal_state(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= temporal_config.count { return; }
    let stride = select(2u, 4u, MOTION_MODEL == 2u);
    let base = index * stride;
    let status_offset = select(6u, 12u, MOTION_MODEL == 2u);
    let status_vector = status_offset / 4u;
    let status_lane = status_offset % 4u;
    let raw_status = raw_state.values[base + status_vector][status_lane];
    let final_status = output_state.values[base + status_vector][status_lane];
    let raw_accepted = raw_status == 1.0 || raw_status == 2.0;
    let final_accepted = final_status == 1.0 || final_status == 2.0;
    if final_accepted && (temporal_config.use_filtered != 0u || raw_accepted) {
        for (var word = 0u; word < stride; word += 1u) {
            var value = raw_state.values[base + word];
            if temporal_config.use_filtered != 0u {
                value = output_state.values[base + word];
            }
            temporal_state.values[base + word] = value;
        }
        temporal_ages.values[index] = 0u;
        if temporal_config.padding == 0u {
            atomicAdd(&temporal_counts.values[1], 1u);
        }
    } else {
        let age = temporal_ages.values[index];
        // UINT_MAX is the never-set sentinel; valid ages saturate below it.
        if age != 0xffffffffu {
            let new_age = min(age, 0xfffffffdu) + 1u;
            temporal_ages.values[index] = new_age;
            if new_age > temporal_config.max_age {
                temporal_state.values[base + status_vector][status_lane] = -1.0;
            }
        }
    }
}
