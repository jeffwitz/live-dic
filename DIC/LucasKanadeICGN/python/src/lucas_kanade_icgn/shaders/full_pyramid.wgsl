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

@group(0) @binding(0) var reduce_x_source: texture_2d<f32>;
@group(0) @binding(1) var reduce_x_destination: texture_storage_2d<r32float, write>;

@group(1) @binding(0) var reduce_y_source: texture_2d<f32>;
@group(1) @binding(1) var reduce_y_destination: texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(0) var<storage, read> source_state: Float4Buffer;
@group(2) @binding(1) var<storage, read_write> destination_seed: Float4Buffer;
@group(2) @binding(2) var<storage, read_write> destination_active: UintBuffer;
@group(2) @binding(3) var<uniform> seed_config: SeedConfig;

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

fn accumulate_seed(
    source_index: u32,
    weight: f32,
    source_position: vec2<f32>,
    destination_position: vec2<f32>,
    translation_scale: f32,
    accumulated_a: ptr<function, vec4<f32>>,
    accumulated_b: ptr<function, vec2<f32>>,
    weight_sum: ptr<function, f32>,
) {
    let source_a = source_state.values[source_index * 2u];
    let source_b = source_state.values[source_index * 2u + 1u];
    if source_b.z < -0.5 {
        return;
    }
    let delta = destination_position - source_position;
    let weighted = weight;
    (*accumulated_a).x += weighted
        * (source_a.x + source_a.y * delta.x + source_a.z * delta.y)
        * translation_scale;
    (*accumulated_a).y += weighted * source_a.y;
    (*accumulated_a).z += weighted * source_a.z;
    (*accumulated_a).w += weighted
        * (source_a.w + source_b.x * delta.x + source_b.y * delta.y)
        * translation_scale;
    (*accumulated_b).x += weighted * source_b.x;
    (*accumulated_b).y += weighted * source_b.y;
    *weight_sum += weighted;
}

@compute @workgroup_size(256)
fn transfer_seed(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let destination_index = global_id.x;
    if destination_index >= seed_config.destination_grid_count {
        return;
    }
    let destination_x = destination_index % seed_config.destination_grid_width;
    let destination_y = destination_index / seed_config.destination_grid_width;
    let source_offset = (1.0 - seed_config.source_scale) * 0.5;
    let destination_offset = (1.0 - seed_config.destination_scale) * 0.5;
    let destination_center = vec2<f32>(
        8.0 + 16.0 * f32(destination_x),
        8.0 + 16.0 * f32(destination_y),
    );
    let source_coordinate = (
        seed_config.destination_scale * destination_center
        + vec2<f32>(destination_offset - source_offset)
    ) / seed_config.source_scale;
    let source_grid_coordinate = (source_coordinate - vec2<f32>(8.0)) / 16.0;
    let maximum_lower = vec2<i32>(
        i32(seed_config.source_grid_width) - 2,
        i32(seed_config.source_grid_height) - 2,
    );
    let lower = clamp(vec2<i32>(floor(source_grid_coordinate)), vec2<i32>(0), maximum_lower);
    let fraction = clamp(source_grid_coordinate - vec2<f32>(lower), vec2<f32>(0.0), vec2<f32>(1.0));
    let translation_scale = seed_config.source_scale / seed_config.destination_scale;
    var accumulated_a = vec4<f32>(0.0);
    var accumulated_b = vec2<f32>(0.0);
    var weight_sum = 0.0;
    for (var corner = 0; corner < 4; corner = corner + 1) {
        let offset = vec2<i32>(corner & 1, corner >> 1);
        let source_grid = lower + offset;
        let source_index = u32(source_grid.y) * seed_config.source_grid_width
            + u32(source_grid.x);
        let weight_x = select(1.0 - fraction.x, fraction.x, offset.x == 1);
        let weight_y = select(1.0 - fraction.y, fraction.y, offset.y == 1);
        let source_center = vec2<f32>(
            8.0 + 16.0 * f32(source_grid.x),
            8.0 + 16.0 * f32(source_grid.y),
        );
        accumulate_seed(
            source_index,
            weight_x * weight_y,
            source_center,
            source_coordinate,
            translation_scale,
            &accumulated_a,
            &accumulated_b,
            &weight_sum,
        );
    }
    let has_seed = weight_sum > 0.0;
    if has_seed {
        accumulated_a /= weight_sum;
        accumulated_b /= weight_sum;
    }
    destination_seed.values[destination_index * 2u] = accumulated_a;
    destination_seed.values[destination_index * 2u + 1u] = vec4<f32>(
        accumulated_b,
        0.0,
        0.0,
    );
    // A missing coarse estimate is not a reason to suppress a potentially
    // valid finer subset. Run it from the zero seed instead.
    destination_active.values[destination_index] = 1u;
}
