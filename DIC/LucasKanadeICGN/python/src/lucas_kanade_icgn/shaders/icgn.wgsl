struct Config {
    width: u32,
    height: u32,
    grid_width: u32,
    grid_count: u32,
    iterations: u32,
    pixel_count: u32,
    tolerance: f32,
    zncc_min: f32,
}

struct FloatBuffer {
    values: array<f32>,
}

struct Float4Buffer {
    values: array<vec4<f32>>,
}

struct UintBuffer {
    values: array<u32>,
}

struct WarpUpdate {
    translation: vec2<f32>,
    linear: vec4<f32>,
}

@group(0) @binding(0) var source_image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read_write> temporary: FloatBuffer;
@group(0) @binding(3) var<storage, read_write> coefficients: FloatBuffer;
@group(0) @binding(4) var<storage, read> reference_data: Float4Buffer;
@group(0) @binding(5) var<storage, read> inverse_hessian: Float4Buffer;
@group(0) @binding(6) var<storage, read> metadata: Float4Buffer;
@group(0) @binding(7) var<storage, read_write> state: Float4Buffer;
@group(0) @binding(8) var<storage, read> fir_weights: FloatBuffer;
@group(0) @binding(9) var<storage, read_write> filtered_state: Float4Buffer;
@group(0) @binding(10) var<storage, read> seed_state: Float4Buffer;
@group(0) @binding(11) var<storage, read> active_mask: UintBuffer;

const FIR_WEIGHT_RADIUS: i32 = 15;
const FIR_RADIUS: i32 = 12;
const PREFILTER_X_WIDTH: i32 = 32;
const PREFILTER_X_HEIGHT: i32 = 8;
const PREFILTER_X_STRIDE: i32 = PREFILTER_X_WIDTH + 2 * FIR_RADIUS;
const PREFILTER_Y_WIDTH: i32 = 8;
const PREFILTER_Y_HEIGHT: i32 = 32;
const PREFILTER_TILE_PIXELS: u32 = 448u;

// 0: historical Gaussian + cubic B-spline coefficients
// 1: direct bilinear interpolation
// 2: direct Keys cubic convolution (a = -0.5)
override INTERPOLATION_MODE: u32 = 0u;
// 0: six-parameter affine warp, 1: translation-only warp.
override MOTION_MODEL: u32 = 0u;
// Square subset width. Repository hosts expose the validated 8, 16, and 32
// specializations while preserving 16 as the default pipeline constant.
override SUBSET_SIZE: i32 = 16;
// Keep standalone consumers safe. Repository hosts specialize this override
// to zero for dimensions where one reflection already covers every tile load.
override PREFILTER_CLAMP_HALO: u32 = 1u;

var<workgroup> partial_a: array<vec4<f32>, 16>;
var<workgroup> partial_b: array<vec4<f32>, 16>;
var<workgroup> partial_c: array<vec4<f32>, 16>;
var<workgroup> shared_translation: vec2<f32>;
var<workgroup> shared_linear: vec4<f32>;
var<workgroup> shared_status: vec2<f32>;
var<workgroup> prefilter_tile: array<f32, 496>;

fn mirror_coordinate(coordinate: i32, size: i32) -> i32 {
    var reflected = coordinate;
    if coordinate < 0 {
        reflected = -coordinate;
    } else if coordinate >= size {
        reflected = 2 * size - 2 - coordinate;
    }
    // Partial edge workgroups load a few halo cells that no valid output uses.
    // A single reflection can leave those speculative coordinates out of range
    // only below 44 pixels. Hosts specialize this override per axis so the
    // common large-image pipeline retains the original arithmetic exactly.
    if PREFILTER_CLAMP_HALO != 0u {
        return clamp(reflected, 0, size - 1);
    }
    return reflected;
}

@compute @workgroup_size(32, 8)
fn prefilter_x(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let width = i32(config.width);
    let height = i32(config.height);
    let base_x = i32(workgroup_id.x) * PREFILTER_X_WIDTH;
    let base_y = i32(workgroup_id.y) * PREFILTER_X_HEIGHT;

    for (var tile_index = local_index; tile_index < PREFILTER_TILE_PIXELS; tile_index += 256u) {
        let tile_x = i32(tile_index % u32(PREFILTER_X_STRIDE));
        let tile_y = i32(tile_index / u32(PREFILTER_X_STRIDE));
        let sample_x = mirror_coordinate(base_x + tile_x - FIR_RADIUS, width);
        let sample_y = min(base_y + tile_y, height - 1);
        prefilter_tile[tile_index] =
            textureLoad(source_image, vec2<i32>(sample_x, sample_y), 0).x * 255.0;
    }
    workgroupBarrier();

    let x = base_x + i32(local_id.x);
    let y = base_y + i32(local_id.y);
    if x >= width || y >= height {
        return;
    }
    var result = 0.0;
    for (var tap = -FIR_RADIUS; tap <= FIR_RADIUS; tap = tap + 1) {
        let tile_index = i32(local_id.y) * PREFILTER_X_STRIDE
            + i32(local_id.x) + FIR_RADIUS + tap;
        let sample = prefilter_tile[tile_index];
        result = result + fir_weights.values[u32(tap + FIR_WEIGHT_RADIUS)] * sample;
    }
    temporary.values[u32(y * width + x)] = result;
}

@compute @workgroup_size(8, 32)
fn prefilter_y(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let width = i32(config.width);
    let height = i32(config.height);
    let base_x = i32(workgroup_id.x) * PREFILTER_Y_WIDTH;
    let base_y = i32(workgroup_id.y) * PREFILTER_Y_HEIGHT;

    for (var tile_index = local_index; tile_index < PREFILTER_TILE_PIXELS; tile_index += 256u) {
        let tile_x = i32(tile_index % u32(PREFILTER_Y_WIDTH));
        let tile_y = i32(tile_index / u32(PREFILTER_Y_WIDTH));
        let sample_x = min(base_x + tile_x, width - 1);
        let sample_y = mirror_coordinate(base_y + tile_y - FIR_RADIUS, height);
        prefilter_tile[tile_index] = temporary.values[u32(sample_y * width + sample_x)];
    }
    workgroupBarrier();

    let x = base_x + i32(local_id.x);
    let y = base_y + i32(local_id.y);
    if x >= width || y >= height {
        return;
    }
    var result = 0.0;
    for (var tap = -FIR_RADIUS; tap <= FIR_RADIUS; tap = tap + 1) {
        let tile_index = (i32(local_id.y) + FIR_RADIUS + tap) * PREFILTER_Y_WIDTH
            + i32(local_id.x);
        result = result + fir_weights.values[u32(tap + FIR_WEIGHT_RADIUS)]
            * prefilter_tile[tile_index];
    }
    coefficients.values[u32(y * width + x)] = result;
}

fn metadata_value(subset_index: u32, component: u32) -> f32 {
    return metadata.values[subset_index * 4u + component / 4u][component % 4u];
}

fn hessian_value(subset_index: u32, component: u32) -> f32 {
    return inverse_hessian.values[subset_index * 9u + component / 4u][component % 4u];
}

fn cubic_weights(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;
    let inverse = 1.0 - t;
    var weights: array<f32, 4>;
    weights[0] = inverse * inverse * inverse / 6.0;
    weights[1] = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
    weights[2] = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
    weights[3] = t3 / 6.0;
    return weights;
}

fn sample_cubic(position: vec2<f32>) -> f32 {
    let floored = floor(position);
    let base = vec2<i32>(floored);
    let fraction = position - floored;
    let weight_x = cubic_weights(fraction.x);
    let weight_y = cubic_weights(fraction.y);
    let width = i32(config.width);
    var value = 0.0;
    for (var yy = 0; yy < 4; yy = yy + 1) {
        var row = 0.0;
        for (var xx = 0; xx < 4; xx = xx + 1) {
            let x = base.x + xx - 1;
            let y = base.y + yy - 1;
            row = row + weight_x[xx] * coefficients.values[u32(y * width + x)];
        }
        value = value + weight_y[yy] * row;
    }
    return value;
}

fn source_value(x: i32, y: i32) -> f32 {
    return textureLoad(source_image, vec2<i32>(x, y), 0).x * 255.0;
}

fn sample_linear(position: vec2<f32>) -> f32 {
    let floored = floor(position);
    let base = vec2<i32>(floored);
    let fraction = position - floored;
    let top = mix(
        source_value(base.x, base.y),
        source_value(base.x + 1, base.y),
        fraction.x,
    );
    let bottom = mix(
        source_value(base.x, base.y + 1),
        source_value(base.x + 1, base.y + 1),
        fraction.x,
    );
    return mix(top, bottom, fraction.y);
}

fn keys_cubic_weights(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;
    var weights: array<f32, 4>;
    weights[0] = -0.5 * t + t2 - 0.5 * t3;
    weights[1] = 1.0 - 2.5 * t2 + 1.5 * t3;
    weights[2] = 0.5 * t + 2.0 * t2 - 1.5 * t3;
    weights[3] = -0.5 * t2 + 0.5 * t3;
    return weights;
}

fn sample_keys_cubic(position: vec2<f32>) -> f32 {
    let floored = floor(position);
    let base = vec2<i32>(floored);
    let fraction = position - floored;
    let weight_x = keys_cubic_weights(fraction.x);
    let weight_y = keys_cubic_weights(fraction.y);
    var value = 0.0;
    for (var yy = 0; yy < 4; yy = yy + 1) {
        var row = 0.0;
        for (var xx = 0; xx < 4; xx = xx + 1) {
            row += weight_x[xx] * source_value(base.x + xx - 1, base.y + yy - 1);
        }
        value += weight_y[yy] * row;
    }
    return value;
}

fn sample_target(position: vec2<f32>) -> f32 {
    if INTERPOLATION_MODE == 1u {
        return sample_linear(position);
    }
    if INTERPOLATION_MODE == 2u {
        return sample_keys_cubic(position);
    }
    return sample_cubic(position);
}

fn solve_increment(
    subset_index: u32,
    reduced_j: array<f32, 6>,
    current_mean: f32,
    ratio: f32,
) -> array<f32, 6> {
    if MOTION_MODEL == 1u {
        let residual_x = metadata_value(subset_index, 2u)
            - ratio * (reduced_j[0] - current_mean * metadata_value(subset_index, 8u));
        let residual_y = metadata_value(subset_index, 5u)
            - ratio * (reduced_j[3] - current_mean * metadata_value(subset_index, 11u));
        var translation_delta = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        translation_delta[0] = -(
            hessian_value(subset_index, 0u) * residual_x
            + hessian_value(subset_index, 3u) * residual_y
        );
        translation_delta[3] = -(
            hessian_value(subset_index, 18u) * residual_x
            + hessian_value(subset_index, 21u) * residual_y
        );
        return translation_delta;
    }
    var residual: array<f32, 6>;
    for (var component = 0u; component < 6u; component = component + 1u) {
        residual[component] = metadata_value(subset_index, 2u + component)
            - ratio * (reduced_j[component]
            - current_mean * metadata_value(subset_index, 8u + component));
    }
    var delta: array<f32, 6>;
    for (var row = 0u; row < 6u; row = row + 1u) {
        var value = 0.0;
        for (var column = 0u; column < 6u; column = column + 1u) {
            value = value + hessian_value(subset_index, 6u * row + column) * residual[column];
        }
        delta[row] = -value;
    }
    return delta;
}

fn increment_determinant(delta: array<f32, 6>) -> f32 {
    if MOTION_MODEL == 1u {
        return 1.0;
    }
    return (1.0 + delta[1]) * (1.0 + delta[5]) - delta[2] * delta[4];
}

fn compose_inverse_update(
    translation: vec2<f32>,
    linear: vec4<f32>,
    delta: array<f32, 6>,
    determinant: f32,
) -> WarpUpdate {
    if MOTION_MODEL == 1u {
        return WarpUpdate(
            translation - vec2<f32>(delta[0], delta[3]),
            vec4<f32>(0.0),
        );
    }
    let inverse11 = (1.0 + delta[5]) / determinant;
    let inverse12 = -delta[2] / determinant;
    let inverse21 = -delta[4] / determinant;
    let inverse22 = (1.0 + delta[1]) / determinant;
    let warp11 = 1.0 + linear.x;
    let warp12 = linear.y;
    let warp21 = linear.z;
    let warp22 = 1.0 + linear.w;
    let inverse_translation_x = -(inverse11 * delta[0] + inverse12 * delta[3]);
    let inverse_translation_y = -(inverse21 * delta[0] + inverse22 * delta[3]);
    let updated_translation = translation + vec2<f32>(
        warp11 * inverse_translation_x + warp12 * inverse_translation_y,
        warp21 * inverse_translation_x + warp22 * inverse_translation_y,
    );
    let updated11 = warp11 * inverse11 + warp12 * inverse21;
    let updated12 = warp11 * inverse12 + warp12 * inverse22;
    let updated21 = warp21 * inverse11 + warp22 * inverse21;
    let updated22 = warp21 * inverse12 + warp22 * inverse22;
    return WarpUpdate(
        updated_translation,
        vec4<f32>(updated11 - 1.0, updated12, updated21, updated22 - 1.0),
    );
}

fn corner_change(delta: array<f32, 6>) -> f32 {
    if MOTION_MODEL == 1u {
        return length(vec2<f32>(delta[0], delta[3]));
    }
    var maximum = 0.0;
    for (var corner = 0; corner < 4; corner = corner + 1) {
        let half_extent = subset_half_extent();
        let x = select(-half_extent, half_extent, corner == 1 || corner == 3);
        let y = select(-half_extent, half_extent, corner >= 2);
        let change = length(vec2<f32>(
            delta[0] + delta[1] * x + delta[2] * y,
            delta[3] + delta[4] * x + delta[5] * y,
        ));
        maximum = max(maximum, change);
    }
    return maximum;
}

fn convergence_status(change: f32, zncc: f32) -> f32 {
    if change >= config.tolerance {
        return 0.0;
    }
    return select(-2.0, 1.0, zncc >= config.zncc_min);
}

fn solve_subset(
    subset_index: u32,
    lane: u32,
    initial_translation: vec2<f32>,
    initial_linear: vec4<f32>,
    enabled: bool,
    preserve_disabled: bool,
) {
    if lane == 0u {
        shared_translation = initial_translation;
        shared_linear = select(initial_linear, vec4<f32>(0.0), MOTION_MODEL == 1u);
        let valid = enabled && metadata_value(subset_index, 1u) >= 0.5;
        shared_status = vec2<f32>(select(-1.0, 0.0, valid), 0.0);
    }
    workgroupBarrier();

    let grid_x = i32(subset_index % config.grid_width);
    let grid_y = i32(subset_index / config.grid_width);
    let center = (vec2<f32>(f32(grid_x), f32(grid_y)) + vec2<f32>(0.5))
        * f32(SUBSET_SIZE);
    let half_extent = subset_half_extent();

    for (var iteration = 0u; iteration < 30u; iteration = iteration + 1u) {
        if lane == 0u {
            var inside = true;
            if MOTION_MODEL == 1u {
                let minimum = center - vec2<f32>(0.5 + half_extent) + shared_translation;
                let maximum = center + vec2<f32>(half_extent - 0.5) + shared_translation;
                inside = minimum.x >= 2.0 && minimum.y >= 2.0
                    && maximum.x <= f32(config.width) - 3.0
                    && maximum.y <= f32(config.height) - 3.0;
            } else {
                for (var corner = 0; corner < 4; corner = corner + 1) {
                    let local_x = select(-half_extent, half_extent, corner == 1 || corner == 3);
                    let local_y = select(-half_extent, half_extent, corner >= 2);
                    let local = vec2<f32>(local_x, local_y);
                    let position = center - vec2<f32>(0.5) + local + shared_translation + vec2<f32>(
                        shared_linear.x * local_x + shared_linear.y * local_y,
                        shared_linear.z * local_x + shared_linear.w * local_y,
                    );
                    inside = inside && position.x >= 2.0 && position.y >= 2.0
                        && position.x <= f32(config.width) - 3.0
                        && position.y <= f32(config.height) - 3.0;
                }
            }
            if !inside {
                shared_status.x = -1.0;
            }
        }
        workgroupBarrier();
        let iteration_status = workgroupUniformLoad(&shared_status);
        if iteration >= config.iterations || abs(iteration_status.x) > 0.5 {
            break;
        }

        var sum_current = 0.0;
        var sum_current2 = 0.0;
        var sum_reference_current = 0.0;
        var sum_j_current = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        // Sixteen lanes cover every square size. For 16x16 this is the same
        // per-lane column traversal as the original specialized kernel.
        let subset_pixels = SUBSET_SIZE * SUBSET_SIZE;
        for (var pixel = i32(lane); pixel < subset_pixels; pixel = pixel + 16) {
            let column = pixel % SUBSET_SIZE;
            let row = pixel / SUBSET_SIZE;
            let local_x = f32(column) - half_extent;
            let local_y = f32(row) - half_extent;
            var displacement = shared_translation;
            if MOTION_MODEL == 0u {
                displacement = displacement + vec2<f32>(
                    shared_linear.x * local_x + shared_linear.y * local_y,
                    shared_linear.z * local_x + shared_linear.w * local_y,
                );
            }
            let current = sample_target(
                center - vec2<f32>(0.5) + vec2<f32>(local_x, local_y) + displacement
            );
            let pixel_x = grid_x * SUBSET_SIZE + column;
            let pixel_y = grid_y * SUBSET_SIZE + row;
            let reference_value = reference_data.values[u32(pixel_y * i32(config.width) + pixel_x)];
            let f = reference_value.x;
            let gradient_x = reference_value.y;
            let gradient_y = reference_value.z;
            let gxg = gradient_x * current;
            let gyg = gradient_y * current;
            sum_current = sum_current + current;
            sum_current2 = sum_current2 + current * current;
            sum_reference_current = sum_reference_current + f * current;
            sum_j_current[0] = sum_j_current[0] + gxg;
            sum_j_current[3] = sum_j_current[3] + gyg;
            if MOTION_MODEL == 0u {
                sum_j_current[1] = sum_j_current[1] + gxg * local_x;
                sum_j_current[2] = sum_j_current[2] + gxg * local_y;
                sum_j_current[4] = sum_j_current[4] + gyg * local_x;
                sum_j_current[5] = sum_j_current[5] + gyg * local_y;
            }
        }

        partial_a[lane] = vec4<f32>(sum_current, sum_current2, sum_reference_current, sum_j_current[0]);
        partial_b[lane] = vec4<f32>(sum_j_current[1], sum_j_current[2], sum_j_current[3], sum_j_current[4]);
        partial_c[lane] = vec4<f32>(sum_j_current[5], 0.0, 0.0, 0.0);
        workgroupBarrier();

        var stride = 8u;
        loop {
            if lane < stride {
                partial_a[lane] = partial_a[lane] + partial_a[lane + stride];
                partial_b[lane] = partial_b[lane] + partial_b[lane + stride];
                partial_c[lane] = partial_c[lane] + partial_c[lane + stride];
            }
            workgroupBarrier();
            if stride == 1u {
                break;
            }
            stride = stride / 2u;
        }

        if lane == 0u {
            let totals_a = partial_a[0];
            let totals_b = partial_b[0];
            let totals_c = partial_c[0];
            var reduced_j: array<f32, 6>;
            reduced_j[0] = totals_a.w;
            reduced_j[1] = totals_b.x;
            reduced_j[2] = totals_b.y;
            reduced_j[3] = totals_b.z;
            reduced_j[4] = totals_b.w;
            reduced_j[5] = totals_c.x;

            let current_mean = totals_a.x / subset_pixel_count();
            let current_norm = sqrt(max(0.0, totals_a.y - totals_a.x * current_mean));
            let reference_norm = metadata_value(subset_index, 0u);
            shared_status.y = f32(iteration + 1u);

            if current_norm < 1e-9 || reference_norm < 1e-9 {
                shared_status.x = -2.0;
            } else {
                let ratio = reference_norm / current_norm;
                let delta = solve_increment(subset_index, reduced_j, current_mean, ratio);
                let determinant = increment_determinant(delta);
                if abs(determinant) < 1e-8 {
                    shared_status.x = -1.0;
                } else {
                    let update = compose_inverse_update(
                        shared_translation,
                        shared_linear,
                        delta,
                        determinant,
                    );
                    shared_translation = update.translation;
                    shared_linear = update.linear;
                    let zncc = totals_a.z / (reference_norm * current_norm);
                    shared_status.x = convergence_status(corner_change(delta), zncc);
                }
            }
        }
        workgroupBarrier();
    }

    // New fields need an explicit invalid status for disabled workgroups.
    // A selective retry instead retains their existing results so the median
    // pass can still use neighbours that already converged.
    if lane == 0u && (enabled || !preserve_disabled) {
        state.values[subset_index * 2u] = vec4<f32>(
            shared_translation.x,
            shared_linear.x,
            shared_linear.y,
            shared_translation.y,
        );
        state.values[subset_index * 2u + 1u] = vec4<f32>(
            shared_linear.z,
            shared_linear.w,
            shared_status.x,
            shared_status.y,
        );
    }
}

@compute @workgroup_size(16)
fn solve(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let subset_index = workgroup_id.y * config.grid_width + workgroup_id.x;
    solve_subset(
        subset_index,
        local_id.x,
        vec2<f32>(0.0),
        vec4<f32>(0.0),
        true,
        false,
    );
}

fn solve_from_seed(
    subset_index: u32,
    lane: u32,
    preserve_disabled: bool,
) {
    let seed_a = seed_state.values[subset_index * 2u];
    let seed_b = seed_state.values[subset_index * 2u + 1u];
    solve_subset(
        subset_index,
        lane,
        vec2<f32>(seed_a.x, seed_a.w),
        vec4<f32>(seed_a.y, seed_a.z, seed_b.x, seed_b.y),
        active_mask.values[subset_index] != 0u,
        preserve_disabled,
    );
}

@compute @workgroup_size(16)
fn solve_seeded(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let subset_index = workgroup_id.y * config.grid_width + workgroup_id.x;
    solve_from_seed(subset_index, local_id.x, false);
}

@compute @workgroup_size(16)
fn retry_seeded(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let subset_index = workgroup_id.y * config.grid_width + workgroup_id.x;
    solve_from_seed(subset_index, local_id.x, true);
}

fn median_eight(values: ptr<function, array<f32, 8>>, count: u32) -> f32 {
    for (var phase = 0u; phase < 8u; phase = phase + 1u) {
        for (var index = 0u; index < 7u; index = index + 1u) {
            if index + 1u < count && (*values)[index] > (*values)[index + 1u] {
                let saved = (*values)[index];
                (*values)[index] = (*values)[index + 1u];
                (*values)[index + 1u] = saved;
            }
        }
    }
    if count % 2u == 1u {
        return (*values)[count / 2u];
    }
    return 0.5 * ((*values)[count / 2u - 1u] + (*values)[count / 2u]);
}

@compute @workgroup_size(256)
fn median_filter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let subset_index = global_id.x;
    if subset_index >= config.grid_count {
        return;
    }
    let state_index = subset_index * 2u;
    let own_a = state.values[state_index];
    let own_b = state.values[state_index + 1u];
    filtered_state.values[state_index] = own_a;
    filtered_state.values[state_index + 1u] = own_b;
    if own_b.z < 0.5 {
        return;
    }

    let point_x = i32(subset_index % config.grid_width);
    let point_y = i32(subset_index / config.grid_width);
    let grid_height = i32(config.grid_count / config.grid_width);
    var predicted_x = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var predicted_y = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var parameter_1 = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var parameter_2 = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var parameter_4 = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var parameter_5 = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var count = 0u;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let neighbor_x = point_x + dx;
            let neighbor_y = point_y + dy;
            if neighbor_x < 0 || neighbor_y < 0
                || neighbor_x >= i32(config.grid_width) || neighbor_y >= grid_height {
                continue;
            }
            let neighbor_index = u32(neighbor_y) * config.grid_width + u32(neighbor_x);
            let neighbor_state_index = neighbor_index * 2u;
            let neighbor_a = state.values[neighbor_state_index];
            let neighbor_b = state.values[neighbor_state_index + 1u];
            if neighbor_b.z < 0.5 {
                continue;
            }
            let delta_x = -f32(SUBSET_SIZE) * f32(dx);
            let delta_y = -f32(SUBSET_SIZE) * f32(dy);
            let p1 = neighbor_a.y;
            let p2 = neighbor_a.z;
            let p4 = neighbor_b.x;
            let p5 = neighbor_b.y;
            predicted_x[count] = neighbor_a.x + p1 * delta_x + p2 * delta_y;
            predicted_y[count] = neighbor_a.w + p4 * delta_x + p5 * delta_y;
            parameter_1[count] = p1;
            parameter_2[count] = p2;
            parameter_4[count] = p4;
            parameter_5[count] = p5;
            count = count + 1u;
        }
    }
    if count < 4u {
        return;
    }

    var values_x = predicted_x;
    var values_y = predicted_y;
    let median_x = median_eight(&values_x, count);
    let median_y = median_eight(&values_y, count);
    for (var index = 0u; index < count; index = index + 1u) {
        values_x[index] = abs(predicted_x[index] - median_x);
        values_y[index] = abs(predicted_y[index] - median_y);
    }
    let spread_x = median_eight(&values_x, count);
    let spread_y = median_eight(&values_y, count);
    let flagged = abs(own_a.x - median_x) / (spread_x + 0.1) > 2.0
        || abs(own_a.w - median_y) / (spread_y + 0.1) > 2.0;
    if !flagged {
        return;
    }

    filtered_state.values[state_index] = vec4<f32>(
        median_x,
        median_eight(&parameter_1, count),
        median_eight(&parameter_2, count),
        median_y,
    );
    filtered_state.values[state_index + 1u] = vec4<f32>(
        median_eight(&parameter_4, count),
        median_eight(&parameter_5, count),
        2.0,
        own_b.w,
    );
}

fn subset_half_extent() -> f32 {
    return 0.5 * (f32(SUBSET_SIZE) - 1.0);
}

fn subset_pixel_count() -> f32 {
    return f32(SUBSET_SIZE * SUBSET_SIZE);
}
