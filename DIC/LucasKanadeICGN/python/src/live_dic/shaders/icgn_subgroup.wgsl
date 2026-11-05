// Affine IC-GN specialization for native devices with subgroup support.
//
// This is a direct WGSL port of the former GLSL/SPIR-V specialization.  Its
// intentionally narrow contract is affine motion, cubic B-spline sampling,
// 16 x 16 non-overlapping subsets, and uniform spatial weighting.  The
// portable shader remains the source for every other configuration.

struct Config {
    width: u32,
    height: u32,
    grid_width: u32,
    grid_count: u32,
    iterations: u32,
    pixel_count: u32,
    tolerance: f32,
    zncc_min: f32,
    inverse_offset: u32, // vec4 offset after metadata and optional ROI tables
    padding0: u32,
    padding1: u32,
    padding2: u32,
}

struct FloatBuffer {
    values: array<f32>,
}

struct Float4Buffer {
    values: array<vec4<f32>>,
}

@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read_write> temporary: FloatBuffer;
@group(0) @binding(4) var<storage, read> reference_data: Float4Buffer;
@group(0) @binding(6) var<storage, read> metadata: Float4Buffer;
@group(0) @binding(7) var<storage, read_write> state: Float4Buffer;

const SUBSET_SIZE: i32 = 16;
const SUBSET_PIXELS: f32 = 256.0;
const SUBSET_HALF_EXTENT: f32 = 7.5;

override ROI_ENABLED: u32 = 0u;

fn metadata_value(subset_index: u32, component: u32) -> f32 {
    return metadata.values[subset_index * 4u + component / 4u][component % 4u];
}

fn hessian_value(subset_index: u32, component: u32) -> f32 {
    return metadata.values[config.inverse_offset + subset_index * 9u + component / 4u][component % 4u];
}

fn cubic_weights(value: f32) -> vec4<f32> {
    let value2 = value * value;
    let value3 = value2 * value;
    let inverse = 1.0 - value;
    return vec4<f32>(
        inverse * inverse * inverse / 6.0,
        (3.0 * value3 - 6.0 * value2 + 4.0) / 6.0,
        (-3.0 * value3 + 3.0 * value2 + 3.0 * value + 1.0) / 6.0,
        value3 / 6.0,
    );
}

fn sample_cubic(position: vec2<f32>) -> f32 {
    let floored = floor(position);
    let base = vec2<i32>(floored) - vec2<i32>(1);
    let fraction = position - floored;
    let weight_x = cubic_weights(fraction.x);
    let weight_y = cubic_weights(fraction.y);
    var result = 0.0;
    for (var iy = 0; iy < 4; iy += 1) {
        for (var ix = 0; ix < 4; ix += 1) {
            let x = base.x + ix;
            let y = base.y + iy;
            let index = u32(y * i32(config.width) + x);
            result += temporary.values[index] * weight_x[ix] * weight_y[iy];
        }
    }
    return result;
}

@compute @workgroup_size(16)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let subset_index = workgroup_id.y * config.grid_width + workgroup_id.x;
    let lane = local_id.x;
    if subset_index >= config.grid_count {
        return;
    }

    var current_translation = vec2<f32>(0.0);
    var current_linear = vec4<f32>(0.0);
    var rejected_status = -1.0;
    if ROI_ENABLED != 0u {
        rejected_status = min(-1.0, metadata_value(subset_index, 1u));
    }
    var current_status = select(rejected_status, 0.0, metadata_value(subset_index, 1u) >= 0.5);
    var current_iterations = 0.0;
    let grid_x = i32(subset_index % config.grid_width);
    let grid_y = i32(subset_index / config.grid_width);
    let center = (vec2<f32>(f32(grid_x), f32(grid_y)) + vec2<f32>(0.5)) * 16.0;

    for (var iteration = 0u; iteration < 30u; iteration += 1u) {
        if iteration >= config.iterations || abs(current_status) > 0.5 {
            break;
        }

        var inside = true;
        for (var corner = 0; corner < 4; corner += 1) {
            let local_x = select(-SUBSET_HALF_EXTENT, SUBSET_HALF_EXTENT, corner == 1 || corner == 3);
            let local_y = select(-SUBSET_HALF_EXTENT, SUBSET_HALF_EXTENT, corner >= 2);
            let local = vec2<f32>(local_x, local_y);
            let position = center - vec2<f32>(0.5) + local + current_translation + vec2<f32>(
                current_linear.x * local_x + current_linear.y * local_y,
                current_linear.z * local_x + current_linear.w * local_y,
            );
            inside = inside
                && position.x >= 2.0
                && position.y >= 2.0
                && position.x <= f32(config.width) - 3.0
                && position.y <= f32(config.height) - 3.0;
        }
        if !inside {
            current_status = -1.0;
            break;
        }

        var sum_current = 0.0;
        var sum_current2 = 0.0;
        var sum_reference_current = 0.0;
        var sum_j_current = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let local_x = f32(lane) - SUBSET_HALF_EXTENT;
        for (var row = 0; row < SUBSET_SIZE; row += 1) {
            let local_y = f32(row) - SUBSET_HALF_EXTENT;
            let displacement = current_translation + vec2<f32>(
                current_linear.x * local_x + current_linear.y * local_y,
                current_linear.z * local_x + current_linear.w * local_y,
            );
            let current = sample_cubic(
                center - vec2<f32>(0.5) + vec2<f32>(local_x, local_y) + displacement,
            );
            let pixel_x = grid_x * SUBSET_SIZE + i32(lane);
            let pixel_y = grid_y * SUBSET_SIZE + row;
            let reference_value = reference_data.values[
                u32(pixel_y * i32(config.width) + pixel_x)
            ];
            let reference = reference_value.x;
            let gradient_x = reference_value.y;
            let gradient_y = reference_value.z;
            let gxg = gradient_x * current;
            let gyg = gradient_y * current;
            sum_current += current;
            sum_current2 += current * current;
            sum_reference_current += reference * current;
            sum_j_current[0] += gxg;
            sum_j_current[1] += gxg * local_x;
            sum_j_current[2] += gxg * local_y;
            sum_j_current[3] += gyg;
            sum_j_current[4] += gyg * local_x;
            sum_j_current[5] += gyg * local_y;
        }

        let totals_a = subgroupAdd(vec4<f32>(
            sum_current,
            sum_current2,
            sum_reference_current,
            sum_j_current[0],
        ));
        let totals_b = subgroupAdd(vec4<f32>(
            sum_j_current[1],
            sum_j_current[2],
            sum_j_current[3],
            sum_j_current[4],
        ));
        let totals_c = subgroupAdd(vec4<f32>(sum_j_current[5], 0.0, 0.0, 0.0));
        let reduced_j = array<f32, 6>(
            totals_a.w,
            totals_b.x,
            totals_b.y,
            totals_b.z,
            totals_b.w,
            totals_c.x,
        );

        let current_mean = totals_a.x / SUBSET_PIXELS;
        let current_norm = sqrt(max(0.0, totals_a.y - totals_a.x * current_mean));
        let reference_norm = metadata_value(subset_index, 0u);
        current_iterations = f32(iteration + 1u);
        if current_norm < 1e-9 || reference_norm < 1e-9 {
            current_status = -2.0;
            break;
        }

        let ratio = reference_norm / current_norm;
        var residual: array<f32, 6>;
        for (var component = 0u; component < 6u; component += 1u) {
            residual[component] = metadata_value(subset_index, 2u + component)
                - ratio * (
                    reduced_j[component]
                    - current_mean * metadata_value(subset_index, 8u + component)
                );
        }

        var delta: array<f32, 6>;
        for (var hessian_row = 0u; hessian_row < 6u; hessian_row += 1u) {
            var value = 0.0;
            for (var column = 0u; column < 6u; column += 1u) {
                value += hessian_value(subset_index, 6u * hessian_row + column)
                    * residual[column];
            }
            delta[hessian_row] = -value;
        }

        let da11 = 1.0 + delta[1];
        let da12 = delta[2];
        let da21 = delta[4];
        let da22 = 1.0 + delta[5];
        let determinant = da11 * da22 - da12 * da21;
        if abs(determinant) < 1e-8 {
            current_status = -1.0;
            break;
        }

        let inverse11 = da22 / determinant;
        let inverse12 = -da12 / determinant;
        let inverse21 = -da21 / determinant;
        let inverse22 = da11 / determinant;
        let warp11 = 1.0 + current_linear.x;
        let warp12 = current_linear.y;
        let warp21 = current_linear.z;
        let warp22 = 1.0 + current_linear.w;
        let inverse_translation_x = -(inverse11 * delta[0] + inverse12 * delta[3]);
        let inverse_translation_y = -(inverse21 * delta[0] + inverse22 * delta[3]);
        let updated_translation = current_translation + vec2<f32>(
            warp11 * inverse_translation_x + warp12 * inverse_translation_y,
            warp21 * inverse_translation_x + warp22 * inverse_translation_y,
        );
        let updated11 = warp11 * inverse11 + warp12 * inverse21;
        let updated12 = warp11 * inverse12 + warp12 * inverse22;
        let updated21 = warp21 * inverse11 + warp22 * inverse21;
        let updated22 = warp21 * inverse12 + warp22 * inverse22;

        var maximum_corner_change = 0.0;
        for (var corner = 0; corner < 4; corner += 1) {
            let x = select(-SUBSET_HALF_EXTENT, SUBSET_HALF_EXTENT, corner == 1 || corner == 3);
            let y = select(-SUBSET_HALF_EXTENT, SUBSET_HALF_EXTENT, corner >= 2);
            let change = length(vec2<f32>(
                delta[0] + delta[1] * x + delta[2] * y,
                delta[3] + delta[4] * x + delta[5] * y,
            ));
            maximum_corner_change = max(maximum_corner_change, change);
        }

        current_translation = updated_translation;
        current_linear = vec4<f32>(
            updated11 - 1.0,
            updated12,
            updated21,
            updated22 - 1.0,
        );
        if maximum_corner_change < config.tolerance {
            let zncc = totals_a.z / (reference_norm * current_norm);
            current_status = select(-2.0, 1.0, zncc >= config.zncc_min);
        }
    }

    if lane == 0u {
        let state_index = subset_index * 2u;
        state.values[state_index] = vec4<f32>(
            current_translation.x,
            current_linear.x,
            current_linear.y,
            current_translation.y,
        );
        state.values[state_index + 1u] = vec4<f32>(
            current_linear.z,
            current_linear.w,
            current_status,
            current_iterations,
        );
    }
}
