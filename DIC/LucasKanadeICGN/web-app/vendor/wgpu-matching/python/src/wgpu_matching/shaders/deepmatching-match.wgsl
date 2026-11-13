struct Params {
    a: u32, b: u32, c: u32, d: u32,
    e: u32, f: u32, g: u32, h: u32,
    i: u32, j: u32, k: u32, l: u32,
    m: u32, n: u32, o: u32, p: u32,
}

struct QueryBuffer { values: array<vec4<u32>> }
struct FloatBuffer { values: array<f32> }
struct ResultBuffer { values: array<vec4<f32>> }

@group(0) @binding(0) var source_first: texture_2d<f32>;
@group(0) @binding(1) var target_first: texture_2d<f32>;
@group(0) @binding(2) var source_second: texture_2d<f32>;
@group(0) @binding(3) var target_second: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> queries: QueryBuffer;
@group(0) @binding(7) var<storage, read_write> response_maps: FloatBuffer;
@group(0) @binding(8) var<uniform> params: Params;
@group(0) @binding(9) var<storage, read_write> results: ResultBuffer;

fn descriptor_dot(source_point: vec2<i32>, target_point: vec2<i32>) -> f32 {
    let source_a = textureLoad(source_first, source_point, 0);
    let target_a = textureLoad(target_first, target_point, 0);
    let source_b = textureLoad(source_second, source_point, 0);
    let target_b = textureLoad(target_second, target_point, 0);
    // The INRIA ninth HOG component is spatially constant (0.3) by default.
    return dot(source_a, target_a) + dot(source_b, target_b) + 0.09;
}

// Direct score of a complete seed patch at each target translation. This keeps
// the patch's spatial support together instead of reducing all cell maps to a
// single coarse root map.
@compute @workgroup_size(8, 8)
fn correlate_patches(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let query_index = group.z;
    if query_index >= params.f || id.x >= params.a || id.y >= params.b { return; }

    let query = queries.values[query_index];
    let source_origin = vec2<i32>(query.xy);
    let target_origin = vec2<i32>(query.zw) + vec2<i32>(id.xy);
    let radius = i32(params.h);
    if radius > 0 && any(abs(target_origin - source_origin) > vec2<i32>(radius)) {
        let output_index = query_index * params.g + id.y * params.a + id.x;
        response_maps.values[output_index] = -1.0;
        return;
    }
    var score = 0.0;
    for (var dy = 0u; dy < params.e; dy = dy + 1u) {
        for (var dx = 0u; dx < params.e; dx = dx + 1u) {
            let offset = vec2<i32>(i32(dx), i32(dy));
            score += descriptor_dot(source_origin + offset, target_origin + offset);
        }
    }
    let output_index = query_index * params.g + id.y * params.a + id.x;
    response_maps.values[output_index] = score / f32(params.e * params.e);
}

var<workgroup> best_scores: array<f32, 64>;
var<workgroup> best_indices: array<u32, 64>;

@compute @workgroup_size(64)
fn select_best_patch(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let query_index = group.x;
    if query_index >= params.e { return; }
    let area = params.a * params.b;
    let base = query_index * params.d + params.c;
    var local_score = -1.0;
    var local_index = 0xffffffffu;
    for (var candidate = lane; candidate < area; candidate = candidate + 64u) {
        let score = response_maps.values[base + candidate];
        if score > local_score || (score == local_score && candidate < local_index) {
            local_score = score;
            local_index = candidate;
        }
    }
    best_scores[lane] = local_score;
    best_indices[lane] = local_index;
    workgroupBarrier();

    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if lane < stride {
            let other = lane + stride;
            if best_scores[other] > best_scores[lane]
                || (best_scores[other] == best_scores[lane]
                    && best_indices[other] < best_indices[lane]) {
                best_scores[lane] = best_scores[other];
                best_indices[lane] = best_indices[other];
            }
        }
        workgroupBarrier();
    }

    let winner_score = best_scores[0];
    let winner = best_indices[0];
    let winner_x = winner % params.a;
    let winner_y = winner / params.a;
    local_score = -1.0;
    local_index = 0xffffffffu;
    for (var candidate = lane; candidate < area; candidate = candidate + 64u) {
        let x = candidate % params.a;
        let y = candidate / params.a;
        let dx = abs(i32(x) - i32(winner_x));
        let dy = abs(i32(y) - i32(winner_y));
        if dx <= i32(params.f) && dy <= i32(params.f) { continue; }
        let score = response_maps.values[base + candidate];
        if score > local_score || (score == local_score && candidate < local_index) {
            local_score = score;
            local_index = candidate;
        }
    }
    best_scores[lane] = local_score;
    best_indices[lane] = local_index;
    workgroupBarrier();

    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if lane < stride {
            let other = lane + stride;
            if best_scores[other] > best_scores[lane]
                || (best_scores[other] == best_scores[lane]
                    && best_indices[other] < best_indices[lane]) {
                best_scores[lane] = best_scores[other];
                best_indices[lane] = best_indices[other];
            }
        }
        workgroupBarrier();
    }

    if lane == 0u {
        let second_score = best_scores[0];
        let center_index = winner_y * params.a + winner_x;
        var subpixel_x = 0.0;
        var subpixel_y = 0.0;
        if winner_x > 0u && winner_x + 1u < params.a {
            let left_score = response_maps.values[base + center_index - 1u];
            let right_score = response_maps.values[base + center_index + 1u];
            let denominator = left_score - 2.0 * winner_score + right_score;
            if denominator < -1e-6 {
                subpixel_x = clamp(0.5 * (left_score - right_score) / denominator, -0.5, 0.5);
            }
        }
        if winner_y > 0u && winner_y + 1u < params.b {
            let up_score = response_maps.values[base + center_index - params.a];
            let down_score = response_maps.values[base + center_index + params.a];
            let denominator = up_score - 2.0 * winner_score + down_score;
            if denominator < -1e-6 {
                subpixel_y = clamp(0.5 * (up_score - down_score) / denominator, -0.5, 0.5);
            }
        }
        results.values[query_index] = vec4<f32>(
            f32(winner_x) + subpixel_x, f32(winner_y) + subpixel_y,
            winner_score, second_score
        );
    }
}
