struct Params {
    a: u32, b: u32, c: u32, d: u32,
    e: u32, f: u32, g: u32, h: u32,
    i: u32, j: u32, k: u32, l: u32,
    m: u32, n: u32, o: u32, p: u32,
}

struct QueryBuffer { values: array<vec4<u32>> }
struct FloatBuffer { values: array<f32> }
struct Int4Buffer { values: array<vec4<i32>> }
struct Float4Buffer { values: array<vec4<f32>> }
struct SearchState {
    x: i32,
    y: i32,
    score: f32,
    pad: f32,
}
struct SearchStateBuffer { values: array<SearchState> }

@group(0) @binding(0) var source_first: texture_2d<f32>;
@group(0) @binding(1) var target_first: texture_2d<f32>;
@group(0) @binding(2) var source_second: texture_2d<f32>;
@group(0) @binding(3) var target_second: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> atom_queries: QueryBuffer;
@group(0) @binding(7) var<storage, read_write> atom_maps: FloatBuffer;
@group(0) @binding(8) var<uniform> params: Params;

@group(0) @binding(10) var<storage, read> child_maps: FloatBuffer;
@group(0) @binding(11) var<storage, read_write> parent_maps: FloatBuffer;
@group(0) @binding(12) var<storage, read> parent_children: Int4Buffer;
@group(0) @binding(13) var<storage, read> child_info: Int4Buffer;
@group(0) @binding(14) var<storage, read> parent_info: Int4Buffer;
@group(0) @binding(15) var<storage, read> child_weights: Float4Buffer;
@group(0) @binding(16) var<storage, read> border_factors: Float4Buffer;
@group(0) @binding(20) var<storage, read> top_maps: FloatBuffer;
@group(0) @binding(21) var<storage, read> atomic_paths: Int4Buffer;
@group(0) @binding(22) var<storage, read_write> search_states: SearchStateBuffer;
@group(0) @binding(24) var<storage, read> lower_maps: FloatBuffer;

fn descriptor_dot(source_point: vec2<i32>, target_point: vec2<i32>) -> vec3<f32> {
    let source_a = textureLoad(source_first, source_point, 0);
    let target_a = textureLoad(target_first, target_point, 0);
    let source_b = textureLoad(source_second, source_point, 0);
    let target_b = textureLoad(target_second, target_point, 0);
    let dot_value = dot(source_a, target_a) + dot(source_b, target_b) + 0.09;
    let source_norm = dot(source_a, source_a) + dot(source_b, source_b) + 0.09;
    let target_norm = dot(target_a, target_a) + dot(target_b, target_b) + 0.09;
    return vec3<f32>(dot_value, source_norm, target_norm);
}

@compute @workgroup_size(8, 8)
fn correlate_atomic(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let atom = group.z;
    if atom >= params.e || id.x >= params.a || id.y >= params.b { return; }

    let query = atom_queries.values[atom];
    let output_index = atom * params.f + id.y * params.a + id.x;
    if query.x == 0xffffffffu {
        atom_maps.values[output_index] = 0.0;
        return;
    }

    let source_start = vec2<i32>(query.xy);
    let target_center = vec2<i32>(query.zw + id.xy);
    let target_start = target_center - vec2<i32>(2, 2);
    let target_size = textureDimensions(target_first);
    if target_start.x < 0 || target_start.y < 0
        || target_start.x + 4 > i32(target_size.x)
        || target_start.y + 4 > i32(target_size.y) {
        atom_maps.values[output_index] = 0.0;
        return;
    }

    var dot_sum = 0.0;
    var source_sq = 0.0;
    var target_sq = 0.0;
    for (var dy = 0; dy < 4; dy = dy + 1) {
        for (var dx = 0; dx < 4; dx = dx + 1) {
            let v = descriptor_dot(source_start + vec2<i32>(dx, dy),
                target_start + vec2<i32>(dx, dy));
            dot_sum += v.x;
            source_sq += v.y;
            target_sq += v.z;
        }
    }
    let score = dot_sum / max(sqrt(source_sq * target_sq), 1e-8);
    atom_maps.values[output_index] = pow(max(score, 0.0), 1.4);
}

fn pooled_value(child_index: u32, x: i32, y: i32) -> f32 {
    let pool_width = i32(params.c);
    let pool_height = i32(params.d);
    if x < 0 || y < 0 || x >= pool_width || y >= pool_height { return 0.0; }

    let info = child_info.values[child_index];
    let sample_x = clamp(2 * x - info.z, 0, i32(params.a) - 1);
    let sample_y = clamp(2 * y - info.w, 0, i32(params.b) - 1);
    var maximum = 0.0;
    for (var oy = -1; oy <= 1; oy = oy + 1) {
        let py = sample_y + oy;
        if py < 0 || py >= i32(params.b) { continue; }
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let px = sample_x + ox;
            if px < 0 || px >= i32(params.a) { continue; }
            let index = child_index * params.g + u32(py) * params.a + u32(px);
            maximum = max(maximum, child_maps.values[index]);
        }
    }
    return maximum;
}

@compute @workgroup_size(8, 8)
fn aggregate_parent(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let parent = group.z;
    if parent >= params.e || id.x >= params.h || id.y >= params.i { return; }

    let children = parent_children.values[parent];
    let info = parent_info.values[parent];
    let weights = child_weights.values[parent];
    var value = 0.0;
    for (var child_slot = 0; child_slot < 4; child_slot = child_slot + 1) {
        let child = children[child_slot];
        if child < 0 { continue; }

        let dx = (child_slot % 2) * 2 - 1;
        let dy = (child_slot / 2) * 2 - 1;
        let pooled_offset = child_info.values[u32(child)].xy;
        let trans = vec2<i32>(dx, dy) + info.xy - pooled_offset;
        let sampled = pooled_value(u32(child), i32(id.x) + trans.x, i32(id.y) + trans.y);
        value += sampled * weights[child_slot];
    }

    let region_x = select(select(1u, 0u, id.x == 0u), 2u, id.x + 1u == params.h);
    let region_y = select(select(1u, 0u, id.y == 0u), 2u, id.y + 1u == params.i);
    let factors = border_factors.values[parent * 3u + region_y];
    value *= factors[region_x];
    parent_maps.values[parent * params.f + id.y * params.h + id.x] = pow(max(value, 0.0), 1.4);
}

@compute @workgroup_size(64)
fn initialize_topdown(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let candidate = id.x;
    if candidate >= params.e { return; }
    let top_index = candidate / params.c;
    let x = i32(top_index % params.a) + i32(params.f);
    let y = i32(top_index / params.a) + i32(params.g);
    search_states.values[candidate] = SearchState(
        x, y, top_maps.values[top_index], 0.0);
}

@compute @workgroup_size(64)
fn descend_topdown_one_level(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let candidate = id.x;
    if candidate >= params.e { return; }
    let atom = candidate % params.d;
    let path_index = atom * params.g + params.f;
    let path = atomic_paths.values[path_index];
    let state = search_states.values[candidate];
    if state.score <= 0.0 {
        search_states.values[candidate] = SearchState(0, 0, 0.0, 0.0);
        return;
    }

    let slot = path.y;
    let dx = (slot % 2) * 2 - 1;
    let dy = (slot / 2) * 2 - 1;
    let expected_x = state.x * 2 + dx * 2 - path.z;
    let expected_y = state.y * 2 + dy * 2 - path.w;
    let min_x = max(0, expected_x - 1);
    let min_y = max(0, expected_y - 1);
    let max_x = min(i32(params.a), expected_x + 2);
    let max_y = min(i32(params.b), expected_y + 2);

    var best_score = 0.0;
    var best_x = 0;
    var best_y = 0;
    for (var py = min_y; py < max_y; py = py + 1) {
        for (var px = min_x; px < max_x; px = px + 1) {
            let map_index = u32(path.x) * params.c + u32(py) * params.a + u32(px);
            let value = lower_maps.values[map_index];
            if value > best_score {
                best_score = value;
                best_x = px;
                best_y = py;
            }
        }
    }
    if best_score <= 0.0 {
        search_states.values[candidate] = SearchState(0, 0, 0.0, 0.0);
    } else {
        search_states.values[candidate] = SearchState(
            best_x + path.z, best_y + path.w, state.score + best_score, 0.0);
    }
}
