struct Params {
    grid_width: u32,
    grid_height: u32,
    descriptor_width: u32,
    descriptor_height: u32,
    patch_size: u32,
    radius: i32,
    seed: u32,
    iteration: u32,
    random_radius: u32,
    random_trials: u32,
    initialize: u32,
    scale_previous: u32,
    initial_hypotheses: u32,
    padding_2: u32,
    padding_3: u32,
}
struct PointBuffer { values: array<vec2<i32>> }
struct State { dx: i32, dy: i32, cost: f32, valid: u32 }
struct StateBuffer { values: array<State> }

@group(0) @binding(0) var source_first: texture_2d<f32>;
@group(0) @binding(1) var source_second: texture_2d<f32>;
@group(0) @binding(2) var target_first: texture_2d<f32>;
@group(0) @binding(3) var target_second: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> points: PointBuffer;
@group(0) @binding(5) var<storage, read> previous_states: StateBuffer;
@group(0) @binding(6) var<storage, read_write> next_states: StateBuffer;
@group(0) @binding(7) var<uniform> params: Params;

fn hash32(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn random_signed(key: u32, radius: u32) -> i32 {
    if radius == 0u { return 0; }
    let span = radius * 2u + 1u;
    return i32(hash32(key) % span) - i32(radius);
}

fn stratified_signed(key: u32, radius: u32, trial: u32, trial_count: u32, axis: u32) -> i32 {
    let span = radius * 2u + 1u;
    let side = u32(ceil(sqrt(f32(trial_count))));
    let cell = select(trial / side, trial % side, axis == 1u);
    let first = (cell * span) / side;
    let last = ((cell + 1u) * span) / side;
    let cell_span = max(1u, last - first);
    return i32(first + hash32(key ^ (axis * 0x9e3779b9u)) % cell_span) - i32(radius);
}

fn evaluate(index: u32, dx: i32, dy: i32) -> State {
    if dx < -params.radius || dx > params.radius || dy < -params.radius || dy > params.radius {
        return State(dx, dy, 2.0, 0u);
    }
    let center = points.values[index];
    let half = i32(params.patch_size / 2u);
    let source_min = center - vec2<i32>(half);
    let target_center = center + vec2<i32>(dx, dy);
    let target_min = target_center - vec2<i32>(half);
    let max_point = vec2<i32>(i32(params.descriptor_width), i32(params.descriptor_height));
    if any(source_min < vec2<i32>(0)) || any(target_min < vec2<i32>(0))
        || any(source_min + vec2<i32>(i32(params.patch_size)) > max_point)
        || any(target_min + vec2<i32>(i32(params.patch_size)) > max_point) {
        return State(dx, dy, 2.0, 0u);
    }

    var dot_sum = 0.0;
    var source_norm = 0.0;
    var target_norm = 0.0;
    for (var y = 0; y < i32(params.patch_size); y = y + 1) {
        for (var x = 0; x < i32(params.patch_size); x = x + 1) {
            let offset = vec2<i32>(x, y);
            let a0 = textureLoad(source_first, source_min + offset, 0);
            let a1 = textureLoad(source_second, source_min + offset, 0);
            let b0 = textureLoad(target_first, target_min + offset, 0);
            let b1 = textureLoad(target_second, target_min + offset, 0);
            dot_sum += dot(a0, b0) + dot(a1, b1);
            source_norm += dot(a0, a0) + dot(a1, a1);
            target_norm += dot(b0, b0) + dot(b1, b1);
        }
    }
    let similarity = dot_sum / max(sqrt(source_norm * target_norm), 1e-12);
    return State(dx, dy, 1.0 - clamp(similarity, 0.0, 1.0), 1u);
}

fn consider(best: State, candidate: State) -> State {
    if candidate.valid != 0u && candidate.cost < best.cost { return candidate; }
    return best;
}

@compute @workgroup_size(64)
fn patchmatch_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = params.grid_width * params.grid_height;
    if index >= count { return; }

    var best = State(0, 0, 2.0, 0u);
    var key = hash32(index ^ params.seed ^ (params.iteration * 0x9e3779b9u));
    if params.initialize != 0u {
        best = consider(best, evaluate(index, 0, 0));
    } else {
        let multiplier = select(1, 2, params.scale_previous != 0u);
        let inherited = previous_states.values[index];
        best = consider(best, evaluate(index, inherited.dx * multiplier, inherited.dy * multiplier));
        let gx = i32(index % params.grid_width);
        let gy = i32(index / params.grid_width);
        if gx > 0 {
            let neighbor = previous_states.values[index - 1u];
            best = consider(best, evaluate(index, neighbor.dx * multiplier, neighbor.dy * multiplier));
        }
        if gx + 1 < i32(params.grid_width) {
            let neighbor = previous_states.values[index + 1u];
            best = consider(best, evaluate(index, neighbor.dx * multiplier, neighbor.dy * multiplier));
        }
        if gy > 0 {
            let neighbor = previous_states.values[index - params.grid_width];
            best = consider(best, evaluate(index, neighbor.dx * multiplier, neighbor.dy * multiplier));
        }
        if gy + 1 < i32(params.grid_height) {
            let neighbor = previous_states.values[index + params.grid_width];
            best = consider(best, evaluate(index, neighbor.dx * multiplier, neighbor.dy * multiplier));
        }
    }

    let current_radius = select(params.random_radius, u32(max(params.radius, 0)), params.initialize != 0u);
    let trial_count = select(params.random_trials, params.initial_hypotheses, params.initialize != 0u);
    for (var trial = 0u; trial < trial_count; trial = trial + 1u) {
        key = hash32(key + trial + params.seed + 0x9e3779b9u);
        if params.initialize != 0u {
            let candidate = vec2<i32>(
                stratified_signed(key, current_radius, trial, trial_count, 0u),
                stratified_signed(hash32(key ^ 0xa511e9b3u), current_radius, trial,
                    trial_count, 1u));
            best = consider(best, evaluate(index, candidate.x, candidate.y));
        } else {
            let candidate = vec2<i32>(best.dx, best.dy) + vec2<i32>(
                random_signed(key, current_radius),
                random_signed(hash32(key ^ 0xa511e9b3u), current_radius));
            best = consider(best, evaluate(index, candidate.x, candidate.y));
        }
    }
    next_states.values[index] = best;
}

@compute @workgroup_size(64)
fn evaluate_oracle(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = params.grid_width * params.grid_height;
    if index >= count { return; }
    let candidate = previous_states.values[index];
    next_states.values[index] = evaluate(index, candidate.dx, candidate.dy);
}
