// Experimental translation seed selection, not a replacement measurement.
// Score hypotheses on the destination patch; compact only improved seeds.
// One immutable snapshot per pass gives Jacobi propagation across workgroups.
struct Floats { values: array<vec4<f32>> }
struct UintBuffer { values: array<u32> }
struct WorkList {
    count: atomic<u32>, x: u32, y: u32, z: u32, indices: array<u32>,
}
struct Config {
    width: u32, height: u32, grid_width: u32, count: u32,
    size: u32, stride: u32, sample_step: u32, padding: u32,
    min_gain: f32, min_displacement: f32, padding2: f32, padding3: f32,
}
@group(0) @binding(0) var reference: texture_2d<f32>;
@group(0) @binding(1) var target_image: texture_2d<f32>;
@group(0) @binding(2) var<uniform> config: Config;
@group(0) @binding(3) var<storage, read> snapshot: Floats;
// Includes own bit 4 iff ROI-selected AND reference numerically admissible.
@group(0) @binding(4) var<storage, read> permissions: UintBuffer;
@group(0) @binding(5) var<storage, read_write> seeds: Floats;
@group(0) @binding(6) var<storage, read_write> work: WorkList;
@group(0) @binding(7) var<storage, read_write> scores: Floats;
@group(0) @binding(8) var<storage, read> prior: Floats;
// Separate from the writable work list, so it may supply indirect arguments
// while choose writes its new work list in the same dispatch.
@group(0) @binding(9) var<storage, read_write> continuation: UintBuffer;
override FAILED_ONLY: bool = false;
// Optional exact memoization inside one frame/level. The first pass resets
// every visited cache slot; subsequent passes reuse a score only for the same
// candidate displacement. Cache records follow the public diagnostic records
// in scores: count vec4 diagnostics, then 11 vec4 records per subset.
override CACHE_SCORES: bool = false;
override REUSE_SCORES: bool = false;
// 0 neighbours, 1 median hypothesis only, 2 neighbours plus median.
override CANDIDATE_MODE: u32 = 0u;
var<workgroup> candidates: array<vec4<f32>, 11>;
var<workgroup> sums: array<vec4<f32>, 64>;
var<workgroup> sum_g2: array<f32, 64>;
var<workgroup> best: vec4<f32>; // dx, dy, rho, own_rho

fn median(v: ptr<function, array<f32, 9>>, n: u32) -> f32 {
    for (var i = 1u; i < n; i++) {
        let value = (*v)[i];
        var j = i;
        loop {
            if j == 0u { break; }
            if (*v)[j - 1u] <= value { break; }
            (*v)[j] = (*v)[j - 1u]; j--;
        }
        (*v)[j] = value;
    }
    return 0.5 * ((*v)[(n - 1u) / 2u] + (*v)[n / 2u]);
}

fn linear_sample(position: vec2<f32>) -> f32 {
    let p = vec2<i32>(floor(position));
    let d = fract(position);
    let a = textureLoad(target_image, p, 0).x;
    let b = textureLoad(target_image, p + vec2<i32>(1, 0), 0).x;
    let c = textureLoad(target_image, p + vec2<i32>(0, 1), 0).x;
    let e = textureLoad(target_image, p + vec2<i32>(1, 1), 0).x;
    return 255.0 * mix(mix(a, b, d.x), mix(c, e, d.x), d.y) - 128.0;
}

@compute @workgroup_size(64)
fn choose(@builtin(workgroup_id) group: vec3<u32>,
          @builtin(local_invocation_index) lane: u32) {
    let index = group.y * config.grid_width + group.x;
    if index >= config.count { return; }
    if (permissions.values[index] & 16u) == 0u { return; }
    if FAILED_ONLY && snapshot.values[2u * index + 1u].z >= 0.5 { return; }
    let own = snapshot.values[2u * index].xw;
    if lane == 0u {
        for (var k = 0u; k < 11u; k++) {
            candidates[k] = vec4<f32>(0.0);
            if CACHE_SCORES && !REUSE_SCORES {
                scores.values[config.count + index * 11u + k] = vec4<f32>(0.0);
            }
        }
        candidates[0] = vec4<f32>(own, 1.0, 0.0);
        candidates[1] = vec4<f32>(prior.values[2u * index].xw, 1.0, 0.0);
        var us: array<f32, 9>;
        var vs: array<f32, 9>;
        var n = 0u;
        var slot = 2u;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let bit = u32((dy + 1) * 3 + dx + 1);
                let x = i32(group.x) + dx;
                let y = i32(group.y) + dy;
                let inside = x >= 0 && y >= 0 && x < i32(config.grid_width)
                    && y < i32(config.count / config.grid_width);
                if inside && (permissions.values[index] & (1u << bit)) != 0u {
                    let neighbour = u32(y) * config.grid_width + u32(x);
                    if (permissions.values[neighbour] & 16u) != 0u {
                        let p = snapshot.values[2u * neighbour].xw;
                        if dx != 0 || dy != 0 {
                            if CANDIDATE_MODE != 1u {
                                candidates[slot] = vec4<f32>(p, 1.0, 0.0);
                            }
                        }
                        if snapshot.values[2u * neighbour + 1u].z >= 0.5 {
                            us[n] = p.x; vs[n] = p.y; n++;
                        }
                    }
                }
                if dx != 0 || dy != 0 { slot++; }
            }
        }
        if CANDIDATE_MODE != 0u && n > 0u {
            candidates[10] = vec4<f32>(median(&us, n), median(&vs, n), 1.0, 0.0);
        }
        // Piggyback exact cache keys/scores on the existing candidate broadcast.
        // z=2 means reuse w; no extra per-candidate barrier is necessary.
        if CACHE_SCORES && REUSE_SCORES {
            for (var k = 0u; k < 11u; k++) {
                let previous = scores.values[config.count + index * 11u + k];
                if candidates[k].z != 0.0 && previous.w > 0.5 && all(previous.xy == candidates[k].xy) {
                    candidates[k] = vec4<f32>(candidates[k].xy, 2.0, previous.z);
                }
            }
        }
        best = vec4<f32>(own, -2.0, -2.0);
    }
    workgroupBarrier();
    let origin = vec2<f32>(group.xy * config.stride);
    let side = (config.size + config.sample_step - 1u) / config.sample_step;
    let total = side * side;
    for (var k = 0u; k < 11u; k++) {
        let candidate = workgroupUniformLoad(&candidates[k]);
        let p = candidate.xy;
        let minimum = origin + p;
        let maximum = minimum + vec2<f32>(f32(config.size - 1u));
        if candidate.z == 0.0 || !all(abs(p) <= vec2<f32>(1e12))
            // Match the existing IC-GN image-support guard. An initial warp
            // outside it is rejected before the first solver iteration.
            || any(minimum < vec2<f32>(2.0))
            || any(maximum > vec2<f32>(f32(config.width - 3u), f32(config.height - 3u))) {
            continue;
        }
        if CACHE_SCORES && REUSE_SCORES && candidate.z == 2.0 {
            if lane == 0u {
                let rho = candidate.w;
                if k == 0u { best.w = rho; }
                if rho > best.z { best = vec4<f32>(p, rho, best.w); }
            }
            workgroupBarrier();
            continue;
        }
        var s = vec4<f32>(0.0);
        var g2 = 0.0;
        for (var pixel = lane; pixel < total; pixel += 64u) {
            let offset = vec2<f32>(f32(pixel % side), f32(pixel / side)) * f32(config.sample_step);
            let f = 255.0 * textureLoad(reference, vec2<i32>(origin + offset), 0).x - 128.0;
            let g = linear_sample(origin + offset + p);
            s += vec4<f32>(f, g, f * g, f * f);
            g2 += g * g;
        }
        sums[lane] = s; sum_g2[lane] = g2;
        workgroupBarrier();
        for (var offset = 32u; offset > 0u; offset /= 2u) {
            if lane < offset { sums[lane] += sums[lane + offset]; sum_g2[lane] += sum_g2[lane + offset]; }
            workgroupBarrier();
        }
        if lane == 0u {
            let t = sums[0];
            let vf = max(0.0, t.w - t.x * t.x / f32(total));
            let vg = max(0.0, sum_g2[0] - t.y * t.y / f32(total));
            var rho = -2.0;
            if vf > 1e-9 && vg > 1e-9 { rho = (t.z - t.x * t.y / f32(total)) / sqrt(vf * vg); }
            if CACHE_SCORES {
                scores.values[config.count + index * 11u + k] = vec4<f32>(p, rho, 1.0);
            }
            if k == 0u { best.w = rho; }
            if rho > best.z { best = vec4<f32>(p, rho, best.w); }
        }
        workgroupBarrier();
    }
    if lane == 0u {
        let improved = best.z > best.w + config.min_gain
            && length(best.xy - own) > config.min_displacement;
        scores.values[index] = vec4<f32>(best.w, best.z, select(0.0, 1.0, improved), 0.0);
        if improved {
            seeds.values[2u * index] = vec4<f32>(best.x, 0.0, 0.0, best.y);
            seeds.values[2u * index + 1u] = vec4<f32>(0.0);
            let ordinal = atomicAdd(&work.count, 1u);
            work.indices[ordinal] = index;
        }
    }
}

@compute @workgroup_size(1)
fn finalize() {
    let count = atomicLoad(&work.count);
    work.x = min(count, 65535u);
    work.y = max(1u, (count + 65534u) / 65535u);
    work.z = 1u;
}

// An empty work list means that the state will not change. Later passes then
// have exactly the same candidates and cannot produce new work.
@compute @workgroup_size(1)
fn finalize_chain() {
    let count = atomicLoad(&work.count);
    work.x = min(count, 65535u);
    work.y = max(1u, (count + 65534u) / 65535u);
    work.z = 1u;
    continuation.values[0] = select(0u, config.grid_width, count > 0u);
    continuation.values[1] = config.count / config.grid_width;
    continuation.values[2] = 1u;
}

// An accepted original measurement is retained if its trial re-solve fails.
@compute @workgroup_size(256)
fn retain_accepted(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= config.count { return; }
    if snapshot.values[2u * index + 1u].z >= 0.5 && seeds.values[2u * index + 1u].z < 0.5 {
        seeds.values[2u * index] = snapshot.values[2u * index];
        seeds.values[2u * index + 1u] = snapshot.values[2u * index + 1u];
    }
}

// A separate initialization-only control. Unlike the measurement median,
// unfinished (status 0) coarse estimates can contribute. An invalid own seed
// may be initialized from >=4 admissible neighbours, never reported as solved.
@compute @workgroup_size(64)
fn median_seed(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= config.count { return; }
    let a = snapshot.values[2u * index];
    let b = snapshot.values[2u * index + 1u];
    seeds.values[2u * index] = a;
    seeds.values[2u * index + 1u] = b;
    if (permissions.values[index] & 16u) == 0u { return; }
    let x = i32(index % config.grid_width);
    let y = i32(index / config.grid_width);
    var us: array<f32, 9>;
    var vs: array<f32, 9>;
    var n = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx = x + dx;
            let ny = y + dy;
            let bit = u32((dy + 1) * 3 + dx + 1);
            if nx < 0 || ny < 0 || nx >= i32(config.grid_width)
                || ny >= i32(config.count / config.grid_width) { continue; }
            let neighbour = u32(ny) * config.grid_width + u32(nx);
            if (permissions.values[index] & (1u << bit)) == 0u
                || (permissions.values[neighbour] & 16u) == 0u { continue; }
            if snapshot.values[2u * neighbour + 1u].z < -0.5 { continue; }
            let p = snapshot.values[2u * neighbour].xw;
            if !all(abs(p) <= vec2<f32>(1e12)) { continue; }
            us[n] = p.x; vs[n] = p.y; n++;
        }
    }
    if n >= 4u {
        seeds.values[2u * index] = vec4<f32>(median(&us, n), 0.0, 0.0, median(&vs, n));
        // Mark seed availability only in this private initialization buffer.
        // Raw coarse measurements/statuses remain untouched.
        seeds.values[2u * index + 1u] = vec4<f32>(0.0, 0.0, max(b.z, 0.0), b.w);
    }
}

// Opt-in affine frontier initialization. Each work item uses the immutable
// previous-pass snapshot and the median of first-order 8-neighbour transports.
// It produces seeds and a compact work list, never accepted measurements.
@compute @workgroup_size(64)
fn affine_frontier(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= config.count || (permissions.values[index] & 16u) == 0u { return; }
    if snapshot.values[2u * index + 1u].z >= 0.5 { return; }
    let gx = i32(index % config.grid_width);
    let gy = i32(index / config.grid_width);
    var t0: array<f32, 9>;
    var g0: array<f32, 9>;
    var g1: array<f32, 9>;
    var t1: array<f32, 9>;
    var g2: array<f32, 9>;
    var g3: array<f32, 9>;
    var count = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; }
            let x = gx + dx;
            let y = gy + dy;
            let bit = u32((dy + 1) * 3 + dx + 1);
            if x < 0 || y < 0 || x >= i32(config.grid_width)
                || y >= i32(config.count / config.grid_width)
                || (permissions.values[index] & (1u << bit)) == 0u { continue; }
            let neighbor = u32(y) * config.grid_width + u32(x);
            let p = snapshot.values[2u * neighbor];
            let q = snapshot.values[2u * neighbor + 1u];
            if (permissions.values[neighbor] & 16u) == 0u || q.z != 1.0 { continue; }
            let delta = vec2<f32>(f32(-dx), f32(-dy)) * f32(config.stride);
            t0[count] = p.x + dot(p.yz, delta);
            t1[count] = p.w + dot(q.xy, delta);
            g0[count] = p.y; g1[count] = p.z;
            g2[count] = q.x; g3[count] = q.y;
            count++;
        }
    }
    if count == 0u { return; }
    seeds.values[2u * index] = vec4<f32>(median(&t0, count), median(&g0, count),
                                         median(&g1, count), median(&t1, count));
    seeds.values[2u * index + 1u] = vec4<f32>(median(&g2, count), median(&g3, count), 0.0, 0.0);
    let ordinal = atomicAdd(&work.count, 1u);
    work.indices[ordinal] = index;
}
