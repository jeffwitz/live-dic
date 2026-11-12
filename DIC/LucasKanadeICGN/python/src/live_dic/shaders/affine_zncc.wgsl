// Optional final affine ZNCC for uniform full windows.
struct Config {
    width: u32, height: u32, grid_width: u32, grid_count: u32,
    iterations: u32, pixel_count: u32, tolerance: f32, zncc_min: f32,
    inverse_offset: u32, padding0: u32, padding1: u32, padding2: u32,
}
struct F4 { values: array<vec4<f32>> }
struct Floats { values: array<f32> }
@group(0) @binding(0) var target_image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> reference: F4;
@group(0) @binding(2) var<storage, read> state: F4;
@group(0) @binding(3) var<uniform> config: Config;
@group(0) @binding(4) var<storage, read_write> output: Floats;
@group(0) @binding(5) var<storage, read> coefficients: Floats;
override SUBSET_SIZE: u32 = 16u;
override SUBSET_STRIDE: u32 = 16u;
// Same interpolation modes as the solver: B-spline, bilinear, Keys cubic.
override INTERPOLATION_MODE: u32 = 0u;
var<workgroup> accum: array<vec4<f32>, 64>;
var<workgroup> acc2: array<vec2<f32>, 64>;

fn weights(t: f32) -> array<f32, 4> {
    let t2 = t*t; let t3 = t2*t;
    if INTERPOLATION_MODE == 2u {
        return array<f32, 4>(-0.5*t+t2-0.5*t3, 1.0-2.5*t2+1.5*t3,
                             0.5*t+2.0*t2-1.5*t3, -0.5*t2+0.5*t3);
    }
    let inverse = 1.0-t;
    return array<f32, 4>(inverse*inverse*inverse/6.0, (3.0*t3-6.0*t2+4.0)/6.0,
                         (-3.0*t3+3.0*t2+3.0*t+1.0)/6.0, t3/6.0);
}

fn sample(t: vec2<f32>) -> f32 {
    let low = vec2<i32>(floor(t)); let part = fract(t);
    if INTERPOLATION_MODE == 1u {
        let a = textureLoad(target_image, low, 0).x;
        let b = textureLoad(target_image, low + vec2<i32>(1, 0), 0).x;
        let c = textureLoad(target_image, low + vec2<i32>(0, 1), 0).x;
        let d = textureLoad(target_image, low + vec2<i32>(1, 1), 0).x;
        return 255.0 * mix(mix(a, b, part.x), mix(c, d, part.x), part.y);
    }
    let wx = weights(part.x); let wy = weights(part.y);
    var value = 0.0;
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let pos = low + vec2<i32>(i-1, j-1);
            var pixel = 0.0;
            if INTERPOLATION_MODE == 0u {
                pixel = coefficients.values[u32(pos.y) * config.width + u32(pos.x)];
            } else { pixel = textureLoad(target_image, pos, 0).x * 255.0; }
            value += wx[i] * wy[j] * pixel;
        }
    }
    return value;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = group.y * config.grid_width + group.x;
    if index >= config.grid_count { return; }
    let p = state.values[index * 2u];
    let q = state.values[index * 2u + 1u];
    if q.z != 1.0 {
        if lane == 0u { output.values[index] = -2.0; }
        return;
    }
    let origin = vec2<u32>(group.xy) * SUBSET_STRIDE;
    let half = (f32(SUBSET_SIZE) - 1.0) * 0.5;
    var sums = vec4<f32>(0.0);
    var sums2 = vec2<f32>(0.0);
    for (var k = lane; k < SUBSET_SIZE * SUBSET_SIZE; k += 64u) {
        let pos = origin + vec2<u32>(k % SUBSET_SIZE, k / SUBSET_SIZE);
        let delta = vec2<f32>(f32(k % SUBSET_SIZE) - half, f32(k / SUBSET_SIZE) - half);
        let t = vec2<f32>(pos) + vec2<f32>(p.x + dot(p.yz, delta), p.w + dot(q.xy, delta));
        let valid = all(t >= vec2<f32>(2.0)) &&
            t.x <= f32(config.width - 3u) && t.y <= f32(config.height - 3u);
        if valid {
            let g = sample(t);
            let f = reference.values[pos.y * config.width + pos.x].x;
            sums += vec4<f32>(f, g, f*f, g*g);
            sums2 += vec2<f32>(f*g, 1.0);
        }
    }
    accum[lane] = sums;
    acc2[lane] = sums2;
    workgroupBarrier();
    for (var offset = 32u; offset > 0u; offset /= 2u) {
        if lane < offset { accum[lane] += accum[lane + offset]; acc2[lane] += acc2[lane + offset]; }
        workgroupBarrier();
    }
    if lane == 0u {
        let s = accum[0]; let extra = acc2[0]; let n = extra.y;
        let vf = s.z - s.x*s.x / max(n, 1.0);
        let vg = s.w - s.y*s.y / max(n, 1.0);
        if n == f32(SUBSET_SIZE * SUBSET_SIZE) && vf > 1e-9 && vg > 1e-9 {
            output.values[index] = clamp((extra.x - s.x*s.y/n) / sqrt(vf*vg), -1.0, 1.0);
        } else { output.values[index] = -2.0; }
    }
}
