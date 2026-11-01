// Optional GPU reference preparation. The IC-GN solver/layout remain unchanged.
// Gaussian intensity, recursive cubic coefficients, gradients, H and H^-1 are
// recomputed for each reference. Geometry and ROI permissions stay static.
struct Config {
    width: u32, height: u32, grid_width: u32, grid_count: u32,
    inverse_offset: u32, valid_offset: u32, count_index: u32, padding: u32,
}
struct Floats { values: array<f32>, }
struct Pixels { values: array<vec4<f32>>, }
struct Words { values: array<u32>, }
struct AtomicWords { values: array<atomic<u32>>, }
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> cfg: Config;
@group(0) @binding(2) var<storage, read_write> scratch: Floats;
@group(0) @binding(3) var<storage, read_write> reference: Pixels;
@group(0) @binding(4) var<storage, read_write> metadata: Floats;
@group(0) @binding(5) var<storage, read> selected: Words;
@group(0) @binding(6) var<storage, read_write> validity: AtomicWords;

override SUBSET_SIZE: u32 = 16u;
override SUBSET_STRIDE: u32 = 16u;
override MOTION_MODEL: u32 = 0u; // 0 affine, 1 translation
override INTERPOLATION_MODE: u32 = 0u; // 0 B-spline, 1 linear, 2 Keys cubic
override ROI_ENABLED: bool = false;
const LANES: u32 = 64u;
const POLE: f32 = -0.2679491924311227;
const GAUSSIAN: array<f32, 7> = array<f32, 7>(
    0.004433048175243746, 0.054005582622414484, 0.2420362293761143,
    0.3990502796524549, 0.2420362293761143, 0.054005582622414484,
    0.004433048175243746);
var<workgroup> products: array<f32, 34u * LANES>;
var<workgroup> statistics: array<f32, 34>;
var<workgroup> augmented: array<f32, 72>;
var<workgroup> factors: array<f32, 6>;
var<workgroup> pivot_row: u32;
var<workgroup> pivot_value: f32;
var<workgroup> matrix_ok: bool;
var<workgroup> bad_inverse: atomic<u32>;
var<workgroup> mean_parts: array<f32, 64>;
var<workgroup> mean: f32;

fn mirror(x: i32, size: u32) -> u32 {
    let period = 2 * i32(size) - 2;
    let wrapped = ((x % period) + period) % period;
    return u32(min(wrapped, period - wrapped));
}
fn intensity(x: u32, y: u32) -> f32 {
    return floor(textureLoad(image, vec2<i32>(i32(x), i32(y)), 0).x * 255.0 + 0.5);
}
fn coefficient(x: i32, y: i32) -> f32 {
    let index = mirror(y, cfg.height) * cfg.width + mirror(x, cfg.width);
    return scratch.values[2u * cfg.width * cfg.height + index];
}
fn raw(x: i32, y: i32) -> f32 {
    let index = mirror(y, cfg.height) * cfg.width + mirror(x, cfg.width);
    return scratch.values[cfg.width * cfg.height + index];
}

@compute @workgroup_size(16, 16)
fn gaussian_x(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cfg.width || id.y >= cfg.height { return; }
    var value = 0.0;
    for(var tap = 0u; tap < 7u; tap++) {
        value += GAUSSIAN[tap] * intensity(mirror(i32(id.x) + i32(tap) - 3, cfg.width), id.y);
    }
    scratch.values[id.y * cfg.width + id.x] = value;
}
@compute @workgroup_size(16, 16)
fn gaussian_y(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cfg.width || id.y >= cfg.height { return; }
    var value = 0.0;
    for(var tap = 0u; tap < 7u; tap++) {
        value += GAUSSIAN[tap] * scratch.values[mirror(i32(id.y) + i32(tap) - 3, cfg.height) * cfg.width + id.x];
    }
    scratch.values[cfg.width * cfg.height + id.y * cfg.width + id.x] = value;
}
@compute @workgroup_size(16, 16)
fn direct_intensity(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cfg.width || id.y >= cfg.height { return; }
    scratch.values[cfg.width * cfg.height + id.y * cfg.width + id.x] = intensity(id.x, id.y);
}

// Preserve the CPU causal/anti-causal boundary initialization. One invocation
// owns a complete line; different lines are independent. No CPU image readback.
fn filter_line(offset: u32, count: u32, stride: u32, initialize: bool) {
    let plane = cfg.width * cfg.height;
    let base = 2u * plane + offset;
    if initialize {
        for(var i = 0u; i < count; i++) {
            scratch.values[base + i * stride] = 6.0 * scratch.values[plane + offset + i * stride];
        }
    } else {
        for(var i = 0u; i < count; i++) { scratch.values[base + i * stride] *= 6.0; }
    }
    var value = scratch.values[base];
    var power = POLE;
    for(var i = 1u; i < min(count, 21u); i++) {
        value += power * scratch.values[base + i * stride]; power *= POLE;
    }
    scratch.values[base] = value;
    for(var i = 1u; i < count; i++) {
        scratch.values[base + i * stride] += POLE * scratch.values[base + (i - 1u) * stride];
    }
    let last = base + (count - 1u) * stride;
    scratch.values[last] = POLE / (POLE * POLE - 1.0)
        * (POLE * scratch.values[last - stride] + scratch.values[last]);
    for(var i = i32(count) - 2; i >= 0; i--) {
        let position = base + u32(i) * stride;
        scratch.values[position] = POLE * (scratch.values[position + stride] - scratch.values[position]);
    }
}
@compute @workgroup_size(64)
fn spline_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < cfg.height { filter_line(id.x * cfg.width, cfg.width, 1u, true); }
}
@compute @workgroup_size(64)
fn spline_columns(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < cfg.width { filter_line(id.x, cfg.height, cfg.width, false); }
}

@compute @workgroup_size(16, 16)
fn gradients(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cfg.width || id.y >= cfg.height { return; }
    let covered_width = ((cfg.width - SUBSET_SIZE) / SUBSET_STRIDE) * SUBSET_STRIDE + SUBSET_SIZE;
    let covered_height = ((cfg.height - SUBSET_SIZE) / SUBSET_STRIDE) * SUBSET_STRIDE + SUBSET_SIZE;
    let index = id.y * cfg.width + id.x;
    if id.x >= covered_width || id.y >= covered_height { reference.values[index] = vec4<f32>(0.0); return; }
    let x = i32(id.x); let y = i32(id.y);
    var gx: f32; var gy: f32;
    if INTERPOLATION_MODE == 0u {
        gx = (0.5 * (coefficient(x + 1, y - 1) - coefficient(x - 1, y - 1))
            + 2.0 * (coefficient(x + 1, y) - coefficient(x - 1, y))
            + 0.5 * (coefficient(x + 1, y + 1) - coefficient(x - 1, y + 1))) / 6.0;
        gy = (0.5 * (coefficient(x - 1, y + 1) - coefficient(x - 1, y - 1))
            + 2.0 * (coefficient(x, y + 1) - coefficient(x, y - 1))
            + 0.5 * (coefficient(x + 1, y + 1) - coefficient(x + 1, y - 1))) / 6.0;
    } else {
        gx = 0.5 * (raw(x + 1, y) - raw(x - 1, y));
        gy = 0.5 * (raw(x, y + 1) - raw(x, y - 1));
    }
    reference.values[index] = vec4<f32>(raw(x, y), gx, gy, 0.0);
}

// Each lane accumulates pixel products; shared reductions form the Hessian.
// Cooperatively invert the 6x6 matrix to avoid a private 6x12 array spilling
// registers in every invocation. No subgroup feature or size assumption.
@compute @workgroup_size(64)
fn prepare_subsets(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = group.y * cfg.grid_width + group.x;
    if index >= cfg.grid_count { return; }
    let origin = vec2<u32>(group.x, group.y) * SUBSET_STRIDE;
    let count = SUBSET_SIZE * SUBSET_SIZE;
    let half = 0.5 * f32(SUBSET_SIZE - 1u);
    let anchor = reference.values[origin.y * cfg.width + origin.x].x;
    var delta = 0.0;
    for(var p = lane; p < count; p += LANES) {
        let pixel = (origin.y + p / SUBSET_SIZE) * cfg.width + origin.x + p % SUBSET_SIZE;
        let v = reference.values[pixel];
        delta += v.x - anchor;
    }
    mean_parts[lane] = delta;
    workgroupBarrier();
    for(var step = LANES / 2u; step > 0u; step /= 2u) {
        if lane < step { mean_parts[lane] += mean_parts[lane + step]; }
        workgroupBarrier();
    }
    if lane == 0u { mean = anchor + mean_parts[0] / f32(count); matrix_ok = true; atomicStore(&bad_inverse, 0u); }
    workgroupBarrier();
    var jf0 = vec3<f32>(0.0); var jf1 = vec3<f32>(0.0);
    var sj0 = vec3<f32>(0.0); var sj1 = vec3<f32>(0.0);
    var h00 = vec3<f32>(0.0); var h01 = vec3<f32>(0.0);
    var h10 = vec2<f32>(0.0); var h11 = vec3<f32>(0.0);
    var h20 = 0.0; var h21 = vec3<f32>(0.0);
    var h30 = vec3<f32>(0.0); var h40 = vec2<f32>(0.0); var h50 = 0.0;
    var energy = 0.0;
    for(var p = lane; p < count; p += LANES) {
        let pixel = (origin.y+p/SUBSET_SIZE)*cfg.width+origin.x+p%SUBSET_SIZE;
        let v = reference.values[pixel]; let f = v.x-mean;
        let xy = vec3<f32>(1.0,f32(p%SUBSET_SIZE)-half,f32(p/SUBSET_SIZE)-half);
        var a = v.y*xy; var b = v.z*xy;
        if MOTION_MODEL == 1u { a = vec3<f32>(v.y,0.0,0.0); b = vec3<f32>(v.z,0.0,0.0); }
        jf0 += a*f; jf1 += b*f; sj0 += a; sj1 += b; energy += f*f;
        h00 += a*a.x; h01 += b*a.x; h10 += a.yz*a.y; h11 += b*a.y;
        h20 += a.z*a.z; h21 += b*a.z; h30 += b*b.x; h40 += b.yz*b.y; h50 += b.z*b.z;
    }
    for(var c = 0u; c < 3u; c++) {
        products[c*LANES+lane] = jf0[c]; products[(c+3u)*LANES+lane] = jf1[c];
        products[(c+6u)*LANES+lane] = sj0[c]; products[(c+9u)*LANES+lane] = sj1[c];
        products[(c+12u)*LANES+lane] = h00[c]; products[(c+15u)*LANES+lane] = h01[c];
        products[(c+20u)*LANES+lane] = h11[c]; products[(c+24u)*LANES+lane] = h21[c];
        products[(c+27u)*LANES+lane] = h30[c];
    }
    for(var c = 0u; c < 2u; c++) {
        products[(c+18u)*LANES+lane] = h10[c]; products[(c+30u)*LANES+lane] = h40[c];
    }
    products[23u*LANES+lane] = h20; products[32u*LANES+lane] = h50; products[33u*LANES+lane] = energy;
    workgroupBarrier();
    if lane < 34u {
        var sum = 0.0; var correction = 0.0;
        for(var p = 0u; p < LANES; p++) {
            let adjusted = products[lane*LANES+p]-correction; let next = sum+adjusted;
            correction = (next-sum)-adjusted; sum = next;
        }
        statistics[lane] = sum;
    }
    let inverse_base = cfg.inverse_offset + index*36u;
    if lane < 36u { metadata.values[inverse_base+lane] = 0.0; }
    workgroupBarrier();
    if MOTION_MODEL == 1u {
        if lane == 0u {
            let xx = statistics[12]; let xy = statistics[15]; let yy = statistics[27];
            let det = xx*yy-xy*xy;
            matrix_ok = abs(det) > 2.220446049250313e-16*max(abs(xx*yy),1.0) && abs(det) <= 3.402823e38;
            if matrix_ok {
                metadata.values[inverse_base] = yy/det; metadata.values[inverse_base+3u] = -xy/det;
                metadata.values[inverse_base+18u] = -xy/det; metadata.values[inverse_base+21u] = xx/det;
            }
        }
    } else {
        for(var p = lane; p < 72u; p += LANES) {
            let r = p/12u; let c = p%12u;
            if c < 6u {
                let lo = min(r,c); let hi = max(r,c);
                augmented[p] = statistics[12u+lo*(11u-lo)/2u+hi];
            } else { augmented[p] = select(0.0,1.0,c-6u == r); }
        }
        workgroupBarrier();
        for(var c = 0u; c < 6u; c++) {
            if lane == 0u && matrix_ok {
                pivot_row = c;
                for(var r = c+1u; r < 6u; r++) { if abs(augmented[r*12u+c]) > abs(augmented[pivot_row*12u+c]) { pivot_row = r; } }
                pivot_value = augmented[pivot_row*12u+c];
                matrix_ok = abs(pivot_value) >= 1e-14 && abs(pivot_value) <= 3.402823e38;
            }
            workgroupBarrier();
            if matrix_ok && pivot_row != c && lane < 12u {
                let saved = augmented[c*12u+lane]; augmented[c*12u+lane] = augmented[pivot_row*12u+lane]; augmented[pivot_row*12u+lane] = saved;
            }
            workgroupBarrier();
            if matrix_ok && lane < 12u { augmented[c*12u+lane] /= pivot_value; }
            workgroupBarrier();
            if lane < 6u { factors[lane] = augmented[lane*12u+c]; }
            workgroupBarrier();
            if matrix_ok {
                for(var p = lane; p < 72u; p += LANES) {
                    let row = p/12u;
                    if row != c { augmented[p] -= factors[row]*augmented[c*12u+p%12u]; }
                }
            }
            workgroupBarrier();
        }
        if lane < 36u && matrix_ok {
            if !(abs(augmented[(lane/6u)*12u+6u+lane%6u]) <= 3.402823e38) { atomicOr(&bad_inverse,1u); }
        }
        workgroupBarrier();
        if lane == 0u { matrix_ok = matrix_ok && atomicLoad(&bad_inverse) == 0u; }
        workgroupBarrier();
        if lane < 36u && matrix_ok { metadata.values[inverse_base+lane] = augmented[(lane/6u)*12u+6u+lane%6u]; }
    }
    workgroupBarrier();
    if lane == 0u {
        let valid = matrix_ok && statistics[33] > 1e-12;
        var included = true; if ROI_ENABLED { included = selected.values[index] != 0u; }
        let base = index*16u;
        metadata.values[base] = sqrt(statistics[33]); metadata.values[base+1u] = select(-3.0,select(0.0,1.0,valid),included);
        for(var p = 0u; p < 12u; p++) { metadata.values[base+2u+p] = statistics[p]; }
        metadata.values[base+14u] = mean; metadata.values[base+15u] = 0.0;
        atomicStore(&validity.values[cfg.valid_offset+index],select(0u,1u,valid));
        if valid && included { atomicAdd(&validity.values[cfg.count_index],1u); }
    }
    if SUBSET_STRIDE == SUBSET_SIZE {
        for(var p = lane; p < count; p += LANES) {
            let pixel = (origin.y+p/SUBSET_SIZE)*cfg.width+origin.x+p%SUBSET_SIZE;
            reference.values[pixel].x -= mean;
        }
    }
}
