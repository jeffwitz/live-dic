// Display diagnostics only; this shader never modifies an IC-GN field.
struct Values { values: array<f32> }
struct Rows { values: array<vec4<f32>> }
struct Config {
    count: u32, state_stride: u32, parameters: u32, grid_width: u32,
    center: f32, subset_stride: f32, image_center_x: f32, image_center_y: f32,
    truth_x: f32, truth_y: f32, truth_enabled: f32, padding: f32,
    decimation: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> field: Values;
@group(0) @binding(1) var<storage, read_write> summaries: Rows;
@group(0) @binding(2) var<uniform> config: Config;
@group(0) @binding(3) var<storage, read_write> decimated: Values;
var<workgroup> a: array<vec4<f32>, 256>;
var<workgroup> b: array<vec4<f32>, 256>;
var<workgroup> c: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn statistics(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    var aa = vec4<f32>(0.0); var bb = vec4<f32>(0.0); var cc = vec4<f32>(0.0);
    if id.x < config.count {
        let base = id.x * config.state_stride;
        let status = field.values[base + config.parameters];
        let selected = status != -3.0;
        aa.z = select(0.0, 1.0, selected);
        if status >= 0.5 {
            let u = field.values[base]; let v = field.values[base + 3u];
            aa.x = 1.0; aa.y = field.values[base + config.parameters + 1u];
            aa.w = select(0.0, 1.0, status > 1.5);
            bb = vec4<f32>(abs(u), abs(v), length(vec2<f32>(u, v)), 0.0);
            for (var k = 1u; k < 6u; k++) {
                if k != 3u { bb.w = max(bb.w, abs(field.values[base + k])); }
            }
            let x = f32(id.x % config.grid_width) * config.subset_stride + config.center;
            let y = f32(id.x / config.grid_width) * config.subset_stride + config.center;
            let delta = vec2<f32>(u - config.truth_x * (x - config.image_center_x), v - config.truth_y * (y - config.image_center_y));
            cc.x = dot(delta, delta) * config.truth_enabled;
        }
    }
    a[lane] = aa; b[lane] = bb; c[lane] = cc;
    workgroupBarrier();
    for (var offset = 128u; offset > 0u; offset /= 2u) {
        if lane < offset { a[lane] += a[lane + offset]; b[lane] = max(b[lane], b[lane + offset]); c[lane] += c[lane + offset]; }
        workgroupBarrier();
    }
    if lane == 0u { summaries.values[group.x * 3u] = a[0]; summaries.values[group.x * 3u + 1u] = b[0]; summaries.values[group.x * 3u + 2u] = c[0]; }
}
@compute @workgroup_size(256)
fn decimate(@builtin(global_invocation_id) id: vec3<u32>) {
    let source = id.x * config.decimation;
    if source >= config.count { return; }
    for (var k = 0u; k < config.state_stride; k++) {
        decimated.values[id.x * config.state_stride + k] = field.values[source * config.state_stride + k];
    }
}
