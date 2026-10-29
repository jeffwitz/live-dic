// Minimal native subgroup qualification shader.
//
// wgpu-native exposes subgroup operations through the requested `subgroup`
// device feature.  Naga 27 (used by wgpu-py 0.32) accepts the operations and
// built-ins directly; it does not accept the browser-side `enable subgroups;`
// directive.  Keep this source native-only and deliberately small so feature
// failures remain easy to diagnose.

struct ProbeOutput {
    values: array<vec4<f32>, 16>,
}

@group(0) @binding(0)
var<storage, read_write> probe_output: ProbeOutput;

@compute @workgroup_size(16)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_lane: u32,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    let value = f32(local_id.x + 1u);
    let reduced = subgroupAdd(vec4<f32>(value, 1.0, value * value, 1.0));
    if subgroup_lane == 0u {
        let subgroup_index = local_id.x / subgroup_size;
        probe_output.values[subgroup_index] = vec4<f32>(
            f32(subgroup_size),
            reduced.x,
            reduced.y,
            reduced.z,
        );
    }
}
