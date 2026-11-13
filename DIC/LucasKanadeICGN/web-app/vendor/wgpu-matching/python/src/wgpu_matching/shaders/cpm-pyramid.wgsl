struct Params {
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
}

@group(0) @binding(0) var source_first: texture_2d<f32>;
@group(0) @binding(1) var source_second: texture_2d<f32>;
@group(0) @binding(2) var target_first: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var target_second: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn downsample_hog(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.target_width || id.y >= params.target_height { return; }
    let source_origin = vec2<i32>(id.xy * 2u);
    let source_limit = vec2<i32>(i32(params.source_width) - 1, i32(params.source_height) - 1);
    var first = vec4<f32>(0.0);
    var second = vec4<f32>(0.0);
    for (var y = 0; y < 2; y = y + 1) {
        for (var x = 0; x < 2; x = x + 1) {
            let p = min(source_origin + vec2<i32>(x, y), source_limit);
            first += textureLoad(source_first, p, 0);
            second += textureLoad(source_second, p, 0);
        }
    }
    textureStore(target_first, vec2<i32>(id.xy), first * 0.25);
    textureStore(target_second, vec2<i32>(id.xy), second * 0.25);
}
