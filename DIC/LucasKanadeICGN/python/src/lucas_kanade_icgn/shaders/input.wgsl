// Input conversion only. BT.709 decimal coefficients expressed as exact integer
// ratios after recovering source bytes; ties round up to the nearest uint8.
@group(0) @binding(0) var color_image: texture_2d<f32>;
@group(0) @binding(1) var gray_image: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(16, 16)
fn luminance(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(gray_image);
    if any(id.xy >= dimensions) { return; }
    let rgb = vec3<u32>(round(textureLoad(color_image, vec2<i32>(id.xy), 0).rgb * 255.0));
    let gray = (2126u * rgb.r + 7152u * rgb.g + 722u * rgb.b + 5000u) / 10000u;
    let value = f32(gray) / 255.0;
    textureStore(gray_image, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
