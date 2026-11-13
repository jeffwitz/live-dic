struct ImageConfig {
    width: u32,
    height: u32,
    padding0: u32,
    padding1: u32,
}

struct DescriptorConfig {
    image_width: u32,
    image_height: u32,
    output_width: u32,
    output_height: u32,
    scale: u32,
    tile_side: u32,
    work_side: u32,
    halo: u32,
    batch_start: u32,
    batch_count: u32,
    tiles_x: u32,
    total_tiles: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
}

struct FloatBuffer {
    values: array<f32>,
}

@group(0) @binding(0) var blur_source: texture_2d<f32>;
@group(0) @binding(1) var blur_target: texture_storage_2d<{{GRAY_STORAGE_FORMAT}}, write>;
@group(0) @binding(2) var<uniform> image_config: ImageConfig;

@group(0) @binding(3) var blurred_image: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> scratch_a: FloatBuffer;
@group(0) @binding(5) var<storage, read_write> scratch_b: FloatBuffer;
@group(0) @binding(6) var<uniform> descriptor_config: DescriptorConfig;
@group(0) @binding(7) var descriptor_first: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var descriptor_second: texture_storage_2d<rgba16float, write>;

const GAUSSIAN_WEIGHTS = array<f32, 5>(135.0, 607.0, 1000.0, 607.0, 135.0);
const GAUSSIAN_WEIGHT_SUM = 2484.0;
const SQRT_HALF = 0.7071067811865476;

fn bounded_image_point(point: vec2<i32>) -> vec2<i32> {
    return clamp(point, vec2<i32>(0),
        vec2<i32>(i32(image_config.width) - 1, i32(image_config.height) - 1));
}

fn quantize_gray(value: f32) -> f32 {
    return floor(clamp(value, 0.0, 1.0) * 255.0) / 255.0;
}

@compute @workgroup_size(16, 16)
fn presmooth_horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= image_config.width || id.y >= image_config.height { return; }
    let point = vec2<i32>(id.xy);
    var sum = 0.0;
    for (var k = -2; k <= 2; k = k + 1) {
        let sample_point = bounded_image_point(point + vec2<i32>(k, 0));
        sum += textureLoad(blur_source, sample_point, 0).x
            * GAUSSIAN_WEIGHTS[u32(k + 2)];
    }
    let gray = quantize_gray(sum / GAUSSIAN_WEIGHT_SUM);
    textureStore(blur_target, point, vec4<f32>(gray, gray, gray, 1.0));
}

@compute @workgroup_size(16, 16)
fn presmooth_vertical(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= image_config.width || id.y >= image_config.height { return; }
    let point = vec2<i32>(id.xy);
    var sum = 0.0;
    for (var k = -2; k <= 2; k = k + 1) {
        let sample_point = bounded_image_point(point + vec2<i32>(0, k));
        sum += textureLoad(blur_source, sample_point, 0).x
            * GAUSSIAN_WEIGHTS[u32(k + 2)];
    }
    let gray = quantize_gray(sum / GAUSSIAN_WEIGHT_SUM);
    textureStore(blur_target, point, vec4<f32>(gray, gray, gray, 1.0));
}

fn bounded_descriptor_image_point(point: vec2<i32>) -> vec2<i32> {
    return clamp(point, vec2<i32>(0),
        vec2<i32>(i32(descriptor_config.image_width) - 1,
                  i32(descriptor_config.image_height) - 1));
}

fn gray_value(point: vec2<i32>) -> f32 {
    return textureLoad(blurred_image, bounded_descriptor_image_point(point), 0).x * 255.0;
}

fn orientation(bin: u32) -> vec2<f32> {
    switch bin {
        case 0u: { return vec2<f32>(0.0, 1.0); }
        case 1u: { return vec2<f32>(SQRT_HALF, SQRT_HALF); }
        case 2u: { return vec2<f32>(1.0, 0.0); }
        case 3u: { return vec2<f32>(SQRT_HALF, -SQRT_HALF); }
        case 4u: { return vec2<f32>(0.0, -1.0); }
        case 5u: { return vec2<f32>(-SQRT_HALF, -SQRT_HALF); }
        case 6u: { return vec2<f32>(-1.0, 0.0); }
        default: { return vec2<f32>(-SQRT_HALF, SQRT_HALF); }
    }
}

fn pure_gradient(point: vec2<i32>) -> vec2<f32> {
    let bounded = bounded_descriptor_image_point(point);
    let x = bounded.x;
    let y = bounded.y;
    let width = i32(descriptor_config.image_width);
    let height = i32(descriptor_config.image_height);
    var gx = 0.0;
    var gy = 0.0;
    if width > 1 {
        if x == 0 {
            gx = gray_value(vec2<i32>(x + 1, y)) - gray_value(vec2<i32>(x, y));
        } else if x == width - 1 {
            gx = gray_value(vec2<i32>(x, y)) - gray_value(vec2<i32>(x - 1, y));
        } else {
            gx = gray_value(vec2<i32>(x + 1, y)) - gray_value(vec2<i32>(x - 1, y));
        }
    }
    if height > 1 {
        if y == 0 {
            gy = gray_value(vec2<i32>(x, y + 1)) - gray_value(vec2<i32>(x, y));
        } else if y == height - 1 {
            gy = gray_value(vec2<i32>(x, y)) - gray_value(vec2<i32>(x, y - 1));
        } else {
            gy = gray_value(vec2<i32>(x, y + 1)) - gray_value(vec2<i32>(x, y - 1));
        }
    }
    return vec2<f32>(gx, gy);
}

fn tile_origin(tile_index: u32) -> vec2<u32> {
    return vec2<u32>(tile_index % descriptor_config.tiles_x,
                     tile_index / descriptor_config.tiles_x)
        * descriptor_config.tile_side;
}

fn scratch_index(tile_slot: u32, local: vec2<u32>, channel: u32) -> u32 {
    let tile_area = descriptor_config.work_side * descriptor_config.work_side;
    return (tile_slot * tile_area + local.y * descriptor_config.work_side + local.x) * 8u
        + channel;
}

@compute @workgroup_size(8, 8)
fn build_hog(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let tile_slot = group.z;
    if tile_slot >= descriptor_config.batch_count
        || id.x >= descriptor_config.work_side || id.y >= descriptor_config.work_side {
        return;
    }
    let tile_index = descriptor_config.batch_start + tile_slot;
    if tile_index >= descriptor_config.total_tiles { return; }
    let origin = tile_origin(tile_index) * descriptor_config.scale;
    let point = vec2<i32>(origin) + vec2<i32>(id.xy)
        - vec2<i32>(i32(descriptor_config.halo));
    let gradient = pure_gradient(point);
    var values: array<f32, 8>;
    for (var bin = 0u; bin < 8u; bin = bin + 1u) {
        values[bin] = max(0.0, dot(gradient, orientation(bin)));
    }
    let base = scratch_index(tile_slot, id.xy, 0u);
    for (var channel = 0u; channel < 8u; channel = channel + 1u) {
        scratch_a.values[base + channel] = values[channel];
    }
}

@compute @workgroup_size(8, 8)
fn smooth_horizontal(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let tile_slot = group.z;
    if tile_slot >= descriptor_config.batch_count
        || id.x >= descriptor_config.work_side || id.y >= descriptor_config.work_side {
        return;
    }
    let base = scratch_index(tile_slot, id.xy, 0u);
    for (var channel = 0u; channel < 8u; channel = channel + 1u) {
        var sum = 0.0;
        for (var k = -2; k <= 2; k = k + 1) {
            let x = u32(clamp(i32(id.x) + k, 0, i32(descriptor_config.work_side) - 1));
            let source = scratch_index(tile_slot, vec2<u32>(x, id.y), channel);
            sum += scratch_a.values[source] * GAUSSIAN_WEIGHTS[u32(k + 2)];
        }
        scratch_b.values[base + channel] = sum / GAUSSIAN_WEIGHT_SUM;
    }
}

@compute @workgroup_size(8, 8)
fn smooth_vertical(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let tile_slot = group.z;
    if tile_slot >= descriptor_config.batch_count
        || id.x >= descriptor_config.work_side || id.y >= descriptor_config.work_side {
        return;
    }
    let base = scratch_index(tile_slot, id.xy, 0u);
    for (var channel = 0u; channel < 8u; channel = channel + 1u) {
        var sum = 0.0;
        for (var k = -2; k <= 2; k = k + 1) {
            let y = u32(clamp(i32(id.y) + k, 0, i32(descriptor_config.work_side) - 1));
            let source = scratch_index(tile_slot, vec2<u32>(id.x, y), channel);
            sum += scratch_b.values[source] * GAUSSIAN_WEIGHTS[u32(k + 2)];
        }
        scratch_a.values[base + channel] = sum / GAUSSIAN_WEIGHT_SUM;
    }
}

@compute @workgroup_size(8, 8)
fn sigmoid_smooth_horizontal(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let tile_slot = group.z;
    if tile_slot >= descriptor_config.batch_count
        || id.x >= descriptor_config.work_side || id.y >= descriptor_config.work_side {
        return;
    }
    let base = scratch_index(tile_slot, id.xy, 0u);
    for (var channel = 0u; channel < 8u; channel = channel + 1u) {
        var sum = 0.0;
        for (var k = -2; k <= 2; k = k + 1) {
            let x = u32(clamp(i32(id.x) + k, 0, i32(descriptor_config.work_side) - 1));
            let source = scratch_index(tile_slot, vec2<u32>(x, id.y), channel);
            let value = 2.0 / (1.0 + exp(-0.2 * scratch_a.values[source])) - 1.0;
            sum += value * GAUSSIAN_WEIGHTS[u32(k + 2)];
        }
        scratch_b.values[base + channel] = sum / GAUSSIAN_WEIGHT_SUM;
    }
}

@compute @workgroup_size(8, 8)
fn average_pool(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>
) {
    let tile_slot = group.z;
    if tile_slot >= descriptor_config.batch_count
        || id.x >= descriptor_config.tile_side || id.y >= descriptor_config.tile_side {
        return;
    }
    let tile_index = descriptor_config.batch_start + tile_slot;
    if tile_index >= descriptor_config.total_tiles { return; }
    let output_point = tile_origin(tile_index) + id.xy;
    if output_point.x >= descriptor_config.output_width
        || output_point.y >= descriptor_config.output_height {
        return;
    }

    var first = vec4<f32>(0.0);
    var second = vec4<f32>(0.0);
    for (var dy = 0u; dy < descriptor_config.scale; dy = dy + 1u) {
        for (var dx = 0u; dx < descriptor_config.scale; dx = dx + 1u) {
            let local = vec2<u32>(descriptor_config.halo) + id.xy * descriptor_config.scale
                + vec2<u32>(dx, dy);
            let base = scratch_index(tile_slot, local, 0u);
            first += vec4<f32>(
                scratch_a.values[base], scratch_a.values[base + 1u],
                scratch_a.values[base + 2u], scratch_a.values[base + 3u]
            );
            second += vec4<f32>(
                scratch_a.values[base + 4u], scratch_a.values[base + 5u],
                scratch_a.values[base + 6u], scratch_a.values[base + 7u]
            );
        }
    }
    let inverse_area = 1.0 / f32(descriptor_config.scale * descriptor_config.scale);
    textureStore(descriptor_first, vec2<i32>(output_point), first * inverse_area);
    textureStore(descriptor_second, vec2<i32>(output_point), second * inverse_area);
}
