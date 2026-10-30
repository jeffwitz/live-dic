"""Run one portable WGSL kernel through wgpu-py, without a window."""

import numpy as np
import wgpu

SOURCE = """
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;

@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&output_values)) {
        return;
    }
    output_values[i] = 2.0 * input_values[i];
}
"""


def main() -> None:
    # A non-multiple of 64 makes the final group's bounds guard necessary.
    values = np.arange(1003, dtype=np.float32)
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No WebGPU adapter is available")
    device = adapter.request_device_sync()
    source_buffer = device.create_buffer_with_data(
        data=values, usage=wgpu.BufferUsage.STORAGE
    )
    destination_buffer = device.create_buffer(
        size=values.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    try:
        shader = device.create_shader_module(code=SOURCE)
        pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": shader, "entry_point": "scale"}
        )
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": source_buffer}},
                {"binding": 1, "resource": {"buffer": destination_buffer}},
            ],
        )
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((values.size + 63) // 64)
        compute_pass.end()
        device.queue.submit([encoder.finish()])

        # Convenience readback: includes the required copy and CPU wait.
        raw = device.queue.read_buffer(destination_buffer)
        result = np.frombuffer(raw, dtype=np.float32).copy()
        np.testing.assert_array_equal(result, 2.0 * values)
        print(f"{adapter.info['device']}: {result.size} correct values")
    finally:
        source_buffer.destroy()
        destination_buffer.destroy()
        device.destroy()


if __name__ == "__main__":
    main()
