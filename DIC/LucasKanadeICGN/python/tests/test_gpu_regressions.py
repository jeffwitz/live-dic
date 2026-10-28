"""Opt-in regressions that execute the packaged WGSL on a real wgpu adapter.

Run with ICGN_RUN_GPU_TESTS=1 and the package's ``native`` extra installed.
Once requested, missing wgpu, an unavailable adapter, or shader compilation
errors are failures, never skips. These tests enforce exact reduction and
state-preservation contracts; they do not qualify DIC accuracy or throughput.
"""

from __future__ import annotations

import os
import struct
import unittest
from typing import Any

import numpy as np

from lucas_kanade_icgn import ICGNWGPUFullPyramid
from lucas_kanade_icgn.wgpu_backend import _reduce_half


def _integer_reduction(image: np.ndarray) -> np.ndarray:
    """Independent exact oracle: one integer 5x5 convolution, then truncation."""

    height, width = image.shape
    reduced_height, reduced_width = height // 2, width // 2
    padded = np.pad(image.astype(np.uint32), 2, mode="reflect")
    total = np.zeros((reduced_height, reduced_width), dtype=np.uint32)
    taps = (1, 4, 6, 4, 1)
    for row, weight_y in enumerate(taps):
        for column, weight_x in enumerate(taps):
            total += (
                weight_y
                * weight_x
                * padded[
                    row : row + 2 * reduced_height : 2,
                    column : column + 2 * reduced_width : 2,
                ]
            )
    return (total // 256).astype(np.uint8)


@unittest.skipUnless(
    os.environ.get("ICGN_RUN_GPU_TESTS") == "1",
    "set ICGN_RUN_GPU_TESTS=1 to execute the native GPU regressions",
)
class GPURegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Use production pipeline creation and layouts, including both seeded
        # entry points. An unsupported runtime must fail an explicitly enabled run.
        cls.runtime = ICGNWGPUFullPyramid(pyramid_levels=2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime.close()
        cls.runtime.device.destroy()

    def _gpu_reduce(self, image: np.ndarray) -> np.ndarray:
        device, wgpu = self.runtime.device, self.runtime._wgpu
        height, width = image.shape
        reduced_height, reduced_width = height // 2, width // 2
        textures = []
        try:
            source = device.create_texture(
                size=(width, height, 1),
                format=wgpu.TextureFormat.r8unorm,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            textures.append(source)
            intermediate = device.create_texture(
                size=(reduced_width, height, 1),
                format=wgpu.TextureFormat.r32float,
                usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            textures.append(intermediate)
            destination = device.create_texture(
                size=(reduced_width, reduced_height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
            )
            textures.append(destination)
            device.queue.write_texture(
                {"texture": source, "mip_level": 0, "origin": (0, 0, 0)},
                np.ascontiguousarray(image),
                {"offset": 0, "bytes_per_row": width, "rows_per_image": height},
                (width, height, 1),
            )
            horizontal = device.create_bind_group(
                layout=self.runtime._reduce_x_layout,
                entries=[
                    {"binding": 0, "resource": source.create_view()},
                    {"binding": 1, "resource": intermediate.create_view()},
                ],
            )
            vertical = device.create_bind_group(
                layout=self.runtime._reduce_y_layout,
                entries=[
                    {"binding": 0, "resource": intermediate.create_view()},
                    {"binding": 1, "resource": destination.create_view()},
                ],
            )
            encoder = device.create_command_encoder(label="GPU reduction regression")
            for pipeline, group, bind_group, dispatch_height in (
                (self.runtime._reduce_x_pipeline, 0, horizontal, height),
                (self.runtime._reduce_y_pipeline, 1, vertical, reduced_height),
            ):
                compute_pass = encoder.begin_compute_pass()
                compute_pass.set_pipeline(pipeline)
                compute_pass.set_bind_group(group, bind_group)
                compute_pass.dispatch_workgroups(
                    (reduced_width + 15) // 16,
                    (dispatch_height + 15) // 16,
                    1,
                )
                compute_pass.end()
            device.queue.submit([encoder.finish()])
            row_bytes = ((4 * reduced_width + 255) // 256) * 256
            raw = device.queue.read_texture(
                {"texture": destination, "mip_level": 0, "origin": (0, 0, 0)},
                {"offset": 0, "bytes_per_row": row_bytes, "rows_per_image": reduced_height},
                (reduced_width, reduced_height, 1),
            )
            rgba = np.ndarray(
                shape=(reduced_height, reduced_width, 4),
                dtype=np.uint8,
                buffer=raw,
                strides=(row_bytes, 4, 1),
            )
            return rgba[..., 0].copy()
        finally:
            for texture in textures:
                texture.destroy()

    def test_reduction_matches_cpu_and_integer_oracle(self) -> None:
        # Every uint8 value appears in a flat band wide enough to contain
        # interior samples. The odd final row/column also exercises truncation.
        band_row = np.concatenate((np.repeat(np.arange(256, dtype=np.uint8), 8), [255]))
        bands = np.tile(band_row.astype(np.uint8), (65, 1))
        impulse = np.zeros((65, 97), dtype=np.uint8)
        impulse[0, 0] = impulse[32, 48] = impulse[-1, -1] = 255
        y, x = np.indices((67, 99))
        ramp = ((3 * x + 5 * y) % 256).astype(np.uint8)
        noise = np.random.default_rng(20261028).integers(0, 256, (65, 97), dtype=np.uint8)
        for name, image in (
            ("all_256_gray_levels", bands),
            ("impulses_and_borders", impulse),
            ("two_dimensional_ramp", ramp),
            ("deterministic_noise", noise),
        ):
            cpu_image, gpu_image = image, image
            for level in range(1, 3):
                with self.subTest(pattern=name, level=level):
                    expected = _integer_reduction(cpu_image)
                    np.testing.assert_array_equal(_reduce_half(cpu_image), expected)
                    actual = self._gpu_reduce(gpu_image)
                    np.testing.assert_array_equal(actual, expected)
                    cpu_image, gpu_image = expected, actual

    def _seeded_state(self, *, retry: bool) -> tuple[np.ndarray, ...]:
        """Read solver state directly, before median filtering can hide a loss.

        One interior subset is enabled with valid metadata and zero target
        energy. Its deterministic rejection proves that the dispatch ran; all
        other subsets carry distinguishable previously accepted observations.
        """

        root = self.runtime._root
        device, wgpu = root.device, root._wgpu
        width = height = 64
        grid_width = 4
        grid_count = 16
        previous = np.arange(grid_count * 8, dtype=np.float32).reshape(grid_count, 8) / 16
        previous[:, 6] = 1 + np.arange(grid_count) % 2
        previous[:, 7] = 2 + np.arange(grid_count)
        seeds = np.zeros((grid_count, 8), dtype=np.float32)
        seeds[:, 0], seeds[:, 3] = 0.25, -0.5
        active = np.zeros(grid_count, dtype=np.uint32)
        active[5] = 1  # Interior subset (1, 1), surrounded by inactive neighbours.
        metadata = np.zeros((grid_count, 16), dtype=np.float32)
        metadata[5, :2] = 1.0  # Positive reference norm and valid reference.
        config = struct.pack("<6I2f", width, height, grid_width, grid_count, 1, 4096, 1e-3, 0.8)
        data = {
            1: config,
            2: np.zeros(width * height, dtype=np.float32),
            3: np.zeros(width * height, dtype=np.float32),
            4: np.zeros((width * height, 4), dtype=np.float32),
            5: np.zeros((grid_count, 36), dtype=np.float32),
            6: metadata,
            7: previous,
            8: np.zeros(31, dtype=np.float32),
            9: np.zeros((grid_count, 8), dtype=np.float32),
            10: seeds,
            11: active,
        }
        buffers: dict[int, Any] = {}
        source = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING,
        )
        try:
            for binding, values in data.items():
                usage = wgpu.BufferUsage.UNIFORM if binding == 1 else wgpu.BufferUsage.STORAGE
                if binding == 7:
                    usage |= wgpu.BufferUsage.COPY_SRC
                buffers[binding] = device.create_buffer_with_data(data=values, usage=usage)
            bind_group = device.create_bind_group(
                layout=root._seeded_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": source.create_view()},
                    *(
                        root._buffer_resource(binding, buffer)
                        for binding, buffer in buffers.items()
                    ),
                ],
            )
            encoder = device.create_command_encoder(label="GPU seeded-state regression")
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(root._retry_solver if retry else root._seeded_solver)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(grid_count, 1, 1)
            compute_pass.end()
            device.queue.submit([encoder.finish()])
            result = np.frombuffer(device.queue.read_buffer(buffers[7]), dtype=np.float32)
            return previous, seeds, active.astype(bool), result.reshape(grid_count, 8).copy()
        finally:
            for buffer in buffers.values():
                buffer.destroy()
            source.destroy()

    def test_fresh_seeded_solve_initializes_disabled_subsets(self) -> None:
        _, seeds, active, result = self._seeded_state(retry=False)
        np.testing.assert_array_equal(result[:, :6], seeds[:, :6])
        np.testing.assert_array_equal(result[~active, 6], -1.0)
        np.testing.assert_array_equal(result[~active, 7], 0.0)
        np.testing.assert_array_equal(result[active, 6], -2.0)
        np.testing.assert_array_equal(result[active, 7], 1.0)

    def test_selective_retry_preserves_inactive_neighbours_before_median(self) -> None:
        previous, seeds, active, result = self._seeded_state(retry=True)
        np.testing.assert_array_equal(result[~active], previous[~active])
        np.testing.assert_array_equal(result[active, :6], seeds[active, :6])
        np.testing.assert_array_equal(result[active, 6], -2.0)
        np.testing.assert_array_equal(result[active, 7], 1.0)


if __name__ == "__main__":
    unittest.main()
