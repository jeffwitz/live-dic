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

from lucas_kanade_icgn import ICGNWGPU, ICGNWGPUFullPyramid
from lucas_kanade_icgn.wgpu_backend import (
    FIR_RADIUS,
    FIR_RUNTIME_RADIUS,
    _fir_weights,
    _reduce_half,
)


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


def _cpu_prefilter(image: np.ndarray) -> np.ndarray:
    """Mirror the shader's separable float32 FIR accumulation order."""

    trim = FIR_RADIUS - FIR_RUNTIME_RADIUS
    weights = _fir_weights()[trim:-trim]
    radius = weights.size // 2
    horizontal = np.zeros(image.shape, dtype=np.float32)
    padded_x = np.pad(image.astype(np.float32), ((0, 0), (radius, radius)), mode="reflect")
    for tap, weight in enumerate(weights):
        horizontal += weight * padded_x[:, tap : tap + image.shape[1]]
    result = np.zeros(image.shape, dtype=np.float32)
    padded_y = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    for tap, weight in enumerate(weights):
        result += weight * padded_y[tap : tap + image.shape[0], :]
    return result


def _fractional_pair(
    width: int,
    height: int,
    translation: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a continuous texture before and after a known translation."""

    y, x = np.mgrid[:height, :width].astype(np.float64)

    def texture(sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
        values = (
            127.0
            + 34.0 * np.sin(0.173 * sample_x + 0.071 * sample_y)
            + 29.0 * np.cos(0.043 * sample_x - 0.191 * sample_y)
            + 23.0 * np.sin(0.311 * sample_x - 0.137 * sample_y + 0.4)
            + 17.0 * np.cos(0.229 * sample_x + 0.257 * sample_y - 0.8)
        )
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)

    dx, dy = translation
    return texture(x, y), texture(x - dx, y - dy)


def _quadratic_pair(
    width: int,
    height: int,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a continuous texture through a known quadratic displacement."""

    target_y, target_x = np.mgrid[:height, :width].astype(np.float64)
    origin_x = 0.5 * (width - 1)
    origin_y = 0.5 * (height - 1)

    def texture(sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
        values = (
            127.0
            + 34.0 * np.sin(0.173 * sample_x + 0.071 * sample_y)
            + 29.0 * np.cos(0.043 * sample_x - 0.191 * sample_y)
            + 23.0 * np.sin(0.311 * sample_x - 0.137 * sample_y + 0.4)
            + 17.0 * np.cos(0.229 * sample_x + 0.257 * sample_y - 0.8)
        )
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)

    reference_x = target_x.copy()
    reference_y = target_y.copy()
    for _ in range(20):
        x = reference_x - origin_x
        y = reference_y - origin_y
        u = (
            parameters[0]
            + parameters[1] * x
            + parameters[2] * y
            + 0.5 * parameters[6] * x * x
            + parameters[7] * x * y
            + 0.5 * parameters[8] * y * y
        )
        v = (
            parameters[3]
            + parameters[4] * x
            + parameters[5] * y
            + 0.5 * parameters[9] * x * x
            + parameters[10] * x * y
            + 0.5 * parameters[11] * y * y
        )
        residual_x = reference_x + u - target_x
        residual_y = reference_y + v - target_y
        j11 = 1.0 + parameters[1] + parameters[6] * x + parameters[7] * y
        j12 = parameters[2] + parameters[7] * x + parameters[8] * y
        j21 = parameters[4] + parameters[9] * x + parameters[10] * y
        j22 = 1.0 + parameters[5] + parameters[10] * x + parameters[11] * y
        determinant = j11 * j22 - j12 * j21
        reference_x -= (j22 * residual_x - j12 * residual_y) / determinant
        reference_y -= (-j21 * residual_x + j11 * residual_y) / determinant
    return texture(target_x, target_y), texture(reference_x, reference_y)


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

    def test_interpolation_modes_recover_fractional_translation(self) -> None:
        truth = (0.5, -0.375)
        reference, target = _fractional_pair(128, 128, truth)
        thresholds = {"bspline": 0.03, "linear": 0.05, "cubic": 0.03}
        for mode, threshold in thresholds.items():
            with self.subTest(mode=mode):
                engine = ICGNWGPU(interpolation=mode, prefer_subgroups=False)
                try:
                    engine.set_reference(reference)
                    result = engine.register(target)
                    self.assertGreater(result.accepted, 0)
                    self.assertLess(result.rmse(truth), threshold)
                finally:
                    engine.close()
                    engine.device.destroy()

    def test_tuned_full_pyramid_defaults_recover_fractional_translation(self) -> None:
        truth = (0.5, -0.375)
        reference, target = _fractional_pair(256, 256, truth)
        thresholds = {"bspline": 0.03, "linear": 0.05, "cubic": 0.03}
        for mode, threshold in thresholds.items():
            with self.subTest(mode=mode):
                engine = ICGNWGPUFullPyramid(pyramid_levels=2, interpolation=mode)
                try:
                    self.assertEqual(engine.iterations_per_level, (8, 16))
                    self.assertEqual(engine.tolerance_per_level, (5e-3, 5e-3))
                    engine.set_reference(reference)
                    result = engine.register(target)
                    self.assertGreater(result.accepted, 180)
                    self.assertLess(result.rmse(truth), threshold)
                finally:
                    engine.close()
                    engine.device.destroy()

    def test_configurable_subset_sizes_run_the_full_pyramid(self) -> None:
        truth = (0.5, -0.375)
        reference, target = _fractional_pair(256, 256, truth)
        for subset_size in (8, 32):
            with self.subTest(subset_size=subset_size):
                engine = ICGNWGPUFullPyramid(
                    pyramid_levels=2,
                    subset_size=subset_size,
                    interpolation="linear",
                )
                try:
                    info = engine.set_reference(reference)
                    result = engine.register(target)
                    expected_grid = (256 // subset_size, 256 // subset_size)
                    self.assertEqual(info.subset_size, subset_size)
                    self.assertEqual(result.subset_size, subset_size)
                    self.assertEqual(result.parameters.shape[:2], expected_grid[::-1])
                    self.assertGreater(result.accepted, 0)
                    self.assertLess(result.rmse(truth), 0.08)
                finally:
                    engine.close()
                    engine.device.destroy()

    def test_full_pyramid_stream_preserves_order_and_synchronous_results(self) -> None:
        translations = ((0.25, -0.125), (0.5, -0.375), (-0.375, 0.25))
        pairs = [_fractional_pair(128, 128, truth) for truth in translations]
        reference = pairs[0][0]
        targets = [pair[1] for pair in pairs]
        engine = ICGNWGPUFullPyramid(pyramid_levels=2, motion_model="translation")
        try:
            engine.set_reference(reference)
            synchronous = [engine.register(target) for target in targets]
            streamed = list(engine.register_stream(targets, pipeline_depth=2))
            self.assertEqual(len(streamed), len(synchronous))
            for expected, actual, truth in zip(
                synchronous,
                streamed,
                translations,
                strict=True,
            ):
                np.testing.assert_array_equal(actual.status, expected.status)
                np.testing.assert_allclose(
                    actual.parameters, expected.parameters, rtol=0.0, atol=1e-6
                )
                self.assertLess(actual.rmse(truth), 0.05)
                self.assertGreater(actual.timing["resultAgeWallMs"], 0.0)
                self.assertGreaterEqual(actual.timing["completionWaitWallMs"], 0.0)
        finally:
            engine.close()
            engine.device.destroy()

    def test_translation_model_recovers_fractional_shift_without_strain(self) -> None:
        truth = (0.5, -0.375)
        reference, target = _fractional_pair(256, 256, truth)
        engine = ICGNWGPUFullPyramid(
            pyramid_levels=2,
            motion_model="translation",
        )
        try:
            info = engine.set_reference(reference)
            result = engine.register(target)
            self.assertEqual(info.config["motionModel"], "translation")
            self.assertEqual(result.config["motionModel"], "translation")
            self.assertGreater(result.accepted, 180)
            self.assertLess(result.rmse(truth), 0.03)
            np.testing.assert_array_equal(result.parameters[..., (1, 2, 4, 5)], 0.0)
        finally:
            engine.close()
            engine.device.destroy()

    def test_quadratic_model_recovers_second_order_displacement(self) -> None:
        truth = np.asarray(
            (
                0.5,
                0.002,
                -0.001,
                -0.3,
                0.001,
                0.002,
                8e-4,
                -5e-4,
                4e-4,
                -4e-4,
                6e-4,
                -7e-4,
            ),
            dtype=np.float64,
        )
        reference, target = _quadratic_pair(256, 256, truth)
        engine = ICGNWGPUFullPyramid(
            pyramid_levels=3,
            subset_size=32,
            interpolation="cubic",
            motion_model="quadratic",
            iterations_per_level=(8, 12, 20),
            tolerance_per_level=(1e-2, 5e-3, 2e-3),
        )
        try:
            engine.set_reference(reference, zncc_min=0.7)
            result = engine.register(target)
            mask = result.accepted_mask
            grid_y, grid_x = np.mgrid[: result.status.shape[0], : result.status.shape[1]]
            local_x = (grid_x + 0.5) * 32 - 0.5 - 127.5
            local_y = (grid_y + 0.5) * 32 - 0.5 - 127.5
            expected = np.zeros_like(result.parameters, dtype=np.float64)
            expected[..., 0] = (
                truth[0]
                + truth[1] * local_x
                + truth[2] * local_y
                + 0.5 * truth[6] * local_x**2
                + truth[7] * local_x * local_y
                + 0.5 * truth[8] * local_y**2
            )
            expected[..., 1] = truth[1] + truth[6] * local_x + truth[7] * local_y
            expected[..., 2] = truth[2] + truth[7] * local_x + truth[8] * local_y
            expected[..., 3] = (
                truth[3]
                + truth[4] * local_x
                + truth[5] * local_y
                + 0.5 * truth[9] * local_x**2
                + truth[10] * local_x * local_y
                + 0.5 * truth[11] * local_y**2
            )
            expected[..., 4] = truth[4] + truth[9] * local_x + truth[10] * local_y
            expected[..., 5] = truth[5] + truth[10] * local_x + truth[11] * local_y
            expected[..., 6:] = truth[6:]
            error = result.parameters.astype(np.float64) - expected
            displacement_rmse = np.sqrt(np.mean(np.sum(error[..., (0, 3)][mask] ** 2, axis=1)))
            curvature_rmse = np.sqrt(np.mean(error[..., 6:][mask] ** 2))
            self.assertGreater(result.accepted, 30)
            self.assertLess(displacement_rmse, 0.02)
            self.assertLess(curvature_rmse, 2e-4)
        finally:
            engine.close()
            engine.device.destroy()

    def test_three_level_pyramid_prefilter_handles_small_halo_levels(self) -> None:
        truth = (0.5, -0.375)
        reference, target = _fractional_pair(64, 64, truth)
        engine = ICGNWGPUFullPyramid(pyramid_levels=3)
        try:
            engine.set_reference(reference)
            result = engine.register(target)
            self.assertGreater(result.accepted, 0)
            self.assertTrue(np.isfinite(result.parameters).all())
        finally:
            engine.close()
            engine.device.destroy()

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

    def test_tiled_prefilter_matches_cpu_on_partial_workgroups(self) -> None:
        # Deliberately avoid multiples of 16 so both right and bottom workgroup
        # boundaries exercise the cooperative halo loads.
        image = np.random.default_rng(20261029).integers(0, 256, (35, 37), dtype=np.uint8)
        root = self.runtime._root
        root.set_reference(image)
        slot = root._pipeline_slots[0]
        root._upload_frame(slot, image)
        encoder = root.device.create_command_encoder(label="GPU tiled-prefilter regression")
        passes = (
            (root._prefilter_x, ((image.shape[1] + 31) // 32, (image.shape[0] + 7) // 8, 1)),
            (root._prefilter_y, ((image.shape[1] + 7) // 8, (image.shape[0] + 31) // 32, 1)),
        )
        for pipeline, workgroups in passes:
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, slot.bind_group)
            compute_pass.dispatch_workgroups(*workgroups)
            compute_pass.end()
        root.device.queue.submit([encoder.finish()])
        actual = np.frombuffer(
            root.device.queue.read_buffer(slot.buffers["coefficients"]),
            dtype=np.float32,
        ).reshape(image.shape)
        np.testing.assert_allclose(actual, _cpu_prefilter(image), rtol=2e-6, atol=2e-5)

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
