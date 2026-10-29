from __future__ import annotations

import unittest
from importlib.resources import files
from inspect import signature
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lucas_kanade_icgn import (
    DEFAULT_FULL_PYRAMID_LEVELS,
    DEFAULT_IN_FLIGHT_FRAMES,
    DEFAULT_PYRAMID_LEVELS,
    ICGNWGPUFullPyramid,
    RegistrationResult,
)
from lucas_kanade_icgn.full_pyramid_backend import (
    _DEFAULT_ITERATION_SCHEDULES,
    _DEFAULT_TOLERANCES,
    _schedule,
)
from lucas_kanade_icgn.subset import normalize_subset_stride, subset_grid_shape
from lucas_kanade_icgn.wgpu_backend import (
    FIR_RADIUS,
    FIR_RUNTIME_RADIUS,
    ICGNWGPU,
    _compare_solver_outputs,
    _fir_weights,
    _pyramid_dependency_mask,
    _reduce_half,
    _reference_arrays,
    _transfer_pyramid_seed,
    _validated_config,
)


class NativeReferenceTests(unittest.TestCase):
    def test_solver_autotuning_accepts_numerically_equivalent_output(self) -> None:
        portable = np.zeros((200, 8), dtype=np.float32)
        portable[:, 0] = 2.0
        portable[:, 3] = -1.0
        portable[:, 6] = 1.0
        candidate = portable.copy()
        candidate[:, :6] += np.float32(5e-4)

        diagnostics = _compare_solver_outputs(
            portable,
            candidate,
            grid_count=portable.shape[0],
        )

        self.assertTrue(diagnostics["valid"])
        self.assertEqual(diagnostics["reason"], "compatible")

    def test_solver_autotuning_rejects_incompatible_output(self) -> None:
        portable = np.zeros((200, 8), dtype=np.float32)
        portable[:, 6] = 1.0
        candidate = portable.copy()
        candidate[:10, 6] = -1.0

        diagnostics = _compare_solver_outputs(
            portable,
            candidate,
            grid_count=portable.shape[0],
        )

        self.assertFalse(diagnostics["valid"])
        self.assertEqual(diagnostics["reason"], "accepted subset masks disagree")

    def test_solver_autotuning_accepts_matching_rejected_output(self) -> None:
        portable = np.zeros((4, 8), dtype=np.float32)
        portable[:, 6] = -2.0
        candidate = portable.copy()

        diagnostics = _compare_solver_outputs(
            portable,
            candidate,
            grid_count=portable.shape[0],
        )

        self.assertTrue(diagnostics["valid"])
        self.assertEqual(diagnostics["reason"], "compatible with no accepted subsets")

    def test_full_pyramid_has_explicit_coarse_to_fine_defaults(self) -> None:
        default = signature(ICGNWGPUFullPyramid).parameters["pyramid_levels"].default
        self.assertEqual(default, DEFAULT_FULL_PYRAMID_LEVELS)
        self.assertEqual(default, 2)
        self.assertEqual(_DEFAULT_ITERATION_SCHEDULES[1], (30,))
        self.assertEqual(_DEFAULT_ITERATION_SCHEDULES[2], (8, 16))
        self.assertEqual(_DEFAULT_ITERATION_SCHEDULES[3], (6, 8, 16))
        self.assertEqual(_DEFAULT_TOLERANCES, (1e-2, 5e-3, 5e-3))
        self.assertEqual(
            _schedule(None, level_count=2, defaults=(8, 12, 30), name="iterations"),
            (12, 30),
        )

    def test_full_pyramid_schedule_requires_one_value_per_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "one value per pyramid level"):
            _schedule((8, 12), level_count=3, defaults=(8, 12, 30), name="iterations")

    def test_invalid_full_pyramid_reference_config_preserves_state(self) -> None:
        engine = object.__new__(ICGNWGPUFullPyramid)
        engine.pyramid_levels = 2
        engine.iterations_per_level = (12, 30)
        engine.tolerance_per_level = (0.005, 0.001)
        engine._zncc_min = 0.8
        engine._reference_ready = True
        engine._clear_level_resources = mock.Mock()

        with self.assertRaisesRegex(ValueError, "between 1 and 30"):
            engine.set_reference(
                np.zeros((64, 64), dtype=np.uint8),
                iterations_per_level=(0, 30),
            )

        self.assertEqual(engine.iterations_per_level, (12, 30))
        self.assertEqual(engine.tolerance_per_level, (0.005, 0.001))
        self.assertEqual(engine._zncc_min, 0.8)
        self.assertTrue(engine._reference_ready)
        engine._clear_level_resources.assert_not_called()

    def test_native_engine_defaults_to_one_image_scale(self) -> None:
        default = signature(ICGNWGPU).parameters["pyramid_levels"].default
        self.assertEqual(default, DEFAULT_PYRAMID_LEVELS)
        self.assertEqual(default, 1)

    def test_subset_size_is_validated_before_gpu_creation(self) -> None:
        for engine_type in (ICGNWGPU, ICGNWGPUFullPyramid):
            for value in (7, 12, 64, True, 16.0):
                with (
                    self.subTest(engine=engine_type.__name__, value=value),
                    self.assertRaisesRegex(ValueError, "subset_size"),
                ):
                    engine_type(subset_size=value)

    def test_subset_stride_defaults_to_size_and_validates_overlap(self) -> None:
        self.assertEqual(normalize_subset_stride(None, subset_size=32), 32)
        self.assertEqual(normalize_subset_stride(16, subset_size=32), 16)
        self.assertEqual(
            subset_grid_shape(3840, 2160, subset_size=32, subset_stride=16),
            (239, 134),
        )
        for value in (0, 33, True, 16.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "subset_stride"):
                normalize_subset_stride(value, subset_size=32)

    def test_overlapping_reference_layout_keeps_per_subset_means(self) -> None:
        image = np.arange(64 * 96, dtype=np.uint8).reshape(64, 96)
        reference, inverse, metadata = _reference_arrays(
            image,
            subset_size=32,
            subset_stride=16,
            interpolation="linear",
        )
        grid_width, grid_height = 5, 3
        reference_pixels = reference.reshape(64, 96, 4)
        metadata_rows = metadata.reshape(grid_height, grid_width, 16)
        self.assertEqual(inverse.shape, (grid_width * grid_height * 36,))
        np.testing.assert_allclose(reference_pixels[..., 0], image, rtol=0.0, atol=0.0)
        expected_means = np.asarray(
            [
                [image[y : y + 32, x : x + 32].mean() for x in range(0, 65, 16)]
                for y in range(0, 33, 16)
            ]
        )
        np.testing.assert_allclose(metadata_rows[..., 14], expected_means, atol=1e-5)

    def test_reference_layout_supports_all_subset_sizes(self) -> None:
        image = np.arange(64 * 96, dtype=np.uint8).reshape(64, 96)
        for subset_size in (8, 16, 32):
            with self.subTest(subset_size=subset_size):
                reference, inverse, metadata = _reference_arrays(
                    image,
                    subset_size=subset_size,
                )
                grid_count = (96 // subset_size) * (64 // subset_size)
                self.assertEqual(reference.shape, (64 * 96 * 4,))
                self.assertEqual(inverse.shape, (grid_count * 36,))
                self.assertEqual(metadata.shape, (grid_count * 16,))
                self.assertTrue(np.isfinite(reference).all())
                self.assertTrue(np.isfinite(inverse).all())
                self.assertTrue(np.isfinite(metadata).all())

    def test_pyramid_level_count_is_validated_before_gpu_creation(self) -> None:
        for levels in (0, 4, 1.5, True):
            with (
                self.subTest(levels=levels),
                self.assertRaisesRegex(
                    ValueError,
                    "pyramid_levels",
                ),
            ):
                ICGNWGPU(pyramid_levels=levels)

    def test_half_reduction_preserves_constant_image(self) -> None:
        image = np.full((32, 48), 137, dtype=np.uint8)
        reduced = _reduce_half(image)
        self.assertEqual(reduced.shape, (16, 24))
        np.testing.assert_array_equal(reduced, 137)

    def test_half_reduction_matches_full_five_tap_convolution(self) -> None:
        image = np.random.default_rng(20261028).integers(
            0,
            256,
            size=(32, 48),
            dtype=np.uint8,
        )
        source = image.astype(np.float32)
        taps = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float32) / 16.0
        padded_x = np.pad(source, ((0, 0), (2, 2)), mode="reflect")
        horizontal = sum(
            taps[index] * padded_x[:, index : index + image.shape[1]] for index in range(5)
        )[:, ::2]
        padded_y = np.pad(horizontal, ((2, 2), (0, 0)), mode="reflect")
        expected = sum(
            taps[index] * padded_y[index : index + image.shape[0]] for index in range(5)
        )[::2]
        np.testing.assert_array_equal(
            _reduce_half(image),
            np.clip(expected, 0.0, 255.0).astype(np.uint8),
        )

    def test_pyramid_transfer_scales_translation_and_respects_mask(self) -> None:
        parameters = np.zeros((2, 2, 6), dtype=np.float32)
        parameters[..., 0] = 2.0
        parameters[..., 3] = -1.0
        result = RegistrationResult(
            parameters=parameters,
            status=np.ones((2, 2), dtype=np.float32),
            iterations=np.ones((2, 2), dtype=np.float32),
            accepted=4,
            replaced=0,
            mean_translation=(2.0, -1.0),
            mean_iterations=1.0,
            timing={},
            config={},
            width=32,
            height=32,
            subset_size=16,
        )
        requested = np.zeros((2, 2), dtype=np.bool_)
        requested[0, 1] = True
        seeds, active = _transfer_pyramid_seed(
            result,
            source_available=np.ones((2, 2), dtype=np.bool_),
            source_scale=2,
            destination_scale=1,
            destination_grid=(2, 2),
            requested=requested,
        )
        self.assertEqual(int(active.sum()), 1)
        np.testing.assert_allclose(seeds[0, 1, [0, 3]], (4.0, -2.0))

    def test_pyramid_transfer_ignores_uncomputed_zero_status(self) -> None:
        parameters = np.zeros((2, 2, 6), dtype=np.float32)
        parameters[0, 0, 0] = 4.0
        result = RegistrationResult(
            parameters=parameters,
            status=np.zeros((2, 2), dtype=np.float32),
            iterations=np.zeros((2, 2), dtype=np.float32),
            accepted=0,
            replaced=0,
            mean_translation=None,
            mean_iterations=None,
            timing={},
            config={},
            width=32,
            height=32,
            subset_size=16,
        )
        requested = np.zeros((2, 2), dtype=np.bool_)
        requested[0, 0] = True
        available = np.zeros((2, 2), dtype=np.bool_)
        available[0, 0] = True
        seeds, active = _transfer_pyramid_seed(
            result,
            source_available=available,
            source_scale=2,
            destination_scale=1,
            destination_grid=(2, 2),
            requested=requested,
        )
        self.assertEqual(int(active[0, 0]), 1)
        self.assertEqual(float(seeds[0, 0, 0]), 8.0)

    def test_pyramid_transfer_falls_back_to_zero_seed(self) -> None:
        result = RegistrationResult(
            parameters=np.zeros((2, 2, 6), dtype=np.float32),
            status=np.zeros((2, 2), dtype=np.float32),
            iterations=np.zeros((2, 2), dtype=np.float32),
            accepted=0,
            replaced=0,
            mean_translation=None,
            mean_iterations=None,
            timing={},
            config={},
            width=32,
            height=32,
            subset_size=16,
        )
        requested = np.zeros((2, 2), dtype=np.bool_)
        requested[1, 1] = True
        seeds, active = _transfer_pyramid_seed(
            result,
            source_available=np.zeros((2, 2), dtype=np.bool_),
            source_scale=2,
            destination_scale=1,
            destination_grid=(2, 2),
            requested=requested,
        )
        self.assertEqual(int(active[1, 1]), 1)
        np.testing.assert_array_equal(seeds[1, 1], 0.0)

    def test_quadratic_pyramid_transfer_recenters_and_rescales_curvature(self) -> None:
        parameters = np.zeros((2, 2, 12), dtype=np.float32)
        parameters[..., 6] = 0.02
        result = RegistrationResult(
            parameters=parameters,
            status=np.ones((2, 2), dtype=np.float32),
            iterations=np.ones((2, 2), dtype=np.float32),
            accepted=4,
            replaced=0,
            mean_translation=(0.0, 0.0),
            mean_iterations=1.0,
            timing={},
            config={"motionModel": "quadratic"},
        )

        seeds, active = _transfer_pyramid_seed(
            result,
            source_available=np.ones((2, 2), dtype=np.bool_),
            source_scale=2,
            destination_scale=1,
            destination_grid=(1, 1),
            requested=np.ones((1, 1), dtype=np.bool_),
        )

        self.assertEqual(seeds.shape, (1, 1, 16))
        self.assertEqual(active[0, 0], 1)
        self.assertAlmostEqual(float(seeds[0, 0, 0]), 0.28125, places=6)
        self.assertAlmostEqual(float(seeds[0, 0, 1]), -0.075, places=6)
        self.assertAlmostEqual(float(seeds[0, 0, 6]), 0.01, places=6)

    def test_pyramid_dependency_mask_keeps_four_interpolation_sources(self) -> None:
        requested = np.zeros((4, 4), dtype=np.bool_)
        requested[2, 2] = True
        mask = _pyramid_dependency_mask(
            source_grid=(4, 4),
            source_scale=2,
            destination_scale=1,
            requested=requested,
        )
        self.assertEqual(int(mask.sum()), 4)

    def test_sequence_apis_default_to_two_images_in_flight(self) -> None:
        self.assertEqual(DEFAULT_IN_FLIGHT_FRAMES, 2)
        for method in (
            ICGNWGPU.register_stream,
            ICGNWGPU.register_many,
            ICGNWGPUFullPyramid.register_stream,
        ):
            with self.subTest(method=method.__name__):
                default = signature(method).parameters["pipeline_depth"].default
                self.assertEqual(default, DEFAULT_IN_FLIGHT_FRAMES)

    def test_pipeline_depth_is_a_positive_integer(self) -> None:
        for engine_type in (ICGNWGPU, ICGNWGPUFullPyramid):
            engine = object.__new__(engine_type)
            engine._reference_ready = True
            for depth in (0, -1, 1.5, True):
                with (
                    self.subTest(engine=engine_type.__name__, depth=depth),
                    self.assertRaisesRegex(
                        ValueError,
                        "pipeline_depth",
                    ),
                ):
                    next(engine.register_stream((), pipeline_depth=depth))

    def test_stream_retains_recovery_frame_when_input_buffer_is_reused(self) -> None:
        engine = object.__new__(ICGNWGPU)
        engine.pyramid_levels = 2
        engine._solver = object()
        engine._upload_frame = mock.Mock(return_value=0.1)
        engine._encode_registration = mock.Mock(return_value="command")
        engine.device = SimpleNamespace(queue=SimpleNamespace(submit=mock.Mock()))
        frame = np.full((16, 16), 7, dtype=np.uint8)
        pending = engine._submit_slot(object(), frame, {"iterations": 30})
        frame.fill(9)
        self.assertFalse(np.shares_memory(pending.frame, frame))
        np.testing.assert_array_equal(pending.frame, 7)

    def test_invalid_reference_configuration_preserves_prepared_state(self) -> None:
        engine = object.__new__(ICGNWGPU)
        engine.subset_size = 16
        engine.height = 64
        engine.width = 64
        engine.grid_height = 4
        engine.grid_width = 4
        engine._reference_ready = True
        engine._config = {"iterations": 30, "tolerance": 1e-3, "znccMin": 0.8}
        engine._close_pyramid_engines = mock.Mock()
        with self.assertRaisesRegex(ValueError, "iterations"):
            engine.set_reference(np.zeros((128, 128), dtype=np.uint8), iterations=0)
        self.assertEqual((engine.height, engine.width), (64, 64))
        self.assertTrue(engine._reference_ready)
        engine._close_pyramid_engines.assert_not_called()

    def test_shader_is_packaged_as_a_runtime_resource(self) -> None:
        shader = files("lucas_kanade_icgn").joinpath("shaders/icgn.wgsl")
        subgroup_source = files("lucas_kanade_icgn").joinpath("shaders/icgn_subgroup.comp")
        subgroup_binary = files("lucas_kanade_icgn").joinpath("shaders/icgn_subgroup.spv")
        probe_binary = files("lucas_kanade_icgn").joinpath("shaders/subgroup_probe.spv")
        pyramid_shader = files("lucas_kanade_icgn").joinpath("shaders/full_pyramid.wgsl")
        shader_text = shader.read_text(encoding="utf-8")
        self.assertIn("fn solve", shader_text)
        self.assertIn("override PREFILTER_CLAMP_HALO", shader_text)
        self.assertIn("return clamp(reflected, 0, size - 1)", shader_text)
        self.assertIn("array<vec4<f32>>", shader_text)
        for helper in (
            "solve_standard_increment",
            "solve_quadratic_increment",
            "compose_standard_inverse_update",
            "compose_quadratic_inverse",
            "standard_corner_change",
            "quadratic_change",
            "convergence_status",
            "solve_seeded",
        ):
            self.assertIn(f"fn {helper}", shader_text)
        self.assertIn("workgroupUniformLoad", shader_text)
        self.assertIn("subgroupAdd", subgroup_source.read_text(encoding="utf-8"))
        self.assertEqual(subgroup_binary.read_bytes()[:4], b"\x03\x02#\x07")
        self.assertEqual(probe_binary.read_bytes()[:4], b"\x03\x02#\x07")
        pyramid_text = pyramid_shader.read_text(encoding="utf-8")
        for entry_point in ("reduce_x", "reduce_y", "transfer_seed"):
            self.assertIn(f"fn {entry_point}", pyramid_text)
        self.assertIn("texture_storage_2d<r32float, write>", pyramid_text)
        self.assertNotIn("rgba16float", pyramid_text)

    def test_solver_controls_are_validated(self) -> None:
        self.assertEqual(_validated_config(30, 1e-3, 0.8)["iterations"], 30)
        for values in ((0, 1e-3, 0.8), (31, 1e-3, 0.8), (30, 0.0, 0.8), (30, 1e-3, 1.1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                _validated_config(*values)

    def test_fir_coefficients_are_finite_and_normalized(self) -> None:
        weights = _fir_weights()
        self.assertEqual(weights.shape, (31,))
        self.assertEqual(weights.dtype, np.float32)
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        trim = FIR_RADIUS - FIR_RUNTIME_RADIUS
        omitted = np.concatenate((weights[:trim], weights[-trim:]))
        self.assertLess(float(np.abs(omitted).sum()), 7e-8)

    def test_reference_arrays_have_gpu_buffer_layout(self) -> None:
        random = np.random.default_rng(20261028)
        image = random.integers(0, 256, size=(32, 48), dtype=np.uint8)
        reference, inverse, metadata = _reference_arrays(image)
        self.assertEqual(reference.shape, (32 * 48 * 4,))
        self.assertEqual(inverse.shape, (2 * 3 * 36,))
        self.assertEqual(metadata.shape, (2 * 3 * 16,))
        self.assertTrue(np.isfinite(reference).all())
        self.assertTrue(np.isfinite(inverse).all())
        self.assertTrue(np.all(metadata.reshape(-1, 16)[:, 1] == 1.0))

    def test_translation_reference_uses_analytic_two_parameter_inverse(self) -> None:
        image = np.random.default_rng(20261029).integers(0, 256, size=(32, 48), dtype=np.uint8)
        reference, inverse, metadata = _reference_arrays(
            image,
            motion_model="translation",
        )
        gradients = reference.reshape(32, 48, 4)[..., 1:3]
        inverse_blocks = inverse.reshape(-1, 6, 6)
        metadata_blocks = metadata.reshape(-1, 16)
        active = np.array([0, 3])
        inactive = np.array([1, 2, 4, 5])

        np.testing.assert_array_equal(inverse_blocks[:, inactive, :], 0.0)
        np.testing.assert_array_equal(inverse_blocks[:, :, inactive], 0.0)
        np.testing.assert_array_equal(metadata_blocks[:, 2:8][:, inactive], 0.0)
        np.testing.assert_array_equal(metadata_blocks[:, 8:14][:, inactive], 0.0)

        for grid_y in range(2):
            for grid_x in range(3):
                index = grid_y * 3 + grid_x
                tile = gradients[
                    grid_y * 16 : (grid_y + 1) * 16,
                    grid_x * 16 : (grid_x + 1) * 16,
                ].reshape(-1, 2)
                hessian = tile.T @ tile
                analytic_inverse = inverse_blocks[index][np.ix_(active, active)]
                np.testing.assert_allclose(
                    hessian @ analytic_inverse,
                    np.eye(2),
                    rtol=2e-5,
                    atol=2e-5,
                )

    def test_quadratic_reference_uses_twelve_parameter_layout(self) -> None:
        image = np.random.default_rng(91).integers(0, 256, size=(32, 32), dtype=np.uint8)

        reference, inverse, metadata = _reference_arrays(
            image,
            interpolation="linear",
            motion_model="quadratic",
        )

        self.assertEqual(reference.size, 32 * 32 * 4)
        self.assertEqual(inverse.size, 4 * 12 * 12)
        self.assertEqual(metadata.size, 4 * 28)
        self.assertTrue(np.all(np.isfinite(inverse)))
        self.assertTrue(np.all(np.isfinite(metadata)))

    def test_reference_interpolation_modes_are_finite_and_distinct(self) -> None:
        random = np.random.default_rng(20261029)
        image = random.integers(0, 256, size=(32, 48), dtype=np.uint8)
        default = _reference_arrays(image)
        explicit_bspline = _reference_arrays(image, interpolation="bspline")
        for default_array, explicit_array in zip(default, explicit_bspline, strict=True):
            np.testing.assert_array_equal(default_array, explicit_array)

        linear = _reference_arrays(image, interpolation="linear")
        cubic = _reference_arrays(image, interpolation="cubic")
        for prepared in (linear, cubic):
            self.assertTrue(all(np.isfinite(array).all() for array in prepared))
        # Both direct interpolants have the same integer-grid reference model.
        for linear_array, cubic_array in zip(linear, cubic, strict=True):
            np.testing.assert_array_equal(linear_array, cubic_array)
        self.assertFalse(np.array_equal(default[0], linear[0]))

    def test_unknown_reference_interpolation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "interpolation"):
            _reference_arrays(np.zeros((32, 32), dtype=np.uint8), interpolation="nearest")

    def test_native_interpolation_is_validated_before_gpu_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "interpolation"):
            ICGNWGPU(interpolation="nearest")
        with self.assertRaisesRegex(ValueError, "prefer_subgroups"):
            ICGNWGPU(interpolation="linear", prefer_subgroups=True)
        with self.assertRaisesRegex(ValueError, "interpolation"):
            ICGNWGPUFullPyramid(interpolation="nearest")

    def test_native_motion_model_is_validated_before_gpu_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "motion_model"):
            ICGNWGPU(motion_model="rigid")
        with self.assertRaisesRegex(ValueError, "motion_model"):
            ICGNWGPUFullPyramid(motion_model="rigid")
        with self.assertRaisesRegex(ValueError, "prefer_subgroups"):
            ICGNWGPU(motion_model="translation", prefer_subgroups=True)

    def test_flat_reference_has_finite_rejected_inverse_hessians(self) -> None:
        image = np.full((64, 64), 137, dtype=np.uint8)
        _, inverse, metadata = _reference_arrays(image)

        self.assertTrue(np.isfinite(inverse).all())
        self.assertTrue(np.all(inverse == 0.0))
        self.assertTrue(np.all(metadata.reshape(-1, 16)[:, 1] == 0.0))


if __name__ == "__main__":
    unittest.main()
