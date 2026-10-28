from __future__ import annotations

import unittest
from importlib.resources import files
from inspect import signature

import numpy as np

from lucas_kanade_icgn import (
    DEFAULT_IN_FLIGHT_FRAMES,
    DEFAULT_PYRAMID_LEVELS,
    RegistrationResult,
)
from lucas_kanade_icgn.wgpu_backend import (
    ICGNWGPU,
    _fir_weights,
    _pyramid_dependency_mask,
    _reduce_half,
    _reference_arrays,
    _transfer_pyramid_seed,
    _validated_config,
)


class NativeReferenceTests(unittest.TestCase):
    def test_native_engine_defaults_to_one_image_scale(self) -> None:
        default = signature(ICGNWGPU).parameters["pyramid_levels"].default
        self.assertEqual(default, DEFAULT_PYRAMID_LEVELS)
        self.assertEqual(default, 1)

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
            source_scale=2,
            destination_scale=1,
            destination_grid=(2, 2),
            requested=requested,
        )
        self.assertEqual(int(active.sum()), 1)
        np.testing.assert_allclose(seeds[0, 1, [0, 3]], (4.0, -2.0))

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
        for method in (ICGNWGPU.register_stream, ICGNWGPU.register_many):
            with self.subTest(method=method.__name__):
                default = signature(method).parameters["pipeline_depth"].default
                self.assertEqual(default, DEFAULT_IN_FLIGHT_FRAMES)

    def test_pipeline_depth_is_a_positive_integer(self) -> None:
        engine = object.__new__(ICGNWGPU)
        engine._reference_ready = True
        for depth in (0, -1, 1.5, True):
            with (
                self.subTest(depth=depth),
                self.assertRaisesRegex(
                    ValueError,
                    "pipeline_depth",
                ),
            ):
                next(engine.register_stream((), pipeline_depth=depth))

    def test_shader_is_packaged_as_a_runtime_resource(self) -> None:
        shader = files("lucas_kanade_icgn").joinpath("shaders/icgn.wgsl")
        subgroup_source = files("lucas_kanade_icgn").joinpath("shaders/icgn_subgroup.comp")
        subgroup_binary = files("lucas_kanade_icgn").joinpath("shaders/icgn_subgroup.spv")
        probe_binary = files("lucas_kanade_icgn").joinpath("shaders/subgroup_probe.spv")
        shader_text = shader.read_text(encoding="utf-8")
        self.assertIn("fn solve", shader_text)
        self.assertIn("array<vec4<f32>>", shader_text)
        for helper in (
            "solve_increment",
            "compose_inverse_update",
            "corner_change",
            "convergence_status",
            "solve_seeded",
        ):
            self.assertIn(f"fn {helper}", shader_text)
        self.assertIn("workgroupUniformLoad", shader_text)
        self.assertIn("subgroupAdd", subgroup_source.read_text(encoding="utf-8"))
        self.assertEqual(subgroup_binary.read_bytes()[:4], b"\x03\x02#\x07")
        self.assertEqual(probe_binary.read_bytes()[:4], b"\x03\x02#\x07")

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


if __name__ == "__main__":
    unittest.main()
