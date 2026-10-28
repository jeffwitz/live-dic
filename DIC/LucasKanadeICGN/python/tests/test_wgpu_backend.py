from __future__ import annotations

import unittest
from importlib.resources import files
from inspect import signature

import numpy as np

from lucas_kanade_icgn import DEFAULT_IN_FLIGHT_FRAMES
from lucas_kanade_icgn.wgpu_backend import (
    ICGNWGPU,
    _fir_weights,
    _reference_arrays,
    _validated_config,
)


class NativeReferenceTests(unittest.TestCase):
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
