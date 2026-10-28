from __future__ import annotations

import unittest
from importlib.resources import files

import numpy as np

from lucas_kanade_icgn.wgpu_backend import (
    _fir_weights,
    _reference_arrays,
    _validated_config,
)


class NativeReferenceTests(unittest.TestCase):
    def test_shader_is_packaged_as_a_runtime_resource(self) -> None:
        shader = files("lucas_kanade_icgn").joinpath("shaders/icgn.wgsl")
        self.assertIn("fn solve", shader.read_text(encoding="utf-8"))

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
