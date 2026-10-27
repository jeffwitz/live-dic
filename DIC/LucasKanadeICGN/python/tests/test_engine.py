from __future__ import annotations

import asyncio
import unittest

import numpy as np

from lucas_kanade_icgn.engine import (
    PARAMETER_ORDER,
    BenchmarkResult,
    RegistrationResult,
    _as_uint8_gray,
    _run_async_blocking,
)


class ImageConversionTests(unittest.TestCase):
    def test_float_image_in_unit_interval_is_scaled(self) -> None:
        image = np.full((512, 512), 0.5, dtype=np.float32)
        converted = _as_uint8_gray(image)
        self.assertEqual(converted.dtype, np.uint8)
        self.assertEqual(int(converted[0, 0]), 127)

    def test_wrong_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "512"):
            _as_uint8_gray(np.zeros((64, 64), dtype=np.uint8))

    def test_unsupported_channel_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "3 or 4 channels"):
            _as_uint8_gray(np.zeros((512, 512, 2), dtype=np.uint8))


class RegistrationResultTests(unittest.TestCase):
    def test_payload_layout_and_rmse(self) -> None:
        parameters = np.zeros((4, 6), dtype=np.float32)
        parameters[:, 0] = 2.0
        parameters[:, 3] = -1.0
        payload = {
            "grid": [2, 2],
            "parameters": parameters.ravel().tolist(),
            "status": [1, 1, 2, 0],
            "iterations": [4, 5, 6, 30],
            "accepted": 3,
            "replaced": 1,
            "meanTranslation": [2, -1],
            "meanIterations": 5,
            "timing": {"solveWallMs": 4.2},
            "config": {"iterations": 30},
            "width": 512,
            "height": 512,
            "subsetSize": 16,
        }
        result = RegistrationResult.from_payload(payload)
        self.assertEqual(result.parameters.shape, (2, 2, len(PARAMETER_ORDER)))
        self.assertEqual(result.displacement.shape, (2, 2, 2))
        self.assertEqual(int(result.accepted_mask.sum()), 3)
        self.assertEqual(int(result.replaced_mask.sum()), 1)
        self.assertEqual(result.rmse((2, -1)), 0.0)

    def test_benchmark_payload_keeps_timing_samples(self) -> None:
        registration = {
            "grid": [1, 1],
            "parameters": [0.0] * len(PARAMETER_ORDER),
            "status": [1],
            "iterations": [3],
            "accepted": 1,
            "replaced": 0,
            "meanTranslation": [0, 0],
            "meanIterations": 3,
            "width": 512,
            "height": 512,
            "subsetSize": 16,
        }
        payload = {
            "repeats": 2,
            "warmups": 1,
            "gpuTimes": [3.5, 3.7],
            "wallTimes": [16.0, 17.0],
            "gpuMedianMs": 3.7,
            "gpuP95Ms": 3.7,
            "wallMedianMs": 17.0,
            "wallP95Ms": 17.0,
            "device": "test device",
            "registration": registration,
        }
        result = BenchmarkResult.from_payload(payload)
        np.testing.assert_allclose(result.gpu_times_ms, [3.5, 3.7])
        np.testing.assert_allclose(result.wall_times_ms, [16.0, 17.0])
        self.assertEqual(result.registration.parameters.shape, (1, 1, 6))


class NotebookLoopTests(unittest.TestCase):
    def test_blocking_bridge_inside_running_loop(self) -> None:
        async def outer() -> int:
            return _run_async_blocking(asyncio.sleep(0, result=7))

        self.assertEqual(asyncio.run(outer()), 7)


if __name__ == "__main__":
    unittest.main()
