from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lucas_kanade_icgn.engine import (
    PARAMETER_ORDER,
    BenchmarkResult,
    RegistrationResult,
    _as_uint8_gray,
    _find_runtime_assets,
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


class RuntimeAssetTests(unittest.TestCase):
    def test_packaged_runtime_is_complete(self) -> None:
        assets = _find_runtime_assets(None)
        self.assertEqual(assets.headless_path, "headless.html")
        for filename in ("headless.html", "gpu-io.js", "LucasKanadeICGN.js"):
            self.assertTrue((assets.root / filename).is_file(), filename)
        page = (assets.root / "headless.html").read_text(encoding="utf-8")
        self.assertNotIn("https://", page)
        self.assertNotIn("http://", page)

    def test_explicit_direct_asset_directory_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in ("headless.html", "gpu-io.js", "LucasKanadeICGN.js"):
                (root / filename).touch()
            assets = _find_runtime_assets(root)
        self.assertEqual(assets.root, root)
        self.assertEqual(assets.headless_path, "headless.html")

    def test_packaged_three_tsl_runtime_is_complete_and_offline(self) -> None:
        assets = _find_runtime_assets(None, "three-tsl")
        self.assertEqual(assets.headless_path, "headless.html")
        for filename in (
            "headless.html",
            "LucasKanadeICGNThree.js",
            "vendor/three.core.js",
            "vendor/three.webgpu.js",
            "vendor/three.tsl.js",
            "vendor/THREE-LICENSE.txt",
        ):
            self.assertTrue((assets.root / filename).is_file(), filename)
        page = (assets.root / "headless.html").read_text(encoding="utf-8")
        self.assertNotIn("https://", page)
        self.assertNotIn("http://", page)

    def test_browser_demo_uses_local_script_dependencies(self) -> None:
        module_root = Path(__file__).resolve().parents[2]
        page = (module_root / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('<script src="http://', page)
        self.assertNotIn('<script src="https://', page)
        for relative in (
            "../../gpu-io.js",
            "../../tweakpane.js",
            "./vendor/tweakpane-plugin-essentials-0.1.0.min.js",
            "./LucasKanadeICGN.js",
        ):
            self.assertIn(f'src="{relative}"', page)
            self.assertTrue((module_root / relative).resolve().is_file(), relative)


if __name__ == "__main__":
    unittest.main()
