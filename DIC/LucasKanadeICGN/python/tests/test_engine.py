from __future__ import annotations

import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from lucas_kanade_icgn.engine import (
    PARAMETER_ORDER,
    BenchmarkResult,
    ICGNError,
    ICGNWebGL,
    RegistrationResult,
    _array_data_url,
    _as_uint8_gray,
    _chromium_backend_arguments,
    _chromium_base_arguments,
    _chromium_candidates,
    _find_chromium,
    _find_runtime_assets,
    _run_async_blocking,
    _subprocess_session_arguments,
    register_pair,
)


class ImageConversionTests(unittest.TestCase):
    def test_higher_bit_depth_array_requires_explicit_conversion(self) -> None:
        image = np.linspace(0, 4095, 64 * 64, dtype=np.uint16).reshape(64, 64)
        with self.assertRaisesRegex(ValueError, "Convert higher-bit-depth images explicitly"):
            _as_uint8_gray(image)

    def test_higher_bit_depth_pillow_image_requires_explicit_conversion(self) -> None:
        image = Image.fromarray(np.full((64, 64), 1024, dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "Convert higher-bit-depth images explicitly"):
            _as_uint8_gray(image)

    def test_integer_array_in_uint8_range_is_accepted(self) -> None:
        image = np.full((64, 64), 137, dtype=np.uint16)
        converted = _as_uint8_gray(image)
        self.assertEqual(converted.dtype, np.uint8)
        self.assertTrue(np.all(converted == 137))

    def test_float_image_in_unit_interval_is_scaled(self) -> None:
        image = np.full((512, 512), 0.5, dtype=np.float32)
        converted = _as_uint8_gray(image)
        self.assertEqual(converted.dtype, np.uint8)
        self.assertEqual(int(converted[0, 0]), 127)

    def test_arbitrary_frame_dimensions_are_preserved(self) -> None:
        image = np.zeros((64, 96), dtype=np.uint8)
        self.assertEqual(_as_uint8_gray(image).shape, (64, 96))

    def test_uhd_frame_is_accepted(self) -> None:
        image = np.zeros((2160, 3840), dtype=np.uint8)
        self.assertEqual(_as_uint8_gray(image).shape, (2160, 3840))

    def test_normalized_array_can_be_encoded_without_rescaling(self) -> None:
        image = np.full((16, 32), 73, dtype=np.uint8)
        self.assertTrue(_array_data_url(image).startswith("data:image/png;base64,"))

    def test_too_small_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 16"):
            _as_uint8_gray(np.zeros((15, 64), dtype=np.uint8))

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
            "timing": {"solveWallMs": 4.2, "gpuMs": None},
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
        self.assertNotIn("gpuMs", result.timing)

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
    def test_browser_discovery_is_lazy(self) -> None:
        with mock.patch(
            "lucas_kanade_icgn.engine._find_chromium",
            side_effect=AssertionError("browser discovery must wait for start()"),
        ):
            engine = ICGNWebGL()
        self.assertIsNone(engine.chromium)

    def test_missing_browser_does_not_allocate_runtime_resources(self) -> None:
        engine = ICGNWebGL()
        with (
            mock.patch(
                "lucas_kanade_icgn.engine._find_chromium",
                side_effect=ICGNError("missing browser"),
            ),
            self.assertRaisesRegex(ICGNError, "missing browser"),
        ):
            engine.start()
        self.assertIsNone(engine._server)
        self.assertIsNone(engine._temporary)
        self.assertIsNone(engine._log_handle)

    def test_browser_candidates_cover_supported_platform_families(self) -> None:
        self.assertIn("chromium", _chromium_candidates("linux"))
        self.assertIn("chrome.exe", _chromium_candidates("win32"))
        self.assertTrue(
            any(".app/Contents/MacOS/" in path for path in _chromium_candidates("darwin"))
        )

    def test_browser_discovery_uses_platform_candidates(self) -> None:
        with mock.patch(
            "lucas_kanade_icgn.engine.shutil.which",
            side_effect=lambda name: "C:/Chrome/chrome.exe" if name == "chrome.exe" else None,
        ):
            self.assertEqual(_find_chromium(None, platform="win32"), "C:/Chrome/chrome.exe")

    def test_browser_gpu_arguments_do_not_force_vulkan_off_linux(self) -> None:
        for platform in ("win32", "darwin"):
            with self.subTest(platform=platform):
                arguments = _chromium_backend_arguments(platform, "webgpu-wgsl", False)
                self.assertIn("--enable-unsafe-webgpu", arguments)
                self.assertFalse(any("vulkan" in argument.lower() for argument in arguments))
                self.assertNotIn("--use-angle=gl", _chromium_base_arguments(platform))

    def test_linux_browser_gpu_arguments_keep_native_vulkan_path(self) -> None:
        arguments = _chromium_backend_arguments("linux", "webgpu-wgsl", False)
        self.assertIn("--use-angle=vulkan", arguments)
        self.assertIn("--use-angle=gl", _chromium_base_arguments("linux"))

    def test_browser_process_session_arguments_are_platform_safe(self) -> None:
        self.assertEqual(_subprocess_session_arguments("win32"), {})
        self.assertEqual(_subprocess_session_arguments("linux"), {"start_new_session": True})
        self.assertEqual(_subprocess_session_arguments("darwin"), {"start_new_session": True})

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "webgpu-wgsl"):
            ICGNWebGL(backend="unknown")

    def test_benchmark_target_requires_prepared_reference(self) -> None:
        engine = ICGNWebGL()
        with self.assertRaisesRegex(ICGNError, "set_reference"):
            engine.benchmark_target(np.zeros((64, 64), dtype=np.uint8))

    def test_public_api_defaults_to_three_tsl(self) -> None:
        engine_default = inspect.signature(ICGNWebGL).parameters["backend"].default
        one_shot_default = inspect.signature(register_pair).parameters["backend"].default
        self.assertEqual(engine_default, "three-tsl")
        self.assertEqual(one_shot_default, "three-tsl")
        self.assertIn("pyramid_levels", inspect.signature(register_pair).parameters)

    def test_shared_wgsl_backend_defaults_to_two_pyramid_levels(self) -> None:
        engine = ICGNWebGL(backend="webgpu-wgsl")
        self.assertEqual(engine.pyramid_levels, 2)
        self.assertIsNone(engine.iterations_per_level)
        self.assertEqual(
            engine._options(iterations=None, tolerance=None, zncc_min=None)["pyramidLevels"],
            2,
        )

    def test_interpolation_mode_is_validated_and_forwarded(self) -> None:
        for mode in ("bspline", "linear", "cubic"):
            with self.subTest(mode=mode):
                engine = ICGNWebGL(backend="webgpu-wgsl", interpolation=mode)
                self.assertEqual(engine.interpolation, mode)
                self.assertEqual(
                    engine._options(iterations=None, tolerance=None, zncc_min=None)[
                        "interpolation"
                    ],
                    mode,
                )
        with self.assertRaisesRegex(ValueError, "interpolation"):
            ICGNWebGL(interpolation="nearest")
        with self.assertRaisesRegex(ValueError, "supports only"):
            ICGNWebGL(backend="gpuio", interpolation="linear")

    def test_browser_pyramid_options_are_validated(self) -> None:
        for levels in (0, 4, 1.5, True):
            with self.subTest(levels=levels), self.assertRaisesRegex(ValueError, "pyramid_levels"):
                ICGNWebGL(backend="webgpu-wgsl", pyramid_levels=levels)
        with self.assertRaisesRegex(ValueError, "only.*webgpu-wgsl"):
            ICGNWebGL(backend="three-tsl", pyramid_levels=2)
        with self.assertRaisesRegex(ValueError, "one value per pyramid level"):
            ICGNWebGL(
                backend="webgpu-wgsl",
                pyramid_levels=3,
                iterations_per_level=(8, 20),
            )
        for schedule in ((12.5, 30), (True, 30)):
            with (
                self.subTest(schedule=schedule),
                self.assertRaisesRegex(ValueError, "iterations_per_level"),
            ):
                ICGNWebGL(
                    backend="webgpu-wgsl",
                    pyramid_levels=2,
                    iterations_per_level=schedule,
                )

    def test_browser_motion_model_is_validated_and_forwarded(self) -> None:
        with self.assertRaisesRegex(ValueError, "motion_model"):
            ICGNWebGL(motion_model="rigid")
        with self.assertRaisesRegex(ValueError, "supports only"):
            ICGNWebGL(backend="gpuio", motion_model="translation")

        engine = ICGNWebGL(backend="three-tsl", motion_model="translation")
        options = engine._options(iterations=30, tolerance=1e-3, zncc_min=0.8)
        self.assertEqual(options["motionModel"], "translation")

    def test_three_webgl_fallback_is_explicitly_testable(self) -> None:
        engine = ICGNWebGL(backend="three-tsl", force_three_webgl=True)
        self.assertTrue(engine.force_three_webgl)
        self.assertFalse(any("webgpu" in argument.lower() for argument in engine.chromium_args))
        with self.assertRaisesRegex(ValueError, "only.*three-tsl"):
            ICGNWebGL(backend="gpuio", force_three_webgl=True)

    def test_packaged_runtime_is_complete(self) -> None:
        assets = _find_runtime_assets(None, "gpuio")
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
            assets = _find_runtime_assets(root, "gpuio")
            self.assertTrue(assets.root.samefile(root))
        self.assertEqual(assets.headless_path, "headless.html")

    def test_packaged_three_tsl_runtime_is_complete_and_offline(self) -> None:
        assets = _find_runtime_assets(None, "three-tsl")
        self.assertEqual(assets.headless_path, "three/headless.html")
        for filename in (
            "three/headless.html",
            "three/LucasKanadeICGNThree.js",
            "three/vendor/three.core.js",
            "three/vendor/three.webgpu.js",
            "three/vendor/three.tsl.js",
            "three/vendor/THREE-LICENSE.txt",
            "common/ReferencePreparation.js",
        ):
            self.assertTrue((assets.root / filename).is_file(), filename)
        implementation = (assets.root / "three/LucasKanadeICGNThree.js").read_text(encoding="utf-8")
        self.assertIn("makeSerialSolverKernel", implementation)
        self.assertIn("forceWebGL", implementation)
        self.assertIn("reflected.clamp(0, size - 1)", implementation)
        self.assertIn("validateSubgroupSolver", implementation)
        self.assertIn("portable-workgroup-subgroup-mismatch", implementation)
        page = (assets.root / assets.headless_path).read_text(encoding="utf-8")
        self.assertNotIn("https://", page)
        self.assertNotIn("http://", page)

    def test_explicit_checkout_three_directory_resolves_canonical_source(self) -> None:
        module_root = Path(__file__).resolve().parents[2]
        assets = _find_runtime_assets(module_root / "three", "three-tsl")
        self.assertEqual(assets.root, module_root)
        self.assertEqual(assets.headless_path, "three/headless.html")

    def test_default_runtime_is_three_tsl(self) -> None:
        assets = _find_runtime_assets(None)
        self.assertEqual(assets.root.name, "web")
        self.assertTrue((assets.root / "three/LucasKanadeICGNThree.js").is_file())

    def test_shared_wgsl_runtime_is_complete_and_offline(self) -> None:
        assets = _find_runtime_assets(None, "webgpu-wgsl")
        self.assertEqual(assets.headless_path, "web/webgpu/headless.html")
        for filename in (
            "web/webgpu/headless.html",
            "web/webgpu/LucasKanadeICGNWebGPU.js",
            "web/common/ReferencePreparation.js",
            "shaders/icgn.wgsl",
            "shaders/full_pyramid.wgsl",
        ):
            self.assertTrue((assets.root / filename).is_file(), filename)
        page = (assets.root / assets.headless_path).read_text(encoding="utf-8")
        self.assertNotIn("https://", page)
        self.assertNotIn("http://", page)
        runtime = (assets.root / "web/webgpu/LucasKanadeICGNWebGPU.js").read_text(encoding="utf-8")
        self.assertIn("format: 'r32float'", runtime)
        self.assertIn("PREFILTER_CLAMP_HALO", runtime)
        self.assertIn("prefilterXSafe", runtime)
        self.assertNotIn("rgba16float", runtime)

    def test_checkout_three_entry_point_delegates_to_packaged_source(self) -> None:
        module_root = Path(__file__).resolve().parents[2]
        entry_point = module_root / "three/LucasKanadeICGNThree.js"
        source = entry_point.read_text(encoding="utf-8")
        self.assertIn("python/src/lucas_kanade_icgn/web/three", source)
        self.assertEqual(source.count("import "), 1)

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
