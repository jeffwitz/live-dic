"""Opt-in native solver geometry, with scientific checks at native UHD."""

import os
import unittest

import numpy as np
import test_temporal_init
from PIL import Image
from temporal_cases import FIXTURES

from lucas_kanade_icgn import ICGNWGPUFullPyramid


class WorkgroupArgumentsTests(unittest.TestCase):
    def test_invalid_sizes_fail_before_device_creation(self):
        for value in (None, True, np.bool_(True), 8, 64, 32.5, "32"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "solver_workgroup_size"),
            ):
                ICGNWGPUFullPyramid(solver_workgroup_size=value)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests explicitly disabled")
class WideTemporalTests(test_temporal_init.TemporalGpuTests):
    """Apply the same ordering/reset/ROI/capture contracts to 32 lanes."""

    def engine(self, **kwargs):
        return super().engine(solver_workgroup_size=32, **kwargs)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests explicitly disabled")
class UHDWorkgroupTests(unittest.TestCase):
    def test_native_uhd_motion_models_and_roi_keep_scientific_accuracy(self):
        with Image.open(FIXTURES / "reference.png") as image:
            reference = np.array(image)
        target = np.roll(reference, (-1, 1), axis=(0, 1))
        mask = np.ones_like(reference, bool)
        mask[:, :63] = False
        mask[600:900, 1601:1603] = False
        for model, policy, size, weighting in (
            ("translation", None, 16, "uniform"),
            ("affine", "full", 16, "uniform"),
            ("affine", "partial", 16, "uniform"),
            ("quadratic", None, 16, "uniform"),
            ("affine", "partial", 17, "gaussian"),
        ):
            for lanes in (16, 32):
                with (
                    self.subTest(model=model, policy=policy, lanes=lanes),
                    ICGNWGPUFullPyramid(
                        motion_model=model,
                        solver_workgroup_size=lanes,
                        outlier_policy="median",
                        initialization="temporal",
                        subset_size=size,
                        subset_stride=16,
                        spatial_weighting=weighting,
                    ) as engine,
                ):
                    info = engine.set_reference(
                        reference,
                        **({"roi_mask": mask, "roi_policy": policy} if policy else {}),
                    )
                    result = list(engine.register_stream([target] * 3, pipeline_depth=3))[-1]
                    y, x = np.indices(result.status.shape) * 16 + (size - 1) / 2
                    population = (x >= 64) & (x < 3776) & (y >= 64) & (y < 2096)
                    selected = population if policy is None else population & info.roi_selected
                    accepted = selected & result.accepted_mask
                    self.assertGreater(accepted.sum() / selected.sum(), 0.90)
                    error = np.linalg.norm(result.displacement[accepted] - (1, -1), axis=-1)
                    self.assertLess(float(np.sqrt(np.mean(error**2))), 0.05)
                    self.assertLess(float(np.percentile(error, 95)), 0.10)
                    if policy:
                        np.testing.assert_array_equal(result.status[~result.roi_selected], -3)
                    if lanes == 32:
                        self.assertEqual(result.config["solverWorkgroupSize"], 32)


if __name__ == "__main__":
    unittest.main()
