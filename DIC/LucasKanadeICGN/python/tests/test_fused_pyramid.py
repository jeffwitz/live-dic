"""Native dispatch fusion and optional temporal correction contracts."""

import os
import unittest

import numpy as np
import test_temporal_init
from PIL import Image
from temporal_cases import FIXTURES

from lucas_kanade_icgn import ICGNWGPUFullPyramid


class CorrectionArgumentsTests(unittest.TestCase):
    def test_invalid_correction_configuration_fails_before_device_creation(self):
        for options in (
            {"initialization": "zero"},
            {"initialization": "temporal", "pyramid_levels": 1},
            {"initialization": "temporal", "pyramid_levels": 3},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "temporal_fine_correction"),
            ):
                ICGNWGPUFullPyramid(temporal_fine_correction=True, **options)
        for value in (None, 0, 1, "yes"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                ICGNWGPUFullPyramid(temporal_fine_correction=value)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests explicitly disabled")
class CorrectedTemporalTests(test_temporal_init.TemporalGpuTests):
    """The same reset/ordering/expiry/ROI tests also cover coarse history."""

    def engine(self, **kwargs):
        enabled = (
            kwargs.get("initialization") == "temporal" and kwargs.get("pyramid_levels", 2) == 2
        )
        return super().engine(temporal_fine_correction=enabled, **kwargs)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests explicitly disabled")
class UHDFusionTests(unittest.TestCase):
    def test_native_uhd_models_and_roi_preserve_scientific_accuracy(self):
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
            for corrected in (False, True):
                with (
                    self.subTest(model=model, policy=policy, correction=corrected),
                    ICGNWGPUFullPyramid(
                        motion_model=model,
                        temporal_fine_correction=corrected,
                        outlier_policy="median",
                        initialization="temporal",
                        subset_size=size,
                        subset_stride=16,
                        spatial_weighting=weighting,
                    ) as engine,
                ):
                    info = engine.set_reference(
                        reference, **({"roi_mask": mask, "roi_policy": policy} if policy else {})
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
                    if corrected:
                        self.assertTrue(result.config["temporalFineCorrection"])


if __name__ == "__main__":
    unittest.main()
