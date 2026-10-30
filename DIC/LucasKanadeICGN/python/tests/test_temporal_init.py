"""Temporal history contracts; GPU execution is explicitly opt-in."""

import os
import unittest

import numpy as np
from PIL import Image
from temporal_cases import FIXTURES, tensile_stretches, warp
from test_gpu_regressions import _fractional_pair

from lucas_kanade_icgn import ICGNWGPUFullPyramid


class TemporalArgumentsTests(unittest.TestCase):
    def test_invalid_arguments_are_rejected_before_device_creation(self):
        for value in (None, "", "predictive", 1):
            with (
                self.subTest(initialization=value),
                self.assertRaisesRegex(ValueError, "initialization"),
            ):
                ICGNWGPUFullPyramid(initialization=value)
        for value in (-1, True, 1.5, "2", 2**32):
            with self.subTest(age=value), self.assertRaisesRegex(ValueError, "temporal_max_age"):
                ICGNWGPUFullPyramid(temporal_max_age=value)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests explicitly disabled")
class TemporalGpuTests(unittest.TestCase):
    def setUp(self):
        self.reference, _ = _fractional_pair(256, 256, (0, 0))
        self.targets = [_fractional_pair(256, 256, (0.3 * i, -0.2 * i))[1] for i in range(1, 5)]

    def engine(self, **kwargs):
        engine = ICGNWGPUFullPyramid(**kwargs)
        self.addCleanup(engine.close)
        engine.set_reference(self.reference)
        return engine

    def assert_field_equal(self, a, b):
        for name in ("parameters", "status", "iterations"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)

    def test_zero_default_and_first_temporal_frame_match_bitwise(self):
        for model in ("translation", "affine", "quadratic"):
            for levels in (1, 2, 3):
                with self.subTest(model=model, levels=levels):
                    options = dict(motion_model=model, pyramid_levels=levels)
                    default = self.engine(**options)
                    zero = self.engine(initialization="zero", **options)
                    temporal = self.engine(initialization="temporal", **options)
                    a = default.register(self.targets[0])
                    self.assert_field_equal(a, zero.register(self.targets[0]))
                    b = temporal.register(self.targets[0])
                    self.assert_field_equal(a, b)
                    self.assertEqual(b.timing["temporallySeededCoarseSubsets"], 0)
                    self.assertFalse(default._temporal_buffers)
                    default.close()
                    zero.close()
                    temporal.close()

    def test_ordered_stream_matches_sequential_at_all_depths_and_roi_policies(self):
        y, x = np.indices(self.reference.shape)
        mask = (x > 8) & (x < 248) & (y > 8) & (y < 248)
        mask[40:70, 128] = False
        for policy in (None, "full", "partial"):
            engine = self.engine(initialization="temporal")
            if policy:
                engine.set_reference(self.reference, roi_mask=mask, roi_policy=policy)
            sequential = [engine.register(target) for target in self.targets]
            self.assertGreater(sequential[-1].timing["temporallySeededCoarseSubsets"], 0)
            for depth in (1, 2, 3):
                engine.reset_temporal_state()
                streamed = list(engine.register_stream(self.targets, pipeline_depth=depth))
                for a, b in zip(sequential, streamed, strict=True):
                    self.assert_field_equal(a, b)
                    if policy:
                        np.testing.assert_array_equal(b.status[~b.roi_selected], -3)
            engine.close()

    def test_explicit_reset_and_new_reference_clear_history(self):
        engine = self.engine(initialization="temporal", pyramid_levels=1)
        first = engine.register(self.targets[0])
        self.assertGreater(
            engine.register(self.targets[0]).timing["temporallySeededCoarseSubsets"], 0
        )
        engine.reset_temporal_state()
        self.assert_field_equal(first, engine.register(self.targets[0]))
        engine.set_reference(self.reference)
        fresh = engine.register(self.targets[0])
        self.assert_field_equal(first, fresh)
        self.assertEqual(fresh.timing["temporallySeededCoarseSubsets"], 0)

    def test_corrupted_frame_preserves_history_and_age_expiration(self):
        black = np.zeros_like(self.reference)
        for max_age, expected_seed in ((None, True), (0, False), (1, True)):
            engine = self.engine(initialization="temporal", temporal_max_age=max_age)
            first = engine.register(self.targets[0])
            expected = engine.register(self.targets[0]) if expected_seed else first
            engine.reset_temporal_state()
            engine.register(self.targets[0])
            corrupted = engine.register(black)
            self.assertEqual(corrupted.accepted, 0)
            self.assertEqual(corrupted.timing["temporalAcceptedUpdates"], 0)
            recovered = engine.register(self.targets[0])
            self.assertEqual(recovered.timing["temporallySeededCoarseSubsets"] > 0, expected_seed)
            self.assertEqual(recovered.accepted, first.accepted)
            self.assert_field_equal(recovered, expected)
            engine.close()

    def test_empty_roi_does_not_submit_or_create_seeds(self):
        engine = self.engine(initialization="temporal")
        engine.set_reference(self.reference, roi_mask=np.zeros_like(self.reference, dtype=bool))
        for result in engine.register_stream(self.targets, pipeline_depth=3):
            self.assertEqual(result.accepted, 0)
            np.testing.assert_array_equal(result.status, -3)

    def test_ordered_tensile_sequence_extends_capture_range(self):
        # Crop permitted for this contract test; qualification remains native UHD.
        with Image.open(FIXTURES / "reference.png") as image:
            reference = np.array(image)[800:1312, 1600:2112]
        stretches = tensile_stretches()[::2]
        stretches = np.vstack((stretches, tensile_stretches()[-1]))
        engines = [self.engine(initialization=mode) for mode in ("zero", "temporal")]
        for engine in engines:
            engine.set_reference(reference)
        results = None
        for stretch in stretches:
            target = warp(reference, stretch)
            results = [engine.register(target) for engine in engines]
        zero, temporal = results
        y, x = np.indices(temporal.status.shape) * 16 + 7.5
        support = (x >= 32) & (x < 480) & (y >= 48) & (y < 464)
        truth = np.stack(((x - 256) * (stretch[0] - 1), (y - 256) * (stretch[1] - 1)), -1)
        good = support & temporal.accepted_mask
        self.assertGreater(good.sum() / support.sum(), 0.9)
        self.assertLess((support & zero.accepted_mask).sum() / support.sum(), 0.8)
        error = temporal.displacement[good] - truth[good]
        self.assertLess(np.sqrt(np.mean(np.sum(error**2, axis=-1))), 0.05)


if __name__ == "__main__":
    unittest.main()
