"""Opt-in latency policies preserve measurements and explicit fallbacks."""

import os
import unittest

import numpy as np
import test_temporal_init
from test_gpu_regressions import _fractional_pair

from lucas_kanade_icgn import ICGNWGPUFullPyramid


class LatencyArgumentsTests(unittest.TestCase):
    def test_prediction_validation_precedes_gpu_creation(self):
        for value in (None, True, "linear", "", []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "prediction"):
                ICGNWGPUFullPyramid(temporal_prediction=value)
        with self.assertRaisesRegex(ValueError, "requires temporal"):
            ICGNWGPUFullPyramid(temporal_prediction="velocity")

    def test_abort_and_adaptive_arguments_are_validated_before_gpu_creation(self):
        for name in ("divergence_patience", "divergence_min_iterations"):
            for value in (0, -1, True, 1.5, "3", 31):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    ICGNWGPUFullPyramid(**{name: value})
        for options in (
            {"pyramid_policy": "unknown"},
            {"pyramid_policy": "adaptive"},
            {"pyramid_policy": "adaptive", "initialization": "temporal", "pyramid_levels": 1},
            {"adaptive_acceptance": float("nan")},
            {"adaptive_acceptance": 0},
            {"adaptive_acceptance": 1.01},
            {"adaptive_acceptance": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ICGNWGPUFullPyramid(**options)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests explicitly disabled")
class VelocityTemporalTests(test_temporal_init.TemporalGpuTests):
    """Run ordering, corruption, expiry, reset and cropped-tensile contracts."""

    def engine(self, **kwargs):
        prediction = "velocity" if kwargs.get("initialization") == "temporal" else "none"
        return super().engine(temporal_prediction=prediction, **kwargs)

    def test_prediction_requires_two_measurements_and_stops_after_corruption(self):
        engine = self.engine(initialization="temporal", outlier_policy="median")
        first = engine.register(self.targets[0])
        second = engine.register(self.targets[1])
        self.assertEqual(first.timing["velocityPredictedSubsets"], 0)
        self.assertEqual(second.timing["velocityPredictedSubsets"], 0)
        third = engine.register(self.targets[2])
        expected = int(np.count_nonzero(first.accepted_mask & second.accepted_mask))
        self.assertEqual(third.timing["velocityPredictedSubsets"], expected)
        self.assertEqual(third.timing["temporalAcceptedUpdates"], third.accepted)
        black = engine.register(np.zeros_like(self.reference))
        self.assertEqual(black.timing["temporalAcceptedUpdates"], 0)
        recovered = engine.register(self.targets[2])
        self.assertEqual(recovered.timing["velocityPredictedSubsets"], 0)
        self.assertGreater(recovered.accepted, 0)
        engine.reset_temporal_state()
        self.assert_field_equal(first, engine.register(self.targets[0]))


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests explicitly disabled")
class AbortAndAdaptiveTests(unittest.TestCase):
    def test_abort_preserves_early_convergence_and_disabled_path(self):
        reference, target = _fractional_pair(256, 256, (4.3, -2.4))
        fields = []
        for patience in (None, 30, 2):
            with ICGNWGPUFullPyramid(
                pyramid_levels=1, outlier_policy="disabled", divergence_patience=patience
            ) as engine:
                engine.set_reference(reference)
                fields.append(engine.register(target))
        baseline, inactive, stopped = fields
        for name in ("parameters", "status", "iterations"):
            np.testing.assert_array_equal(getattr(baseline, name), getattr(inactive, name))
            # min=4, patience=2: the first possible abort is iteration 5.
            early = baseline.accepted_mask & (baseline.iterations <= 5)
            np.testing.assert_array_equal(
                getattr(baseline, name)[early], getattr(stopped, name)[early]
            )
        self.assertTrue(np.any((stopped.status == 0) & (stopped.iterations < 30)))
        self.assertTrue(np.isfinite(stopped.parameters).all())

    def test_adaptive_feedback_switches_resets_and_is_identical_at_all_depths(self):
        reference, _ = _fractional_pair(256, 256, (0, 0))
        targets = [_fractional_pair(256, 256, (i * 0.2, -i * 0.1))[1] for i in range(4)]
        sequence = [*targets, np.zeros_like(reference), targets[-1], targets[-1]]
        with ICGNWGPUFullPyramid(
            initialization="temporal",
            temporal_prediction="velocity",
            pyramid_policy="adaptive",
            # The historical rejected outer ring occupies 60/256 positions.
            adaptive_acceptance=0.7,
            temporal_max_age=0,
            outlier_policy="median",
            temporal_fine_correction=True,
        ) as engine:
            engine.set_reference(reference)
            expected = None
            for depth in (1, 2, 3):
                engine.reset_temporal_state()
                outputs = list(engine.register_stream(sequence, pipeline_depth=depth))
                levels = [r.timing["pyramidLevels"] for r in outputs]
                self.assertEqual(levels[:2], [2, 1])
                # Individual failed windows may expire even on a healthy
                # noisy field; that must also request the coarse solve.
                self.assertEqual(levels[5], 2)  # after the corrupted image
                self.assertGreater(outputs[4].timing["temporalExpiredSubsets"], 0)
                if expected is None:
                    expected = outputs
                for old, new in zip(expected, outputs, strict=True):
                    for name in ("parameters", "status", "iterations"):
                        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
                    self.assertEqual(old.timing["pyramidLevels"], new.timing["pyramidLevels"])
            engine.set_reference(reference)
            self.assertEqual(engine.register(targets[0]).timing["pyramidLevels"], 2)


if __name__ == "__main__":
    unittest.main()
