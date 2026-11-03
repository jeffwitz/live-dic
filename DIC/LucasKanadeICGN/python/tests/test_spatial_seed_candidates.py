"""Independent integer-shift and ROI controls for the optional CPU research oracle."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "benchmarks/spatial_seed_reference.py"
spec = importlib.util.spec_from_file_location("spatial_seed_reference", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional for research benchmarks")
class SpatialCandidateTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.random.default_rng(42).integers(0, 256, (64, 64), dtype=np.uint8)
        self.target = np.roll(self.reference, (-1, 2), axis=(0, 1))
        selected = np.zeros((5, 5), bool)
        selected[2, 2:4] = True
        self.roi = SimpleNamespace(selected=selected, neighbour_bits=np.zeros((5, 5), np.uint32))
        self.parameters = np.zeros((5, 5, 6), np.float32)
        self.parameters[2, 3, [0, 3]] = [2, -1]

    def test_scored_neighbour_recovers_known_shift_without_mutating_raw_field(self):
        self.roi.neighbour_bits[2, 2] = 1 << 5
        original = self.parameters.copy()
        seeds, scores = module.select_translation_candidates(
            self.reference, self.target, self.parameters, self.roi, size=8, stride=8
        )
        np.testing.assert_array_equal(seeds[2, 2, [0, 3]], [2, -1])
        self.assertGreater(scores[2, 2], 0.999)
        np.testing.assert_array_equal(original, self.parameters)
        self.assertFalse(seeds[~self.roi.selected].any())

    def test_forbidden_link_never_supplies_a_candidate(self):
        seeds, _ = module.select_translation_candidates(
            self.reference, self.target, self.parameters, self.roi, size=8, stride=8
        )
        np.testing.assert_array_equal(seeds[2, 2, [0, 3]], [0, 0])

    def test_good_own_estimate_wins_over_a_bad_neighbour(self):
        self.roi.neighbour_bits[2, 2] = 1 << 5
        self.parameters[2, 2, [0, 3]] = [2, -1]
        self.parameters[2, 3, [0, 3]] = [100000, -100000]
        seeds, _ = module.select_translation_candidates(
            self.reference, self.target, self.parameters, self.roi, size=8, stride=8
        )
        np.testing.assert_array_equal(seeds[2, 2, [0, 3]], [2, -1])
        self.assertTrue(np.isfinite(seeds).all())
