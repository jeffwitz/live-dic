"""Offline native-UHD plate data and independent analytic-reference checks."""

import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from plate_hole_reference import (
    CX,
    CY,
    RADIUS,
    displacement,
    grid,
    hole_mask,
    strain,
    von_mises_strain,
)

from lucas_kanade_icgn import ICGNWGPUFullPyramid

ASSETS = Path(__file__).resolve().parents[2] / "assets/uhd_plate_hole"


class PlateReferenceTests(unittest.TestCase):
    def test_native_images_are_original_and_have_analytic_loads(self):
        manifest = json.loads((ASSETS / "manifest.json").read_text())
        self.assertEqual((manifest["width"], manifest["height"]), (3840, 2160))
        for row in manifest["images"]:
            path = ASSETS / row["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["file_sha256"])
            with Image.open(path) as image:
                self.assertEqual(image.size, (3840, 2160))
                self.assertEqual(image.mode, "L")
        self.assertEqual(
            [(r["primary_MPa"], r["secondary_MPa"]) for r in manifest["images"]],
            [(0, 0), (250, 0), (250, 80)],
        )
        for name, record in manifest["source_files"].items():
            path = ASSETS / "provenance" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
        self.assertEqual(
            hashlib.sha256((ASSETS / "provenance/original_truth.csv").read_bytes()).hexdigest(),
            manifest["original_truth_csv_sha256"],
        )

    def test_numpy_truth_matches_original_cpp_at_boundary_and_interior_points(self):
        rows = np.genfromtxt(ASSETS / "provenance/original_truth.csv", delimiter=",", names=True)
        expected_u = np.stack((rows["ux"], rows["uy"]), axis=-1)
        expected_e = np.stack((rows["exx"], rows["eyy"], rows["exy"]), axis=-1)
        u = displacement(rows["x"], rows["y"], rows["primary"], rows["secondary"])
        e = strain(rows["x"], rows["y"], rows["primary"], rows["secondary"])
        np.testing.assert_allclose(u, expected_u, atol=2e-13, rtol=1e-12)
        np.testing.assert_allclose(e, expected_e, atol=2e-15, rtol=1e-12)
        np.testing.assert_allclose(von_mises_strain(e), rows["vm"], atol=2e-15, rtol=1e-12)

    def test_strain_matches_displacement_derivatives_independently(self):
        x = np.array([2040.0, 2007.5, 56.0, 3784.0])
        y = np.array([1080.0, 1167.5, 120.0, 2040.0])
        step = 0.001
        dx = (displacement(x + step, y, 250, 80) - displacement(x - step, y, 250, 80)) / (2 * step)
        dy = (displacement(x, y + step, 250, 80) - displacement(x, y - step, 250, 80)) / (2 * step)
        expected = np.stack((dx[:, 0], dy[:, 1], 0.5 * (dx[:, 1] + dy[:, 0])), axis=-1)
        np.testing.assert_allclose(strain(x, y, 250, 80), expected, atol=2e-11, rtol=1e-6)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "explicit GPU opt-in")
class UHDPlateRegressionTests(unittest.TestCase):
    def test_full_and_partial_plate_masks_on_the_actual_uhd_pair(self):
        with Image.open(ASSETS / "reference.png") as image:
            reference = np.array(image, dtype=np.uint8)
        with Image.open(ASSETS / "frame_095.png") as image:
            target = np.array(image, dtype=np.uint8)
        x, y = grid()
        radius = np.hypot(x - CX, y - CY)
        # An unchanged predeclared far-field population, not a post-hoc
        # exclusion of numerical failures around the hole.
        population = (radius >= 288) & (x >= 32) & (x < 3808) & (y >= 32) & (y < 2128)
        expected = displacement(x[population], y[population], 250)
        for policy in ("full", "partial"):
            with ICGNWGPUFullPyramid(motion_model="affine", outlier_policy="median") as engine:
                info = engine.set_reference(reference, roi_mask=hole_mask(), roi_policy=policy)
                result = engine.register(target)
                self.assertEqual((result.height, result.width), (2160, 3840))
                self.assertTrue(np.isfinite(result.parameters).all())
                self.assertTrue(np.all(result.status[~info.roi_selected] == -3))
                self.assertGreaterEqual(np.mean(result.accepted_mask[population]), 0.9)
                accepted = result.accepted_mask[population]
                error = result.displacement[population][accepted] - expected[accepted]
                self.assertLess(float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))), 0.05)
                if policy == "partial":
                    self.assertEqual(info.roi.partial_indices.size, 24)
                    self.assertTrue(np.all(result.accepted_mask.ravel()[info.roi.partial_indices]))
        self.assertEqual(RADIUS, 112)
