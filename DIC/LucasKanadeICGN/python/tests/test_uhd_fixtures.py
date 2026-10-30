"""Offline native-UHD fixtures and numerical regression against tensile truth."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from uhd_partial_cases import TRANSLATION, UHD_SHAPE, boundary_mask, crack_pair, translation_pair

from lucas_kanade_icgn import ICGNWGPU, ICGNWGPUFullPyramid

FIXTURES = Path(__file__).resolve().parents[2] / "assets" / "uhd_tensile"


def _manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def _image(name: str) -> np.ndarray:
    with Image.open(FIXTURES / name) as image:
        return np.array(image, dtype=np.uint8)


def _truth() -> dict[int, dict[str, str]]:
    with (FIXTURES / "machine_truth.csv").open(newline="", encoding="utf-8") as stream:
        return {int(row["frame"]): row for row in csv.DictReader(stream)}


def _interior(result, margin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_y, grid_x = np.indices(result.status.shape)
    x = grid_x * result.subset_stride + (result.subset_size - 1) / 2
    y = grid_y * result.subset_stride + (result.subset_size - 1) / 2
    valid = (x >= margin) & (x < result.width - margin)
    valid &= (y >= margin) & (y < result.height - margin)
    return valid, x, y


class UHDFixtureIntegrityTests(unittest.TestCase):
    def test_original_uhd_images_and_provenance_are_present_and_unchanged(self) -> None:
        manifest = _manifest()
        self.assertEqual((manifest["width"], manifest["height"]), (3840, 2160))
        self.assertEqual({row["frame"] for row in manifest["images"]}, {0, 40, 100, 199})
        for row in manifest["images"]:
            with self.subTest(file=row["file"]):
                path = FIXTURES / row["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
                with Image.open(path) as image:
                    self.assertEqual(image.size, (3840, 2160))
                    self.assertEqual(image.mode, "L")
        for name, expected in manifest["provenance_files"].items():
            with self.subTest(file=name):
                self.assertEqual(
                    hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), expected
                )

    def test_archived_truth_covers_each_selected_frame(self) -> None:
        truth = _truth()
        self.assertEqual(set(truth), set(range(200)))
        self.assertEqual(float(truth[0]["stretch_x"]), 1.0)
        self.assertEqual(float(truth[0]["stretch_y"]), 1.0)
        for row in _manifest()["images"]:
            with self.subTest(frame=row["frame"]):
                state = truth[row["frame"]]
                self.assertAlmostEqual(
                    float(state["engineering_axial"]), float(state["stretch_x"]) - 1
                )
                self.assertAlmostEqual(
                    float(state["engineering_transverse"]), float(state["stretch_y"]) - 1
                )


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "set ICGN_RUN_GPU_TESTS=1")
class UHDRegistrationRegressionTests(unittest.TestCase):
    def test_uhd_partial_other_sizes_overlap_and_gaussian_supports(self) -> None:
        reference, target = translation_pair("archived-speckle")
        roi = boundary_mask()
        configurations = (
            (15, 7, "translation", "bspline", "uniform"),
            (31, 16, "affine", "cubic", "uniform"),
            (32, 32, "affine", "bspline", "uniform"),
            (33, 16, "affine", "bspline", "gaussian"),
        )
        for size, stride, model, interpolation, weighting in configurations:
            with (
                self.subTest(size=size, stride=stride, model=model, weighting=weighting),
                ICGNWGPU(
                    motion_model=model,
                    subset_size=size,
                    subset_stride=stride,
                    interpolation=interpolation,
                    spatial_weighting=weighting,
                    prefer_subgroups=False,
                    outlier_policy="disabled",
                ) as engine,
            ):
                info = engine.set_reference(
                    reference, roi_mask=roi, roi_policy="partial", tolerance=0.005
                )
                result = engine.register(target)
                self.assertEqual((result.height, result.width), UHD_SHAPE)
                partial = np.zeros(result.status.shape, bool)
                partial.ravel()[info.roi.partial_indices] = True
                self.assertTrue(np.isfinite(result.parameters).all())
                self.assertFalse(np.any(result.status == -4))
                for population in (partial, info.roi_selected & ~partial):
                    accepted = population & result.accepted_mask
                    self.assertGreater(population.sum(), 0)
                    self.assertGreaterEqual(accepted.sum() / population.sum(), 0.90)
                    distance = np.linalg.norm(result.displacement[accepted] - TRANSLATION, axis=-1)
                    self.assertLess(float(np.sqrt(np.mean(distance**2))), 0.05)
                    self.assertLess(float(np.percentile(distance, 95)), 0.10)

    def test_uhd_fractional_partial_roi_translation_8_and_16(self) -> None:
        roi = boundary_mask()
        for scene in ("analytic-wave", "archived-speckle"):
            reference, target = translation_pair(scene)
            self.assertEqual(reference.shape, UHD_SHAPE)
            self.assertEqual(target.shape, UHD_SHAPE)
            for size in (8, 16):
                with (
                    self.subTest(scene=scene, size=size),
                    ICGNWGPUFullPyramid(
                        motion_model="translation",
                        subset_size=size,
                        interpolation="bspline",
                        outlier_policy="median",
                    ) as engine,
                ):
                    info = engine.set_reference(reference, roi_mask=roi, roi_policy="partial")
                    result = engine.register(target)
                    self.assertTrue(np.isfinite(result.parameters).all())
                    self.assertFalse(np.any(result.status == -4))
                    partial = np.zeros(result.status.shape, bool)
                    partial.ravel()[info.roi.partial_indices] = True
                    for population in (partial, info.roi_selected & ~partial):
                        self.assertGreater(population.sum(), 0)
                        accepted = population & result.accepted_mask
                        self.assertGreaterEqual(accepted.sum() / population.sum(), 0.90)
                        distance = np.linalg.norm(
                            result.displacement[accepted] - TRANSLATION, axis=-1
                        )
                        self.assertLess(float(np.sqrt(np.mean(distance**2))), 0.05)
                        self.assertLess(float(np.percentile(distance, 95)), 0.10)

    def test_uhd_visibility_limits_mixing_at_a_crack_tip(self) -> None:
        reference, target, roi, point = crack_pair()
        self.assertEqual(reference.shape, UHD_SHAPE)
        self.assertEqual(target.shape, UHD_SHAPE)
        for model in ("translation", "affine"):
            errors = {}
            for truncation in ("none", "visibility"):
                with (
                    self.subTest(model=model, truncation=truncation),
                    ICGNWGPU(
                        motion_model=model,
                        subset_size=17,
                        interpolation="linear",
                        outlier_policy="disabled",
                        prefer_subgroups=False,
                    ) as engine,
                ):
                    info = engine.set_reference(
                        reference,
                        roi_mask=roi,
                        roi_policy="partial",
                        roi_truncation=truncation,
                        tolerance=0.005,
                    )
                    result = engine.register(target)
                    self.assertTrue(info.roi_selected[point])
                    self.assertEqual(result.status[point], 1)
                    self.assertTrue(np.isfinite(result.parameters).all())
                    errors[truncation] = np.linalg.norm(result.displacement[point] - TRANSLATION)
            self.assertGreater(errors["none"], 0.1)
            self.assertLess(errors["visibility"], 0.05)

    def test_uhd_partial_roi_complete_and_boundary_populations(self) -> None:
        reference, target = _image("reference.png"), _image("frame_040.png")
        y, x = np.indices(reference.shape)
        roi = ((x - 1920) / 1600) ** 2 + ((y - 1080) / 900) ** 2 <= 1
        roi[400:1205, 1930:1932] = False
        state = _truth()[40]
        ex, ey = float(state["stretch_x"]) - 1, float(state["stretch_y"]) - 1
        for model in ("translation", "affine"):
            for truncation in ("none", "visibility"):
                with (
                    self.subTest(model=model, truncation=truncation),
                    ICGNWGPUFullPyramid(motion_model=model, outlier_policy="median") as engine,
                ):
                    info = engine.set_reference(
                        reference, roi_mask=roi, roi_policy="partial", roi_truncation=truncation
                    )
                    current = target if model == "affine" else np.roll(reference, (-1, 2), (0, 1))
                    result = engine.register(current)
                    self.assertFalse(np.any(result.status == -4))
                    self.assertTrue(np.isfinite(result.parameters).all())
                    partial = np.zeros(result.status.shape, bool)
                    partial.ravel()[info.roi.partial_indices] = True
                    _, cx, cy = _interior(result, 32)
                    truth = (
                        np.stack((ex * (cx - 1920), ey * (cy - 1080)), axis=-1)
                        if model == "affine"
                        else np.broadcast_to([2.0, -1.0], result.displacement.shape)
                    )
                    for population in (partial, info.roi_selected & ~partial):
                        self.assertGreater(population.sum(), 0)
                        accepted = population & result.accepted_mask
                        self.assertGreaterEqual(accepted.sum() / population.sum(), 0.90)
                        distance = np.linalg.norm(
                            result.displacement[accepted] - truth[accepted], axis=-1
                        )
                        self.assertLess(float(np.sqrt(np.mean(distance**2))), 0.05)
                        self.assertLess(float(np.percentile(distance, 95)), 0.10)
                    # Existing complete-window median gradient bound stays in
                    # force; boundary gradients are reported independently.
                    if model == "affine":
                        complete = result.accepted_mask & ~partial
                        error = result.parameters[..., (1, 2, 4, 5)][complete] - [ex, 0, 0, ey]
                        self.assertLess(float(np.sqrt(np.mean(error**2))), 0.002)

    def test_uhd_roi_preserves_affine_precision_and_pure_translation(self) -> None:
        reference, target = _image("reference.png"), _image("frame_040.png")
        limits = _manifest()["regression_limits"]
        state = _truth()[40]
        ex, ey = float(state["stretch_x"]) - 1, float(state["stretch_y"]) - 1
        roi = np.zeros(reference.shape, bool)
        roi[160:2000, 160:3680] = True
        roi[880:1200, 1800:2040] = False
        for model in ("affine", "translation"):
            for policy in ("disabled", "median"):
                with (
                    self.subTest(model=model, policy=policy),
                    ICGNWGPUFullPyramid(motion_model=model, outlier_policy=policy) as engine,
                ):
                    info = engine.set_reference(reference, roi_mask=roi)
                    current = target if model == "affine" else np.roll(reference, (-1, 2), (0, 1))
                    result = engine.register(current)
                    interior, x, y = _interior(result, limits["interior_margin_px"])
                    population = interior & info.roi_selected
                    accepted = population & result.accepted_mask
                    self.assertGreaterEqual(accepted.sum() / population.sum(), 0.90)
                    self.assertTrue(np.all(result.status[~info.roi_selected] == -3))
                    truth = (
                        np.stack((ex * (x - 1920), ey * (y - 1080)), axis=-1)
                        if model == "affine"
                        else np.broadcast_to([2.0, -1.0], result.displacement.shape)
                    )
                    distance = np.linalg.norm(
                        result.displacement[accepted] - truth[accepted], axis=-1
                    )
                    self.assertLess(float(np.sqrt(np.mean(distance**2))), 0.05)
                    self.assertLess(float(np.percentile(distance, 95)), 0.10)
                    if model == "affine" and policy == "median":
                        error = result.parameters[..., (1, 2, 4, 5)][accepted] - [ex, 0, 0, ey]
                        self.assertLess(float(np.sqrt(np.mean(error**2))), 0.002)

    def test_all_archived_uhd_images_recover_identity(self) -> None:
        limits = _manifest()["regression_limits"]
        with ICGNWGPUFullPyramid(
            motion_model="translation",
            subset_size=16,
            subset_stride=16,
            interpolation="bspline",
            outlier_policy="median",
        ) as engine:
            try:
                for row in _manifest()["images"]:
                    with self.subTest(file=row["file"]):
                        image = _image(row["file"])
                        engine.set_reference(image)
                        result = engine.register(image)
                        self.assertEqual(result.parameters.shape, (135, 240, 6))
                        self.assertTrue(np.isfinite(result.parameters).all())
                        interior, _, _ = _interior(result, limits["interior_margin_px"])
                        mask = result.accepted_mask & interior
                        self.assertGreaterEqual(
                            mask.sum() / interior.sum(),
                            limits["identity_minimum_accepted_fraction"],
                        )
                        displacement = result.parameters[..., (0, 3)][mask]
                        self.assertLess(
                            float(np.sqrt(np.mean(np.sum(displacement**2, axis=-1)))),
                            limits["identity_displacement_rmse_px"],
                        )
                        np.testing.assert_array_equal(result.parameters[..., (1, 2, 4, 5)], 0)
            finally:
                engine.close()

    def test_uhd_affine_frame_040_displacement_and_median_gradients(self) -> None:
        limits = _manifest()["regression_limits"]
        state = _truth()[40]
        ex, ey = float(state["stretch_x"]) - 1, float(state["stretch_y"]) - 1
        reference, target = _image("reference.png"), _image("frame_040.png")
        # Keep raw and explicitly smoothed measurements qualified independently.
        for policy in ("disabled", "median"):
            with self.subTest(policy=policy):
                engine = ICGNWGPUFullPyramid(
                    motion_model="affine",
                    subset_size=16,
                    subset_stride=16,
                    interpolation="bspline",
                    outlier_policy=policy,
                )
                try:
                    engine.set_reference(reference)
                    result = engine.register(target)
                    self.assertTrue(np.isfinite(result.parameters).all())
                    self.assertEqual(result.parameters.shape, (135, 240, 6))
                    interior, x, y = _interior(result, limits["interior_margin_px"])
                    mask = interior & result.accepted_mask
                    self.assertGreaterEqual(
                        mask.sum() / interior.sum(),
                        limits["affine_frame_040_minimum_accepted_fraction"],
                    )
                    expected = np.stack((ex * (x - 1920), ey * (y - 1080)), axis=-1)
                    displacement_error = result.parameters[..., (0, 3)][mask] - expected[mask]
                    distance = np.linalg.norm(displacement_error, axis=-1)
                    self.assertLess(
                        float(np.sqrt(np.mean(distance**2))),
                        limits["affine_frame_040_displacement_rmse_px"],
                    )
                    self.assertLess(
                        float(np.percentile(distance, 95)),
                        limits["affine_frame_040_displacement_p95_px"],
                    )
                    gradient_error = result.parameters[..., (1, 2, 4, 5)][mask] - [ex, 0, 0, ey]
                    if policy == "median":
                        # Raw 16/16 derivatives are not qualified to this bound
                        # on these bilinearly rendered images; see fixture README.
                        self.assertLess(
                            float(np.sqrt(np.mean(gradient_error**2))),
                            limits["affine_frame_040_median_gradient_rmse_per_component"],
                        )
                finally:
                    engine.close()
