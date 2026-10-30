"""Independent support geometry and weighted IC-GN qualification for partial ROI."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from test_gpu_regressions import _fractional_pair, _quadratic_pair

from lucas_kanade_icgn import ICGNWGPU, ICGNWGPUFullPyramid
from lucas_kanade_icgn.partial import prepare_partial_reference
from lucas_kanade_icgn.roi import (
    _visibility_paths,
    pack_support,
    prepare_roi,
    unpack_support,
    validate_roi,
)
from lucas_kanade_icgn.wgpu_backend import _reference_arrays


def _bilinear(image, x, y):
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    return (1 - fy) * ((1 - fx) * image[iy, ix] + fx * image[iy, ix + 1]) + fy * (
        (1 - fx) * image[iy + 1, ix] + fx * image[iy + 1, ix + 1]
    )


def _keys_cubic(image, x, y):
    """Independent separable Keys kernel evaluated from sample distances."""
    ix = np.floor(x).astype(int)[:, None] + np.arange(-1, 3)
    iy = np.floor(y).astype(int)[:, None] + np.arange(-1, 3)

    def kernel(distance):
        z = np.abs(distance)
        return np.where(
            z < 1,
            1.5 * z**3 - 2.5 * z**2 + 1,
            np.where(z < 2, -0.5 * z**3 + 2.5 * z**2 - 4 * z + 2, 0),
        )

    pixels = image[iy[:, :, None], ix[:, None, :]]
    return np.einsum("nij,ni,nj->n", pixels, kernel(y[:, None] - iy), kernel(x[:, None] - ix))


def _double_icgn(
    reference, target, support, *, origin, model, weights, tolerance=1e-4, sampler=_bilinear
):
    """Matrix-based double-precision oracle; no production metadata or inverse."""
    size = support.shape[0]
    y, x = np.indices(support.shape, dtype=np.float64)
    local = np.stack((x[support] - (size - 1) / 2, y[support] - (size - 1) / 2))
    pixels = reference[support].astype(np.float64)
    w = weights[support]
    f = pixels[:, 0] - np.average(pixels[:, 0], weights=w)
    gx, gy = pixels[:, 1], pixels[:, 2]
    j = np.stack((gx, gx * local[0], gx * local[1], gy, gy * local[0], gy * local[1]), axis=1)
    active = [0, 3] if model == "translation" else list(range(6))
    j = j[:, active]
    h = j.T @ (w[:, None] * j)
    norm_f = np.sqrt(np.dot(w, f * f))
    a, t = np.eye(2), np.zeros(2)
    center = np.asarray(origin) + (size - 1) / 2
    corners = np.asarray([[-1, -1, 1, 1], [-1, 1, -1, 1]]) * (size - 1) / 2
    for _ in range(30):
        positions = a @ local + center[:, None] + t[:, None]
        g = sampler(target.astype(np.float64), positions[0], positions[1])
        g -= np.average(g, weights=w)
        ratio = norm_f / np.sqrt(np.dot(w, g * g))
        delta = np.zeros(6)
        delta[active] = np.linalg.solve(h, j.T @ (w * (ratio * g - f)))
        increment = np.array([[1 + delta[1], delta[2]], [delta[4], 1 + delta[5]]])
        new_a = a @ np.linalg.inv(increment)
        new_t = t - new_a @ delta[[0, 3]]
        change = (new_a - a) @ corners + (new_t - t)[:, None]
        a, t = new_a, new_t
        if np.max(np.linalg.norm(change, axis=0)) < tolerance:
            break
    return np.array([t[0], a[0, 0] - 1, a[0, 1], t[1], a[1, 0], a[1, 1] - 1])


class PartialGeometryTests(unittest.TestCase):
    def test_bit_layout_is_explicit_and_portable(self):
        mask = np.zeros((33, 33), bool)
        mask.ravel()[[0, 31, 32, 1088]] = True
        words = pack_support(mask)
        self.assertEqual(words[0], 0x80000001)
        self.assertEqual(words[1], 1)
        self.assertEqual(words[-1], 1)
        np.testing.assert_array_equal(unpack_support(words, 33), mask)

    def test_connected_support_and_even_center_ambiguity(self):
        for size in (16, 17):
            mask = np.ones((size, size), bool)
            mask[:, 7] = False
            plan = prepare_roi(mask, size, size, policy="partial")
            if size == 16:
                self.assertFalse(plan.selected.any())
            else:
                self.assertTrue(plan.selected.all())
                support = unpack_support(plan.support_words[0], size)
                self.assertFalse(support[:, :8].any())
                self.assertTrue(support[:, 8:].all())

    def test_visibility_removes_a_connected_wrap_around_a_crack_tip(self):
        mask = np.ones((17, 17), bool)
        mask[:14, 7] = False
        component = prepare_roi(mask, 17, 17, policy="partial")
        clipped = prepare_roi(mask, 17, 17, policy="partial", truncation="visibility")
        before = unpack_support(component.support_words[0], 17)
        after = unpack_support(clipped.support_words[0], 17)
        self.assertTrue(before[8, 3])
        self.assertFalse(after[8, 3])
        self.assertTrue(after[8, 8])
        self.assertTrue(np.all(~after | before))

    def test_visibility_uses_every_pixel_touched_including_corner_touches(self):
        # Independent oracle: inspect each crossing event and every open
        # interval between events, rather than using slab intersections.
        for size in (5, 6):
            center = (size - 1) / 2
            paths = _visibility_paths(size)
            for y in range(size):
                for x in range(size):
                    delta = np.array([x - center, y - center])
                    times = {0.0, 1.0}
                    for d in delta:
                        if d:
                            times.update(
                                float(t)
                                for t in (np.arange(size + 1) - 0.5 - center) / d
                                if 0 <= t <= 1
                            )
                    times = sorted(times)
                    probes = times + [
                        (a + b) / 2 for a, b in zip(times[:-1], times[1:], strict=True)
                    ]
                    expected = np.zeros((size, size), bool)
                    yy, xx = np.indices(expected.shape)
                    for t in probes:
                        q = center + t * delta
                        expected |= (np.abs(xx - q[0]) <= 0.5 + 1e-12) & (
                            np.abs(yy - q[1]) <= 0.5 + 1e-12
                        )
                    np.testing.assert_array_equal(
                        unpack_support(paths[y * size + x], size), expected
                    )

    def test_a_thin_band_can_be_informative_for_translation_but_not_affine(self):
        image, _ = _fractional_pair(51, 51, (0, 0))
        mask = np.zeros(image.shape, bool)
        mask[25] = True
        for model in ("translation", "affine"):
            plan = prepare_roi(mask, 17, 17, policy="partial")
            reference, inverse, metadata = _reference_arrays(
                image, subset_size=17, subset_stride=17, interpolation="linear", motion_model=model
            )
            plan = prepare_partial_reference(
                reference,
                inverse,
                metadata,
                plan,
                width=51,
                height=51,
                size=17,
                stride=17,
                motion_model=model,
                weighting="uniform",
                sigma=4.25,
            )
            rows = metadata.reshape(-1, 16)
            if model == "translation":
                self.assertTrue(np.all(rows[plan.partial_indices, 1] == -4))
            else:
                self.assertTrue(np.all(rows[plan.partial_indices, 1] == 0))
                self.assertTrue(np.all(plan.support_rank[plan.selected] < 6))
            self.assertTrue(np.isfinite(inverse).all())

    def test_truncation_validation_is_explicit(self):
        mask = np.ones((32, 32), bool)
        with self.assertRaises(ValueError):
            validate_roi(mask, "full", mask.shape, "affine", "visibility")
        with self.assertRaises(ValueError):
            validate_roi(mask, "partial", mask.shape, "affine", "unknown")


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "explicit GPU opt-in")
class PartialGPURegressionTests(unittest.TestCase):
    def test_partial_supports_cover_the_other_supported_sizes(self):
        truth = (0.35, -0.2)
        reference, target = _fractional_pair(256, 256, truth)
        mask = np.zeros(reference.shape, bool)
        mask[21:237, 19:235] = True
        mask[80:101, 112:115] = False
        configurations = (
            (8, 8, "translation", "cubic", "uniform"),
            (15, 7, "translation", "bspline", "uniform"),
            (31, 16, "affine", "cubic", "uniform"),
            (32, 32, "affine", "bspline", "uniform"),
            (33, 16, "affine", "bspline", "gaussian"),
        )
        for size, stride, model, interpolation, weighting in configurations:
            with (
                self.subTest(size=size, stride=stride, model=model),
                ICGNWGPU(
                    subset_size=size,
                    subset_stride=stride,
                    motion_model=model,
                    interpolation=interpolation,
                    spatial_weighting=weighting,
                    prefer_subgroups=False,
                    outlier_policy="disabled",
                ) as engine,
            ):
                info = engine.set_reference(
                    reference, roi_mask=mask, roi_policy="partial", tolerance=0.005
                )
                result = engine.register(target)
                partial = np.zeros(result.status.shape, bool)
                partial.ravel()[info.roi.partial_indices] = True
                accepted = result.accepted_mask & partial
                self.assertGreater(np.count_nonzero(accepted), 0)
                error = result.displacement[accepted] - truth
                if size != 8:
                    self.assertLess(np.sqrt(np.mean(np.sum(error**2, axis=-1))), 0.05)
                else:
                    # Small supports on this texture do not meet the 16-pixel
                    # metrology bound (documented separately). Check the
                    # actual algorithm against double precision instead of
                    # hiding this case or loosening that established bound.
                    # A fixed 30-step budget avoids comparing different final
                    # iterates when float32/float64 straddle a stop threshold.
                    info = engine.set_reference(
                        reference, roi_mask=mask, roi_policy="partial", tolerance=1e-8
                    )
                    result = engine.register(target)
                    pixels, _, _ = _reference_arrays(
                        reference,
                        interpolation=interpolation,
                        motion_model=model,
                        subset_size=size,
                        subset_stride=stride,
                    )
                    pixels = pixels.reshape(256, 256, 4)
                    for ordinal, index in enumerate(info.roi.partial_indices):
                        if not info.reference_valid.ravel()[index]:
                            continue
                        gy, gx = divmod(int(index), engine.grid_width)
                        expected = _double_icgn(
                            pixels[
                                gy * stride : gy * stride + size, gx * stride : gx * stride + size
                            ],
                            target,
                            unpack_support(info.roi.support_words[ordinal], size),
                            origin=(gx * stride, gy * stride),
                            model=model,
                            weights=np.ones((size, size)),
                            tolerance=1e-8,
                            sampler=_keys_cubic,
                        )
                        np.testing.assert_allclose(
                            result.parameters[gy, gx], expected, atol=2e-3, rtol=0
                        )
                self.assertFalse(np.any(result.status == -4))

    def test_visibility_limits_mixing_across_a_displacement_jump(self):
        truth = np.array([0.35, -0.2])
        reference, right = _fractional_pair(136, 136, tuple(truth))
        _, left = _fractional_pair(136, 136, tuple(-truth))
        target = right.copy()
        target[:, :56] = left[:, :56]
        mask = np.ones(reference.shape, bool)
        mask[:65, 56] = False  # one-pixel slit ends inside window (3, 3)
        for model in ("translation", "affine"):
            errors = {}
            for truncation in ("none", "visibility"):
                with ICGNWGPU(
                    motion_model=model,
                    subset_size=17,
                    interpolation="linear",
                    prefer_subgroups=False,
                    outlier_policy="disabled",
                ) as engine:
                    engine.set_reference(
                        reference,
                        roi_mask=mask,
                        roi_policy="partial",
                        roi_truncation=truncation,
                        tolerance=0.005,
                    )
                    result = engine.register(target)
                    self.assertEqual(result.status[3, 3], 1)
                    errors[truncation] = np.linalg.norm(result.displacement[3, 3] - truth)
            self.assertGreater(errors["none"], 0.1)
            self.assertLess(errors["visibility"], 0.05)

    def test_partial_windows_preserve_an_admissible_subgroup_specialization(self):
        reference, target = _fractional_pair(256, 256, (0.35, -0.2))
        mask = np.ones(reference.shape, bool)
        mask[90:150, 113:117] = False
        with ICGNWGPU(prefer_subgroups=None, outlier_policy="disabled") as automatic:
            eligible = automatic._subgroup_solver is not None and automatic.subgroup_width >= 16
        if not eligible:
            # Capability rejection itself is tested in the subgroup regressions.
            # Partial ROI still has to run on the portable engine on this device.
            with ICGNWGPU(prefer_subgroups=False, outlier_policy="disabled") as engine:
                info = engine.set_reference(reference, roi_mask=mask, roi_policy="partial")
                self.assertGreater(info.roi.partial_indices.size, 0)
                self.assertGreater(engine.register(target).accepted, 0)
            return
        with ICGNWGPU(prefer_subgroups=True, outlier_policy="disabled") as engine:
            info = engine.set_reference(reference, roi_mask=mask, roi_policy="partial")
            result = engine.register(target)
            self.assertIn("subgroup", engine.solver)
            partial = np.zeros(result.status.shape, bool)
            partial.ravel()[info.roi.partial_indices] = True
            self.assertGreater(np.count_nonzero(result.accepted_mask & partial), 0)
            self.assertGreater(np.count_nonzero(result.accepted_mask & ~partial), 0)
            self.assertLess(result.rmse((0.35, -0.2)), 0.03)

    def test_all_true_and_empty_partial_preserve_the_full_contract(self):
        reference, target = _fractional_pair(128, 128, (0.35, -0.2))
        for factory in (ICGNWGPU, ICGNWGPUFullPyramid):
            with factory(motion_model="translation", outlier_policy="disabled") as engine:
                engine.set_reference(reference)
                baseline = engine.register(target)
                info = engine.set_reference(
                    reference, roi_mask=np.ones(reference.shape, bool), roi_policy="partial"
                )
                result = engine.register(target)
                np.testing.assert_array_equal(result.parameters, baseline.parameters)
                np.testing.assert_array_equal(result.status, baseline.status)
                self.assertEqual(info.roi.partial_indices.size, 0)
                engine.set_reference(
                    reference, roi_mask=np.zeros(reference.shape, bool), roi_policy="partial"
                )
                root = engine if factory is ICGNWGPU else engine._root
                with mock.patch.object(
                    root, "_upload_frame", side_effect=AssertionError("empty ROI upload")
                ):
                    empty = engine.register(target)
                self.assertEqual(empty.accepted, 0)
                self.assertIsNone(empty.rmse((0, 0)))

    def test_partial_retry_and_reconfiguration_do_not_recompile_during_live(self):
        reference, target = _fractional_pair(128, 128, (0.35, -0.2))
        mask = np.ones(reference.shape, bool)
        mask[48:70, 49:70] = False
        with ICGNWGPU(
            motion_model="translation", outlier_policy="disabled", prefer_subgroups=False
        ) as engine:
            info = engine.set_reference(reference, roi_mask=mask, roi_policy="partial")
            root_slot = engine._pipeline_slots[0]
            engine._ensure_seed_resources(root_slot)
            with mock.patch.object(
                engine.device,
                "create_compute_pipeline",
                side_effect=AssertionError("live compilation"),
            ):
                result = engine.register(target)
                seeds = np.zeros((engine.grid_height, engine.grid_width, 8), np.float32)
                active = np.zeros(info.roi_selected.shape, np.uint32)
                active.ravel()[info.roi.partial_indices] = 1
                retried = engine._retry_seeded(root_slot, seeds, active, result)
                np.testing.assert_array_equal(
                    retried.parameters[active == 0], result.parameters[active == 0]
                )
                np.testing.assert_allclose(
                    retried.parameters[active != 0],
                    result.parameters[active != 0],
                    atol=2e-3,
                    rtol=0,
                )
                self.assertTrue(np.all(retried.status[~info.roi_selected] == -3))
            engine.set_reference(reference, roi_mask=mask)  # full, same engine
            self.assertIsNone(engine._partial_solver)
            self.assertLess(engine.roi.selected_count, info.roi.selected_count)
            self.assertFalse(np.any(engine.register(target).status == -4))

    def test_partial_exports_include_the_exact_support_and_diagnostics(self):
        reference, _ = _fractional_pair(128, 128, (0, 0))
        mask = np.ones(reference.shape, bool)
        mask[40:80, 45:49] = False
        with ICGNWGPUFullPyramid(motion_model="affine") as engine:
            info = engine.set_reference(
                reference, roi_mask=mask, roi_policy="partial", roi_truncation="visibility"
            )
            result = engine.register(reference)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "partial.npz"
                result.save_npz(path)
                with np.load(path, allow_pickle=False) as archive:
                    for key, expected in info.roi.archive_fields().items():
                        np.testing.assert_array_equal(archive[key], expected)
                    config = json.loads(str(archive["config_json"]))
                    self.assertEqual(config["roi"]["policy"], "partial")
                    self.assertEqual(config["roi"]["truncation"], "visibility")
                    self.assertEqual(config["roi"]["supportConditionLimit"], 1e8)

    def test_partial_solver_matches_double_precision_cpu_with_and_without_overlap(self):
        source, target = _fractional_pair(128, 128, (0.35, -0.2))
        y, x = np.indices(source.shape)
        mask = (x + y >= 72) & (x < 112) & (y < 112)
        for size, stride in ((16, 16), (17, 8)):
            for model in ("translation", "affine"):
                if model == "affine":
                    source, target = _quadratic_pair(
                        128,
                        128,
                        np.array([0.3, 0.002, 0.001, -0.2, -0.001, 0.0008, 0, 0, 0, 0, 0, 0]),
                    )
                else:
                    source, target = _fractional_pair(128, 128, (0.35, -0.2))
                for weighting in ("uniform", "gaussian"):
                    with (
                        self.subTest(size=size, stride=stride, model=model, weighting=weighting),
                        ICGNWGPU(
                            motion_model=model,
                            interpolation="linear",
                            prefer_subgroups=False,
                            subset_size=size,
                            subset_stride=stride,
                            spatial_weighting=weighting,
                            outlier_policy="disabled",
                        ) as engine,
                    ):
                        info = engine.set_reference(
                            source, roi_mask=mask, roi_policy="partial", tolerance=1e-4
                        )
                        result = engine.register(target)
                        reference, _, _ = _reference_arrays(
                            source,
                            interpolation="linear",
                            motion_model=model,
                            subset_size=size,
                            subset_stride=stride,
                            spatial_weighting=weighting,
                        )
                        reference = reference.reshape(128, 128, 4)
                        candidates = info.roi.partial_indices
                        index = next(
                            int(i)
                            for i in candidates
                            if 0.5 < info.roi.coverage.ravel()[i] < 0.95
                            and info.reference_valid.ravel()[i]
                            and (int(i) // engine.grid_width) * stride >= 2
                            and (int(i) % engine.grid_width) * stride >= 2
                        )
                        gy, gx = divmod(index, engine.grid_width)
                        ordinal = int(np.flatnonzero(candidates == index)[0])
                        support = unpack_support(info.roi.support_words[ordinal], size)
                        yy, xx = np.indices((size, size), dtype=float)
                        weights = (
                            np.exp(
                                -0.5
                                * ((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2)
                                / (size / 4) ** 2
                            )
                            if weighting == "gaussian"
                            else np.ones((size, size))
                        )
                        expected = _double_icgn(
                            reference[
                                gy * stride : gy * stride + size,
                                gx * stride : gx * stride + size,
                            ],
                            target,
                            support,
                            origin=(gx * stride, gy * stride),
                            model=model,
                            weights=weights,
                        )
                        np.testing.assert_allclose(
                            result.parameters[gy, gx], expected, atol=2e-3, rtol=0
                        )
                        self.assertTrue(np.isfinite(result.parameters).all())
                        self.assertFalse(np.any(result.status == -4))

    def test_pyramid_partial_stream_and_visibility(self):
        reference, target = _fractional_pair(256, 256, (0.35, -0.2))
        mask = np.zeros(reference.shape, bool)
        mask[19:235, 21:231] = True
        mask[90:150, 111:114] = False
        for model in ("translation", "affine"):
            for truncation in ("none", "visibility"):
                with ICGNWGPUFullPyramid(motion_model=model, outlier_policy="median") as engine:
                    info = engine.set_reference(
                        reference, roi_mask=mask, roi_policy="partial", roi_truncation=truncation
                    )
                    self.assertGreater(info.roi.partial_indices.size, 0)
                    sync = engine.register(target)
                    identity = engine.register(reference)
                    self.assertGreater(sync.accepted / info.roi.selected_count, 0.9)
                    self.assertLess(sync.rmse((0.35, -0.2)), 0.03)
                    for result, expected in zip(
                        engine.register_stream([target, reference, target], pipeline_depth=3),
                        (sync, identity, sync),
                        strict=True,
                    ):
                        np.testing.assert_array_equal(result.parameters, expected.parameters)
                        np.testing.assert_array_equal(result.status, expected.status)
                        self.assertIs(result.roi, info.roi)
                        self.assertTrue(np.all(result.status[~info.roi_selected] == -3))


if __name__ == "__main__":
    unittest.main()
