"""Full-window ROI contracts, with opt-in execution on the selected wgpu device."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from test_gpu_regressions import _fractional_pair, _quadratic_pair

from lucas_kanade_icgn import ICGNWGPU, ICGNWebGL, ICGNWGPUFullPyramid, register_pair
from lucas_kanade_icgn.roi import prepare_roi, reduce_mask, seed_permissions, validate_roi


class RoiGeometryTests(unittest.TestCase):
    def test_selection_matches_direct_slices_for_holes_odd_sizes_and_strides(self):
        mask = np.ones((83, 97), bool)
        mask[9, 11] = False  # all four corners of its square remain true
        mask[30:46, 30:56] = False
        for size in (8, 15, 16, 17, 31, 32, 33):
            for stride in (4, size, size + 3):
                with self.subTest(size=size, stride=stride):
                    plan = prepare_roi(mask, size, stride)
                    expected = [
                        [
                            mask[y : y + size, x : x + size].all()
                            for x in range(0, mask.shape[1] - size + 1, stride)
                        ]
                        for y in range(0, mask.shape[0] - size + 1, stride)
                    ]
                    np.testing.assert_array_equal(plan.selected, expected)

    def test_validation_and_immutable_ownership(self):
        for value in (np.ones((32, 32), np.uint8), [[True]], "mask.npy"):
            with self.assertRaises(TypeError):
                validate_roi(value, "full", (32, 32), "affine")
        for value in (np.ones((31, 32), bool), np.ones((32, 32, 1), bool)):
            with self.assertRaises(ValueError):
                validate_roi(value, "full", (32, 32), "affine")
        for policy in ("unknown", None):
            with self.assertRaises(ValueError):
                validate_roi(None, policy, (32, 32), "affine")
        self.assertIsNone(validate_roi(None, "partial", (32, 32), "affine"))
        with self.assertRaises(ValueError):
            validate_roi(np.ones((32, 32), bool), "full", (32, 32), "quadratic")
        source = np.ones((32, 32), bool)
        copied = validate_roi(source, "full", source.shape, "affine")
        source[:] = False
        self.assertTrue(copied.all())
        self.assertIs(validate_roi(copied, "full", copied.shape, "affine"), copied)
        with self.assertRaises(ValueError):
            copied.setflags(write=True)
        self.assertEqual(
            prepare_roi(copied, 16, 16).mask_hash, prepare_roi(copied[:, ::-1], 16, 16).mask_hash
        )

        noncanonical = np.full((32, 32), 255, np.uint8).view(np.bool_)
        normalized = validate_roi(noncanonical, "full", noncanonical.shape, "affine")
        self.assertEqual(
            prepare_roi(normalized, 16, 16).mask_hash, prepare_roi(copied, 16, 16).mask_hash
        )

    def test_conservative_reduction_matches_independent_reflected_footprint(self):
        mask = np.ones((35, 37), bool)
        mask[0, 0] = mask[13, 17] = False

        def mirror(i, n):
            return -i if i < 0 else (2 * n - 2 - i if i >= n else i)

        expected = np.array(
            [
                [
                    all(
                        mask[mirror(2 * y + dy, 35), mirror(2 * x + dx, 37)]
                        for dy in range(-2, 3)
                        for dx in range(-2, 3)
                    )
                    for x in range(37 // 2)
                ]
                for y in range(35 // 2)
            ]
        )
        np.testing.assert_array_equal(reduce_mask(mask), expected)
        self.assertTrue(reduce_mask(np.ones_like(mask)).all())

    def test_neighbour_links_cannot_cross_a_one_pixel_crack_between_windows(self):
        mask = np.ones((64, 64), bool)
        mask[:, 31] = False
        plan = prepare_roi(mask, 8, 16)
        self.assertTrue(plan.selected.all())  # crack lies between nominal squares
        self.assertFalse(bool(plan.neighbour_bits[1, 1] & (1 << 5)))
        self.assertTrue(bool(plan.neighbour_bits[1, 0] & (1 << 5)))

    def test_seed_links_and_destinations_respect_static_geometry(self):
        mask = np.ones((128, 128), bool)
        mask[:, 63] = False
        fine = prepare_roi(mask, 16, 16)
        coarse = prepare_roi(reduce_mask(mask), 16, 16)
        flags = seed_permissions(
            coarse,
            fine,
            source_scale=2,
            destination_scale=1,
            size=16,
            stride=16,
            reference_mask=mask,
        )
        np.testing.assert_array_equal((flags >> 4) != 0, fine.selected)
        self.assertTrue(np.any((flags & 15) == 0))

    def test_browser_rejects_roi_before_startup(self):
        engine = ICGNWebGL()
        with self.assertRaises(ValueError):
            engine.set_reference(np.ones((64, 64), np.uint8), roi_mask=np.ones((64, 64), bool))
        with self.assertRaises(ValueError):
            register_pair(np.ones((64, 64)), np.ones((64, 64)), roi_policy="partial")


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "explicit GPU opt-in required")
class RoiGPURegressionTests(unittest.TestCase):
    def test_selected_flat_reference_is_a_numeric_rejection_in_automatic_mode(self):
        flat = np.zeros((128, 128), np.uint8)
        mask = np.ones(flat.shape, bool)
        mask[48:64, 48:64] = False
        with ICGNWGPU() as engine:
            with mock.patch.object(
                engine, "_autotune_solver", side_effect=AssertionError("no data to qualify")
            ):
                info = engine.set_reference(flat, roi_mask=mask)
            result = engine.register(flat)
            self.assertEqual(result.accepted, 0)
            self.assertFalse(info.reference_valid.any())
            self.assertTrue(np.all(result.status[info.roi_selected] == -1))
            self.assertTrue(np.all(result.status[~info.roi_selected] == -3))

    def test_both_native_engines_recover_a_known_affine_field(self):
        p = np.array([0.3, 0.001, 0.0005, -0.2, -0.0003, -0.0007, 0, 0, 0, 0, 0, 0])
        reference, target = _quadratic_pair(256, 256, p)
        mask = np.zeros(reference.shape, bool)
        mask[32:224, 32:224] = True
        mask[96:144, 112:144] = False
        for factory in (ICGNWGPU, ICGNWGPUFullPyramid):
            with (
                self.subTest(engine=factory.__name__),
                factory(
                    motion_model="affine",
                    outlier_policy="median",
                    **({"iterations_per_level": (8, 30)} if factory is ICGNWGPUFullPyramid else {}),
                ) as engine,
            ):
                info = engine.set_reference(
                    reference, roi_mask=mask, **({"tolerance": 5e-3} if factory is ICGNWGPU else {})
                )
                result = engine.register(target)
                y, x = np.indices(result.status.shape)
                x = 16 * x + 7.5 - 127.5
                y = 16 * y + 7.5 - 127.5
                expected = np.stack(
                    (p[0] + p[1] * x + p[2] * y, p[3] + p[4] * x + p[5] * y), axis=-1
                )
                selected = result.accepted_mask
                self.assertGreater(selected.sum() / info.roi_selected.sum(), 0.90)
                error = result.displacement[selected] - expected[selected]
                self.assertLess(float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))), 0.03)
                gradients = result.parameters[..., (1, 2, 4, 5)][selected] - p[[1, 2, 4, 5]]
                self.assertLess(float(np.sqrt(np.mean(gradients**2))), 0.002)

    def test_full_pyramid_all_true_is_the_unmasked_path(self):
        reference, target = _fractional_pair(256, 256, (0.35, -0.2))
        for model in ("translation", "affine"):
            with ICGNWGPUFullPyramid(motion_model=model) as engine:
                engine.set_reference(reference)
                solver = engine._root._solver
                median = engine._root._median_filter
                expected = list(engine.register_stream([target, reference], pipeline_depth=2))
                engine.set_reference(reference, roi_mask=np.ones(reference.shape, bool))
                self.assertIs(engine._root._solver, solver)
                self.assertIs(engine._root._median_filter, median)
                self.assertFalse(engine._roi_link_buffers)
                for a, b in zip(
                    expected,
                    engine.register_stream([target, reference], pipeline_depth=2),
                    strict=True,
                ):
                    np.testing.assert_array_equal(a.parameters, b.parameters)
                    np.testing.assert_array_equal(a.status, b.status)
                    np.testing.assert_array_equal(a.iterations, b.iterations)

    def test_roi_keeps_the_admissible_subgroup_specialization(self):
        reference, target = _fractional_pair(256, 256, (0.35, -0.2))
        with ICGNWGPU(outlier_policy="disabled") as discovery:
            usable = discovery._subgroup_solver is not None
        if not usable:
            # The existing subgroup qualification tests check explicit refusal.
            # ROI must not require subgroups on an incompatible adapter.
            with ICGNWGPU(prefer_subgroups=False) as portable:
                portable.set_reference(reference, roi_mask=np.ones(reference.shape, bool))
            return
        mask = np.ones(reference.shape, bool)
        mask[80:120, 100:140] = False
        with ICGNWGPU(prefer_subgroups=True, outlier_policy="disabled") as engine:
            info = engine.set_reference(reference, roi_mask=mask)
            self.assertEqual(engine.solver, "subgroup")
            result = engine.register(target)
            self.assertLess(result.rmse((0.35, -0.2)), 0.03)
            self.assertTrue(np.all(result.status[~info.roi_selected] == -3))

    def test_gpu_median_cannot_bridge_an_unselected_pixel_gap(self):
        reference, _ = _fractional_pair(128, 128, (0, 0))
        mask = np.zeros(reference.shape, bool)
        mask[16:32, 16:32] = True
        mask[32:48, 32:48] = True
        with ICGNWGPU(
            motion_model="translation",
            subset_size=16,
            subset_stride=16,
            outlier_policy="median",
            prefer_subgroups=False,
        ) as engine:
            info = engine.set_reference(reference, roi_mask=mask)
            packed = np.zeros((engine.grid_height, engine.grid_width, 8), np.float32)
            packed[..., 0] = 100
            packed[..., 6] = np.where(info.roi_selected, 1, -3)
            packed[1, 1, 0] = 3
            engine.device.queue.write_buffer(engine._buffers["state"], 0, packed)
            encoder = engine.device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(engine._median_filter)
            compute.set_bind_group(0, engine._bind_group)
            compute.dispatch_workgroups(1)
            compute.end()
            engine.device.queue.submit([encoder.finish()])
            filtered = np.frombuffer(
                engine.device.queue.read_buffer(engine._buffers["filtered_state"]), np.float32
            ).reshape(packed.shape)
            self.assertEqual(filtered[1, 1, 0], 3)
            np.testing.assert_array_equal(filtered[..., 6], packed[..., 6])

    def test_retained_reference_data_and_invalid_preparation_are_transactional(self):
        reference, _ = _fractional_pair(128, 128, (0, 0))
        mask = np.ones(reference.shape, bool)
        mask[50:54, 60:64] = False
        for size, stride, interpolation, weighting in (
            (15, 8, "linear", "uniform"),
            (17, 9, "cubic", "uniform"),
            (32, 16, "bspline", "gaussian"),
        ):
            with ICGNWGPU(
                subset_size=size,
                subset_stride=stride,
                interpolation=interpolation,
                spatial_weighting=weighting,
                prefer_subgroups=False,
            ) as engine:
                with mock.patch.object(
                    engine, "_allocate_runtime", wraps=engine._allocate_runtime
                ) as allocate:
                    engine.set_reference(reference)
                    original = [value.copy() for value in allocate.call_args.args]
                    info = engine.set_reference(reference, roi_mask=mask)
                    prepared = allocate.call_args.args
                # Pixel data, all inverse Hessians, and retained metadata remain
                # unchanged; the only extra metadata is the static link tail.
                np.testing.assert_array_equal(original[0], prepared[0])
                np.testing.assert_array_equal(original[1], prepared[1])
                selected = info.roi_selected.ravel()
                rows = original[2].reshape(-1, 16)
                np.testing.assert_array_equal(
                    rows[selected], prepared[2][: rows.size].reshape(-1, 16)[selected]
                )
                for bad in (np.zeros(reference.shape, np.uint8), np.zeros((64, 64), bool)):
                    with self.assertRaises((TypeError, ValueError)):
                        engine.set_reference(reference, roi_mask=bad)
                self.assertIs(engine.roi, info.roi)
                result = engine.register(reference)
                self.assertTrue(np.all(result.status[~info.roi_selected] == -3))
                flat = np.zeros_like(reference)
                flat_info = engine.set_reference(flat, roi_mask=mask)
                flat_result = engine.register(flat)
                self.assertFalse(flat_info.reference_valid.any())
                self.assertTrue(np.all(flat_result.status[flat_info.roi_selected] == -1))
                self.assertTrue(np.all(flat_result.status[~flat_info.roi_selected] == -3))

    def test_absent_all_true_and_raw_retained_windows_are_equivalent(self):
        reference, target = _fractional_pair(256, 256, (0.35, -0.2))
        for model in ("translation", "affine"):
            with ICGNWGPU(
                motion_model=model, prefer_subgroups=False, outlier_policy="disabled"
            ) as engine:
                engine.set_reference(reference)
                raw = engine.register(target)
                engine.set_reference(reference, roi_mask=np.ones(reference.shape, bool))
                all_true = engine.register(target)
                np.testing.assert_array_equal(raw.parameters, all_true.parameters)
                np.testing.assert_array_equal(raw.status, all_true.status)
                mask = np.ones(reference.shape, bool)
                mask[55:73, 90:113] = False
                info = engine.set_reference(reference, roi_mask=mask)
                result = engine.register(target)
                selected = info.roi_selected
                np.testing.assert_array_equal(raw.parameters[selected], result.parameters[selected])
                self.assertTrue(np.all(result.status[~selected] == -3))
                self.assertTrue(np.all(result.iterations[~selected] == 0))
                self.assertLess(result.rmse((0.35, -0.2)), 0.03)

    def test_empty_mask_avoids_all_frame_gpu_work_and_exports_without_pickle(self):
        reference, target = _fractional_pair(128, 128, (0.35, -0.2))
        for factory in (ICGNWGPU, ICGNWGPUFullPyramid):
            with factory(motion_model="translation") as engine:
                mask = np.zeros(reference.shape, bool)
                info = engine.set_reference(reference, roi_mask=mask)
                root = engine if factory is ICGNWGPU else engine._root
                with mock.patch.object(
                    root, "_upload_frame", side_effect=AssertionError("unexpected upload")
                ):
                    results = [
                        engine.register(target),
                        *engine.register_stream([target] * 3, pipeline_depth=3),
                    ]
                for result in results:
                    self.assertEqual(result.accepted, 0)
                    self.assertIsNone(result.rmse((0, 0)))
                    self.assertIsNone(result.populations["acceptedFractionSelected"])
                    self.assertTrue(np.isfinite(result.parameters).all())
                    self.assertIs(result.roi, info.roi)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "roi.npz"
                    results[0].save_npz(path)
                    with np.load(path, allow_pickle=False) as archive:
                        np.testing.assert_array_equal(archive["roi_mask"], mask)
                        self.assertEqual(
                            json.loads(str(archive["config_json"]))["roi"]["selected"], 0
                        )

    def test_pyramids_stream_recovery_reconfiguration_and_ownership(self):
        truth = (0.35, -0.2)
        reference, target = _fractional_pair(256, 256, truth)
        mask = np.zeros(reference.shape, bool)
        mask[32:224, 32:224] = True
        mask[100:148, 125:127] = False
        for model in ("translation", "affine"):
            for factory in (ICGNWGPU, ICGNWGPUFullPyramid):
                with (
                    self.subTest(model=model, engine=factory.__name__),
                    factory(
                        motion_model=model,
                        pyramid_levels=2,
                        outlier_policy="median",
                        **(
                            {"iterations_per_level": (8, 30)}
                            if factory is ICGNWGPUFullPyramid
                            else {}
                        ),
                    ) as engine,
                ):
                    info = engine.set_reference(reference, roi_mask=mask)
                    copied = info.roi.mask.copy()
                    mask[:] = True
                    result = engine.register(target)
                    self.assertLess(result.rmse(truth), 0.03)
                    selected = info.roi_selected
                    self.assertGreater(result.accepted / selected.sum(), 0.9)
                    np.testing.assert_array_equal(info.roi.mask, copied)
                    for depth in (1, 3):
                        for streamed in engine.register_stream(
                            [target, reference, target], pipeline_depth=depth
                        ):
                            self.assertTrue(np.all(streamed.status[~selected] == -3))
                            self.assertIs(streamed.roi, info.roi)
                            self.assertTrue(np.isfinite(streamed.parameters).all())
                    engine.set_reference(reference)
                    self.assertIsNone(engine.register(target).roi)
                    mask[:] = copied

    def test_thin_roi_remains_solvable_without_a_coarse_seed(self):
        reference, target = _fractional_pair(256, 256, (0.35, -0.2))
        mask = np.zeros(reference.shape, bool)
        mask[96:112, 32:224] = True
        for model in ("translation", "affine"):
            for levels in (2, 3):
                with ICGNWGPUFullPyramid(
                    motion_model=model, pyramid_levels=levels, outlier_policy="disabled"
                ) as engine:
                    info = engine.set_reference(reference, roi_mask=mask)
                    self.assertFalse(engine._levels[1].roi.selected.any())
                    result = engine.register(target)
                    # Missing seeds must not disable the fine solve. Its usual
                    # iteration budget need not converge every affine window.
                    self.assertTrue(np.all(result.iterations[info.roi_selected] > 0))
                    self.assertGreater(result.accepted, 0)
                    self.assertLess(result.rmse((0.35, -0.2)), 0.03)


if __name__ == "__main__":
    unittest.main()
