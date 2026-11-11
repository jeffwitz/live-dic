"""Public affine seeded solve validation and GPU parity."""

import os
import unittest

import numpy as np

from live_dic import ICGNWGPU, ICGNError


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests disabled")
class SeededSolveTests(unittest.TestCase):
    def test_stream_owns_slot_until_exhausted_or_closed(self):
        image = np.random.default_rng(49).integers(0, 256, (128, 128), dtype=np.uint8)
        engine = ICGNWGPU(
            device_type=os.environ.get("ICGN_TEST_DEVICE_TYPE", "auto"),
            subset_size=16,
            subset_stride=16,
            interpolation="linear",
            outlier_policy="disabled",
        )
        self.addCleanup(engine.close)
        engine.set_reference(image)
        shape = (engine.grid_height, engine.grid_width)
        params = np.zeros((*shape, 6), np.float32)
        active = np.zeros(shape, bool)
        active[3, 3] = True
        stream = engine.register_stream([image, image, image], pipeline_depth=2)
        next(stream)
        with self.assertRaisesRegex(ICGNError, "stream is active"):
            engine.register_seeded(image, params, active)
        with self.assertRaisesRegex(ICGNError, "stream is active"):
            engine.affine_frontier_seeds(params, active)
        with self.assertRaisesRegex(ICGNError, "stream is active"):
            engine.register(image)
        competing = engine.register_stream([image])
        with self.assertRaisesRegex(ICGNError, "stream is active"):
            next(competing)
        stream.close()
        self.assertEqual(engine.register_seeded(image, params, active).status[3, 3], 1)
        self.assertEqual(len(list(engine.register_stream([image]))), 1)
        self.assertEqual(engine.register_seeded(image, params, active).status[3, 3], 1)
        failed = engine.register_stream([image, np.zeros((4, 4), np.uint8)], pipeline_depth=1)
        next(failed)
        with self.assertRaises(ValueError):
            next(failed)
        self.assertEqual(engine.register_seeded(image, params, active).status[3, 3], 1)

    def test_keys_cubic_final_zncc_on_identity(self):
        image = np.random.default_rng(137).integers(0, 256, (128, 128), dtype=np.uint8)
        engine = ICGNWGPU(
            device_type=os.environ.get("ICGN_TEST_DEVICE_TYPE", "auto"),
            subset_size=16,
            subset_stride=16,
            interpolation="cubic",
            outlier_policy="disabled",
        )
        self.addCleanup(engine.close)
        engine.set_reference(image)
        result = engine.register(image, include_zncc=True)
        self.assertGreater(result.accepted, 0)
        np.testing.assert_allclose(result.zncc[result.status == 1], 1, atol=1e-5)

    def test_final_zncc_matches_independent_numpy_bilinear(self):
        rng = np.random.default_rng(132)
        reference = rng.integers(20, 220, (128, 128), dtype=np.uint8)
        target = np.clip(
            np.roll(reference.astype(np.int16), (1, -2), axis=(0, 1))
            + rng.integers(-8, 9, reference.shape),
            0,
            255,
        ).astype(np.uint8)
        engine = ICGNWGPU(
            device_type=os.environ.get("ICGN_TEST_DEVICE_TYPE", "auto"),
            subset_size=16,
            subset_stride=16,
            interpolation="linear",
            outlier_policy="disabled",
        )
        self.addCleanup(engine.close)
        engine.set_reference(reference, zncc_min=0.65)
        shape = (engine.grid_height, engine.grid_width)
        params = np.zeros((*shape, 6), np.float32)
        params[..., 0], params[..., 3] = -2, 1
        active = np.zeros(shape, bool)
        active[3, 3] = True
        result = engine.register_seeded(target, params, active, include_zncc=True)
        self.assertEqual(result.status[3, 3], 1)
        p = result.parameters[3, 3].astype(np.float64)
        y, x = np.mgrid[48:64, 48:64].astype(np.float64)
        dx, dy = x - 55.5, y - 55.5
        xx = x + p[0] + p[1] * dx + p[2] * dy
        yy = y + p[3] + p[4] * dx + p[5] * dy
        ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
        fx, fy = xx - ix, yy - iy
        t = target.astype(np.float64)
        sample = (
            (1 - fx) * (1 - fy) * t[iy, ix]
            + fx * (1 - fy) * t[iy, ix + 1]
            + (1 - fx) * fy * t[iy + 1, ix]
            + fx * fy * t[iy + 1, ix + 1]
        )
        expected = np.corrcoef(reference[48:64, 48:64].ravel(), sample.ravel())[0, 1]
        np.testing.assert_allclose(result.zncc[3, 3], expected, atol=2e-5, rtol=0)

    def test_inactive_windows_remain_inactive_with_default_outlier_policy(self):
        image = np.random.default_rng(46).integers(0, 256, (128, 128), dtype=np.uint8)
        engine = ICGNWGPU(
            device_type=os.environ.get("ICGN_TEST_DEVICE_TYPE", "auto"),
            subset_size=16,
            subset_stride=16,
        )
        self.addCleanup(engine.close)
        engine.set_reference(image)
        shape = (engine.grid_height, engine.grid_width)
        active = np.zeros(shape, bool)
        active[2:6, 2:6] = True
        result = engine.register_seeded(image, np.zeros((*shape, 6), np.float32), active)
        self.assertIsNone(result.zncc)
        self.assertGreater(result.accepted, 0)
        self.assertFalse(np.any(result.status[~active] > 0.5))
        measured = engine.register_seeded(
            image, np.zeros((*shape, 6), np.float32), active, include_zncc=True
        )
        self.assertTrue(np.isfinite(measured.zncc[measured.status == 1]).all())
        self.assertGreater(float(np.nanmin(measured.zncc)), 0.99)
        normal = engine.register(image)
        scored = engine.register(image, include_zncc=True)
        np.testing.assert_array_equal(normal.status, scored.status)
        np.testing.assert_array_equal(normal.parameters, scored.parameters)
        self.assertIsNone(normal.zncc)
        self.assertGreater(float(np.nanmin(scored.zncc)), 0.99)

    def test_active_mask_is_intersected_with_roi(self):
        image = np.random.default_rng(47).integers(0, 256, (128, 128), dtype=np.uint8)
        roi = np.zeros_like(image, dtype=bool)
        roi[:, :64] = True
        engine = ICGNWGPU(
            device_type=os.environ.get("ICGN_TEST_DEVICE_TYPE", "auto"),
            subset_size=16,
            subset_stride=16,
            outlier_policy="disabled",
        )
        self.addCleanup(engine.close)
        engine.set_reference(image, roi_mask=roi)
        shape = (engine.grid_height, engine.grid_width)
        result = engine.register_seeded(
            image, np.zeros((*shape, 6), np.float32), np.ones(shape, bool)
        )
        self.assertGreater(result.accepted, 0)
        self.assertFalse(np.any(result.status[~engine.roi.selected] > 0.5))

    def test_seeded_solve_matches_existing_kernel_and_masks(self):
        image = np.random.default_rng(41).integers(0, 256, (128, 128), dtype=np.uint8)
        target = np.roll(image, (1, -2), axis=(0, 1))
        engine = ICGNWGPU(
            device_type=os.environ.get("ICGN_TEST_DEVICE_TYPE", "auto"),
            subset_size=16,
            subset_stride=16,
            interpolation="linear",
            outlier_policy="disabled",
        )
        self.addCleanup(engine.close)
        engine.set_reference(image, zncc_min=0.65)
        shape = (engine.grid_height, engine.grid_width)
        seeds = np.zeros((*shape, 6), np.float32)
        seeds[..., 0], seeds[..., 3] = -2, 1
        active = np.zeros(shape, bool)
        active[1:-1:2, 1:-1:2] = True
        actual = engine.register_seeded(target, seeds, active, include_zncc=True)
        packed = np.zeros((*shape, 8), np.float32)
        packed[..., :6] = np.where(active[..., None], seeds, 0)
        expected = engine._register_seeded_frame(target, packed, active.astype(np.uint32))
        np.testing.assert_array_equal(actual.status, expected.status)
        np.testing.assert_array_equal(actual.parameters, expected.parameters)
        self.assertGreater(actual.accepted, 0)
        self.assertFalse(np.any(actual.status[~active] > 0.5))
        self.assertFalse(actual.zncc.flags.writeable)
        with self.assertRaises(ValueError):
            actual.zncc.setflags(write=True)
        self.assertTrue(np.isnan(actual.zncc[~active]).all())
        self.assertTrue(np.all(actual.zncc[actual.status == 1] > 0.99))
        with self.assertRaises(ValueError):
            engine.register_seeded(target, seeds, active.astype(np.uint32))
        with self.assertRaises(ValueError):
            engine.register_seeded(target, seeds[..., :5], active)
        seeds[active, 0] = np.nan
        with self.assertRaises(ValueError):
            engine.register_seeded(target, seeds, active)
