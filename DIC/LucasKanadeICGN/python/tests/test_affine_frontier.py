"""GPU test of first-order affine frontier transport."""

import os
import struct
import unittest
from importlib.resources import files

import numpy as np
import wgpu

from live_dic import ICGNWGPU


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests disabled")
class AffineFrontierTests(unittest.TestCase):
    def test_public_frontier_feeds_seeded_solver(self):
        image = np.random.default_rng(46).integers(0, 256, (128, 128), dtype=np.uint8)
        engine = ICGNWGPU(
            device_type="gpu",
            subset_size=16,
            subset_stride=16,
            interpolation="linear",
            outlier_policy="disabled",
        )
        self.addCleanup(engine.close)
        engine.set_reference(image, zncc_min=0.65)
        shape = (engine.grid_height, engine.grid_width)
        params = np.zeros((*shape, 6), np.float32)
        accepted = np.zeros(shape, bool)
        accepted[3, 3] = True
        seeds, active = engine.affine_frontier_seeds(params, accepted)
        self.assertEqual(int(active.sum()), 8)
        self.assertFalse(active[3, 3])
        result = engine.register_seeded(image, seeds, active, include_zncc=True)
        self.assertEqual(int(np.count_nonzero(result.status == 1)), 8)
        np.testing.assert_allclose(result.zncc[active], 1, atol=1e-5)
        _, empty = engine.affine_frontier_seeds(params, np.zeros(shape, bool))
        self.assertFalse(np.any(empty))
        with self.assertRaisesRegex(ValueError, "Boolean"):
            engine.affine_frontier_seeds(params, accepted.astype(np.uint8))

    def test_first_order_transport_matches_numpy(self):
        engine = ICGNWGPU(device_type="gpu")
        self.addCleanup(engine.close)
        device = engine.device
        shader = device.create_shader_module(
            code=files("live_dic").joinpath("shaders/spatial_seed.wgsl").read_text()
        )
        pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": shader, "entry_point": "affine_frontier"}
        )
        rng = np.random.default_rng(27)
        states = rng.normal(0, 0.1, (9, 8)).astype(np.float32)
        states[:, 0] += 2
        states[:, 3] -= 1
        states[:, 6] = 1
        states[4, 6] = 0
        permission = np.full(9, 511, np.uint32)
        cfg = struct.pack("8I4f", 128, 128, 3, 9, 16, 16, 1, 0, 0, 0, 0, 0)
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        config = device.create_buffer_with_data(data=cfg, usage=wgpu.BufferUsage.UNIFORM)
        source = device.create_buffer_with_data(data=states, usage=usage)
        perm = device.create_buffer_with_data(data=permission, usage=usage)
        seed = device.create_buffer_with_data(data=np.zeros((9, 8), np.float32), usage=usage)
        work = device.create_buffer_with_data(data=np.zeros(13, np.uint32), usage=usage)
        self.addCleanup(config.destroy)
        self.addCleanup(source.destroy)
        self.addCleanup(perm.destroy)
        self.addCleanup(seed.destroy)
        self.addCleanup(work.destroy)
        buffers = {2: config, 3: source, 4: perm, 5: seed, 6: work}
        group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": binding,
                    "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
                }
                for binding, buffer in buffers.items()
            ],
        )
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(1)
        compute.end()
        device.queue.submit([encoder.finish()])
        actual = np.frombuffer(device.queue.read_buffer(seed), np.float32).reshape(9, 8)
        work_data = np.frombuffer(device.queue.read_buffer(work), np.uint32)
        candidates = []
        for idx in range(9):
            if idx == 4:
                continue
            dx = (1 - idx % 3) * 16
            dy = (1 - idx // 3) * 16
            p = states[idx].copy()
            p[0] += p[1] * dx + p[2] * dy
            p[3] += p[4] * dx + p[5] * dy
            candidates.append(p[:6])
        np.testing.assert_allclose(actual[4, :6], np.median(candidates, axis=0), atol=1e-6)
        self.assertEqual(work_data[0], 1)
        self.assertEqual(work_data[4], 4)
