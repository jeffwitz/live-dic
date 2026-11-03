"""GPU research controls: independent seeds, compact solves and coarse medians."""

import importlib.util
import os
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from lucas_kanade_icgn.wgpu_backend import _transfer_pyramid_seed

PATH = Path(__file__).resolve().parents[2] / "benchmarks/gpu_spatial_seed.py"
SPEC = importlib.util.spec_from_file_location("gpu_spatial_seed", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "GPU tests disabled")
class GpuSpatialSeedTests(unittest.TestCase):
    def read_storage(self, engine, buffer):
        """Cold shader copy: production seed buffers need no COPY_SRC flag."""
        device = engine.device
        shader = device.create_shader_module(
            code="""
            struct Data { values: array<u32> }
            @group(0) @binding(0) var<storage, read> source: Data;
            @group(0) @binding(1) var<storage, read_write> output: Data;
            @compute @workgroup_size(64)
            fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
                if id.x < arrayLength(&source.values) { output.values[id.x] = source.values[id.x]; }
            }
        """
        )
        output = device.create_buffer(
            size=buffer.size,
            usage=engine._wgpu.BufferUsage.STORAGE | engine._wgpu.BufferUsage.COPY_SRC,
        )
        try:
            pipeline = device.create_compute_pipeline(
                layout="auto", compute={"module": shader, "entry_point": "copy"}
            )
            group = device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=[engine._buffer_resource(0, buffer), engine._buffer_resource(1, output)],
            )
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups((buffer.size // 4 + 63) // 64)
            compute.end()
            device.queue.submit([encoder.finish()])
            return device.queue.read_buffer(output)
        finally:
            output.destroy()

    def engine(self, **kwargs):
        engine = MODULE.GpuSpatialPyramid(subset_size=16, subset_stride=16, **kwargs)
        self.addCleanup(engine.close)
        return engine

    def test_only_improved_seed_is_compacted_and_other_states_are_untouched(self):
        f = np.random.default_rng(31).integers(0, 256, (128, 128), dtype=np.uint8)
        g = np.roll(f, (-1, 2), axis=(0, 1))
        engine = self.engine(strategy="spatial")
        mask = np.zeros(f.shape, bool)
        mask[32:96, 32:96] = True
        engine.prepare_research(f, roi_mask=mask)
        engine.register(g)
        slot = engine._full_slots[0].level_slots[0]
        r = engine.research[0]
        states = np.zeros((8, 8, 8), np.float32)
        states[..., 0] = 2
        states[..., 3] = -1
        states[..., 6] = 1
        states[..., 7] = 4
        states[3, 3, [0, 3, 6]] = [0, 0, -1]
        engine.device.queue.write_buffer(slot.buffers["state"], 0, states)
        engine.device.queue.write_buffer(r["snapshot"], 0, states)
        encoder = engine.device.create_command_encoder()
        encoder.clear_buffer(r["work"])
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(r["choose"])
        compute.set_bind_group(0, r["choose_group"])
        compute.dispatch_workgroups(8, 8)
        compute.set_pipeline(r["finalize"])
        compute.set_bind_group(0, r["finalize_group"])
        compute.dispatch_workgroups(1)
        compute.set_pipeline(r["compact"])
        compute.set_bind_group(0, r["compact_group"])
        compute.dispatch_workgroups_indirect(r["work"], 4)
        compute.end()
        engine.device.queue.submit([encoder.finish()])
        work = np.frombuffer(engine.device.queue.read_buffer(r["work"]), np.uint32)
        np.testing.assert_array_equal(work[:5], [1, 1, 1, 1, 27])
        seeds = np.frombuffer(
            self.read_storage(engine, slot.buffers["seed_state"]), np.float32
        ).reshape(8, 8, 8)
        np.testing.assert_array_equal(seeds[3, 3, [0, 3]], [2, -1])
        output = np.frombuffer(
            engine.device.queue.read_buffer(slot.buffers["state"]), np.float32
        ).reshape(8, 8, 8)
        untouched = np.ones((8, 8), bool)
        untouched[3, 3] = False
        np.testing.assert_array_equal(output[untouched], states[untouched])
        self.assertGreater(output[3, 3, 6], 0.5)
        np.testing.assert_allclose(output[3, 3, [0, 3]], [2, -1], atol=0.03)

    def test_coarse_median_is_correct_and_is_actually_transferred_to_fine_seeds(self):
        f = np.random.default_rng(32).integers(0, 256, (128, 128), dtype=np.uint8)
        engine = self.engine(strategy="median")
        engine.prepare_research(f)
        engine.register(f)
        coarse = engine._levels[1]
        coarse_slot = engine._full_slots[0].level_slots[1]
        fine_slot = engine._full_slots[0].level_slots[0]
        y, x = np.indices((4, 4))
        raw = np.zeros((4, 4, 8), np.float32)
        raw[..., 0] = x
        raw[..., 3] = y
        raw[..., 6] = 1
        raw[..., 7] = 5
        raw[2, 2, 0] = 99
        raw[0, 0, 6] = -1
        expected = raw.copy()
        for j in range(4):
            for i in range(4):
                if raw[j, i, 6] < 0.5:
                    continue
                block = raw[max(0, j - 1) : j + 2, max(0, i - 1) : i + 2]
                values = block[block[..., 6] >= 0.5]
                expected[j, i, 0] = np.median(values[:, 0])
                expected[j, i, 3] = np.median(values[:, 3])
        engine.device.queue.write_buffer(coarse_slot.buffers["state"], 0, raw)
        encoder = engine.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(engine.research[1]["median"])
        compute.set_bind_group(0, coarse_slot.bind_group)
        compute.dispatch_workgroups(1)
        compute.set_pipeline(engine._seed_pipeline)
        compute.set_bind_group(2, engine.median_edges[0])
        compute.dispatch_workgroups(1)
        compute.end()
        engine.device.queue.submit([encoder.finish()])
        filtered = engine.device.queue.read_buffer(coarse_slot.buffers["filtered_state"])
        np.testing.assert_array_equal(
            np.frombuffer(filtered, np.float32).reshape(4, 4, 8), expected
        )
        coarse_result = coarse._result_from_raw(filtered, timing={}, config=coarse._config)
        coarse_result = replace(
            coarse_result, parameters=expected[..., :6], status=expected[..., 6]
        )
        expected_seeds, _ = _transfer_pyramid_seed(
            coarse_result,
            source_available=np.ones((4, 4), bool),
            source_scale=2,
            destination_scale=1,
            destination_grid=(8, 8),
            requested=np.ones((8, 8), bool),
            subset_size=16,
            subset_stride=16,
        )
        actual = np.frombuffer(
            self.read_storage(engine, fine_slot.buffers["seed_state"]), np.float32
        ).reshape(8, 8, 8)
        np.testing.assert_allclose(actual[..., :6], expected_seeds[..., :6], atol=1e-6)

    def test_indirect_count_handles_empty_and_two_dimensional_worklists(self):
        f = np.random.default_rng(33).integers(0, 256, (64, 64), dtype=np.uint8)
        engine = self.engine()
        engine.prepare_research(f)
        r = engine.research[0]
        for count, expected in ((0, [0, 1, 1]), (65536, [65535, 2, 1])):
            engine.device.queue.write_buffer(r["work"], 0, np.array([count], np.uint32))
            encoder = engine.device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(r["finalize"])
            compute.set_bind_group(0, r["finalize_group"])
            compute.dispatch_workgroups(1)
            compute.end()
            engine.device.queue.submit([encoder.finish()])
            actual = np.frombuffer(engine.device.queue.read_buffer(r["work"], 4, 12), np.uint32)
            np.testing.assert_array_equal(actual, expected)

    def test_seed_median_includes_unfinished_estimates_without_changing_raw_measurements(self):
        f = np.random.default_rng(34).integers(0, 256, (128, 128), dtype=np.uint8)
        engine = self.engine(strategy="seed-median")
        engine.prepare_research(f)
        r = engine.research[1]
        slot = engine._full_slots[0].level_slots[1]
        y, x = np.indices((4, 4))
        raw = np.zeros((4, 4, 8), np.float32)
        raw[..., 0] = x
        raw[..., 3] = y
        raw[2, 2, 0] = 99
        raw[2, 2, 6] = -1
        expected = raw.copy()
        for j in range(4):
            for i in range(4):
                block = raw[max(0, j - 1) : j + 2, max(0, i - 1) : i + 2]
                values = block[block[..., 6] >= -0.5]
                if len(values) >= 4:
                    expected[j, i, 0] = np.median(values[:, 0])
                    expected[j, i, 3] = np.median(values[:, 3])
                    expected[j, i, 6] = max(raw[j, i, 6], 0)
        engine.device.queue.write_buffer(r["snapshot"], 0, raw)
        engine.device.queue.write_buffer(slot.buffers["state"], 0, raw)
        encoder = engine.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(r["seed_median"])
        compute.set_bind_group(0, r["seed_median_group"])
        compute.dispatch_workgroups(1)
        compute.end()
        engine.device.queue.submit([encoder.finish()])
        actual = np.frombuffer(
            engine.device.queue.read_buffer(slot.buffers["filtered_state"]), np.float32
        ).reshape(4, 4, 8)
        np.testing.assert_array_equal(actual, expected)
        original = np.frombuffer(
            engine.device.queue.read_buffer(slot.buffers["state"]), np.float32
        ).reshape(4, 4, 8)
        np.testing.assert_array_equal(original, raw)

    def test_fixed_affine_prior_is_active_and_reset_after_spatial_trials(self):
        f = np.random.default_rng(35).integers(0, 256, (256, 256), dtype=np.uint8)
        g = np.roll(f, (-6, 12), axis=(0, 1))
        engine = self.engine()
        engine.prepare_research(f, affine={"dx": 12, "dy": -6, "linear": [0, 0, 0, 0]})
        first = engine.register(g)
        self.assertGreater(first.accepted, 100)
        np.testing.assert_allclose(
            np.median(first.parameters[first.status > 0.5][:, [0, 3]], axis=0), [12, -6], atol=0.01
        )
        engine.strategy = "spatial"
        engine.register(g)
        engine.strategy = "baseline"
        second = engine.register(g)
        np.testing.assert_array_equal(first.parameters, second.parameters)
        np.testing.assert_array_equal(first.status, second.status)
