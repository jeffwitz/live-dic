from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lucas_kanade_icgn.engine import RegistrationResult
from lucas_kanade_icgn.full_pyramid_backend import ICGNWGPUFullPyramid, _FullPyramidSlot
from lucas_kanade_icgn.wgpu_backend import (
    ICGNWGPU,
    _PendingRegistration,
    _RegistrationSlot,
)


class _Clock:
    """Deterministic operation durations without sleeping or using a GPU."""

    def __init__(self, milliseconds: float = 0.0) -> None:
        self.seconds = milliseconds / 1_000.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, milliseconds: float) -> None:
        self.seconds += milliseconds / 1_000.0


def _raw_result() -> bytes:
    return np.asarray([2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 5.0], dtype=np.float32).tobytes()


class _Readback:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.size = len(_raw_result())

    def map_sync(self, mode: object) -> None:
        self.clock.advance(4.0)

    def read_mapped(self, *, copy: bool) -> bytes:
        self.clock.advance(2.0)
        return _raw_result()

    def unmap(self) -> None:
        self.clock.advance(1.0)


def _engine(clock: _Clock, *, pyramid_levels: int = 1) -> ICGNWGPU:
    engine = object.__new__(ICGNWGPU)
    engine._reference_ready = True
    engine.pyramid_levels = pyramid_levels
    engine.height = engine.width = 16
    engine.grid_height = engine.grid_width = 1
    engine._config = {"iterations": 30, "tolerance": 1e-3, "znccMin": 0.8}
    engine._pipeline_slots = [object()]
    engine._solver = object()
    engine._wgpu = SimpleNamespace(MapMode=SimpleNamespace(READ=1))
    prepare_target = engine._prepare_target
    result_from_raw = engine._result_from_raw

    def prepare(target: object) -> np.ndarray:
        clock.advance(2.0)
        return prepare_target(target)

    def upload(slot: object, frame: np.ndarray) -> float:
        clock.advance(3.0)
        return 3.0

    def execute(solver: object) -> bytes:
        clock.advance(4.0)
        return _raw_result()

    def build_result(*args: object, **kwargs: object) -> RegistrationResult:
        result = result_from_raw(*args, **kwargs)
        clock.advance(5.0)
        return result

    def recover(slot: object, frame: np.ndarray, result: RegistrationResult) -> RegistrationResult:
        clock.advance(7.0)
        # Recovery may replace both the result and its stage-timing dictionary.
        return replace(result, timing={**result.timing, "pyramidWallMs": 7.0})

    engine._prepare_target = prepare
    engine._update_config = lambda: None
    engine._upload_frame = upload
    engine._execute_registration = execute
    engine._result_from_raw = build_result
    engine._recover_pyramid = recover
    return engine


class ResultViewTests(unittest.TestCase):
    def test_displacement_is_a_writable_view_of_packed_parameters(self) -> None:
        packed = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
        parameters = packed[..., :6]
        result = RegistrationResult(
            parameters=parameters,
            status=packed[..., 6],
            iterations=packed[..., 7],
            accepted=6,
            replaced=0,
            mean_translation=None,
            mean_iterations=None,
            timing={},
            config={},
        )

        displacement = result.displacement
        self.assertEqual(displacement.shape, (2, 3, 2))
        self.assertTrue(np.shares_memory(displacement, parameters))
        np.testing.assert_array_equal(displacement, parameters[..., (0, 3)])
        displacement[0, 0, 1] = 42.0
        self.assertEqual(float(parameters[0, 0, 3]), 42.0)


class NativeTimingTests(unittest.TestCase):
    def test_register_total_includes_result_statistics_and_optional_recovery(self) -> None:
        for levels, expected_ms in ((1, 14.0), (2, 21.0)):
            with self.subTest(pyramid_levels=levels):
                clock = _Clock()
                engine = _engine(clock, pyramid_levels=levels)
                with mock.patch("lucas_kanade_icgn.wgpu_backend.time.perf_counter", clock):
                    result = engine.register(np.zeros((16, 16), dtype=np.uint8))

                self.assertAlmostEqual(result.timing["totalWallMs"], expected_ms)
                self.assertAlmostEqual(result.timing["uploadWallMs"], 3.0)
                self.assertAlmostEqual(result.timing["executeReadbackWallMs"], 4.0)
                self.assertEqual(result.mean_translation, (2.0, -1.0))

    def test_stream_total_includes_result_statistics_and_optional_recovery(self) -> None:
        for levels, expected_ms in ((1, 22.0), (2, 29.0)):
            with self.subTest(pyramid_levels=levels):
                clock = _Clock(milliseconds=10.0)
                engine = _engine(clock, pyramid_levels=levels)
                slot = _RegistrationSlot(None, {}, None, _Readback(clock))
                pending = _PendingRegistration(
                    slot=slot,
                    submitted_at=0.0,
                    upload_wall_ms=3.0,
                    submit_wall_ms=1.0,
                    config=dict(engine._config),
                    frame=np.zeros((16, 16), dtype=np.uint8),
                )
                with mock.patch("lucas_kanade_icgn.wgpu_backend.time.perf_counter", clock):
                    result = engine._retire_slot(pending)

                self.assertAlmostEqual(result.timing["pipelineLatencyWallMs"], 17.0)
                self.assertAlmostEqual(result.timing["totalWallMs"], expected_ms)

    def test_full_pyramid_total_includes_result_statistics(self) -> None:
        clock = _Clock()
        root = _engine(clock)
        slot = _RegistrationSlot(
            None,
            {"filtered_state": object()},
            object(),
            _Readback(clock),
        )
        root._pipeline_slots = [slot]
        root._portable_solver = object()
        root._encode_registration_passes = lambda *args, **kwargs: clock.advance(4.0)
        encoder = SimpleNamespace(
            copy_buffer_to_buffer=lambda *args: clock.advance(1.0),
            finish=lambda: object(),
        )
        device = SimpleNamespace(
            create_command_encoder=lambda **kwargs: encoder,
            queue=SimpleNamespace(submit=lambda commands: clock.advance(1.0)),
        )
        engine = object.__new__(ICGNWGPUFullPyramid)
        engine._reference_ready = True
        engine._root = root
        engine._levels = [root]
        engine.pyramid_levels = 1
        engine._wgpu = root._wgpu
        engine.device = device
        engine._encode_reductions = lambda command_encoder, edges: None
        engine._full_slots = [_FullPyramidSlot((slot,), [], {})]

        with mock.patch("lucas_kanade_icgn.full_pyramid_backend.time.perf_counter", clock):
            result = engine.register(np.zeros((16, 16), dtype=np.uint8))

        self.assertAlmostEqual(result.timing["totalWallMs"], 23.0)
        self.assertAlmostEqual(result.timing["executeReadbackWallMs"], 8.0)
        self.assertAlmostEqual(result.timing["inputPreparationWallMs"], 2.0)
        self.assertAlmostEqual(result.timing["encodingWallMs"], 5.0)
        self.assertAlmostEqual(result.timing["submissionWallMs"], 1.0)
        self.assertAlmostEqual(result.timing["completionWaitWallMs"], 4.0)
        self.assertAlmostEqual(result.timing["readbackCopyWallMs"], 3.0)
        self.assertAlmostEqual(result.timing["pipelineLatencyWallMs"], 18.0)
        self.assertAlmostEqual(result.timing["resultBuildWallMs"], 5.0)
        self.assertAlmostEqual(result.timing["resultAgeWallMs"], 23.0)
        self.assertEqual(result.timing["queueSubmissions"], 1.0)


class SeededPipelineTests(unittest.TestCase):
    def test_fresh_and_retry_pipelines_share_layout_but_have_distinct_entry_points(self) -> None:
        engine = object.__new__(ICGNWGPU)
        engine._seeded_solver = None
        engine._retry_solver = None
        engine._shader = object()
        engine._create_bind_group_layout = mock.Mock(return_value=object())
        engine.device = SimpleNamespace(
            create_pipeline_layout=mock.Mock(return_value=object()),
            create_compute_pipeline=mock.Mock(side_effect=["fresh", "preserving"]),
        )

        engine._ensure_seeded_solver()
        engine._ensure_seeded_solver()

        calls = engine.device.create_compute_pipeline.call_args_list
        self.assertEqual(
            [call.kwargs["compute"]["entry_point"] for call in calls],
            ["solve_seeded", "retry_seeded"],
        )
        self.assertIs(calls[0].kwargs["layout"], calls[1].kwargs["layout"])
        self.assertEqual(engine._seeded_solver, "fresh")
        self.assertEqual(engine._retry_solver, "preserving")

    def test_retry_uses_preserving_pipeline_without_clearing_existing_state(self) -> None:
        clock = _Clock()
        engine = _engine(clock)
        engine._ensure_seed_resources = lambda slot: clock.advance(2.0)
        engine._retry_solver = object()
        engine._seeded_solver = object()
        engine._median_filter = object()
        first_pass, median_pass = mock.Mock(), mock.Mock()
        encoder = mock.Mock()
        encoder.begin_compute_pass.side_effect = [first_pass, median_pass]

        def read_buffer(buffer: object) -> bytes:
            clock.advance(4.0)
            return _raw_result()

        engine.device = SimpleNamespace(
            create_command_encoder=lambda **kwargs: encoder,
            queue=SimpleNamespace(
                write_buffer=lambda *args: clock.advance(0.5),
                submit=lambda commands: clock.advance(1.0),
                read_buffer=read_buffer,
            ),
        )
        slot = _RegistrationSlot(
            None,
            {name: object() for name in ("state", "filtered_state", "seed_state", "active_mask")},
            object(),
            None,
            object(),
        )
        previous = engine._result_from_raw(_raw_result(), timing={"totalWallMs": 10.0}, config={})
        with mock.patch("lucas_kanade_icgn.wgpu_backend.time.perf_counter", clock):
            result = engine._retry_seeded(
                slot,
                np.zeros((1, 1, 8), dtype=np.float32),
                np.ones((1, 1), dtype=np.uint32),
                previous,
            )

        first_pass.set_pipeline.assert_called_once_with(engine._retry_solver)
        median_pass.set_pipeline.assert_called_once_with(engine._median_filter)
        encoder.clear_buffer.assert_called_once_with(slot.buffers["filtered_state"])
        self.assertAlmostEqual(result.timing["seededRetryWallMs"], 13.0)
        self.assertAlmostEqual(result.timing["totalWallMs"], 23.0)

    def test_shared_level_reuses_both_seeded_pipelines(self) -> None:
        engine = object.__new__(ICGNWGPU)
        for name in (
            "adapter",
            "device",
            "device_name",
            "subgroup_width",
            "_shader",
            "_bind_group_layout",
            "_pipeline_layout",
            "_prefilter_x_fast",
            "_prefilter_y_fast",
            "_prefilter_x",
            "_prefilter_y",
            "_portable_solver",
            "_subgroup_solver",
            "_median_filter",
            "_seeded_bind_group_layout",
            "_seeded_solver",
            "_retry_solver",
        ):
            setattr(engine, name, object())
        engine._prefilter_safe_pipelines = {}
        engine._wgpu = SimpleNamespace(TextureFormat=SimpleNamespace(r8unorm="r8unorm"))

        level = engine._new_shared_level()

        self.assertIs(level.device, engine.device)
        self.assertIs(level._seeded_solver, engine._seeded_solver)
        self.assertIs(level._retry_solver, engine._retry_solver)
        self.assertIs(level._seeded_bind_group_layout, engine._seeded_bind_group_layout)
        self.assertIs(level._prefilter_safe_pipelines, engine._prefilter_safe_pipelines)
        self.assertFalse(level._reference_ready)


if __name__ == "__main__":
    unittest.main()
