"""GPU-resident coarse-to-fine native WebGPU IC-GN backend.

Unlike the selective recovery path in :mod:`wgpu_backend`, this backend builds
every target pyramid level on the GPU, keeps intermediate images and affine
states resident, and submits the complete coarse-to-fine solve in one command
buffer.  Only the finest target image is uploaded and only the finest result
is read back.
"""

from __future__ import annotations

import operator
import struct
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from .engine import ICGNError, ImageInput, ReferenceInfo, RegistrationResult, _as_uint8_gray
from .interpolation import normalize_interpolation
from .motion import normalize_motion_model
from .subset import DEFAULT_SUBSET_SIZE, normalize_subset_size
from .wgpu_backend import (
    DEFAULT_IN_FLIGHT_FRAMES,
    ICGNWGPU,
    _reduce_half,
    _validated_config,
)

DEFAULT_FULL_PYRAMID_LEVELS = 2
# Extra scales improve the seed, so the default three-level path can use
# smaller iteration ceilings than the one- and two-level paths.  The solver
# still exits early as soon as each subset converges.
_DEFAULT_ITERATION_SCHEDULES = {
    1: (30,),
    2: (8, 16),
    3: (6, 8, 16),
}
_DEFAULT_TOLERANCES = (1e-2, 5e-3, 5e-3)


@dataclass(slots=True)
class _ReductionEdge:
    """GPU resources that reduce one adjacent image-pyramid level."""

    destination_width: int
    destination_height: int
    source_height: int
    intermediate: Any
    horizontal_bind_group: Any
    vertical_bind_group: Any


@dataclass(slots=True)
class _SeedEdge:
    """GPU resources that transfer one coarse affine field to a finer grid."""

    destination_count: int
    config: Any
    bind_group: Any


@dataclass(slots=True)
class _FullPyramidSlot:
    """All mutable GPU resources required by one in-flight target."""

    level_slots: tuple[Any, ...]
    reduction_edges: list[_ReductionEdge]
    seed_edges: dict[int, _SeedEdge]


@dataclass(slots=True)
class _PendingFullPyramidRegistration:
    """One submitted target awaiting its final readback."""

    slot: _FullPyramidSlot
    accepted_at: float
    input_preparation_wall_ms: float
    upload_wall_ms: float
    encoding_wall_ms: float
    submission_wall_ms: float


def _level_count(value: int) -> int:
    try:
        count = operator.index(value)
    except TypeError as error:
        raise ValueError("pyramid_levels must be an integer from 1 to 3") from error
    if isinstance(value, bool) or not 1 <= count <= 3:
        raise ValueError("pyramid_levels must be an integer from 1 to 3")
    return count


def _schedule(
    values: tuple[int, ...] | tuple[float, ...] | None,
    *,
    level_count: int,
    defaults: tuple[int, ...] | tuple[float, ...],
    name: str,
) -> tuple[int, ...] | tuple[float, ...]:
    selected = tuple(defaults[-level_count:] if values is None else values)
    if len(selected) != level_count:
        raise ValueError(f"{name} must contain one value per pyramid level")
    return selected


class ICGNWGPUFullPyramid:
    """Full GPU-resident image pyramid with a coarse-to-fine IC-GN solve.

    The default uses two levels because it provides the largest measured 4K60
    margin while retaining the robust coarse-to-fine capture range.  Three
    levels remain available for larger expected displacements.

    ``iterations_per_level`` and ``tolerance_per_level`` are ordered from the
    coarsest level to the finest level.  Reference preparation is performed
    once per level by :meth:`set_reference`; per target, all reductions,
    prefilters, solves, seed transfers, and median filters stay on one WebGPU
    device and are encoded into one command buffer. ``linear`` and ``cubic``
    interpolation omit the per-level prefilters; ``bspline`` remains the
    compatibility default. ``subset_size`` selects an 8, 16, or 32 pixel
    square and also sets the field spacing.
    """

    def __init__(
        self,
        *,
        pyramid_levels: int = DEFAULT_FULL_PYRAMID_LEVELS,
        iterations_per_level: tuple[int, ...] | None = None,
        tolerance_per_level: tuple[float, ...] | None = None,
        power_preference: str = "high-performance",
        interpolation: str = "bspline",
        motion_model: str = "affine",
        subset_size: int = DEFAULT_SUBSET_SIZE,
    ) -> None:
        self.interpolation = normalize_interpolation(interpolation)
        self.motion_model = normalize_motion_model(motion_model)
        self.subset_size = normalize_subset_size(subset_size)
        self.pyramid_levels = _level_count(pyramid_levels)
        iteration_values = _schedule(
            iterations_per_level,
            level_count=self.pyramid_levels,
            defaults=_DEFAULT_ITERATION_SCHEDULES[self.pyramid_levels],
            name="iterations_per_level",
        )
        tolerance_values = _schedule(
            tolerance_per_level,
            level_count=self.pyramid_levels,
            defaults=_DEFAULT_TOLERANCES,
            name="tolerance_per_level",
        )
        self.iterations_per_level = tuple(int(value) for value in iteration_values)
        self.tolerance_per_level = tuple(float(value) for value in tolerance_values)
        for iterations, tolerance in zip(
            self.iterations_per_level,
            self.tolerance_per_level,
            strict=True,
        ):
            _validated_config(iterations, tolerance, 0.8)

        # The seeded solver currently uses the portable 16-lane workgroup
        # kernel at every level.  Sharing one root engine also guarantees that
        # every level belongs to exactly the same wgpu device and queue.
        self._root = ICGNWGPU(
            power_preference=power_preference,
            prefer_subgroups=False,
            pyramid_levels=1,
            interpolation=self.interpolation,
            motion_model=self.motion_model,
            subset_size=self.subset_size,
        )
        self._root._ensure_seeded_solver()
        self._wgpu = self._root._wgpu
        self.device = self._root.device
        self.device_name = self._root.device_name
        self._levels: list[ICGNWGPU] = [self._root]
        self._reduction_edges: list[_ReductionEdge] = []
        self._seed_edges: dict[int, _SeedEdge] = {}
        self._full_slots: list[_FullPyramidSlot] = []
        self._reference_ready = False
        self._zncc_min = 0.8
        self._create_orchestration_pipelines()

    def _create_orchestration_pipelines(self) -> None:
        wgpu = self._wgpu
        shader_path = files("lucas_kanade_icgn").joinpath("shaders/full_pyramid.wgsl")
        shader = self.device.create_shader_module(code=shader_path.read_text(encoding="utf-8"))
        self._empty_layout = self.device.create_bind_group_layout(entries=[])
        self._reduce_x_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "storage_texture": {
                        "access": wgpu.StorageTextureAccess.write_only,
                        "format": wgpu.TextureFormat.r32float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
            ]
        )
        self._reduce_y_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.unfilterable_float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "storage_texture": {
                        "access": wgpu.StorageTextureAccess.write_only,
                        "format": wgpu.TextureFormat.rgba8unorm,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
            ]
        )
        storage_read = wgpu.BufferBindingType.read_only_storage
        storage_write = wgpu.BufferBindingType.storage
        self._seed_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": storage_read},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": storage_write},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": storage_write},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        reduce_x_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._reduce_x_layout]
        )
        reduce_y_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._empty_layout, self._reduce_y_layout]
        )
        seed_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._empty_layout, self._empty_layout, self._seed_layout]
        )
        self._reduce_x_pipeline = self.device.create_compute_pipeline(
            layout=reduce_x_pipeline_layout,
            compute={"module": shader, "entry_point": "reduce_x"},
        )
        self._reduce_y_pipeline = self.device.create_compute_pipeline(
            layout=reduce_y_pipeline_layout,
            compute={"module": shader, "entry_point": "reduce_y"},
        )
        self._seed_pipeline = self.device.create_compute_pipeline(
            layout=seed_pipeline_layout,
            compute={
                "module": shader,
                "entry_point": "transfer_seed",
                "constants": {"SUBSET_SIZE": self.subset_size},
            },
        )

    @staticmethod
    def _buffer_resource(binding: int, buffer: Any) -> dict[str, Any]:
        return {
            "binding": binding,
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
        }

    def _clear_level_resources(self) -> None:
        for slot in self._full_slots:
            for edge in slot.reduction_edges:
                edge.intermediate.destroy()
            for edge in slot.seed_edges.values():
                edge.config.destroy()
        self._full_slots.clear()
        self._reduction_edges.clear()
        self._seed_edges.clear()
        for level in self._levels[1:]:
            level.close()
        self._levels = [self._root]

    def _create_reduction_edges(self, level_slots: tuple[Any, ...]) -> list[_ReductionEdge]:
        wgpu = self._wgpu
        edges: list[_ReductionEdge] = []
        for source, destination, source_slot, destination_slot in zip(
            self._levels[:-1],
            self._levels[1:],
            level_slots[:-1],
            level_slots[1:],
            strict=True,
        ):
            intermediate = self.device.create_texture(
                size=(destination.width, source.height, 1),
                usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
                dimension=wgpu.TextureDimension.d2,
                format=wgpu.TextureFormat.r32float,
                mip_level_count=1,
                sample_count=1,
            )
            horizontal = self.device.create_bind_group(
                layout=self._reduce_x_layout,
                entries=[
                    {"binding": 0, "resource": source_slot.source_texture.create_view()},
                    {"binding": 1, "resource": intermediate.create_view()},
                ],
            )
            vertical = self.device.create_bind_group(
                layout=self._reduce_y_layout,
                entries=[
                    {"binding": 0, "resource": intermediate.create_view()},
                    {"binding": 1, "resource": destination_slot.source_texture.create_view()},
                ],
            )
            edges.append(
                _ReductionEdge(
                    destination_width=destination.width,
                    destination_height=destination.height,
                    source_height=source.height,
                    intermediate=intermediate,
                    horizontal_bind_group=horizontal,
                    vertical_bind_group=vertical,
                )
            )
        return edges

    def _create_seed_edges(self, level_slots: tuple[Any, ...]) -> dict[int, _SeedEdge]:
        wgpu = self._wgpu
        edges: dict[int, _SeedEdge] = {}
        # Internal levels are ordered fine-to-coarse.  Each edge is keyed by
        # its finer destination index.
        for destination_index in range(self.pyramid_levels - 2, -1, -1):
            destination = self._levels[destination_index]
            source = self._levels[destination_index + 1]
            destination_slot = level_slots[destination_index]
            source_slot = level_slots[destination_index + 1]
            destination._ensure_seed_resources(destination_slot)
            source_scale = float(1 << (destination_index + 1))
            destination_scale = float(1 << destination_index)
            packed = struct.pack(
                "<4I4f",
                source.grid_width,
                source.grid_height,
                destination.grid_width,
                destination.grid_width * destination.grid_height,
                source_scale,
                destination_scale,
                0.0,
                0.0,
            )
            config = self.device.create_buffer_with_data(
                data=packed,
                usage=wgpu.BufferUsage.UNIFORM,
            )
            bind_group = self.device.create_bind_group(
                layout=self._seed_layout,
                entries=[
                    self._buffer_resource(0, source_slot.buffers["filtered_state"]),
                    self._buffer_resource(1, destination_slot.buffers["seed_state"]),
                    self._buffer_resource(2, destination_slot.buffers["active_mask"]),
                    self._buffer_resource(3, config),
                ],
            )
            edges[destination_index] = _SeedEdge(
                destination_count=destination.grid_width * destination.grid_height,
                config=config,
                bind_group=bind_group,
            )
        return edges

    def _create_full_slot(self, slot_index: int) -> _FullPyramidSlot:
        level_slots = tuple(level._pipeline_slots[slot_index] for level in self._levels)
        return _FullPyramidSlot(
            level_slots=level_slots,
            reduction_edges=self._create_reduction_edges(level_slots),
            seed_edges=self._create_seed_edges(level_slots),
        )

    def _ensure_full_slots(self, depth: int) -> list[_FullPyramidSlot]:
        if depth < 1:
            raise ValueError("pipeline_depth must be at least 1")
        for level in self._levels:
            level._ensure_pipeline_slots(depth)
        while len(self._full_slots) < depth:
            self._full_slots.append(self._create_full_slot(len(self._full_slots)))
        return self._full_slots[:depth]

    def set_reference(
        self,
        image: ImageInput,
        *,
        iterations_per_level: tuple[int, ...] | None = None,
        tolerance_per_level: tuple[float, ...] | None = None,
        zncc_min: float = 0.8,
    ) -> ReferenceInfo:
        """Prepare all scale-specific references and persistent GPU storage."""

        started = time.perf_counter()
        new_iterations = self.iterations_per_level
        if iterations_per_level is not None:
            values = _schedule(
                iterations_per_level,
                level_count=self.pyramid_levels,
                defaults=_DEFAULT_ITERATION_SCHEDULES[self.pyramid_levels],
                name="iterations_per_level",
            )
            new_iterations = tuple(int(value) for value in values)
        new_tolerances = self.tolerance_per_level
        if tolerance_per_level is not None:
            values = _schedule(
                tolerance_per_level,
                level_count=self.pyramid_levels,
                defaults=_DEFAULT_TOLERANCES,
                name="tolerance_per_level",
            )
            new_tolerances = tuple(float(value) for value in values)
        new_zncc_min = float(zncc_min)
        for iterations, tolerance in zip(
            new_iterations,
            new_tolerances,
            strict=True,
        ):
            _validated_config(iterations, tolerance, new_zncc_min)

        reference = _as_uint8_gray(image)
        references = [reference]
        for _ in range(1, self.pyramid_levels):
            references.append(_reduce_half(references[-1]))

        self.iterations_per_level = new_iterations
        self.tolerance_per_level = new_tolerances
        self._zncc_min = new_zncc_min
        self._reference_ready = False
        self._clear_level_resources()
        reference_build_wall_ms = 0.0
        for level_index, level_reference in enumerate(references):
            engine = (
                self._root
                if level_index == 0
                else self._root._new_shared_level(gpu_writable_source=True)
            )
            schedule_index = self.pyramid_levels - 1 - level_index
            info = engine.set_reference(
                level_reference,
                iterations=self.iterations_per_level[schedule_index],
                tolerance=self.tolerance_per_level[schedule_index],
                zncc_min=self._zncc_min,
            )
            reference_build_wall_ms += info.reference_build_wall_ms
            if level_index:
                self._levels.append(engine)

        first_slot = self._create_full_slot(0)
        self._full_slots = [first_slot]
        # Preserve the private aliases used by the existing GPU regression
        # tests and by downstream diagnostic code.
        self._reduction_edges = first_slot.reduction_edges
        self._seed_edges = first_slot.seed_edges
        self._reference_ready = True
        total_wall_ms = (time.perf_counter() - started) * 1_000.0
        fine = self._root
        return ReferenceInfo(
            preparation_wall_ms=total_wall_ms,
            runtime_build_wall_ms=total_wall_ms - reference_build_wall_ms,
            reference_build_wall_ms=reference_build_wall_ms,
            encoding_wall_ms=0.0,
            total_wall_ms=total_wall_ms,
            config={
                **fine._config,
                "interpolation": self.interpolation,
                "motionModel": self.motion_model,
            },
            device=self.device_name,
            grid=(fine.grid_width, fine.grid_height),
            width=fine.width,
            height=fine.height,
            subset_size=self.subset_size,
        )

    def _encode_reductions(self, encoder: Any, edges: list[_ReductionEdge]) -> None:
        for edge in edges:
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(self._reduce_x_pipeline)
            compute_pass.set_bind_group(0, edge.horizontal_bind_group)
            compute_pass.dispatch_workgroups(
                (edge.destination_width + 15) // 16,
                (edge.source_height + 15) // 16,
                1,
            )
            compute_pass.end()
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(self._reduce_y_pipeline)
            compute_pass.set_bind_group(1, edge.vertical_bind_group)
            compute_pass.dispatch_workgroups(
                (edge.destination_width + 15) // 16,
                (edge.destination_height + 15) // 16,
                1,
            )
            compute_pass.end()

    def _encode_slot(self, slot: _FullPyramidSlot) -> tuple[Any, float]:
        encode_started = time.perf_counter()
        encoder = self.device.create_command_encoder(label="DIC-RT full pyramid registration")
        self._encode_reductions(encoder, slot.reduction_edges)
        coarsest_index = self.pyramid_levels - 1
        coarsest = self._levels[coarsest_index]
        coarsest_slot = slot.level_slots[coarsest_index]
        coarsest._encode_registration_passes(
            encoder,
            coarsest_slot,
            coarsest._portable_solver,
            solver_bind_group=coarsest_slot.bind_group,
        )
        for destination_index in range(coarsest_index - 1, -1, -1):
            seed_edge = slot.seed_edges[destination_index]
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(self._seed_pipeline)
            compute_pass.set_bind_group(2, seed_edge.bind_group)
            compute_pass.dispatch_workgroups((seed_edge.destination_count + 255) // 256, 1, 1)
            compute_pass.end()
            destination = self._levels[destination_index]
            destination_slot = slot.level_slots[destination_index]
            destination._encode_registration_passes(
                encoder,
                destination_slot,
                destination._seeded_solver,
                solver_bind_group=destination_slot.seeded_bind_group,
            )
        fine_slot = slot.level_slots[0]
        encoder.copy_buffer_to_buffer(
            fine_slot.buffers["filtered_state"],
            0,
            fine_slot.readback,
            0,
            fine_slot.readback.size,
        )
        command = encoder.finish()
        encoding_wall_ms = (time.perf_counter() - encode_started) * 1_000.0
        return command, encoding_wall_ms

    def _submit_slot(
        self,
        slot: _FullPyramidSlot,
        frame: Any,
        *,
        accepted_at: float,
        input_preparation_wall_ms: float,
    ) -> _PendingFullPyramidRegistration:
        upload_wall_ms = self._root._upload_frame(slot.level_slots[0], frame)
        command, encoding_wall_ms = self._encode_slot(slot)
        submission_started = time.perf_counter()
        self.device.queue.submit([command])
        submission_wall_ms = (time.perf_counter() - submission_started) * 1_000.0
        return _PendingFullPyramidRegistration(
            slot=slot,
            accepted_at=accepted_at,
            input_preparation_wall_ms=input_preparation_wall_ms,
            upload_wall_ms=upload_wall_ms,
            encoding_wall_ms=encoding_wall_ms,
            submission_wall_ms=submission_wall_ms,
        )

    def _retire_slot(self, pending: _PendingFullPyramidRegistration) -> RegistrationResult:
        fine_slot = pending.slot.level_slots[0]
        wait_started = time.perf_counter()
        fine_slot.readback.map_sync(self._wgpu.MapMode.READ)
        completion_wait_wall_ms = (time.perf_counter() - wait_started) * 1_000.0
        copy_started = time.perf_counter()
        raw = fine_slot.readback.read_mapped(copy=True)
        fine_slot.readback.unmap()
        readback_copy_wall_ms = (time.perf_counter() - copy_started) * 1_000.0
        pipeline_latency_wall_ms = (time.perf_counter() - pending.accepted_at) * 1_000.0
        result_started = time.perf_counter()
        execute_readback_wall_ms = (
            pending.submission_wall_ms + completion_wait_wall_ms + readback_copy_wall_ms
        )
        result = self._root._result_from_raw(
            raw,
            timing={
                "inputPreparationWallMs": pending.input_preparation_wall_ms,
                "uploadWallMs": pending.upload_wall_ms,
                "encodingWallMs": pending.encoding_wall_ms,
                "submissionWallMs": pending.submission_wall_ms,
                "completionWaitWallMs": completion_wait_wall_ms,
                "readbackCopyWallMs": readback_copy_wall_ms,
                "executeReadbackWallMs": execute_readback_wall_ms,
                "pipelineLatencyWallMs": pipeline_latency_wall_ms,
                "pyramidLevels": float(self.pyramid_levels),
                "queueSubmissions": 1.0,
            },
            config=self._root._config,
        )
        result_build_wall_ms = (time.perf_counter() - result_started) * 1_000.0
        total_wall_ms = (time.perf_counter() - pending.accepted_at) * 1_000.0
        result.timing = {
            **result.timing,
            "resultBuildWallMs": result_build_wall_ms,
            "resultAgeWallMs": total_wall_ms,
            "totalWallMs": total_wall_ms,
        }
        return result

    def register(self, target: ImageInput) -> RegistrationResult:
        """Register one target with one upload, submission, and final readback."""

        if not self._reference_ready:
            raise ICGNError("Call set_reference() before register().")
        accepted_at = time.perf_counter()
        preparation_started = time.perf_counter()
        frame = self._root._prepare_target(target)
        input_preparation_wall_ms = (time.perf_counter() - preparation_started) * 1_000.0
        pending = self._submit_slot(
            self._full_slots[0],
            frame,
            accepted_at=accepted_at,
            input_preparation_wall_ms=input_preparation_wall_ms,
        )
        return self._retire_slot(pending)

    def register_stream(
        self,
        targets: Iterable[ImageInput],
        *,
        pipeline_depth: int = DEFAULT_IN_FLIGHT_FRAMES,
    ) -> Iterator[RegistrationResult]:
        """Register an ordered stream with bounded full-pyramid work in flight.

        ``pipeline_depth`` limits independent target pyramids that may be
        queued concurrently. Results remain in input order. Increasing it can
        improve steady-state throughput, but also increases result age and GPU
        memory use; it is unrelated to :attr:`pyramid_levels`.
        """

        if not self._reference_ready:
            raise ICGNError("Call set_reference() before register_stream().")
        try:
            depth = operator.index(pipeline_depth)
        except TypeError as error:
            raise ValueError("pipeline_depth must be a positive integer") from error
        if isinstance(pipeline_depth, bool) or depth < 1:
            raise ValueError("pipeline_depth must be a positive integer")

        slots = self._ensure_full_slots(depth)
        available = deque(slots)
        pending: deque[_PendingFullPyramidRegistration] = deque()
        for target in targets:
            if not available:
                completed = pending.popleft()
                result = self._retire_slot(completed)
                available.append(completed.slot)
                yield result
            slot = available.popleft()
            accepted_at = time.perf_counter()
            preparation_started = time.perf_counter()
            frame = self._root._prepare_target(target)
            preparation_wall_ms = (time.perf_counter() - preparation_started) * 1_000.0
            pending.append(
                self._submit_slot(
                    slot,
                    frame,
                    accepted_at=accepted_at,
                    input_preparation_wall_ms=preparation_wall_ms,
                )
            )

        while pending:
            completed = pending.popleft()
            result = self._retire_slot(completed)
            available.append(completed.slot)
            yield result

    def close(self) -> None:
        self._reference_ready = False
        self._clear_level_resources()
        self._root.close()

    def __enter__(self) -> ICGNWGPUFullPyramid:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
