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

import numpy as np

from .engine import ICGNError, ImageInput, ReferenceInfo, RegistrationResult, _as_uint8_gray
from .interpolation import normalize_interpolation
from .motion import (
    MOTION_MODEL_PIPELINE_CONSTANTS,
    normalize_motion_model,
    parameter_count,
    state_float_count,
)
from .outlier import normalize_outlier_policy
from .roi import reduce_mask, seed_permissions, validate_roi
from .subset import DEFAULT_SUBSET_SIZE, normalize_subset_size, normalize_subset_stride
from .weighting import (
    normalize_gaussian_sigma,
    normalize_spatial_weighting,
)
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
    bind_group: Any


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
    temporal: _TemporalSlot | None = None


@dataclass(slots=True)
class _TemporalSlot:
    """Per-submission projection bindings and packed diagnostic counters."""

    projection: Any
    merge: Any
    counts: Any
    seed_config: Any
    refinement: Any | None = None
    coarse_merge: Any | None = None
    prediction: Any | None = None
    save_previous: Any | None = None
    fine_projection: _SeedEdge | None = None


@dataclass(slots=True)
class _PendingFullPyramidRegistration:
    """One submitted target awaiting its final readback."""

    slot: _FullPyramidSlot
    accepted_at: float
    input_preparation_wall_ms: float
    upload_wall_ms: float
    encoding_wall_ms: float
    submission_wall_ms: float
    levels_used: int | None = None


def _level_count(value: int) -> int:
    try:
        count = operator.index(value)
    except TypeError as error:
        raise ValueError("pyramid_levels must be an integer from 1 to 3") from error
    if isinstance(value, bool) or not 1 <= count <= 3:
        raise ValueError("pyramid_levels must be an integer from 1 to 3")
    return count


def _iteration_count(value: int, name: str) -> int:
    try:
        count = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer from 1 to 30") from error
    if isinstance(value, bool) or not 1 <= count <= 30:
        raise ValueError(f"{name} must be an integer from 1 to 30")
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
    prefilters, solves, seed transfers, and optional UOD passes stay on one WebGPU
    device and are encoded into one command buffer. ``linear`` and ``cubic``
    interpolation omit the per-level prefilters; ``bspline`` remains the
    compatibility default. ``subset_size`` supports the established 8, 16,
    and 32 pixel widths plus centred 15, 17, 31, and 33 pixel widths;
    ``subset_stride`` independently selects the field spacing.
    ``spatial_weighting="gaussian"`` enables a centred weighted least-squares
    window with a configurable ``gaussian_sigma`` in pixels. Uniform
    integration remains the compatibility default.

    ``initialization="temporal"`` projects the last accepted fine field onto
    the coarsest grid before resolving every level for every target. Both
    registration methods share this fixed-reference history: callers must
    supply targets chronologically and finish consuming a stream before reset
    or reference replacement. ``temporal_max_age`` expires seeds after more
    than this many rejected frames; ``None`` keeps them until reset or success.

    ``device_type`` accepts ``"auto"``, ``"gpu"``, or ``"cpu"`` with the same
    initialization-time policy as :class:`ICGNWGPU`. All pyramid levels and
    streaming slots share the selected device, including CPU software
    execution. ``execution_device_type`` and ``adapter_info`` report the choice.
    """

    def __init__(
        self,
        *,
        pyramid_levels: int = DEFAULT_FULL_PYRAMID_LEVELS,
        iterations_per_level: tuple[int, ...] | None = None,
        tolerance_per_level: tuple[float, ...] | None = None,
        power_preference: str = "high-performance",
        device_type: str = "auto",
        interpolation: str = "bspline",
        motion_model: str = "affine",
        outlier_policy: str = "replace",
        subset_size: int = DEFAULT_SUBSET_SIZE,
        subset_stride: int | None = None,
        spatial_weighting: str = "uniform",
        gaussian_sigma: float | None = None,
        initialization: str = "zero",
        temporal_max_age: int | None = None,
        temporal_fine_correction: bool = False,
        temporal_prediction: str = "none",
        divergence_patience: int | None = None,
        divergence_min_iterations: int = 4,
        pyramid_policy: str = "fixed",
        adaptive_acceptance: float = 0.95,
    ) -> None:
        if not isinstance(pyramid_policy, str) or pyramid_policy not in ("fixed", "adaptive"):
            raise ValueError("pyramid_policy must be 'fixed' or 'adaptive'")
        if pyramid_policy == "adaptive" and (initialization != "temporal" or pyramid_levels != 2):
            raise ValueError("adaptive pyramid requires temporal initialization and two levels")
        if (
            isinstance(adaptive_acceptance, bool)
            or not isinstance(adaptive_acceptance, (int, float))
            or not np.isfinite(adaptive_acceptance)
            or not 0 < adaptive_acceptance <= 1
        ):
            raise ValueError("adaptive_acceptance must be finite and in (0, 1]")
        self.pyramid_policy = pyramid_policy
        self.adaptive_acceptance = float(adaptive_acceptance)
        self.divergence_patience = (
            None
            if divergence_patience is None
            else _iteration_count(divergence_patience, "divergence_patience")
        )
        self.divergence_min_iterations = _iteration_count(
            divergence_min_iterations, "divergence_min_iterations"
        )
        if not isinstance(temporal_prediction, str) or temporal_prediction not in (
            "none",
            "velocity",
        ):
            raise ValueError("temporal_prediction must be 'none' or 'velocity'")
        if temporal_prediction != "none" and initialization != "temporal":
            raise ValueError("temporal_prediction requires temporal initialization")
        self.temporal_prediction = temporal_prediction
        if not isinstance(temporal_fine_correction, bool):
            raise ValueError("temporal_fine_correction must be a boolean")
        if temporal_fine_correction and (initialization != "temporal" or pyramid_levels != 2):
            raise ValueError(
                "temporal_fine_correction requires temporal initialization and two levels"
            )
        self.temporal_fine_correction = temporal_fine_correction
        if not isinstance(initialization, str) or initialization not in ("zero", "temporal"):
            raise ValueError("initialization must be 'zero' or 'temporal'")
        if temporal_max_age is not None:
            try:
                age = operator.index(temporal_max_age)
            except TypeError as error:
                raise ValueError(
                    "temporal_max_age must be a nonnegative integer or None"
                ) from error
            if isinstance(temporal_max_age, bool) or not 0 <= age <= 0xFFFFFFFE:
                raise ValueError("temporal_max_age must be between 0 and 4294967294 or None")
            temporal_max_age = age
        self.initialization = initialization
        self.temporal_max_age = temporal_max_age
        self._temporal_buffers: dict[str, Any] = {}
        self.interpolation = normalize_interpolation(interpolation)
        self.motion_model = normalize_motion_model(motion_model)
        self.outlier_policy = normalize_outlier_policy(outlier_policy)
        self.subset_size = normalize_subset_size(subset_size)
        self.subset_stride = normalize_subset_stride(
            subset_stride,
            subset_size=self.subset_size,
        )
        self.spatial_weighting = normalize_spatial_weighting(spatial_weighting)
        self.gaussian_sigma = normalize_gaussian_sigma(
            gaussian_sigma,
            subset_size=self.subset_size,
        )
        self.pyramid_levels = _level_count(pyramid_levels)
        self._adaptive_next_levels = self.pyramid_levels
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

        # Every level uses the portable workgroup kernel. Sharing one root
        # engine also guarantees that
        # every level belongs to exactly the same wgpu device and queue.
        self._root = ICGNWGPU(
            power_preference=power_preference,
            device_type=device_type,
            prefer_subgroups=False,
            pyramid_levels=1,
            interpolation=self.interpolation,
            motion_model=self.motion_model,
            outlier_policy=self.outlier_policy,
            subset_size=self.subset_size,
            subset_stride=self.subset_stride,
            spatial_weighting=self.spatial_weighting,
            gaussian_sigma=self.gaussian_sigma,
        )
        if self.divergence_patience is not None:
            root = self._root
            root._solver_extra_constants = {
                "DIVERGENCE_PATIENCE": self.divergence_patience,
                "DIVERGENCE_MIN_ITERATIONS": self.divergence_min_iterations,
            }
            root._portable_solver = root.device.create_compute_pipeline(
                layout=root._pipeline_layout,
                compute={
                    "module": root._shader,
                    "entry_point": "solve",
                    "constants": root._solver_constants(),
                },
            )
            root._base_portable_solver = root._portable_solver
            root._solver = root._portable_solver
        self._root._ensure_seeded_solver()
        self._wgpu = self._root._wgpu
        self.device = self._root.device
        self.device_name = self._root.device_name
        self.device_type = self._root.device_type
        self.execution_device_type = self._root.execution_device_type
        self.adapter_info = dict(self._root.adapter_info)
        self._levels: list[ICGNWGPU] = [self._root]
        self._reduction_edges: list[_ReductionEdge] = []
        self._seed_edges: dict[int, _SeedEdge] = {}
        self._full_slots: list[_FullPyramidSlot] = []
        self._reference_ready = False
        self._roi_link_buffers: dict[int, Any] = {}
        self._roi_seed_pipeline = None
        self._zncc_min = 0.8
        self._create_orchestration_pipelines()
        if self.initialization == "temporal":
            self._create_temporal_pipelines()

    def _create_temporal_pipelines(self) -> None:
        """Compile opt-in orchestration only; the IC-GN equations are unchanged."""
        shader = self.device.create_shader_module(
            code=files("lucas_kanade_icgn").joinpath("shaders/full_pyramid.wgsl").read_text()
        )
        layouts = (
            (
                "read-only-storage",
                "storage",
                "storage",
                "uniform",
                "read-only-storage",
                "storage",
            ),
            ("storage", "storage", "read-only-storage", "read-only-storage", "uniform", "storage"),
        )
        self._temporal_layouts = []
        self._temporal_pipelines = []
        for group, (kinds, entry) in enumerate(
            zip(layouts, ("project_temporal_seed", "update_temporal_state"), strict=True), start=2
        ):
            layout = self.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": binding,
                        "visibility": self._wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": kind},
                    }
                    for binding, kind in enumerate(kinds)
                ]
            )
            pipeline_layout = self.device.create_pipeline_layout(
                bind_group_layouts=[self._empty_layout] * group + [layout]
            )
            constants = {"MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model]}
            if group == 2:
                constants.update(
                    TRACK_SEED_AVAILABILITY=self.temporal_fine_correction,
                    SUBSET_SIZE=self.subset_size,
                    SUBSET_STRIDE=self.subset_stride,
                )
            else:
                constants["REPORT_HISTORY_EXPIRY"] = self.pyramid_policy == "adaptive"
                constants["AGGREGATE_HISTORY_COUNTS"] = self.temporal_prediction == "velocity"
                if self.temporal_prediction == "velocity":
                    entry = "update_temporal_state_grouped"
            pipeline = self.device.create_compute_pipeline(
                layout=pipeline_layout,
                compute={"module": shader, "entry_point": entry, "constants": constants},
            )
            self._temporal_layouts.append(layout)
            self._temporal_pipelines.append(pipeline)
        if self.temporal_prediction == "velocity":
            self._velocity_layouts = []
            self._velocity_pipelines = []
            for group, entry, bindings, kinds in (
                (
                    2,
                    "predict_temporal_state",
                    (0, 1, 5, 6, 7, 9),
                    (
                        "read-only-storage",
                        "storage",
                        "storage",
                        "read-only-storage",
                        "read-only-storage",
                        "read-only-storage",
                    ),
                ),
                (
                    3,
                    "save_previous_temporal_state",
                    (0, 1, 2, 3, 4, 6, 7),
                    (
                        "storage",
                        "storage",
                        "read-only-storage",
                        "read-only-storage",
                        "uniform",
                        "storage",
                        "storage",
                    ),
                ),
            ):
                layout = self.device.create_bind_group_layout(
                    entries=[
                        {
                            "binding": binding,
                            "visibility": self._wgpu.ShaderStage.COMPUTE,
                            "buffer": {"type": kind},
                        }
                        for binding, kind in zip(bindings, kinds, strict=True)
                    ]
                )
                pipeline = self.device.create_compute_pipeline(
                    layout=self.device.create_pipeline_layout(
                        bind_group_layouts=[self._empty_layout] * group + [layout]
                    ),
                    compute={
                        "module": shader,
                        "entry_point": entry,
                        "constants": {
                            "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model]
                        },
                    },
                )
                self._velocity_layouts.append(layout)
                self._velocity_pipelines.append(pipeline)
        if self.temporal_fine_correction:
            bindings = (0, 1, 2, 3, 4, 6, 7, 8, 9)
            kinds = (
                "read-only-storage",
                "storage",
                "storage",
                "uniform",
                "read-only-storage",
                "read-only-storage",
                "read-only-storage",
                "uniform",
                "read-only-storage",
            )
            self._refinement_layout = self.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": binding,
                        "visibility": self._wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": kind},
                    }
                    for binding, kind in zip(bindings, kinds, strict=True)
                ]
            )
            self._refinement_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(
                    bind_group_layouts=[
                        self._empty_layout,
                        self._empty_layout,
                        self._refinement_layout,
                    ]
                ),
                compute={
                    "module": shader,
                    "entry_point": "refine_temporal_seed",
                    "constants": {
                        "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
                        "SUBSET_SIZE": self.subset_size,
                        "SUBSET_STRIDE": self.subset_stride,
                    },
                },
            )

    def _prepare_temporal_state(self, mask: np.ndarray | None) -> None:
        """Allocate one history for the engine, shared across all target slots."""
        root, coarse = self._root, self._levels[-1]
        count = root.grid_width * root.grid_height
        usage = self._wgpu.BufferUsage
        self._temporal_buffers["state"] = self.device.create_buffer(
            size=root._pipeline_slots[0].buffers["state"].size,
            usage=usage.STORAGE | usage.COPY_DST,
        )
        self._temporal_buffers["ages"] = self.device.create_buffer(
            size=count * 4,
            usage=usage.STORAGE | usage.COPY_DST,
        )
        self._temporal_buffers["config"] = self.device.create_buffer_with_data(
            data=struct.pack(
                "<4I",
                count,
                0xFFFFFFFE if self.temporal_max_age is None else self.temporal_max_age,
                int(self.outlier_policy == "replace" and self.temporal_prediction == "none"),
                0,
            ),
            usage=usage.UNIFORM,
        )
        permissions = np.full(coarse.grid_width * coarse.grid_height, 31, dtype=np.uint32)
        if mask is not None and not mask.all():
            permissions = seed_permissions(
                root.roi,
                coarse.roi,
                source_scale=1,
                destination_scale=1 << (self.pyramid_levels - 1),
                size=self.subset_size,
                stride=self.subset_stride,
                reference_mask=mask,
            )
        self._temporal_buffers["links"] = self.device.create_buffer_with_data(
            data=permissions,
            usage=usage.STORAGE,
        )
        if self.pyramid_policy == "adaptive":
            permissions = np.full(count, 31, dtype=np.uint32)
            if root.roi is not None:
                permissions[~root.roi.selected.ravel()] = 0
            self._temporal_buffers["fine_links"] = self.device.create_buffer_with_data(
                data=permissions,
                usage=usage.STORAGE,
            )
        if self.temporal_fine_correction and 0 not in self._roi_link_buffers:
            self._temporal_buffers["refinement_links"] = self.device.create_buffer_with_data(
                data=np.full(count, 31, np.uint32),
                usage=usage.STORAGE,
            )
        if self.temporal_fine_correction:
            coarse_count = coarse.grid_width * coarse.grid_height
            self._temporal_buffers["coarse_state"] = self.device.create_buffer(
                size=coarse_count * state_float_count(self.motion_model) * 4,
                usage=usage.STORAGE | usage.COPY_DST,
            )
            self._temporal_buffers["coarse_ages"] = self.device.create_buffer(
                size=coarse_count * 4,
                usage=usage.STORAGE | usage.COPY_DST,
            )
            self._temporal_buffers["coarse_config"] = self.device.create_buffer_with_data(
                data=struct.pack(
                    "<4I",
                    coarse_count,
                    0xFFFFFFFE if self.temporal_max_age is None else self.temporal_max_age,
                    int(self.outlier_policy == "replace" and self.temporal_prediction == "none"),
                    1,
                ),
                usage=usage.UNIFORM,
            )
        if self.temporal_prediction == "velocity":
            self._temporal_buffers["previous_state"] = self.device.create_buffer(
                size=self._temporal_buffers["state"].size,
                usage=usage.STORAGE | usage.COPY_DST,
            )
            self._temporal_buffers["previous_ages"] = self.device.create_buffer(
                size=count * 4,
                usage=usage.STORAGE | usage.COPY_DST,
            )
            self._temporal_buffers["prediction"] = self.device.create_buffer(
                size=self._temporal_buffers["state"].size,
                usage=usage.STORAGE,
            )
        self.reset_temporal_state()

    def reset_temporal_state(self) -> None:
        """Forget history, in queue order. Finish consuming a stream before reset.

        Both registration APIs share this history and require chronological
        targets. The reference and all pyramid preparations stay unchanged.
        This is a no-op before reference preparation or in zero mode.
        """
        if getattr(self, "pyramid_policy", "fixed") == "adaptive":
            self._adaptive_next_levels = self.pyramid_levels
        if not self._temporal_buffers:
            return
        for prefix in ("", "coarse_", "previous_"):
            if prefix + "ages" not in self._temporal_buffers:
                continue
            ages = self._temporal_buffers[prefix + "ages"]
            self.device.queue.write_buffer(ages, 0, np.full(ages.size // 4, 0xFFFFFFFF, np.uint32))
            state = np.zeros((ages.size // 4, state_float_count(self.motion_model)), np.float32)
            state[:, parameter_count(self.motion_model)] = -1.0
            self.device.queue.write_buffer(self._temporal_buffers[prefix + "state"], 0, state)

    def _create_temporal_slot(
        self, fine_slot: Any, coarse_slot: Any, fine_edge: _SeedEdge | None = None
    ) -> _TemporalSlot:
        coarse = self._levels[-1]
        coarse._ensure_seed_resources(coarse_slot)
        usage = self._wgpu.BufferUsage
        config = self.device.create_buffer_with_data(
            data=struct.pack(
                "<4I4f",
                self._root.grid_width,
                self._root.grid_height,
                coarse.grid_width,
                coarse.grid_width * coarse.grid_height,
                1.0,
                float(1 << (self.pyramid_levels - 1)),
                0.0,
                0.0,
            ),
            usage=usage.UNIFORM,
        )
        counts = self.device.create_buffer(
            size=16, usage=usage.STORAGE | usage.COPY_SRC | usage.COPY_DST
        )
        shared = self._temporal_buffers
        projection = self.device.create_bind_group(
            layout=self._temporal_layouts[0],
            entries=[
                self._buffer_resource(i, buffer)
                for i, buffer in enumerate(
                    (
                        shared["prediction"]
                        if self.temporal_prediction == "velocity"
                        else shared["state"],
                        coarse_slot.buffers["seed_state"],
                        coarse_slot.buffers["active_mask"],
                        config,
                        shared["links"],
                        counts,
                    )
                )
            ],
        )
        merge = self.device.create_bind_group(
            layout=self._temporal_layouts[1],
            entries=[
                self._buffer_resource(i, buffer)
                for i, buffer in enumerate(
                    (
                        shared["state"],
                        shared["ages"],
                        fine_slot.buffers["state"],
                        self._root._output_buffer(fine_slot),
                        shared["config"],
                        counts,
                    )
                )
            ],
        )
        # Append four counters to the existing mapped result, retaining one map.
        size = fine_slot.readback.size
        fine_slot.readback.destroy()
        fine_slot.readback = self.device.create_buffer(
            size=size + 16,
            usage=usage.MAP_READ | usage.COPY_DST,
        )
        refinement = None
        coarse_merge = None
        if self.temporal_fine_correction:
            links = (
                self._roi_link_buffers[0]
                if 0 in self._roi_link_buffers
                else shared["refinement_links"]
            )
            refinement = self.device.create_bind_group(
                layout=self._refinement_layout,
                entries=[
                    self._buffer_resource(binding, buffer)
                    for binding, buffer in zip(
                        (0, 1, 2, 3, 4, 6, 7, 8, 9),
                        (
                            shared["coarse_state"],
                            fine_slot.buffers["seed_state"],
                            fine_slot.buffers["active_mask"],
                            fine_edge.config,
                            links,
                            shared["state"],
                            shared["ages"],
                            shared["config"],
                            shared["coarse_ages"],
                        ),
                        strict=True,
                    )
                ],
            )
            coarse_merge = self.device.create_bind_group(
                layout=self._temporal_layouts[1],
                entries=[
                    self._buffer_resource(i, buffer)
                    for i, buffer in enumerate(
                        (
                            shared["coarse_state"],
                            shared["coarse_ages"],
                            coarse_slot.buffers["state"],
                            coarse._output_buffer(coarse_slot),
                            shared["coarse_config"],
                            counts,
                        )
                    )
                ],
            )
        prediction = save_previous = None
        if self.temporal_prediction == "velocity":
            prediction = self.device.create_bind_group(
                layout=self._velocity_layouts[0],
                entries=[
                    self._buffer_resource(binding, buffer)
                    for binding, buffer in (
                        (0, shared["state"]),
                        (1, shared["prediction"]),
                        (5, counts),
                        (6, shared["previous_state"]),
                        (7, shared["ages"]),
                        (9, shared["previous_ages"]),
                    )
                ],
            )
            save_previous = self.device.create_bind_group(
                layout=self._velocity_layouts[1],
                entries=[
                    self._buffer_resource(binding, buffer)
                    for binding, buffer in (
                        (0, shared["state"]),
                        (1, shared["ages"]),
                        (2, fine_slot.buffers["state"]),
                        (3, self._root._output_buffer(fine_slot)),
                        (4, shared["config"]),
                        (6, shared["previous_state"]),
                        (7, shared["previous_ages"]),
                    )
                ],
            )
        fine_projection = None
        if self.pyramid_policy == "adaptive":
            self._root._ensure_seed_resources(fine_slot)
            fine_config = self.device.create_buffer_with_data(
                data=struct.pack(
                    "<4I4f",
                    self._root.grid_width,
                    self._root.grid_height,
                    self._root.grid_width,
                    self._root.grid_width * self._root.grid_height,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ),
                usage=usage.UNIFORM,
            )
            group = self.device.create_bind_group(
                layout=self._temporal_layouts[0],
                entries=[
                    self._buffer_resource(i, buffer)
                    for i, buffer in enumerate(
                        (
                            shared["prediction"]
                            if self.temporal_prediction == "velocity"
                            else shared["state"],
                            fine_slot.buffers["seed_state"],
                            fine_slot.buffers["active_mask"],
                            fine_config,
                            shared["fine_links"],
                            counts,
                        )
                    )
                ],
            )
            fine_projection = _SeedEdge(
                self._root.grid_width * self._root.grid_height, fine_config, group
            )
        return _TemporalSlot(
            projection,
            merge,
            counts,
            config,
            refinement,
            coarse_merge,
            prediction,
            save_previous,
            fine_projection,
        )

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
        self._reduce_fused_pipeline = self.device.create_compute_pipeline(
            layout=reduce_y_pipeline_layout,
            compute={"module": shader, "entry_point": "reduce_fused"},
        )
        self._seed_pipeline = self.device.create_compute_pipeline(
            layout=seed_pipeline_layout,
            compute={
                "module": shader,
                "entry_point": "transfer_seed",
                "constants": {
                    "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
                    "SUBSET_SIZE": self.subset_size,
                    "SUBSET_STRIDE": self.subset_stride,
                    "TRACK_SEED_AVAILABILITY": self.temporal_fine_correction,
                },
            },
        )

    @staticmethod
    def _buffer_resource(binding: int, buffer: Any) -> dict[str, Any]:
        return {
            "binding": binding,
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
        }

    def _clear_level_resources(self) -> None:
        for buffer in self._temporal_buffers.values():
            buffer.destroy()
        self._temporal_buffers.clear()
        for buffer in self._roi_link_buffers.values():
            buffer.destroy()
        self._roi_link_buffers.clear()
        for slot in self._full_slots:
            if slot.temporal is not None:
                slot.temporal.counts.destroy()
                slot.temporal.seed_config.destroy()
                if slot.temporal.fine_projection is not None:
                    slot.temporal.fine_projection.config.destroy()
            for edge in slot.seed_edges.values():
                edge.config.destroy()
        self._full_slots.clear()
        self._reduction_edges.clear()
        self._seed_edges.clear()
        for level in self._levels[1:]:
            level.close()
        self._levels = [self._root]

    def _create_reduction_edges(self, level_slots: tuple[Any, ...]) -> list[_ReductionEdge]:
        edges: list[_ReductionEdge] = []
        for destination, source_slot, destination_slot in zip(
            self._levels[1:],
            level_slots[:-1],
            level_slots[1:],
            strict=True,
        ):
            bind_group = self.device.create_bind_group(
                layout=self._reduce_y_layout,
                entries=[
                    {"binding": 0, "resource": source_slot.source_texture.create_view()},
                    {"binding": 1, "resource": destination_slot.source_texture.create_view()},
                ],
            )
            edges.append(
                _ReductionEdge(
                    destination_width=destination.width,
                    destination_height=destination.height,
                    bind_group=bind_group,
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
                layout=self._roi_seed_layout
                if destination_index in self._roi_link_buffers
                else self._seed_layout,
                entries=[
                    self._buffer_resource(0, source._output_buffer(source_slot)),
                    self._buffer_resource(1, destination_slot.buffers["seed_state"]),
                    self._buffer_resource(2, destination_slot.buffers["active_mask"]),
                    self._buffer_resource(3, config),
                    *(
                        [self._buffer_resource(4, self._roi_link_buffers[destination_index])]
                        if destination_index in self._roi_link_buffers
                        else []
                    ),
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
        seed_edges = self._create_seed_edges(level_slots)
        return _FullPyramidSlot(
            level_slots=level_slots,
            reduction_edges=self._create_reduction_edges(level_slots),
            seed_edges=seed_edges,
            temporal=self._create_temporal_slot(level_slots[0], level_slots[-1], seed_edges.get(0))
            if self.initialization == "temporal"
            else None,
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
        roi_mask: np.ndarray | None = None,
        roi_policy: str = "full",
        roi_truncation: str = "none",
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
        mask = validate_roi(
            roi_mask, roi_policy, reference.shape, self.motion_model, roi_truncation
        )
        references = [reference]
        masks = [mask]
        for _ in range(1, self.pyramid_levels):
            references.append(_reduce_half(references[-1]))
            masks.append(None if masks[-1] is None else reduce_mask(masks[-1]))

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
                roi_mask=masks[level_index],
                roi_policy=roi_policy,
                roi_truncation=roi_truncation,
            )
            reference_build_wall_ms += info.reference_build_wall_ms
            if level_index:
                self._levels.append(engine)

        if mask is not None and not mask.all() and self.pyramid_levels > 1:
            self._prepare_roi_links(mask)
        self.roi = self._root.roi
        self.reference_valid = self._root.reference_valid
        self._root._roi_level_counts = [
            int(level.roi.selected.sum())
            if level.roi is not None
            else level.grid_width * level.grid_height
            for level in self._levels
        ]
        self._root._pyramid_config = {
            "pyramidLevels": self.pyramid_levels,
            "pyramidStrategy": "full",
            "iterationsPerLevel": list(self.iterations_per_level),
            "tolerancePerLevel": list(self.tolerance_per_level),
        }
        if self.pyramid_policy == "adaptive":
            self._root._pyramid_config.update(
                pyramidPolicy="adaptive",
                adaptiveAcceptance=self.adaptive_acceptance,
                adaptiveFeedback="ordered-single-slot",
            )
            self._adaptive_population = int(np.count_nonzero(self.reference_valid))
        if self.divergence_patience is not None:
            self._root._pyramid_config.update(
                divergencePatience=self.divergence_patience,
                divergenceMinIterations=self.divergence_min_iterations,
            )
        if self.initialization == "temporal":
            self._prepare_temporal_state(mask)
            self._root._pyramid_config.update(
                initialization="temporal",
                temporalMaxAge=self.temporal_max_age,
                temporalSeedLevel="coarsest",
                temporalSeedState=(
                    "filtered"
                    if self.outlier_policy == "replace" and self.temporal_prediction == "none"
                    else "raw"
                ),
                **({"temporalFineCorrection": True} if self.temporal_fine_correction else {}),
                **(
                    {"temporalPrediction": "velocity"}
                    if self.temporal_prediction == "velocity"
                    else {}
                ),
            )

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
                **(
                    {"roi": {**fine.roi.summary(), "selectedPerLevel": fine._roi_level_counts}}
                    if fine.roi is not None
                    else {}
                ),
                "interpolation": self.interpolation,
                "motionModel": self.motion_model,
                "outlierPolicy": self.outlier_policy,
                "subsetStride": self.subset_stride,
                "spatialWeighting": self.spatial_weighting,
                "gaussianSigma": self.gaussian_sigma,
            },
            device=self.device_name,
            grid=(fine.grid_width, fine.grid_height),
            width=fine.width,
            height=fine.height,
            subset_size=self.subset_size,
            subset_stride=self.subset_stride,
            roi=fine.roi,
            reference_valid=fine.reference_valid,
        )

    def _prepare_roi_links(self, mask: np.ndarray) -> None:
        """Build immutable transfer resources once, shared by all stream slots."""
        if self._roi_seed_pipeline is None:
            shader = self.device.create_shader_module(
                code=files("lucas_kanade_icgn")
                .joinpath("shaders/full_pyramid.wgsl")
                .read_text(encoding="utf-8")
            )
            # Preserve the portable layout when ROI is absent/all true.
            entries = [
                {
                    "binding": binding,
                    "visibility": self._wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": kind},
                }
                for binding, kind in enumerate(
                    ("read-only-storage", "storage", "storage", "uniform", "read-only-storage")
                )
            ]
            self._roi_seed_layout = self.device.create_bind_group_layout(entries=entries)
            layout = self.device.create_pipeline_layout(
                bind_group_layouts=[self._empty_layout, self._empty_layout, self._roi_seed_layout]
            )
            self._roi_seed_pipeline = self.device.create_compute_pipeline(
                layout=layout,
                compute={
                    "module": shader,
                    "entry_point": "transfer_seed_roi",
                    "constants": {
                        "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
                        "SUBSET_SIZE": self.subset_size,
                        "SUBSET_STRIDE": self.subset_stride,
                        "TRACK_SEED_AVAILABILITY": self.temporal_fine_correction,
                    },
                },
            )
        for index in range(self.pyramid_levels - 1):
            flags = seed_permissions(
                self._levels[index + 1].roi,
                self._levels[index].roi,
                source_scale=1 << (index + 1),
                destination_scale=1 << index,
                size=self.subset_size,
                stride=self.subset_stride,
                reference_mask=mask,
            )
            self._roi_link_buffers[index] = self.device.create_buffer_with_data(
                data=flags, usage=self._wgpu.BufferUsage.STORAGE
            )

    def _encode_reductions(
        self, encoder: Any, edges: list[_ReductionEdge], *, compute_pass: Any | None = None
    ) -> None:
        for edge in edges:
            current_pass = encoder.begin_compute_pass() if compute_pass is None else compute_pass
            current_pass.set_pipeline(self._reduce_fused_pipeline)
            current_pass.set_bind_group(1, edge.bind_group)
            current_pass.dispatch_workgroups(
                (edge.destination_width + 15) // 16,
                (edge.destination_height + 15) // 16,
                1,
            )
            if compute_pass is None:
                current_pass.end()

    def _encode_slot(self, slot: _FullPyramidSlot) -> tuple[Any, float]:
        encode_started = time.perf_counter()
        encoder = self.device.create_command_encoder(label="DIC-RT full pyramid registration")
        # Clears precede the shared compute pass. Dispatches remain ordered and
        # WebGPU tracks their resource hazards, including image writes/reads.
        for level, level_slot in zip(self._levels, slot.level_slots, strict=True):
            encoder.clear_buffer(level_slot.buffers["state"])
            if level.outlier_policy != "disabled":
                encoder.clear_buffer(level_slot.buffers["filtered_state"])
        if slot.temporal is not None:
            encoder.clear_buffer(slot.temporal.counts)
        compute_pass = encoder.begin_compute_pass()
        levels_used = getattr(self, "_adaptive_next_levels", self.pyramid_levels)
        self._encode_reductions(
            encoder, slot.reduction_edges if levels_used > 1 else [], compute_pass=compute_pass
        )
        coarsest_index = levels_used - 1
        coarsest = self._levels[coarsest_index]
        coarsest_slot = slot.level_slots[coarsest_index]
        if slot.temporal is not None:
            if slot.temporal.prediction is not None:
                compute_pass.set_pipeline(self._velocity_pipelines[0])
                compute_pass.set_bind_group(2, slot.temporal.prediction)
                compute_pass.dispatch_workgroups(
                    (self._root.grid_width * self._root.grid_height + 255) // 256
                )
            compute_pass.set_pipeline(self._temporal_pipelines[0])
            projection = (
                slot.temporal.fine_projection.bind_group
                if levels_used == 1 and slot.temporal.fine_projection is not None
                else slot.temporal.projection
            )
            compute_pass.set_bind_group(2, projection)
            compute_pass.dispatch_workgroups(
                (coarsest.grid_width * coarsest.grid_height + 255) // 256
            )
        coarsest._encode_registration_passes(
            encoder,
            coarsest_slot,
            coarsest._portable_solver if slot.temporal is None else coarsest._seeded_solver,
            solver_bind_group=coarsest_slot.bind_group
            if slot.temporal is None
            else coarsest_slot.seeded_bind_group,
            compute_pass=compute_pass,
        )
        for destination_index in range(coarsest_index - 1, -1, -1):
            seed_edge = slot.seed_edges[destination_index]
            compute_pass.set_pipeline(
                self._roi_seed_pipeline
                if destination_index in self._roi_link_buffers
                else self._seed_pipeline
            )
            compute_pass.set_bind_group(2, seed_edge.bind_group)
            compute_pass.dispatch_workgroups((seed_edge.destination_count + 255) // 256, 1, 1)
            if (
                destination_index == 0
                and slot.temporal is not None
                and slot.temporal.refinement is not None
            ):
                compute_pass.set_pipeline(self._refinement_pipeline)
                compute_pass.set_bind_group(2, slot.temporal.refinement)
                compute_pass.dispatch_workgroups((seed_edge.destination_count + 255) // 256, 1, 1)
            destination = self._levels[destination_index]
            destination_slot = slot.level_slots[destination_index]
            destination._encode_registration_passes(
                encoder,
                destination_slot,
                destination._seeded_solver,
                solver_bind_group=destination_slot.seeded_bind_group,
                compute_pass=compute_pass,
            )
        fine_slot = slot.level_slots[0]
        if slot.temporal is not None:
            if slot.temporal.save_previous is not None:
                compute_pass.set_pipeline(self._velocity_pipelines[1])
                compute_pass.set_bind_group(3, slot.temporal.save_previous)
                compute_pass.dispatch_workgroups(
                    (self._root.grid_width * self._root.grid_height + 255) // 256
                )
            compute_pass.set_pipeline(self._temporal_pipelines[1])
            compute_pass.set_bind_group(3, slot.temporal.merge)
            compute_pass.dispatch_workgroups(
                (self._root.grid_width * self._root.grid_height + 255) // 256
            )
            if slot.temporal.coarse_merge is not None:
                compute_pass.set_bind_group(3, slot.temporal.coarse_merge)
                compute_pass.dispatch_workgroups(
                    (self._levels[-1].grid_width * self._levels[-1].grid_height + 255) // 256
                )
            compute_pass.end()
            encoder.copy_buffer_to_buffer(
                slot.temporal.counts,
                0,
                fine_slot.readback,
                self._root._output_buffer(fine_slot).size,
                16,
            )
        else:
            compute_pass.end()
        encoder.copy_buffer_to_buffer(
            self._root._output_buffer(fine_slot),
            0,
            fine_slot.readback,
            0,
            fine_slot.readback.size if slot.temporal is None else fine_slot.readback.size - 16,
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
            levels_used=getattr(self, "_adaptive_next_levels", self.pyramid_levels),
        )

    def _retire_slot(self, pending: _PendingFullPyramidRegistration) -> RegistrationResult:
        fine_slot = pending.slot.level_slots[0]
        wait_started = time.perf_counter()
        fine_slot.readback.map_sync(self._wgpu.MapMode.READ)
        completion_wait_wall_ms = (time.perf_counter() - wait_started) * 1_000.0
        copy_started = time.perf_counter()
        raw = fine_slot.readback.read_mapped(copy=True)
        fine_slot.readback.unmap()
        temporal_timing = {}
        if pending.slot.temporal is not None:
            counts = np.frombuffer(raw, dtype=np.uint32, count=4, offset=len(raw) - 16)
            temporal_timing = {
                "temporallySeededCoarseSubsets": float(counts[0]),
                "temporalAcceptedUpdates": float(counts[1]),
            }
            if self.temporal_prediction == "velocity":
                temporal_timing["velocityPredictedSubsets"] = float(counts[2])
            if self.pyramid_policy == "adaptive":
                temporal_timing["temporalExpiredSubsets"] = float(counts[3])
            raw = memoryview(raw)[:-16]
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
                "pyramidLevels": float(pending.levels_used or self.pyramid_levels),
                "queueSubmissions": 1.0,
                **temporal_timing,
            },
            config=self._root._config,
        )
        if getattr(self, "pyramid_policy", "fixed") == "adaptive":
            accepted = temporal_timing.get("temporalAcceptedUpdates", 0)
            expired = temporal_timing.get("temporalExpiredSubsets", 0)
            self._adaptive_next_levels = (
                1
                if self._adaptive_population > 0
                and accepted / self._adaptive_population >= self.adaptive_acceptance
                and not expired
                else 2
            )
            result.config["pyramidLevelsUsed"] = pending.levels_used
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
        if self._root.roi is not None and self._root.roi.empty:
            return self._root._empty_roi_result(accepted_at, self._root._config)
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

        # Host feedback must see the same preceding frame at every requested
        # depth. Retire each result before choosing the next level count;
        # this reuses the normal readback and adds no intermediate GPU sync.
        if self.pyramid_policy == "adaptive":
            depth = 1

        if self._root.roi is not None and self._root.roi.empty:
            for target in targets:
                started = time.perf_counter()
                self._root._prepare_target(target)
                yield self._root._empty_roi_result(started, self._root._config)
            return
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
