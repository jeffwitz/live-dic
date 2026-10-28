"""Native WebGPU affine IC-GN backend powered by wgpu-py.

The backend uploads NumPy grayscale bytes directly to an ``r8unorm`` texture,
submits WGSL compute kernels through wgpu-native/Vulkan, and reads one packed
binary result buffer. It deliberately avoids Chromium, PNG encoding, DevTools,
JSON result conversion, and any visible user interface.
"""

from __future__ import annotations

import operator
import struct
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib.resources import files
from statistics import median
from typing import Any

import numpy as np
import numpy.typing as npt

from .engine import ICGNError, ImageInput, ReferenceInfo, RegistrationResult, _as_uint8_gray

SUBSET_SIZE = 16
SUBSET_HALF_EXTENT = 7.5
FIR_RADIUS = 15
MAX_ITERATIONS = 30


@dataclass(slots=True)
class _RegistrationSlot:
    """Per-frame resources that may be in flight independently."""

    source_texture: Any
    buffers: dict[str, Any]
    bind_group: Any
    readback: Any


@dataclass(slots=True)
class _PendingRegistration:
    """CPU timing and configuration retained until one slot is retired."""

    slot: _RegistrationSlot
    submitted_at: float
    upload_wall_ms: float
    submit_wall_ms: float
    config: dict[str, float | int]


def _validated_config(
    iterations: int,
    tolerance: float,
    zncc_min: float,
) -> dict[str, float | int]:
    """Validate public solver controls before packing the GPU uniform."""

    iteration_count = int(iterations)
    tolerance_value = float(tolerance)
    zncc_value = float(zncc_min)
    if not 1 <= iteration_count <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
    if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be a finite positive number")
    if not np.isfinite(zncc_value) or not 0.0 <= zncc_value <= 1.0:
        raise ValueError("zncc_min must be between 0 and 1")
    return {
        "iterations": iteration_count,
        "tolerance": tolerance_value,
        "znccMin": zncc_value,
    }


def _fir_weights() -> npt.NDArray[np.float32]:
    """Return the SYCL-equivalent Gaussian/inverse-spline FIR coefficients."""

    gaussian_radius = 3
    offsets = np.arange(-gaussian_radius, gaussian_radius + 1, dtype=np.float64)
    gaussian = np.exp(-0.5 * offsets * offsets)
    gaussian /= gaussian.sum()
    pole = np.sqrt(3.0) - 2.0
    weights = np.zeros(2 * FIR_RADIUS + 1, dtype=np.float64)
    for gaussian_index, offset in enumerate(range(-gaussian_radius, gaussian_radius + 1)):
        for spline_offset in range(-12, 13):
            weights[offset + spline_offset + FIR_RADIUS] += (
                gaussian[gaussian_index] * np.sqrt(3.0) * pole ** abs(spline_offset)
            )
    return np.ascontiguousarray(weights, dtype=np.float32)


def _gaussian_sigma_one(source: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """Apply the same separable mirrored sigma-one Gaussian as the browser backends."""

    radius = 3
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * offsets * offsets)
    weights /= weights.sum()
    values = source.astype(np.float64, copy=False)
    padded_x = np.pad(values, ((0, 0), (radius, radius)), mode="reflect")
    horizontal = sum(
        weight * padded_x[:, index : index + source.shape[1]]
        for index, weight in enumerate(weights)
    )
    padded_y = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    return sum(
        weight * padded_y[index : index + source.shape[0], :]
        for index, weight in enumerate(weights)
    )


def _spline_filter_axis(coefficients: npt.NDArray[np.float64], axis: int) -> None:
    """Apply the cubic B-spline recursive prefilter along one array axis."""

    values = np.swapaxes(coefficients, axis, -1)
    pole = np.sqrt(3.0) - 2.0
    gain = (1.0 - pole) * (1.0 - 1.0 / pole)
    values *= gain
    count = values.shape[-1]
    horizon = min(count, int(np.ceil(np.log(1e-12) / np.log(abs(pole)))))
    powers = pole ** np.arange(horizon, dtype=np.float64)
    values[..., 0] = np.sum(values[..., :horizon] * powers, axis=-1)
    for index in range(1, count):
        values[..., index] += pole * values[..., index - 1]
    values[..., -1] = pole / (pole * pole - 1.0) * (pole * values[..., -2] + values[..., -1])
    for index in range(count - 2, -1, -1):
        values[..., index] = pole * (values[..., index + 1] - values[..., index])


def _cubic_bspline_prefilter(source: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    coefficients = np.array(source, dtype=np.float64, copy=True, order="C")
    _spline_filter_axis(coefficients, 1)
    _spline_filter_axis(coefficients, 0)
    return coefficients


def _reference_arrays(
    image: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Build reference pixels, inverse Hessians, and solver metadata."""

    height, width = image.shape
    grid_width = width // SUBSET_SIZE
    grid_height = height // SUBSET_SIZE
    filtered = _gaussian_sigma_one(image)
    coefficients = _cubic_bspline_prefilter(filtered)

    padded = np.pad(coefficients, 1, mode="reflect")
    gradient_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    gradient_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    gradient_x = (
        0.5 * (padded[:-2, 2:] - padded[:-2, :-2])
        + 4.0 * gradient_x
        + 0.5 * (padded[2:, 2:] - padded[2:, :-2])
    ) / 6.0
    gradient_y = (
        0.5 * (padded[2:, :-2] - padded[:-2, :-2])
        + 4.0 * gradient_y
        + 0.5 * (padded[2:, 2:] - padded[:-2, 2:])
    ) / 6.0

    covered_height = grid_height * SUBSET_SIZE
    covered_width = grid_width * SUBSET_SIZE
    tiled = filtered[:covered_height, :covered_width].reshape(
        grid_height, SUBSET_SIZE, grid_width, SUBSET_SIZE
    )
    means = tiled.mean(axis=(1, 3))
    centered = filtered.copy()
    centered[:covered_height, :covered_width] -= np.repeat(
        np.repeat(means, SUBSET_SIZE, axis=0), SUBSET_SIZE, axis=1
    )

    reference = np.zeros((height, width, 4), dtype=np.float32)
    reference[..., 0] = centered
    reference[..., 1] = gradient_x
    reference[..., 2] = gradient_y

    grid_count = grid_width * grid_height
    inverse = np.zeros((grid_count, 36), dtype=np.float32)
    metadata = np.zeros((grid_count, 16), dtype=np.float32)
    local_y, local_x = np.meshgrid(
        np.arange(SUBSET_SIZE, dtype=np.float64) - SUBSET_HALF_EXTENT,
        np.arange(SUBSET_SIZE, dtype=np.float64) - SUBSET_HALF_EXTENT,
        indexing="ij",
    )
    local_x = local_x.ravel()
    local_y = local_y.ravel()

    for grid_y in range(grid_height):
        row_slice = slice(grid_y * SUBSET_SIZE, (grid_y + 1) * SUBSET_SIZE)
        f_row = centered[row_slice, :covered_width].reshape(SUBSET_SIZE, grid_width, SUBSET_SIZE)
        gx_row = gradient_x[row_slice, :covered_width].reshape(SUBSET_SIZE, grid_width, SUBSET_SIZE)
        gy_row = gradient_y[row_slice, :covered_width].reshape(SUBSET_SIZE, grid_width, SUBSET_SIZE)
        f_tiles = f_row.transpose(1, 0, 2).reshape(grid_width, -1)
        gx_tiles = gx_row.transpose(1, 0, 2).reshape(grid_width, -1)
        gy_tiles = gy_row.transpose(1, 0, 2).reshape(grid_width, -1)
        jacobian = np.stack(
            (
                gx_tiles,
                gx_tiles * local_x,
                gx_tiles * local_y,
                gy_tiles,
                gy_tiles * local_x,
                gy_tiles * local_y,
            ),
            axis=2,
        )
        hessian = np.einsum("npi,npj->nij", jacobian, jacobian, optimize=True)
        jt_reference = np.einsum("npi,np->ni", jacobian, f_tiles, optimize=True)
        sum_j = jacobian.sum(axis=1)
        energy = np.einsum("np,np->n", f_tiles, f_tiles, optimize=True)
        row_offset = grid_y * grid_width
        try:
            row_inverse = np.linalg.inv(hessian)
            valid = np.isfinite(row_inverse).all(axis=(1, 2)) & (energy > 1e-12)
        except np.linalg.LinAlgError:
            row_inverse = np.zeros_like(hessian)
            valid = np.zeros(grid_width, dtype=bool)
            for index in range(grid_width):
                try:
                    row_inverse[index] = np.linalg.inv(hessian[index])
                    valid[index] = energy[index] > 1e-12
                except np.linalg.LinAlgError:
                    pass
        inverse[row_offset : row_offset + grid_width] = row_inverse.reshape(grid_width, 36)
        metadata[row_offset : row_offset + grid_width, 0] = np.sqrt(energy)
        metadata[row_offset : row_offset + grid_width, 1] = valid.astype(np.float32)
        metadata[row_offset : row_offset + grid_width, 2:8] = jt_reference
        metadata[row_offset : row_offset + grid_width, 8:14] = sum_j

    return (
        np.ascontiguousarray(reference.ravel()),
        np.ascontiguousarray(inverse.ravel()),
        np.ascontiguousarray(metadata.ravel()),
    )


class ICGNWGPU:
    """Persistent native WebGPU IC-GN engine with direct NumPy transfers.

    The target prefilter, affine solver, and spatial median replacement all run
    on the native GPU adapter. By default, reference preparation benchmarks
    the portable and subgroup solvers when both are available. Set
    ``prefer_subgroups`` to ``False`` or ``True`` to force either path for
    qualification.
    """

    def __init__(
        self,
        *,
        power_preference: str = "high-performance",
        prefer_subgroups: bool | None = None,
    ) -> None:
        try:
            import wgpu
        except ImportError as error:
            raise ICGNError(
                "The native backend requires the optional 'wgpu' package. "
                "Install lucas-kanade-icgn-webgl[native]."
            ) from error

        adapters = wgpu.gpu.enumerate_adapters_sync()
        candidates = [
            adapter
            for adapter in adapters
            if adapter.info["backend_type"] == "Vulkan" and adapter.info["adapter_type"] != "CPU"
        ]
        if not candidates:
            adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        else:
            adapter = candidates[0]
        if adapter is None:
            raise ICGNError("wgpu-native could not find a compatible GPU adapter.")

        self._wgpu = wgpu
        self.adapter = adapter
        required_features = []
        if prefer_subgroups is not False and "subgroup" in adapter.features:
            required_features.append("subgroup")
        try:
            self.device = adapter.request_device_sync(
                label="DIC-RT native WebGPU",
                required_features=required_features,
            )
        except Exception as error:
            if not required_features:
                raise
            if prefer_subgroups is True:
                raise ICGNError(
                    "The adapter advertises subgroups but the feature could not be enabled."
                ) from error
            required_features.clear()
            self.device = adapter.request_device_sync(label="DIC-RT native WebGPU")
        self.device_name = f"wgpu-native {adapter.info['backend_type']} ({adapter.info['device']})"
        self.subgroup_width = (
            self._measure_subgroup_width() if "subgroup" in self.device.features else 0
        )
        self._solver_preference = prefer_subgroups
        self.solver_tuning_ms: dict[str, float] = {}
        shader_path = files("lucas_kanade_icgn").joinpath("shaders/icgn.wgsl")
        shader_source = shader_path.read_text(encoding="utf-8")
        shader = self.device.create_shader_module(code=shader_source)
        subgroup_shader = None
        if self.subgroup_width >= SUBSET_SIZE:
            subgroup_path = files("lucas_kanade_icgn").joinpath("shaders/icgn_subgroup.spv")
            subgroup_shader = self.device.create_shader_module(
                label="DIC-RT subgroup compute solver",
                code=subgroup_path.read_bytes(),
            )
        self._bind_group_layout = self._create_bind_group_layout()
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._prefilter_x = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "prefilter_x"},
        )
        self._prefilter_y = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "prefilter_y"},
        )
        self._portable_solver = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "solve"},
        )
        self._subgroup_solver = None
        if subgroup_shader is not None:
            self._subgroup_solver = self.device.create_compute_pipeline(
                layout=pipeline_layout,
                compute={"module": subgroup_shader, "entry_point": "main"},
            )
        if prefer_subgroups is True and self._subgroup_solver is None:
            raise ICGNError("The subgroup solver was requested but is unavailable on this adapter.")
        if prefer_subgroups is True and self._subgroup_solver is not None:
            self.solver = "subgroup"
            self._solver = self._subgroup_solver
        else:
            self.solver = "portable-workgroup"
            self._solver = self._portable_solver
        self._median_filter = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "median_filter"},
        )
        self._reference_ready = False
        self._buffers: dict[str, Any] = {}
        self._source_texture: Any = None
        self._bind_group: Any = None
        self._pipeline_slots: list[_RegistrationSlot] = []
        self.width = 0
        self.height = 0
        self.grid_width = 0
        self.grid_height = 0
        self._config = {"iterations": 30, "tolerance": 1e-3, "znccMin": 0.8}

    def _measure_subgroup_width(self) -> int:
        """Return the hardware subgroup width for a 16-lane workgroup."""

        wgpu = self._wgpu
        try:
            probe_path = files("lucas_kanade_icgn").joinpath("shaders/subgroup_probe.spv")
            shader = self.device.create_shader_module(
                label="DIC-RT subgroup width compute probe",
                code=probe_path.read_bytes(),
            )
            bind_group_layout = self.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.storage},
                    }
                ]
            )
            pipeline_layout = self.device.create_pipeline_layout(
                bind_group_layouts=[bind_group_layout]
            )
            pipeline = self.device.create_compute_pipeline(
                layout=pipeline_layout,
                compute={"module": shader, "entry_point": "main"},
            )
            output = self.device.create_buffer(
                size=4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            bind_group = self.device.create_bind_group(
                layout=bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": output, "offset": 0, "size": 4},
                    }
                ],
            )
            encoder = self.device.create_command_encoder(label="DIC-RT subgroup probe")
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(1, 1, 1)
            compute_pass.end()
            self.device.queue.submit([encoder.finish()])
            raw = self.device.queue.read_buffer(output)
            return int(struct.unpack_from("<I", raw)[0])
        except Exception:
            return 0

    def _create_bind_group_layout(self) -> Any:
        wgpu = self._wgpu
        entries: list[dict[str, Any]] = [
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
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
        for binding in range(2, 10):
            read_only = binding in {4, 5, 6, 8}
            entries.append(
                {
                    "binding": binding,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": (
                            wgpu.BufferBindingType.read_only_storage
                            if read_only
                            else wgpu.BufferBindingType.storage
                        )
                    },
                }
            )
        return self.device.create_bind_group_layout(entries=entries)

    def _storage_buffer(self, data: npt.NDArray[np.float32], *, output: bool = False) -> Any:
        usage = self._wgpu.BufferUsage.STORAGE
        if output:
            usage |= self._wgpu.BufferUsage.COPY_SRC | self._wgpu.BufferUsage.COPY_DST
        return self.device.create_buffer_with_data(data=data, usage=usage)

    def _allocate_runtime(
        self,
        reference: npt.NDArray[np.float32],
        inverse: npt.NDArray[np.float32],
        metadata: npt.NDArray[np.float32],
    ) -> None:
        self._destroy_runtime()
        shared_buffers = {
            "reference": self._storage_buffer(reference),
            "inverse": self._storage_buffer(inverse),
            "metadata": self._storage_buffer(metadata),
            "fir": self._storage_buffer(_fir_weights()),
        }
        primary = self._create_registration_slot(shared_buffers)
        self._pipeline_slots = [primary]
        self._source_texture = primary.source_texture
        self._buffers = primary.buffers
        self._bind_group = primary.bind_group

    def _create_registration_slot(
        self,
        shared_buffers: dict[str, Any],
    ) -> _RegistrationSlot:
        """Allocate one independent target, solver state, and readback set."""

        wgpu = self._wgpu
        pixel_count = self.width * self.height
        grid_count = self.grid_width * self.grid_height
        source_texture = self.device.create_texture(
            size=(self.width, self.height, 1),
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.r8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        buffers = {
            **shared_buffers,
            "config": self.device.create_buffer(
                size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
            ),
            "temporary": self.device.create_buffer(
                size=pixel_count * 4, usage=wgpu.BufferUsage.STORAGE
            ),
            "coefficients": self.device.create_buffer(
                size=pixel_count * 4,
                usage=wgpu.BufferUsage.STORAGE,
            ),
            "state": self.device.create_buffer(
                size=grid_count * 8 * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            ),
            "filtered_state": self.device.create_buffer(
                size=grid_count * 8 * 4,
                usage=(
                    wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
                ),
            ),
        }
        resources = [
            {"binding": 0, "resource": source_texture.create_view()},
            *(
                self._buffer_resource(binding, buffers[name])
                for binding, name in enumerate(
                    (
                        "config",
                        "temporary",
                        "coefficients",
                        "reference",
                        "inverse",
                        "metadata",
                        "state",
                        "fir",
                        "filtered_state",
                    ),
                    start=1,
                )
            ),
        ]
        bind_group = self.device.create_bind_group(
            layout=self._bind_group_layout, entries=resources
        )
        readback = self.device.create_buffer(
            size=grid_count * 8 * 4,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        return _RegistrationSlot(
            source_texture=source_texture,
            buffers=buffers,
            bind_group=bind_group,
            readback=readback,
        )

    @staticmethod
    def _buffer_resource(binding: int, buffer: Any) -> dict[str, Any]:
        return {
            "binding": binding,
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
        }

    def _packed_config(self, config: dict[str, float | int]) -> bytes:
        return struct.pack(
            "<6I2f",
            self.width,
            self.height,
            self.grid_width,
            self.grid_width * self.grid_height,
            int(config["iterations"]),
            self.width * self.height,
            float(config["tolerance"]),
            float(config["znccMin"]),
        )

    def _update_slot_config(
        self,
        slot: _RegistrationSlot,
        config: dict[str, float | int],
    ) -> None:
        self.device.queue.write_buffer(slot.buffers["config"], 0, self._packed_config(config))

    def _update_config(self) -> None:
        self._update_slot_config(self._pipeline_slots[0], self._config)

    def _ensure_pipeline_slots(self, depth: int) -> list[_RegistrationSlot]:
        if depth < 1:
            raise ValueError("pipeline_depth must be at least 1")
        shared_buffers = {
            name: self._buffers[name] for name in ("reference", "inverse", "metadata", "fir")
        }
        while len(self._pipeline_slots) < depth:
            slot = self._create_registration_slot(shared_buffers)
            self._update_slot_config(slot, self._config)
            self._pipeline_slots.append(slot)
        return self._pipeline_slots[:depth]

    def _destroy_runtime(self) -> None:
        """Release dimension-dependent resources without destroying the device."""

        dynamic_names = ("config", "temporary", "coefficients", "state", "filtered_state")
        for slot in self._pipeline_slots:
            slot.source_texture.destroy()
            slot.readback.destroy()
            for name in dynamic_names:
                slot.buffers[name].destroy()
        for name in ("reference", "inverse", "metadata", "fir"):
            buffer = self._buffers.get(name)
            if buffer is not None:
                buffer.destroy()
        self._pipeline_slots.clear()
        self._buffers.clear()
        self._source_texture = None
        self._bind_group = None

    def _encode_registration(
        self,
        slot: _RegistrationSlot,
        solver_pipeline: Any,
        *,
        copy_to_readback: bool,
    ) -> Any:
        """Encode one registration without imposing a CPU/GPU synchronization."""

        encoder = self.device.create_command_encoder(label="DIC-RT native registration")
        encoder.clear_buffer(slot.buffers["state"])
        encoder.clear_buffer(slot.buffers["filtered_state"])
        pixel_workgroups = (self.width * self.height + 255) // 256
        for pipeline, workgroups in (
            (self._prefilter_x, pixel_workgroups),
            (self._prefilter_y, pixel_workgroups),
            (solver_pipeline, self.grid_width * self.grid_height),
            (self._median_filter, (self.grid_width * self.grid_height + 255) // 256),
        ):
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, slot.bind_group)
            compute_pass.dispatch_workgroups(workgroups, 1, 1)
            compute_pass.end()
        if copy_to_readback:
            encoder.copy_buffer_to_buffer(
                slot.buffers["filtered_state"],
                0,
                slot.readback,
                0,
                slot.readback.size,
            )
        return encoder.finish()

    def _execute_registration(self, solver_pipeline: Any) -> memoryview:
        """Execute the primary slot and return its synchronized state readback."""

        slot = self._pipeline_slots[0]
        command = self._encode_registration(slot, solver_pipeline, copy_to_readback=False)
        self.device.queue.submit([command])
        return self.device.queue.read_buffer(self._buffers["filtered_state"])

    def _upload_frame(
        self,
        slot: _RegistrationSlot,
        frame: npt.NDArray[np.uint8],
    ) -> float:
        started = time.perf_counter()
        self.device.queue.write_texture(
            {"texture": slot.source_texture, "mip_level": 0, "origin": (0, 0, 0)},
            frame,
            {"offset": 0, "bytes_per_row": frame.strides[0], "rows_per_image": self.height},
            (self.width, self.height, 1),
        )
        return (time.perf_counter() - started) * 1_000.0

    def _autotune_solver(self, reference: npt.NDArray[np.uint8]) -> None:
        """Select the fastest valid solver on a representative shifted frame."""

        if self._solver_preference is not None or self._subgroup_solver is None:
            return
        target = np.ascontiguousarray(np.roll(reference, shift=(-1, 2), axis=(0, 1)))
        self.device.queue.write_texture(
            {"texture": self._source_texture, "mip_level": 0, "origin": (0, 0, 0)},
            target,
            {"offset": 0, "bytes_per_row": target.strides[0], "rows_per_image": self.height},
            (self.width, self.height, 1),
        )
        candidates = {
            "portable-workgroup": self._portable_solver,
            "subgroup": self._subgroup_solver,
        }
        for pipeline in candidates.values():
            self._execute_registration(pipeline)
        samples: dict[str, list[float]] = {name: [] for name in candidates}
        names = tuple(candidates)
        for repetition in range(3):
            order = names if repetition % 2 == 0 else tuple(reversed(names))
            for name in order:
                started = time.perf_counter()
                self._execute_registration(candidates[name])
                samples[name].append((time.perf_counter() - started) * 1_000.0)
        self.solver_tuning_ms = {name: float(median(timings)) for name, timings in samples.items()}
        self.solver = min(self.solver_tuning_ms, key=self.solver_tuning_ms.__getitem__)
        self._solver = candidates[self.solver]

    def set_reference(
        self,
        image: ImageInput,
        *,
        iterations: int = 30,
        tolerance: float = 1e-3,
        zncc_min: float = 0.8,
    ) -> ReferenceInfo:
        total_start = time.perf_counter()
        frame = _as_uint8_gray(image)
        self.height, self.width = frame.shape
        self.grid_width = self.width // SUBSET_SIZE
        self.grid_height = self.height // SUBSET_SIZE
        if self.grid_width < 1 or self.grid_height < 1:
            raise ValueError("Reference dimensions must each be at least 16 pixels.")
        self._config = _validated_config(iterations, tolerance, zncc_min)
        reference_start = time.perf_counter()
        reference, inverse, metadata = _reference_arrays(frame)
        reference_wall_ms = (time.perf_counter() - reference_start) * 1_000.0
        runtime_start = time.perf_counter()
        self._allocate_runtime(reference, inverse, metadata)
        self._update_config()
        self._autotune_solver(frame)
        runtime_wall_ms = (time.perf_counter() - runtime_start) * 1_000.0
        self._reference_ready = True
        total_wall_ms = (time.perf_counter() - total_start) * 1_000.0
        return ReferenceInfo(
            preparation_wall_ms=reference_wall_ms + runtime_wall_ms,
            runtime_build_wall_ms=runtime_wall_ms,
            reference_build_wall_ms=reference_wall_ms,
            encoding_wall_ms=0.0,
            total_wall_ms=total_wall_ms,
            config=dict(self._config),
            device=self.device_name,
            grid=(self.grid_width, self.grid_height),
            width=self.width,
            height=self.height,
            subset_size=SUBSET_SIZE,
        )

    def _prepare_target(self, target: ImageInput) -> npt.NDArray[np.uint8]:
        frame = _as_uint8_gray(target)
        if frame.shape != (self.height, self.width):
            raise ValueError(
                f"Target dimensions must be {self.width}x{self.height}; "
                f"received {frame.shape[1]}x{frame.shape[0]}."
            )
        return frame

    def _resolve_config(
        self,
        iterations: int | None,
        tolerance: float | None,
        zncc_min: float | None,
    ) -> dict[str, float | int]:
        return _validated_config(
            int(self._config["iterations"]) if iterations is None else iterations,
            float(self._config["tolerance"]) if tolerance is None else tolerance,
            float(self._config["znccMin"]) if zncc_min is None else zncc_min,
        )

    def _result_from_raw(
        self,
        raw: Any,
        *,
        timing: dict[str, float],
        config: dict[str, float | int],
    ) -> RegistrationResult:
        # The NumPy views retain the copied readback memory as their base. No
        # additional full-field copy is made while building the public result.
        packed = np.frombuffer(raw, dtype=np.float32).reshape(
            self.grid_height,
            self.grid_width,
            8,
        )
        parameters = packed[..., :6]
        status = packed[..., 6]
        iteration_field = packed[..., 7]
        accepted_mask = status > 0.5
        accepted = int(np.count_nonzero(accepted_mask))
        mean_translation = None
        mean_iterations = None
        if accepted:
            mean_translation = (
                float(np.mean(parameters[..., 0][accepted_mask], dtype=np.float64)),
                float(np.mean(parameters[..., 3][accepted_mask], dtype=np.float64)),
            )
            mean_iterations = float(np.mean(iteration_field[accepted_mask], dtype=np.float64))
        return RegistrationResult(
            parameters=parameters,
            status=status,
            iterations=iteration_field,
            accepted=accepted,
            replaced=int(np.count_nonzero(status > 1.5)),
            mean_translation=mean_translation,
            mean_iterations=mean_iterations,
            timing=timing,
            config=dict(config),
            width=self.width,
            height=self.height,
            subset_size=SUBSET_SIZE,
        )

    def register(
        self,
        target: ImageInput,
        *,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> RegistrationResult:
        if not self._reference_ready:
            raise ICGNError("Call set_reference() before register().")
        total_start = time.perf_counter()
        frame = self._prepare_target(target)
        self._config = self._resolve_config(iterations, tolerance, zncc_min)
        self._update_config()

        upload_wall_ms = self._upload_frame(self._pipeline_slots[0], frame)

        execution_start = time.perf_counter()
        raw = self._execute_registration(self._solver)
        execution_readback_wall_ms = (time.perf_counter() - execution_start) * 1_000.0
        total_wall_ms = (time.perf_counter() - total_start) * 1_000.0
        return self._result_from_raw(
            raw,
            timing={
                "uploadWallMs": upload_wall_ms,
                "executeReadbackWallMs": execution_readback_wall_ms,
                "totalWallMs": total_wall_ms,
            },
            config=self._config,
        )

    def _submit_slot(
        self,
        slot: _RegistrationSlot,
        frame: npt.NDArray[np.uint8],
        config: dict[str, float | int],
    ) -> _PendingRegistration:
        submitted_at = time.perf_counter()
        upload_wall_ms = self._upload_frame(slot, frame)
        submit_started = time.perf_counter()
        command = self._encode_registration(slot, self._solver, copy_to_readback=True)
        self.device.queue.submit([command])
        submit_wall_ms = (time.perf_counter() - submit_started) * 1_000.0
        return _PendingRegistration(
            slot=slot,
            submitted_at=submitted_at,
            upload_wall_ms=upload_wall_ms,
            submit_wall_ms=submit_wall_ms,
            config=dict(config),
        )

    def _retire_slot(self, pending: _PendingRegistration) -> RegistrationResult:
        wait_started = time.perf_counter()
        pending.slot.readback.map_sync(self._wgpu.MapMode.READ)
        readback_wait_wall_ms = (time.perf_counter() - wait_started) * 1_000.0
        raw = pending.slot.readback.read_mapped(copy=True)
        pending.slot.readback.unmap()
        mapped_at = time.perf_counter()
        pipeline_latency_wall_ms = (mapped_at - pending.submitted_at) * 1_000.0
        timing = {
            "uploadWallMs": pending.upload_wall_ms,
            "submitWallMs": pending.submit_wall_ms,
            "readbackWaitWallMs": readback_wait_wall_ms,
            "executeReadbackWallMs": pipeline_latency_wall_ms - pending.upload_wall_ms,
            "pipelineLatencyWallMs": pipeline_latency_wall_ms,
            "totalWallMs": pipeline_latency_wall_ms,
        }
        result = self._result_from_raw(raw, timing=timing, config=pending.config)
        result.timing["totalWallMs"] = (time.perf_counter() - pending.submitted_at) * 1_000.0
        return result

    def register_stream(
        self,
        targets: Iterable[ImageInput],
        *,
        pipeline_depth: int = 2,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> Iterator[RegistrationResult]:
        """Register a sequence with bounded GPU work and readback in flight.

        A depth of two lets the GPU execute the next target while Python
        converts the previous packed readback into result arrays. Results are
        yielded in input order. The extra in-flight frame improves steady-state
        throughput at the cost of approximately one frame of latency.
        """

        if not self._reference_ready:
            raise ICGNError("Call set_reference() before register_stream().")
        try:
            depth = operator.index(pipeline_depth)
        except TypeError as error:
            raise ValueError("pipeline_depth must be a positive integer") from error
        if isinstance(pipeline_depth, bool) or depth < 1:
            raise ValueError("pipeline_depth must be a positive integer")
        self._config = self._resolve_config(iterations, tolerance, zncc_min)
        slots = self._ensure_pipeline_slots(depth)
        for slot in slots:
            self._update_slot_config(slot, self._config)

        available = deque(slots)
        pending: deque[_PendingRegistration] = deque()
        for target in targets:
            if not available:
                completed = pending.popleft()
                result = self._retire_slot(completed)
                available.append(completed.slot)
                yield result
            slot = available.popleft()
            frame = self._prepare_target(target)
            pending.append(self._submit_slot(slot, frame, self._config))

        while pending:
            completed = pending.popleft()
            result = self._retire_slot(completed)
            available.append(completed.slot)
            yield result

    def register_many(
        self,
        targets: Iterable[ImageInput],
        *,
        pipeline_depth: int = 2,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> list[RegistrationResult]:
        """Return all results from :meth:`register_stream` as a list."""

        return list(
            self.register_stream(
                targets,
                pipeline_depth=pipeline_depth,
                iterations=iterations,
                tolerance=tolerance,
                zncc_min=zncc_min,
            )
        )

    def close(self) -> None:
        self._reference_ready = False
        self._destroy_runtime()

    def __enter__(self) -> ICGNWGPU:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
