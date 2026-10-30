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
from .interpolation import INTERPOLATION_PIPELINE_CONSTANTS, normalize_interpolation
from .motion import (
    MOTION_MODEL_PIPELINE_CONSTANTS,
    normalize_motion_model,
    parameter_count,
    state_float_count,
)
from .outlier import OUTLIER_POLICY_PIPELINE_CONSTANTS, normalize_outlier_policy
from .partial import filter_and_support_data, prepare_partial_reference
from .roi import RoiPlan, immutable, prepare_roi, reduce_mask, seed_permissions, validate_roi
from .subset import (
    DEFAULT_SUBSET_SIZE,
    normalize_subset_size,
    normalize_subset_stride,
    subset_grid_shape,
)
from .weighting import (
    SPATIAL_WEIGHTING_PIPELINE_CONSTANTS,
    normalize_gaussian_sigma,
    normalize_spatial_weighting,
)

FIR_RADIUS = 15
#: Runtime FIR half-width after removing numerically negligible outer taps.
FIR_RUNTIME_RADIUS = 12
MAX_ITERATIONS = 30
#: Recommended maximum number of native sequence registrations in flight.
DEFAULT_IN_FLIGHT_FRAMES = 2
#: Default total number of image scales; one keeps the original fast path.
DEFAULT_PYRAMID_LEVELS = 1


def _normalize_device_type(value: str) -> str:
    """Validate device selection before importing or initializing wgpu."""

    if not isinstance(value, str) or value.strip().lower() not in ("auto", "gpu", "cpu"):
        raise ValueError("device_type must be 'auto', 'gpu', or 'cpu'")
    return value.strip().lower()


def _select_adapter(adapters: Iterable[Any], *, device_type: str, power_preference: str) -> Any:
    """Choose an enumerated adapter without silently crossing an explicit type.

    Keep the established Vulkan preference. Within the same backend rank,
    prefer discrete hardware for performance or integrated hardware for power.
    Automatic CPU fallback occurs only when no hardware adapter is enumerated;
    initialization and runtime errors are deliberately not swallowed here.
    """

    adapters = list(adapters)
    hardware = [adapter for adapter in adapters if adapter.info["adapter_type"] != "CPU"]
    software = [adapter for adapter in adapters if adapter.info["adapter_type"] == "CPU"]
    candidates = software if device_type == "cpu" else hardware
    if device_type == "auto" and not candidates:
        candidates = software
    if not candidates:
        if device_type == "cpu":
            raise ICGNError(
                "No CPU software WebGPU adapter is available. Install a compatible "
                "software driver, for example Mesa lavapipe on Linux; wgpu does not "
                "install it automatically."
            )
        kind = "hardware or CPU software" if device_type == "auto" else "hardware GPU"
        raise ICGNError(f"No compatible {kind} WebGPU adapter is available.")
    preferred_type = "IntegratedGPU" if power_preference == "low-power" else "DiscreteGPU"
    return min(
        candidates,
        key=lambda adapter: (
            adapter.info["backend_type"] != "Vulkan",
            adapter.info["adapter_type"] != preferred_type,
        ),
    )


@dataclass(slots=True)
class _RegistrationSlot:
    """Per-frame resources that may be in flight independently."""

    source_texture: Any
    buffers: dict[str, Any]
    bind_group: Any
    readback: Any
    seeded_bind_group: Any = None


@dataclass(slots=True)
class _PendingRegistration:
    """CPU timing and configuration retained until one slot is retired."""

    slot: _RegistrationSlot
    submitted_at: float
    upload_wall_ms: float
    submit_wall_ms: float
    config: dict[str, float | int]
    frame: npt.NDArray[np.uint8]


def _validate_subgroup_probe(raw: Any, *, workgroup_size: int = 16) -> int:
    """Validate subgroup width and vector reduction results from the WGSL probe."""

    values = np.frombuffer(raw, dtype=np.float32)
    expected_values = workgroup_size * 4
    if values.size != expected_values:
        raise ICGNError(
            f"The subgroup probe returned {values.size} floats; expected {expected_values}."
        )
    rows = values.reshape(workgroup_size, 4)
    populated = rows[rows[:, 1] > 0.0]
    if populated.size == 0:
        raise ICGNError("The subgroup probe did not report any active subgroup.")

    widths = np.rint(populated[:, 0]).astype(np.int64)
    subgroup_width = int(widths[0])
    if subgroup_width <= 0 or np.any(widths != subgroup_width):
        raise ICGNError("The subgroup probe reported inconsistent subgroup widths.")

    expected_subgroups = (workgroup_size + subgroup_width - 1) // subgroup_width
    if populated.shape[0] != expected_subgroups:
        raise ICGNError(
            "The subgroup probe reported an unexpected number of active subgroups "
            f"({populated.shape[0]} instead of {expected_subgroups})."
        )

    for subgroup_index, row in enumerate(populated):
        first_lane = subgroup_index * subgroup_width
        active_lanes = min(subgroup_width, workgroup_size - first_lane)
        lane_values = np.arange(
            first_lane + 1,
            first_lane + active_lanes + 1,
            dtype=np.float32,
        )
        expected = np.asarray(
            (
                float(np.sum(lane_values, dtype=np.float32)),
                float(active_lanes),
                float(np.sum(np.square(lane_values), dtype=np.float32)),
            ),
            dtype=np.float32,
        )
        if not np.allclose(row[1:], expected, rtol=1e-6, atol=1e-6):
            raise ICGNError(
                "The subgroup vector reduction returned an unexpected result "
                f"for subgroup {subgroup_index}: {row[1:].tolist()} instead of "
                f"{expected.tolist()}."
            )
    return subgroup_width


def _compare_solver_outputs(
    portable_raw: Any,
    candidate_raw: Any,
    *,
    grid_count: int,
) -> dict[str, bool | float | int | str | None]:
    """Check an optimized solver against the portable result used for tuning.

    The comparison is deliberately limited to the one-off autotuning frame. It
    therefore protects solver selection without adding work to registrations.
    """

    expected_values = grid_count * 8
    portable = np.frombuffer(portable_raw, dtype=np.float32)
    candidate = np.frombuffer(candidate_raw, dtype=np.float32)
    diagnostics: dict[str, bool | float | int | str | None] = {
        "valid": False,
        "reason": "invalid result size",
        "accepted_portable": 0,
        "accepted_candidate": 0,
        "accepted_mismatch": grid_count,
        "parameter_rmse": None,
        "parameter_max_abs": None,
    }
    if portable.size != expected_values or candidate.size != expected_values:
        return diagnostics

    portable = portable.reshape(grid_count, 8)
    candidate = candidate.reshape(grid_count, 8)
    if not np.all(np.isfinite(portable)) or not np.all(np.isfinite(candidate)):
        diagnostics["reason"] = "non-finite solver output"
        return diagnostics

    portable_accepted = portable[:, 6] > 0.5
    candidate_accepted = candidate[:, 6] > 0.5
    accepted_portable = int(np.count_nonzero(portable_accepted))
    accepted_candidate = int(np.count_nonzero(candidate_accepted))
    accepted_mismatch = int(np.count_nonzero(portable_accepted != candidate_accepted))
    diagnostics.update(
        accepted_portable=accepted_portable,
        accepted_candidate=accepted_candidate,
        accepted_mismatch=accepted_mismatch,
    )
    mismatch_limit = max(1, int(np.ceil(grid_count * 0.005)))
    common = portable_accepted & candidate_accepted
    if accepted_portable == 0:
        if accepted_mismatch:
            diagnostics["reason"] = "accepted subset masks disagree"
            return diagnostics
        diagnostics.update(
            valid=True,
            reason="compatible with no accepted subsets",
            parameter_rmse=0.0,
            parameter_max_abs=0.0,
        )
        return diagnostics
    if accepted_mismatch > mismatch_limit:
        diagnostics["reason"] = "accepted subset masks disagree"
        return diagnostics
    common_count = int(np.count_nonzero(common))
    if common_count == 0 or common_count < accepted_portable - mismatch_limit:
        diagnostics["reason"] = "insufficient common accepted subsets"
        return diagnostics

    difference = candidate[common, :6] - portable[common, :6]
    parameter_rmse = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
    parameter_max_abs = float(np.max(np.abs(difference)))
    diagnostics.update(
        parameter_rmse=parameter_rmse,
        parameter_max_abs=parameter_max_abs,
    )
    if parameter_rmse > 2e-3 or parameter_max_abs > 2e-2:
        diagnostics["reason"] = "affine parameters disagree"
        return diagnostics

    diagnostics["valid"] = True
    diagnostics["reason"] = "compatible"
    return diagnostics


def _solver_execution_failure(
    error: Exception,
) -> dict[str, bool | float | int | str | None]:
    """Return an autotuning diagnostic for a candidate-local execution error."""

    return {
        "valid": False,
        "reason": f"execution failed: {type(error).__name__}: {error}",
        "accepted_portable": 0,
        "accepted_candidate": 0,
        "accepted_mismatch": 0,
        "parameter_rmse": None,
        "parameter_max_abs": None,
    }


def _reduce_half(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Apply the SYSCL five-tap reduction and retain every second sample."""

    source = np.asarray(image, dtype=np.float32)
    height, width = source.shape
    reduced_height = height // 2
    reduced_width = width // 2
    taps = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float32) / 16.0
    padded_x = np.pad(source, ((0, 0), (2, 2)), mode="reflect")
    horizontal = sum(
        taps[index] * padded_x[:, index : index + 2 * reduced_width : 2] for index in range(5)
    )
    padded_y = np.pad(horizontal, ((2, 2), (0, 0)), mode="reflect")
    reduced = sum(
        taps[index] * padded_y[index : index + 2 * reduced_height : 2] for index in range(5)
    )
    return np.ascontiguousarray(np.clip(reduced, 0.0, 255.0).astype(np.uint8))


def _transfer_pyramid_seed(
    result: RegistrationResult,
    *,
    source_available: npt.NDArray[np.bool_],
    source_scale: int,
    destination_scale: int,
    destination_grid: tuple[int, int],
    requested: npt.NDArray[np.bool_],
    subset_size: int = DEFAULT_SUBSET_SIZE,
    subset_stride: int | None = None,
    permissions: np.ndarray | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """Interpolate coarse warps onto requested finer subset centres."""

    destination_width, destination_height = destination_grid
    motion_model = str(result.config.get("motionModel", "affine"))
    component_count = parameter_count(motion_model)
    seeds = np.zeros(
        (destination_height, destination_width, state_float_count(motion_model)),
        dtype=np.float32,
    )
    active = requested.astype(np.uint32, copy=True)
    source_height, source_width = result.status.shape
    if source_available.shape != result.status.shape:
        raise ValueError("source_available must match the source result grid")
    if source_width < 2 or source_height < 2:
        return seeds, active
    source_offset = (1.0 - source_scale) * 0.5
    destination_offset = (1.0 - destination_scale) * 0.5
    translation_scale = source_scale / destination_scale
    curvature_scale = destination_scale / source_scale
    destinations = np.argwhere(requested)
    if destinations.size == 0:
        return seeds, active
    destination_y = destinations[:, 0]
    destination_x = destinations[:, 1]
    subset_stride = normalize_subset_stride(subset_stride, subset_size=subset_size)
    center_offset = 0.5 * subset_size
    x = (
        destination_scale * (center_offset + subset_stride * destination_x)
        + destination_offset
        - source_offset
    ) / source_scale
    y = (
        destination_scale * (center_offset + subset_stride * destination_y)
        + destination_offset
        - source_offset
    ) / source_scale
    grid_x = (x - center_offset) / subset_stride
    grid_y = (y - center_offset) / subset_stride
    lower_x = np.clip(np.floor(grid_x), 0, source_width - 2).astype(np.intp)
    lower_y = np.clip(np.floor(grid_y), 0, source_height - 2).astype(np.intp)
    fraction_x = np.clip(grid_x - lower_x, 0.0, 1.0)
    fraction_y = np.clip(grid_y - lower_y, 0.0, 1.0)
    accumulated = np.zeros((destinations.shape[0], component_count), dtype=np.float64)
    weight_sum = np.zeros(destinations.shape[0], dtype=np.float64)
    for corner in range(4):
        source_x = lower_x + (corner & 1)
        source_y = lower_y + (corner >> 1)
        valid = (result.status[source_y, source_x] >= -0.5) & source_available[source_y, source_x]
        if permissions is not None:
            valid &= (permissions[destination_y, destination_x] & (1 << corner)) != 0
        weight_x = fraction_x if corner & 1 else 1.0 - fraction_x
        weight_y = fraction_y if corner & 2 else 1.0 - fraction_y
        weight = weight_x * weight_y * valid
        parameters = result.parameters[source_y, source_x]
        delta_x = x - (center_offset + subset_stride * source_x)
        delta_y = y - (center_offset + subset_stride * source_y)
        translated = np.array(parameters, dtype=np.float64, copy=True)
        if motion_model == "quadratic":
            delta_x2 = delta_x * delta_x
            delta_xy = delta_x * delta_y
            delta_y2 = delta_y * delta_y
            translated[:, 0] += (
                parameters[:, 1] * delta_x
                + parameters[:, 2] * delta_y
                + 0.5 * parameters[:, 6] * delta_x2
                + parameters[:, 7] * delta_xy
                + 0.5 * parameters[:, 8] * delta_y2
            )
            translated[:, 3] += (
                parameters[:, 4] * delta_x
                + parameters[:, 5] * delta_y
                + 0.5 * parameters[:, 9] * delta_x2
                + parameters[:, 10] * delta_xy
                + 0.5 * parameters[:, 11] * delta_y2
            )
            translated[:, 1] += parameters[:, 6] * delta_x + parameters[:, 7] * delta_y
            translated[:, 2] += parameters[:, 7] * delta_x + parameters[:, 8] * delta_y
            translated[:, 4] += parameters[:, 9] * delta_x + parameters[:, 10] * delta_y
            translated[:, 5] += parameters[:, 10] * delta_x + parameters[:, 11] * delta_y
            translated[:, 6:12] *= curvature_scale
        else:
            translated[:, 0] += parameters[:, 1] * delta_x + parameters[:, 2] * delta_y
            translated[:, 3] += parameters[:, 4] * delta_x + parameters[:, 5] * delta_y
        translated[:, (0, 3)] *= translation_scale
        accumulated += weight[:, None] * translated
        weight_sum += weight
    valid_destinations = weight_sum > 0.0
    selected_y = destination_y[valid_destinations]
    selected_x = destination_x[valid_destinations]
    seeds[selected_y, selected_x, :component_count] = (
        accumulated[valid_destinations] / weight_sum[valid_destinations, None]
    )
    return seeds, active


def _pyramid_dependency_mask(
    *,
    source_grid: tuple[int, int],
    source_scale: int,
    destination_scale: int,
    requested: npt.NDArray[np.bool_],
    subset_size: int = DEFAULT_SUBSET_SIZE,
    subset_stride: int | None = None,
) -> npt.NDArray[np.bool_]:
    """Return the four-source interpolation closure for requested subsets."""

    source_width, source_height = source_grid
    mask = np.zeros((source_height, source_width), dtype=np.bool_)
    destinations = np.argwhere(requested)
    if destinations.size == 0 or source_width < 2 or source_height < 2:
        return mask
    destination_y = destinations[:, 0]
    destination_x = destinations[:, 1]
    source_offset = (1.0 - source_scale) * 0.5
    destination_offset = (1.0 - destination_scale) * 0.5
    subset_stride = normalize_subset_stride(subset_stride, subset_size=subset_size)
    center_offset = 0.5 * subset_size
    x = (
        destination_scale * (center_offset + subset_stride * destination_x)
        + destination_offset
        - source_offset
    ) / source_scale
    y = (
        destination_scale * (center_offset + subset_stride * destination_y)
        + destination_offset
        - source_offset
    ) / source_scale
    lower_x = np.clip(
        np.floor((x - center_offset) / subset_stride),
        0,
        source_width - 2,
    ).astype(np.intp)
    lower_y = np.clip(
        np.floor((y - center_offset) / subset_stride),
        0,
        source_height - 2,
    ).astype(np.intp)
    for corner in range(4):
        mask[lower_y + (corner >> 1), lower_x + (corner & 1)] = True
    return mask


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
    *,
    interpolation: str = "bspline",
    motion_model: str = "affine",
    subset_size: int = DEFAULT_SUBSET_SIZE,
    subset_stride: int | None = None,
    spatial_weighting: str = "uniform",
    gaussian_sigma: float | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Build reference pixels, inverse Hessians, and solver metadata."""

    height, width = image.shape
    subset_size = normalize_subset_size(subset_size)
    subset_stride = normalize_subset_stride(subset_stride, subset_size=subset_size)
    half_extent = 0.5 * (subset_size - 1)
    grid_width, grid_height = subset_grid_shape(
        width,
        height,
        subset_size=subset_size,
        subset_stride=subset_stride,
    )
    mode = normalize_interpolation(interpolation)
    model = normalize_motion_model(motion_model)
    weighting = normalize_spatial_weighting(spatial_weighting)
    sigma = normalize_gaussian_sigma(gaussian_sigma, subset_size=subset_size)
    if mode == "bspline":
        intensity = _gaussian_sigma_one(image)
        gradient_source = _cubic_bspline_prefilter(intensity)
    else:
        # Direct interpolants operate on the acquired samples. Their IC-GN
        # reference gradient is the centered derivative at integer pixels.
        intensity = image.astype(np.float64)
        gradient_source = intensity

    padded = np.pad(gradient_source, 1, mode="reflect")
    gradient_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    gradient_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    if mode == "bspline":
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

    local_y, local_x = np.meshgrid(
        np.arange(subset_size, dtype=np.float64) - half_extent,
        np.arange(subset_size, dtype=np.float64) - half_extent,
        indexing="ij",
    )
    local_x = local_x.ravel()
    local_y = local_y.ravel()
    if weighting == "gaussian":
        integration_weights = np.exp(
            -0.5 * (local_x * local_x + local_y * local_y) / (sigma * sigma)
        )
    else:
        integration_weights = np.ones(local_x.size, dtype=np.float64)
    integration_weight_sum = float(integration_weights.sum(dtype=np.float64))
    integration_weights_2d = integration_weights.reshape(subset_size, subset_size)

    reference = np.zeros((height, width, 4), dtype=np.float32)
    independent_stride = subset_stride != subset_size
    if independent_stride:
        reference[..., 0] = intensity
        intensity_windows = np.lib.stride_tricks.sliding_window_view(
            intensity, (subset_size, subset_size)
        )[::subset_stride, ::subset_stride][:grid_height, :grid_width]
        gradient_x_windows = np.lib.stride_tricks.sliding_window_view(
            gradient_x, (subset_size, subset_size)
        )[::subset_stride, ::subset_stride][:grid_height, :grid_width]
        gradient_y_windows = np.lib.stride_tricks.sliding_window_view(
            gradient_y, (subset_size, subset_size)
        )[::subset_stride, ::subset_stride][:grid_height, :grid_width]
    else:
        covered_height = grid_height * subset_size
        covered_width = grid_width * subset_size
        tiled = intensity[:covered_height, :covered_width].reshape(
            grid_height, subset_size, grid_width, subset_size
        )
        means = (
            np.einsum("aybx,yx->ab", tiled, integration_weights_2d, optimize=True)
            / integration_weight_sum
        )
        centered = intensity.copy()
        centered[:covered_height, :covered_width] -= np.repeat(
            np.repeat(means, subset_size, axis=0), subset_size, axis=1
        )
        reference[..., 0] = centered
    reference[..., 1] = gradient_x
    reference[..., 2] = gradient_y

    grid_count = grid_width * grid_height
    component_count = parameter_count(model)
    inverse_stride = component_count * component_count
    metadata_stride = 28 if model == "quadratic" else 16
    inverse = np.zeros((grid_count, inverse_stride), dtype=np.float32)
    metadata = np.zeros((grid_count, metadata_stride), dtype=np.float32)
    for grid_y in range(grid_height):
        if independent_stride:
            raw_tiles = intensity_windows[grid_y].reshape(grid_width, -1)
            row_means = (
                np.einsum("np,p->n", raw_tiles, integration_weights, optimize=True)
                / integration_weight_sum
            )
            f_tiles = raw_tiles - row_means[:, None]
            gx_tiles = gradient_x_windows[grid_y].reshape(grid_width, -1)
            gy_tiles = gradient_y_windows[grid_y].reshape(grid_width, -1)
        else:
            row_slice = slice(grid_y * subset_size, (grid_y + 1) * subset_size)
            f_row = centered[row_slice, :covered_width].reshape(
                subset_size, grid_width, subset_size
            )
            gx_row = gradient_x[row_slice, :covered_width].reshape(
                subset_size, grid_width, subset_size
            )
            gy_row = gradient_y[row_slice, :covered_width].reshape(
                subset_size, grid_width, subset_size
            )
            f_tiles = f_row.transpose(1, 0, 2).reshape(grid_width, -1)
            gx_tiles = gx_row.transpose(1, 0, 2).reshape(grid_width, -1)
            gy_tiles = gy_row.transpose(1, 0, 2).reshape(grid_width, -1)
            row_means = np.zeros(grid_width, dtype=np.float64)
        energy = np.einsum("np,np,p->n", f_tiles, f_tiles, integration_weights, optimize=True)
        row_offset = grid_y * grid_width
        if model == "translation":
            hessian_xx = np.einsum(
                "np,np,p->n", gx_tiles, gx_tiles, integration_weights, optimize=True
            )
            hessian_xy = np.einsum(
                "np,np,p->n", gx_tiles, gy_tiles, integration_weights, optimize=True
            )
            hessian_yy = np.einsum(
                "np,np,p->n", gy_tiles, gy_tiles, integration_weights, optimize=True
            )
            determinant = hessian_xx * hessian_yy - hessian_xy * hessian_xy
            scale = np.maximum(np.abs(hessian_xx * hessian_yy), 1.0)
            valid = (
                np.isfinite(determinant)
                & (np.abs(determinant) > np.finfo(np.float64).eps * scale)
                & (energy > 1e-12)
            )
            inverse_determinant = np.zeros(grid_width, dtype=np.float64)
            inverse_determinant[valid] = 1.0 / determinant[valid]
            row_inverse = np.zeros((grid_width, 6, 6), dtype=np.float64)
            row_inverse[:, 0, 0] = hessian_yy * inverse_determinant
            row_inverse[:, 0, 3] = -hessian_xy * inverse_determinant
            row_inverse[:, 3, 0] = -hessian_xy * inverse_determinant
            row_inverse[:, 3, 3] = hessian_xx * inverse_determinant
            jt_reference = np.zeros((grid_width, 6), dtype=np.float64)
            jt_reference[:, 0] = np.einsum(
                "np,np,p->n", gx_tiles, f_tiles, integration_weights, optimize=True
            )
            jt_reference[:, 3] = np.einsum(
                "np,np,p->n", gy_tiles, f_tiles, integration_weights, optimize=True
            )
            sum_j = np.zeros((grid_width, 6), dtype=np.float64)
            sum_j[:, 0] = np.einsum("np,p->n", gx_tiles, integration_weights, optimize=True)
            sum_j[:, 3] = np.einsum("np,p->n", gy_tiles, integration_weights, optimize=True)
        else:
            jacobian_components = [
                gx_tiles,
                gx_tiles * local_x,
                gx_tiles * local_y,
                gy_tiles,
                gy_tiles * local_x,
                gy_tiles * local_y,
            ]
            if model == "quadratic":
                half_x2 = 0.5 * local_x * local_x
                xy = local_x * local_y
                half_y2 = 0.5 * local_y * local_y
                jacobian_components.extend(
                    (
                        gx_tiles * half_x2,
                        gx_tiles * xy,
                        gx_tiles * half_y2,
                        gy_tiles * half_x2,
                        gy_tiles * xy,
                        gy_tiles * half_y2,
                    )
                )
            jacobian = np.stack(jacobian_components, axis=2)
            hessian = np.einsum(
                "npi,npj,p->nij", jacobian, jacobian, integration_weights, optimize=True
            )
            jt_reference = np.einsum(
                "npi,np,p->ni", jacobian, f_tiles, integration_weights, optimize=True
            )
            sum_j = np.einsum("npi,p->ni", jacobian, integration_weights, optimize=True)
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
        representable = np.max(np.abs(row_inverse), axis=(1, 2)) <= np.finfo(np.float32).max
        valid &= representable
        # Invalid subsets are rejected through metadata. Keep their unused
        # inverse blocks finite as well so conversion and GPU upload never
        # introduce infinities for flat or nearly singular references.
        row_inverse[~valid] = 0.0
        inverse[row_offset : row_offset + grid_width] = row_inverse.reshape(
            grid_width, inverse_stride
        )
        metadata[row_offset : row_offset + grid_width, 0] = np.sqrt(energy)
        metadata[row_offset : row_offset + grid_width, 1] = valid.astype(np.float32)
        metadata[row_offset : row_offset + grid_width, 2 : 2 + component_count] = jt_reference
        sum_offset = 2 + component_count
        metadata[
            row_offset : row_offset + grid_width,
            sum_offset : sum_offset + component_count,
        ] = sum_j
        metadata[row_offset : row_offset + grid_width, sum_offset + component_count] = row_means
        metadata[
            row_offset : row_offset + grid_width,
            sum_offset + component_count + 1,
        ] = integration_weight_sum

    return (
        np.ascontiguousarray(reference.ravel()),
        np.ascontiguousarray(inverse.ravel()),
        np.ascontiguousarray(metadata.ravel()),
    )


class ICGNWGPU:
    """Persistent native WebGPU IC-GN engine with direct NumPy transfers.

    The target prefilter, motion solver, and optional displacement UOD all run
    on the native GPU adapter. By default, reference preparation benchmarks
    the portable and subgroup solvers when both are available. Set
    ``prefer_subgroups`` to ``False`` or ``True`` to force either path for
    qualification. Direct ``linear`` and Keys ``cubic`` interpolation omit
    target FIR passes and use the portable solver; ``bspline`` is the
    historical default. ``subset_size`` also supports centred odd widths 15,
    17, 31, and 33; ``subset_stride`` independently selects the field spacing.
    ``spatial_weighting="gaussian"`` applies centred Gaussian integration with
    ``gaussian_sigma`` expressed in pixels. The subgroup WGSL path requires
    size and stride 16 plus uniform integration.

    ``device_type="auto"`` prefers hardware and uses an enumerated CPU software
    adapter only when no hardware adapter exists. ``"gpu"`` and ``"cpu"``
    require their respective adapter types. CPU execution runs the same WGSL
    and requires a separately installed software driver. Selection happens at
    construction; it does not recover a lost device during registration.
    ``execution_device_type`` and ``adapter_info`` report the actual choice.
    """

    roi: RoiPlan | None = None
    reference_valid: np.ndarray | None = None
    _partial_solver: Any = None
    _partial_seeded_solver: Any = None
    _partial_retry_solver: Any = None

    def __init__(
        self,
        *,
        power_preference: str = "high-performance",
        device_type: str = "auto",
        prefer_subgroups: bool | None = None,
        pyramid_levels: int = DEFAULT_PYRAMID_LEVELS,
        interpolation: str = "bspline",
        motion_model: str = "affine",
        outlier_policy: str = "replace",
        subset_size: int = DEFAULT_SUBSET_SIZE,
        subset_stride: int | None = None,
        spatial_weighting: str = "uniform",
        gaussian_sigma: float | None = None,
    ) -> None:
        self.device_type = _normalize_device_type(device_type)
        if power_preference not in ("high-performance", "low-power"):
            raise ValueError("power_preference must be 'high-performance' or 'low-power'")
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
        subgroup_configuration_supported = (
            self.interpolation == "bspline"
            and self.motion_model == "affine"
            and self.spatial_weighting == "uniform"
            and self.subset_size == DEFAULT_SUBSET_SIZE
            and self.subset_stride == DEFAULT_SUBSET_SIZE
        )
        if prefer_subgroups is True and (
            self.subset_size != DEFAULT_SUBSET_SIZE or self.subset_stride != DEFAULT_SUBSET_SIZE
        ):
            raise ValueError("prefer_subgroups=True requires subset_size=16 and subset_stride=16")
        if prefer_subgroups is True and (
            self.interpolation != "bspline"
            or self.motion_model != "affine"
            or self.spatial_weighting != "uniform"
        ):
            raise ValueError(
                "prefer_subgroups=True requires interpolation='bspline', "
                "motion_model='affine', and spatial_weighting='uniform'"
            )
        try:
            level_count = operator.index(pyramid_levels)
        except TypeError as error:
            raise ValueError("pyramid_levels must be an integer from 1 to 3") from error
        if isinstance(pyramid_levels, bool) or not 1 <= level_count <= 3:
            raise ValueError("pyramid_levels must be an integer from 1 to 3")
        try:
            import wgpu
        except ImportError as error:
            raise ICGNError(
                "The native backend requires the 'wgpu' package. "
                "Reinstall lucas-kanade-icgn-webgl with its declared dependencies."
            ) from error

        adapter = _select_adapter(
            wgpu.gpu.enumerate_adapters_sync(),
            device_type=self.device_type,
            power_preference=power_preference,
        )
        self.adapter_info = dict(adapter.info)
        self.execution_device_type = "cpu" if self.adapter_info["adapter_type"] == "CPU" else "gpu"
        if prefer_subgroups is True and "subgroup" not in adapter.features:
            raise ICGNError(
                "The subgroup solver was requested, but the adapter does not advertise "
                "wgpu-native's required 'subgroup' feature."
            )

        self._wgpu = wgpu
        self.adapter = adapter
        required_features = []
        if (
            subgroup_configuration_supported
            and prefer_subgroups is not False
            and "subgroup" in adapter.features
        ):
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
        self.subgroup_probe_error = ""
        self.subgroup_width = 0
        if "subgroup" in self.device.features:
            try:
                self.subgroup_width = self._measure_subgroup_width()
            except Exception as error:
                self.subgroup_probe_error = f"{type(error).__name__}: {error}"
                if prefer_subgroups is True:
                    raise ICGNError(
                        "The subgroup solver was requested, but its WGSL qualification "
                        f"probe failed: {self.subgroup_probe_error}"
                    ) from error
        self._solver_preference = prefer_subgroups
        self.solver_tuning_ms: dict[str, float] = {}
        self.solver_validation: dict[str, dict[str, bool | float | int | str | None]] = {}
        shader_path = files("lucas_kanade_icgn").joinpath("shaders/icgn.wgsl")
        shader_source = shader_path.read_text(encoding="utf-8")
        shader = self.device.create_shader_module(code=shader_source)
        self._shader = shader
        subgroup_shader = None
        if subgroup_configuration_supported and self.subgroup_width >= DEFAULT_SUBSET_SIZE:
            subgroup_path = files("lucas_kanade_icgn").joinpath("shaders/icgn_subgroup.wgsl")
            try:
                subgroup_shader = self.device.create_shader_module(
                    label="DIC-RT WGSL subgroup compute solver",
                    code=subgroup_path.read_text(encoding="utf-8"),
                )
            except Exception as error:
                self.subgroup_probe_error = (
                    "The subgroup probe passed, but the specialized WGSL module failed: "
                    f"{type(error).__name__}: {error}"
                )
                if prefer_subgroups is True:
                    raise ICGNError(self.subgroup_probe_error) from error
        self._bind_group_layout = self._create_bind_group_layout()
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._pipeline_layout = pipeline_layout
        self._prefilter_x_fast = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={
                "module": shader,
                "entry_point": "prefilter_x",
                "constants": {"PREFILTER_CLAMP_HALO": 0},
            },
        )
        self._prefilter_y_fast = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={
                "module": shader,
                "entry_point": "prefilter_y",
                "constants": {"PREFILTER_CLAMP_HALO": 0},
            },
        )
        self._prefilter_safe_pipelines: dict[str, Any] = {}
        self._prefilter_x = self._prefilter_x_fast
        self._prefilter_y = self._prefilter_y_fast
        self._portable_solver = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={
                "module": shader,
                "entry_point": "solve",
                "constants": {
                    "INTERPOLATION_MODE": INTERPOLATION_PIPELINE_CONSTANTS[self.interpolation],
                    "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
                    "SUBSET_SIZE": self.subset_size,
                    "SUBSET_STRIDE": self.subset_stride,
                    "SPATIAL_WEIGHTING": SPATIAL_WEIGHTING_PIPELINE_CONSTANTS[
                        self.spatial_weighting
                    ],
                    "GAUSSIAN_SIGMA": self.gaussian_sigma,
                },
            },
        )
        self._subgroup_solver = None
        if subgroup_shader is not None:
            try:
                self._subgroup_solver = self.device.create_compute_pipeline(
                    layout=pipeline_layout,
                    compute={"module": subgroup_shader, "entry_point": "main"},
                )
            except Exception as error:
                self.subgroup_probe_error = (
                    "The subgroup probe passed, but the specialized WGSL pipeline failed: "
                    f"{type(error).__name__}: {error}"
                )
                if prefer_subgroups is True:
                    raise ICGNError(self.subgroup_probe_error) from error
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
            compute={
                "module": shader,
                "entry_point": "median_filter",
                "constants": {
                    "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
                    "OUTLIER_POLICY": OUTLIER_POLICY_PIPELINE_CONSTANTS[self.outlier_policy],
                    "SUBSET_SIZE": self.subset_size,
                    "SUBSET_STRIDE": self.subset_stride,
                },
            },
        )
        self._seeded_bind_group_layout: Any = None
        self._seeded_solver: Any = None
        self._retry_solver: Any = None
        self._reference_ready = False
        self.roi = None
        self.reference_valid = None
        self._roi_seed_permissions = None
        self._default_median_filter = self._median_filter
        self._base_portable_solver = self._portable_solver
        self._base_subgroup_solver = self._subgroup_solver
        self._subgroup_shader = subgroup_shader
        self._roi_solver_pipelines: dict[str, Any] = {}
        self._seeded_roi_mode = False
        self._source_texture_format = wgpu.TextureFormat.r8unorm
        self._source_texture_extra_usage = 0
        self._buffers: dict[str, Any] = {}
        self._source_texture: Any = None
        self._bind_group: Any = None
        self._pipeline_slots: list[_RegistrationSlot] = []
        self.pyramid_levels = level_count
        self._pyramid_engines: list[ICGNWGPU] = []
        self.width = 0
        self.height = 0
        self.grid_width = 0
        self.grid_height = 0
        self._config = {"iterations": 30, "tolerance": 1e-3, "znccMin": 0.8}

    def _new_shared_level(self, *, gpu_writable_source: bool = False) -> ICGNWGPU:
        """Create an independent level runtime on this engine's device.

        This private constructor is used by the full-pyramid backend.  It
        shares immutable pipelines and the device, but not any
        dimension-dependent image, reference, solver-state, or readback
        resource.  Consequently all pyramid levels can be encoded into one
        command buffer without creating extra devices or host round trips.
        """

        level = object.__new__(ICGNWGPU)
        for name in (
            "_wgpu",
            "adapter",
            "device",
            "device_name",
            "subgroup_width",
            "_shader",
            "_bind_group_layout",
            "_pipeline_layout",
            "_prefilter_x_fast",
            "_prefilter_y_fast",
            "_prefilter_safe_pipelines",
            "_prefilter_x",
            "_prefilter_y",
            "_portable_solver",
            "_subgroup_solver",
            "_median_filter",
        ):
            setattr(level, name, getattr(self, name))
        level._solver_preference = False
        level.device_type = getattr(self, "device_type", "auto")
        level.execution_device_type = getattr(self, "execution_device_type", "unknown")
        level.adapter_info = dict(getattr(self, "adapter_info", {}))
        level.interpolation = getattr(self, "interpolation", "bspline")
        level.motion_model = getattr(self, "motion_model", "affine")
        level.outlier_policy = getattr(self, "outlier_policy", "replace")
        level.subset_size = self.subset_size
        level.subset_stride = getattr(self, "subset_stride", self.subset_size)
        level.spatial_weighting = getattr(self, "spatial_weighting", "uniform")
        level.gaussian_sigma = getattr(
            self,
            "gaussian_sigma",
            0.25 * self.subset_size,
        )
        level.solver_tuning_ms = {}
        level.solver = "portable-workgroup"
        level._solver = level._portable_solver
        level._seeded_bind_group_layout = self._seeded_bind_group_layout
        level._seeded_solver = self._seeded_solver
        level._retry_solver = self._retry_solver
        level._reference_ready = False
        level.roi = None
        level.reference_valid = None
        level._roi_seed_permissions = None
        level._default_median_filter = getattr(self, "_default_median_filter", self._median_filter)
        level._base_portable_solver = getattr(self, "_base_portable_solver", self._portable_solver)
        level._base_subgroup_solver = getattr(self, "_base_subgroup_solver", self._subgroup_solver)
        level._subgroup_shader = getattr(self, "_subgroup_shader", None)
        level._roi_solver_pipelines = getattr(self, "_roi_solver_pipelines", {})
        level._seeded_roi_mode = getattr(self, "_seeded_roi_mode", False)
        level._source_texture_format = (
            self._wgpu.TextureFormat.rgba8unorm
            if gpu_writable_source
            else self._wgpu.TextureFormat.r8unorm
        )
        level._source_texture_extra_usage = (
            self._wgpu.TextureUsage.STORAGE_BINDING if gpu_writable_source else 0
        )
        level._buffers = {}
        level._source_texture = None
        level._bind_group = None
        level._pipeline_slots = []
        level.pyramid_levels = 1
        level._pyramid_engines = []
        level.width = 0
        level.height = 0
        level.grid_width = 0
        level.grid_height = 0
        level._config = {"iterations": 30, "tolerance": 1e-3, "znccMin": 0.8}
        return level

    def _prefilter_pipeline(self, axis: str, *, safe_halo: bool) -> Any:
        """Return a dimension-specialized FIR pipeline, compiling safety lazily."""

        if not safe_halo:
            return self._prefilter_x_fast if axis == "x" else self._prefilter_y_fast
        pipeline = self._prefilter_safe_pipelines.get(axis)
        if pipeline is None:
            pipeline = self.device.create_compute_pipeline(
                layout=self._pipeline_layout,
                compute={
                    "module": self._shader,
                    "entry_point": f"prefilter_{axis}",
                    "constants": {"PREFILTER_CLAMP_HALO": 1},
                },
            )
            self._prefilter_safe_pipelines[axis] = pipeline
        return pipeline

    def _measure_subgroup_width(self) -> int:
        """Compile and execute the native WGSL subgroup qualification probe."""

        wgpu = self._wgpu
        probe_path = files("lucas_kanade_icgn").joinpath("shaders/subgroup_probe.wgsl")
        shader = self.device.create_shader_module(
            label="DIC-RT WGSL subgroup qualification probe",
            code=probe_path.read_text(encoding="utf-8"),
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
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
        pipeline = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "main"},
        )
        output_size = 16 * 16
        output = self.device.create_buffer(
            size=output_size,
            usage=(
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
            ),
        )
        bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": output, "offset": 0, "size": output_size},
                }
            ],
        )
        encoder = self.device.create_command_encoder(label="DIC-RT subgroup probe")
        encoder.clear_buffer(output)
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1, 1, 1)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        raw = self.device.queue.read_buffer(output)
        return _validate_subgroup_probe(raw)

    def _create_bind_group_layout(self, *, seeded: bool = False) -> Any:
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
        if seeded:
            for binding in (10, 11):
                entries.append(
                    {
                        "binding": binding,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                    }
                )
        return self.device.create_bind_group_layout(entries=entries)

    def _ensure_seeded_solver(self) -> None:
        """Create fresh seeded and preserving retry pipelines on one layout."""

        roi_mode = self.roi is not None and self.roi.effective
        if (
            self._seeded_solver is not None
            and self._retry_solver is not None
            and roi_mode == self._seeded_roi_mode
        ):
            self._ensure_partial_seeded_pipelines()
            return
        self._seeded_bind_group_layout = self._create_bind_group_layout(seeded=True)
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._seeded_bind_group_layout]
        )
        interpolation = getattr(self, "interpolation", "bspline")
        motion_model = getattr(self, "motion_model", "affine")
        constants = {
            "ROI_ENABLED": int(roi_mode),
            "INTERPOLATION_MODE": INTERPOLATION_PIPELINE_CONSTANTS[interpolation],
            "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[motion_model],
            "SUBSET_SIZE": self.subset_size,
            "SUBSET_STRIDE": getattr(self, "subset_stride", self.subset_size),
            "SPATIAL_WEIGHTING": SPATIAL_WEIGHTING_PIPELINE_CONSTANTS[
                getattr(self, "spatial_weighting", "uniform")
            ],
            "GAUSSIAN_SIGMA": getattr(
                self,
                "gaussian_sigma",
                0.25 * self.subset_size,
            ),
        }
        seeded_solver = self.device.create_compute_pipeline(
            layout=layout,
            compute={
                "module": self._shader,
                "entry_point": "solve_seeded",
                "constants": constants,
            },
        )
        retry_solver = self.device.create_compute_pipeline(
            layout=layout,
            compute={
                "module": self._shader,
                "entry_point": "retry_seeded",
                "constants": constants,
            },
        )
        self._seeded_solver = seeded_solver
        self._retry_solver = retry_solver
        self._seeded_roi_mode = roi_mode
        self._ensure_partial_seeded_pipelines()

    def _partial_constants(self) -> dict:
        return {
            "ROI_ENABLED": 1,
            "PARTIAL_SUPPORT": 1,
            "INTERPOLATION_MODE": INTERPOLATION_PIPELINE_CONSTANTS[self.interpolation],
            "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
            "SUBSET_SIZE": self.subset_size,
            "SUBSET_STRIDE": self.subset_stride,
            "SPATIAL_WEIGHTING": SPATIAL_WEIGHTING_PIPELINE_CONSTANTS[self.spatial_weighting],
            "GAUSSIAN_SIGMA": self.gaussian_sigma,
        }

    def _ensure_partial_seeded_pipelines(self) -> None:
        if self._partial_solver is None:
            return
        if self._partial_seeded_solver is not None and self._partial_retry_solver is not None:
            return
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._seeded_bind_group_layout]
        )
        for entry, attribute in (
            ("solve_partial_seeded", "_partial_seeded_solver"),
            ("retry_partial_seeded", "_partial_retry_solver"),
        ):
            if entry not in self._roi_solver_pipelines:
                self._roi_solver_pipelines[entry] = self.device.create_compute_pipeline(
                    layout=layout,
                    compute={
                        "module": self._shader,
                        "entry_point": entry,
                        "constants": self._partial_constants(),
                    },
                )
            setattr(self, attribute, self._roi_solver_pipelines[entry])

    def _select_roi_solver_pipelines(self) -> None:
        """Keep unmasked solver arithmetic intact through a pipeline constant."""
        self._portable_solver = self._base_portable_solver
        self._subgroup_solver = self._base_subgroup_solver
        self._partial_solver = self._partial_seeded_solver = self._partial_retry_solver = None
        if self.roi is not None and self.roi.effective:
            constants = {
                "ROI_ENABLED": 1,
                "INTERPOLATION_MODE": INTERPOLATION_PIPELINE_CONSTANTS[self.interpolation],
                "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
                "SUBSET_SIZE": self.subset_size,
                "SUBSET_STRIDE": self.subset_stride,
                "SPATIAL_WEIGHTING": SPATIAL_WEIGHTING_PIPELINE_CONSTANTS[self.spatial_weighting],
                "GAUSSIAN_SIGMA": self.gaussian_sigma,
            }
            if "portable" not in self._roi_solver_pipelines:
                self._roi_solver_pipelines["portable"] = self.device.create_compute_pipeline(
                    layout=self._pipeline_layout,
                    compute={
                        "module": self._shader,
                        "entry_point": "solve",
                        "constants": constants,
                    },
                )
            self._portable_solver = self._roi_solver_pipelines["portable"]
            if self._base_subgroup_solver is not None:
                if "subgroup" not in self._roi_solver_pipelines:
                    self._roi_solver_pipelines["subgroup"] = self.device.create_compute_pipeline(
                        layout=self._pipeline_layout,
                        compute={
                            "module": self._subgroup_shader,
                            "entry_point": "main",
                            "constants": {"ROI_ENABLED": 1},
                        },
                    )
                self._subgroup_solver = self._roi_solver_pipelines["subgroup"]
        self._solver = (
            self._subgroup_solver if self._solver_preference is True else self._portable_solver
        )
        self.solver = "subgroup" if self._solver_preference is True else "portable-workgroup"
        if (
            self.roi is not None
            and self.roi.partial_indices is not None
            and self.roi.partial_indices.size
        ):
            if "solve_partial" not in self._roi_solver_pipelines:
                self._roi_solver_pipelines["solve_partial"] = self.device.create_compute_pipeline(
                    layout=self._pipeline_layout,
                    compute={
                        "module": self._shader,
                        "entry_point": "solve_partial",
                        "constants": self._partial_constants(),
                    },
                )
            self._partial_solver = self._roi_solver_pipelines["solve_partial"]
            count = int(self.roi.partial_indices.size)
            self._partial_dispatch = (min(count, 65535), (count + 65534) // 65535, 1)
            # Build every partial entry point before accepting any targets.
            self._ensure_seeded_solver()

    def _storage_buffer(self, data: np.ndarray, *, output: bool = False) -> Any:
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
        self._prefilter_x = self._prefilter_pipeline("x", safe_halo=self.width < 44)
        self._prefilter_y = self._prefilter_pipeline("y", safe_halo=self.height < 44)
        shared_buffers = {
            "reference": self._storage_buffer(reference),
            "inverse": self._storage_buffer(inverse),
            "metadata": self._storage_buffer(metadata),
            "fir": self._storage_buffer(filter_and_support_data(_fir_weights(), self.roi)),
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
        state_stride = state_float_count(self.motion_model)
        source_texture = self.device.create_texture(
            size=(self.width, self.height, 1),
            usage=(
                wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.TEXTURE_BINDING
                | self._source_texture_extra_usage
            ),
            dimension=wgpu.TextureDimension.d2,
            format=self._source_texture_format,
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
                # COPY_SRC lets the opt-in GPU suite validate the tiled FIR
                # coefficients directly without changing the production pass.
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            ),
            "state": self.device.create_buffer(
                size=grid_count * state_stride * 4,
                usage=(
                    wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
                ),
            ),
            "filtered_state": self.device.create_buffer(
                size=grid_count * state_stride * 4,
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
            size=grid_count * state_stride * 4,
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

    def _ensure_seed_resources(self, slot: _RegistrationSlot) -> None:
        """Lazily attach seed and activity buffers to one retry slot."""

        if slot.seeded_bind_group is not None:
            return
        self._ensure_seeded_solver()
        wgpu = self._wgpu
        grid_count = self.grid_width * self.grid_height
        state_stride = state_float_count(self.motion_model)
        slot.buffers["seed_state"] = self.device.create_buffer(
            size=grid_count * state_stride * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        slot.buffers["active_mask"] = self.device.create_buffer(
            size=grid_count * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        resources = [
            {"binding": 0, "resource": slot.source_texture.create_view()},
            *(
                self._buffer_resource(binding, slot.buffers[name])
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
                        "seed_state",
                        "active_mask",
                    ),
                    start=1,
                )
            ),
        ]
        slot.seeded_bind_group = self.device.create_bind_group(
            layout=self._seeded_bind_group_layout,
            entries=resources,
        )

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

        dynamic_names = (
            "config",
            "temporary",
            "coefficients",
            "state",
            "filtered_state",
            "seed_state",
            "active_mask",
        )
        for slot in self._pipeline_slots:
            slot.source_texture.destroy()
            slot.readback.destroy()
            for name in dynamic_names:
                buffer = slot.buffers.get(name)
                if buffer is not None:
                    buffer.destroy()
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
        self._encode_registration_passes(
            encoder,
            slot,
            solver_pipeline,
            solver_bind_group=slot.bind_group,
        )
        if copy_to_readback:
            encoder.copy_buffer_to_buffer(
                self._output_buffer(slot),
                0,
                slot.readback,
                0,
                slot.readback.size,
            )
        return encoder.finish()

    def _encode_registration_passes(
        self,
        encoder: Any,
        slot: _RegistrationSlot,
        solver_pipeline: Any,
        *,
        solver_bind_group: Any,
    ) -> None:
        """Append one level's prefilter, solve, and optional UOD pass."""

        encoder.clear_buffer(slot.buffers["state"])
        if self.outlier_policy != "disabled":
            encoder.clear_buffer(slot.buffers["filtered_state"])
        passes = []
        if self.interpolation == "bspline":
            passes.extend(
                (
                    (
                        self._prefilter_x,
                        ((self.width + 31) // 32, (self.height + 7) // 8, 1),
                        slot.bind_group,
                    ),
                    (
                        self._prefilter_y,
                        ((self.width + 7) // 8, (self.height + 31) // 32, 1),
                        slot.bind_group,
                    ),
                )
            )
        passes.append(
            (
                solver_pipeline,
                (self.grid_width, self.grid_height, 1),
                solver_bind_group,
            )
        )
        if self._partial_solver is not None:
            partial = (
                self._partial_solver
                if solver_bind_group is slot.bind_group
                else self._partial_seeded_solver
            )
            passes.append((partial, self._partial_dispatch, solver_bind_group))
        if self.outlier_policy != "disabled":
            passes.append(
                (
                    self._median_filter,
                    ((self.grid_width * self.grid_height + 255) // 256, 1, 1),
                    slot.bind_group,
                )
            )
        for pipeline, workgroups, bind_group in passes:
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(*workgroups)
            compute_pass.end()

    def _execute_registration(self, solver_pipeline: Any) -> memoryview:
        """Execute the primary slot and return its synchronized state readback."""

        slot = self._pipeline_slots[0]
        command = self._encode_registration(slot, solver_pipeline, copy_to_readback=False)
        self.device.queue.submit([command])
        return self.device.queue.read_buffer(self._output_buffer(slot))

    def _output_buffer(self, slot: _RegistrationSlot) -> Any:
        """Return the raw or UOD-processed state selected by the policy."""

        name = "state" if self.outlier_policy == "disabled" else "filtered_state"
        return slot.buffers[name]

    def _retry_seeded(
        self,
        slot: _RegistrationSlot,
        seeds: npt.NDArray[np.float32],
        active: npt.NDArray[np.uint32],
        previous: RegistrationResult,
    ) -> RegistrationResult:
        """Retry selected subsets from transferred seeds, then apply optional UOD."""

        if self.roi is not None:
            active = active * self.roi.selected
        if not np.any(active):
            return previous
        started = time.perf_counter()
        self._ensure_seed_resources(slot)
        self.device.queue.write_buffer(
            slot.buffers["seed_state"],
            0,
            np.ascontiguousarray(seeds, dtype=np.float32),
        )
        self.device.queue.write_buffer(
            slot.buffers["active_mask"],
            0,
            np.ascontiguousarray(active, dtype=np.uint32),
        )
        encoder = self.device.create_command_encoder(label="DIC-RT seeded pyramid retry")
        if self.outlier_policy != "disabled":
            encoder.clear_buffer(slot.buffers["filtered_state"])
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._retry_solver)
        compute_pass.set_bind_group(0, slot.seeded_bind_group)
        compute_pass.dispatch_workgroups(self.grid_width, self.grid_height, 1)
        compute_pass.end()
        if self._partial_solver is not None:
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(self._partial_retry_solver)
            compute_pass.set_bind_group(0, slot.seeded_bind_group)
            compute_pass.dispatch_workgroups(*self._partial_dispatch)
            compute_pass.end()
        if self.outlier_policy != "disabled":
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(self._median_filter)
            compute_pass.set_bind_group(0, slot.bind_group)
            compute_pass.dispatch_workgroups(
                (self.grid_width * self.grid_height + 255) // 256,
                1,
                1,
            )
            compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        raw = self.device.queue.read_buffer(self._output_buffer(slot))
        timing = dict(previous.timing)
        result = self._result_from_raw(raw, timing=timing, config=self._config)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        timing["seededRetryWallMs"] = elapsed_ms
        timing["totalWallMs"] = timing.get("totalWallMs", 0.0) + elapsed_ms
        return result

    def _register_seeded_frame(
        self,
        frame: npt.NDArray[np.uint8],
        seeds: npt.NDArray[np.float32],
        active: npt.NDArray[np.uint32],
    ) -> RegistrationResult:
        """Prefilter one coarse frame and solve only its active dependency mask."""

        if self.roi is not None:
            active = active * self.roi.selected
        slot = self._pipeline_slots[0]
        self._ensure_seed_resources(slot)
        total_started = time.perf_counter()
        upload_wall_ms = self._upload_frame(slot, frame)
        self.device.queue.write_buffer(
            slot.buffers["seed_state"],
            0,
            np.ascontiguousarray(seeds, dtype=np.float32),
        )
        self.device.queue.write_buffer(
            slot.buffers["active_mask"],
            0,
            np.ascontiguousarray(active, dtype=np.uint32),
        )
        execution_started = time.perf_counter()
        encoder = self.device.create_command_encoder(label="DIC-RT selective coarse solve")
        encoder.clear_buffer(slot.buffers["state"])
        if self.outlier_policy != "disabled":
            encoder.clear_buffer(slot.buffers["filtered_state"])
        passes = []
        if self.interpolation == "bspline":
            passes.extend(
                (
                    (
                        self._prefilter_x,
                        ((self.width + 31) // 32, (self.height + 7) // 8, 1),
                        slot.bind_group,
                    ),
                    (
                        self._prefilter_y,
                        ((self.width + 7) // 8, (self.height + 31) // 32, 1),
                        slot.bind_group,
                    ),
                )
            )
        passes.append(
            (
                self._seeded_solver,
                (self.grid_width, self.grid_height, 1),
                slot.seeded_bind_group,
            )
        )
        if self._partial_solver is not None:
            passes.append(
                (self._partial_seeded_solver, self._partial_dispatch, slot.seeded_bind_group)
            )
        if self.outlier_policy != "disabled":
            passes.append(
                (
                    self._median_filter,
                    ((self.grid_width * self.grid_height + 255) // 256, 1, 1),
                    slot.bind_group,
                )
            )
        for pipeline, workgroups, bind_group in passes:
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(*workgroups)
            compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        raw = self.device.queue.read_buffer(self._output_buffer(slot))
        execution_ms = (time.perf_counter() - execution_started) * 1_000.0
        result = self._result_from_raw(
            raw,
            timing={
                "uploadWallMs": upload_wall_ms,
                "executeReadbackWallMs": execution_ms,
            },
            config=self._config,
        )
        result.timing = {
            **result.timing,
            "totalWallMs": (time.perf_counter() - total_started) * 1_000.0,
        }
        return result

    def _recover_pyramid(
        self,
        slot: _RegistrationSlot,
        frame: npt.NDArray[np.uint8],
        fine_result: RegistrationResult,
    ) -> RegistrationResult:
        """Recover only failed fine subsets through coarse-to-fine seeds."""

        # The outer subset ring cannot satisfy the four-tap interpolation
        # support even at zero motion, so it must not trigger recovery.
        recoverable = np.zeros_like(fine_result.status, dtype=np.bool_)
        if self.grid_width > 2 and self.grid_height > 2:
            recoverable[1:-1, 1:-1] = True
        failed = (fine_result.status < 0.5) & recoverable
        if self.roi is not None:
            failed &= self.roi.selected
        failed_count = int(np.count_nonzero(failed))
        if not self._pyramid_engines or failed_count == 0:
            return fine_result
        started = time.perf_counter()
        reduced_targets: list[npt.NDArray[np.uint8]] = []
        reduced = frame
        for _ in self._pyramid_engines:
            reduced = _reduce_half(reduced)
            reduced_targets.append(reduced)

        requested_by_level: list[npt.NDArray[np.bool_]] = []
        requested = failed
        destination_scale = 1
        for level_index, engine in enumerate(self._pyramid_engines):
            level_scale = 1 << (level_index + 1)
            requested = _pyramid_dependency_mask(
                source_grid=(engine.grid_width, engine.grid_height),
                source_scale=level_scale,
                destination_scale=destination_scale,
                requested=requested,
                subset_size=self.subset_size,
                subset_stride=self.subset_stride,
            )
            requested_by_level.append(requested)
            if engine.roi is not None:
                requested &= engine.roi.selected
            destination_scale = level_scale

        transferred_result: RegistrationResult | None = None
        transferred_scale = 0
        transferred_available: npt.NDArray[np.bool_] | None = None
        for level_index in range(len(self._pyramid_engines) - 1, -1, -1):
            engine = self._pyramid_engines[level_index]
            level_scale = 1 << (level_index + 1)
            level_requested = requested_by_level[level_index]
            if transferred_result is None:
                seeds = np.zeros(
                    (
                        engine.grid_height,
                        engine.grid_width,
                        state_float_count(self.motion_model),
                    ),
                    dtype=np.float32,
                )
                active = level_requested.astype(np.uint32)
            else:
                if transferred_available is None:
                    raise RuntimeError("Missing source availability for pyramid transfer")
                seeds, active = _transfer_pyramid_seed(
                    transferred_result,
                    source_available=transferred_available,
                    source_scale=transferred_scale,
                    destination_scale=level_scale,
                    destination_grid=(engine.grid_width, engine.grid_height),
                    requested=level_requested,
                    subset_size=self.subset_size,
                    subset_stride=self.subset_stride,
                    permissions=engine._roi_seed_permissions,
                )
            level_result = engine._register_seeded_frame(
                reduced_targets[level_index],
                seeds,
                active,
            )
            transferred_result = level_result
            transferred_scale = level_scale
            transferred_available = level_requested

        if transferred_result is None or transferred_available is None:
            return fine_result
        seeds, active = _transfer_pyramid_seed(
            transferred_result,
            source_available=transferred_available,
            source_scale=transferred_scale,
            destination_scale=1,
            destination_grid=(self.grid_width, self.grid_height),
            requested=failed,
            subset_size=self.subset_size,
            subset_stride=self.subset_stride,
            permissions=self._roi_seed_permissions,
        )
        recovered = self._retry_seeded(slot, seeds, active, fine_result)
        recovered_count = int(np.count_nonzero(failed & (recovered.status >= 0.5)))
        preserved = fine_result.status >= 0.5
        parameters = np.array(recovered.parameters, copy=True)
        status = np.array(recovered.status, copy=True)
        iteration_field = np.array(recovered.iterations, copy=True)
        parameters[preserved] = fine_result.parameters[preserved]
        status[preserved] = fine_result.status[preserved]
        iteration_field[preserved] = fine_result.iterations[preserved]
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
        timing = dict(recovered.timing)
        timing.update(
            {
                "pyramidFailedBefore": float(failed_count),
                "pyramidRecovered": float(recovered_count),
                "pyramidLevels": float(self.pyramid_levels),
            }
        )
        result = RegistrationResult(
            parameters=parameters,
            status=status,
            iterations=iteration_field,
            accepted=accepted,
            replaced=int(np.count_nonzero(status > 1.5)),
            mean_translation=mean_translation,
            mean_iterations=mean_iterations,
            timing=timing,
            roi=self.roi,
            reference_valid=self.reference_valid,
            config={
                **recovered.config,
                "interpolation": getattr(self, "interpolation", "bspline"),
                "motionModel": getattr(self, "motion_model", "affine"),
                "outlierPolicy": getattr(self, "outlier_policy", "replace"),
                "subsetStride": self.subset_stride,
                "spatialWeighting": getattr(self, "spatial_weighting", "uniform"),
                "gaussianSigma": getattr(
                    self,
                    "gaussian_sigma",
                    0.25 * self.subset_size,
                ),
            },
            width=self.width,
            height=self.height,
            subset_size=self.subset_size,
            subset_stride=self.subset_stride,
        )
        pyramid_wall_ms = (time.perf_counter() - started) * 1_000.0
        timing["pyramidWallMs"] = pyramid_wall_ms
        timing["totalWallMs"] = fine_result.timing["totalWallMs"] + pyramid_wall_ms
        return result

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

        if self._solver_preference is False or self._subgroup_solver is None:
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
        portable_raw = self._execute_registration(self._portable_solver)
        self.solver_validation = {
            "portable-workgroup": _compare_solver_outputs(
                portable_raw,
                portable_raw,
                grid_count=self.grid_width * self.grid_height,
            )
        }
        try:
            subgroup_raw = self._execute_registration(self._subgroup_solver)
        except Exception as error:
            diagnostics = _solver_execution_failure(error)
            self.solver_validation["subgroup"] = diagnostics
            self.subgroup_probe_error = str(diagnostics["reason"])
            if self._solver_preference is True:
                raise ICGNError(
                    "The forced subgroup solver failed during its validation execution: "
                    f"{diagnostics['reason']}."
                ) from error
            self.solver = "portable-workgroup"
            self._solver = self._portable_solver
            return
        self.solver_validation["subgroup"] = _compare_solver_outputs(
            portable_raw,
            subgroup_raw,
            grid_count=self.grid_width * self.grid_height,
        )
        candidates = {
            name: pipeline
            for name, pipeline in candidates.items()
            if self.solver_validation[name]["valid"]
        }
        if "portable-workgroup" not in candidates:
            raise ICGNError("The portable solver failed its autotuning validation frame.")
        if self._solver_preference is True:
            if "subgroup" not in candidates:
                diagnostics = self.solver_validation["subgroup"]
                raise ICGNError(
                    "The forced subgroup solver failed comparison with the portable "
                    f"solver: {diagnostics['reason']}."
                )
            self.solver = "subgroup"
            self._solver = self._subgroup_solver
            return
        samples: dict[str, list[float]] = {name: [] for name in candidates}
        names = tuple(candidates)
        for repetition in range(3):
            order = names if repetition % 2 == 0 else tuple(reversed(names))
            for name in order:
                started = time.perf_counter()
                try:
                    self._execute_registration(candidates[name])
                except Exception as error:
                    if name == "portable-workgroup":
                        raise
                    diagnostics = _solver_execution_failure(error)
                    self.solver_validation["subgroup"] = diagnostics
                    self.subgroup_probe_error = str(diagnostics["reason"])
                    self.solver_tuning_ms = {
                        "portable-workgroup": float(median(samples["portable-workgroup"]))
                    }
                    self.solver = "portable-workgroup"
                    self._solver = self._portable_solver
                    return
                samples[name].append((time.perf_counter() - started) * 1_000.0)
        self.solver_tuning_ms = {name: float(median(timings)) for name, timings in samples.items()}
        self.solver = min(self.solver_tuning_ms, key=self.solver_tuning_ms.__getitem__)
        self._solver = candidates[self.solver]

    def _close_pyramid_engines(self) -> None:
        for engine in self._pyramid_engines:
            engine.close()
        self._pyramid_engines.clear()

    def _prepare_pyramid(
        self,
        reference: npt.NDArray[np.uint8],
        *,
        zncc_min: float,
    ) -> None:
        """Prepare optional coarse references outside the per-frame path."""

        self._close_pyramid_engines()
        reduced = reference
        mask = None if self.roi is None else self.roi.mask
        for _ in range(1, self.pyramid_levels):
            reduced = _reduce_half(reduced)
            mask = None if mask is None else reduce_mask(mask)
            engine = ICGNWGPU(
                device_type=self.device_type,
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
            engine.set_reference(
                reduced,
                iterations=12,
                tolerance=1e-2,
                zncc_min=zncc_min,
                roi_mask=mask,
                roi_policy="full" if self.roi is None else self.roi.policy,
                roi_truncation="none" if self.roi is None else self.roi.truncation,
            )
            self._pyramid_engines.append(engine)
        levels = [self, *self._pyramid_engines]
        self._roi_level_counts = [
            int(level.roi.selected.sum())
            if level.roi is not None
            else level.grid_width * level.grid_height
            for level in levels
        ]
        if self.roi is not None and self.roi.effective:
            for index, (destination, source) in enumerate(
                zip(levels[:-1], levels[1:], strict=True)
            ):
                destination._roi_seed_permissions = seed_permissions(
                    source.roi,
                    destination.roi,
                    source_scale=1 << (index + 1),
                    destination_scale=1 << index,
                    size=self.subset_size,
                    stride=self.subset_stride,
                    reference_mask=self.roi.mask,
                )

    def set_reference(
        self,
        image: ImageInput,
        *,
        iterations: int = 30,
        tolerance: float = 1e-3,
        zncc_min: float = 0.8,
        roi_mask: np.ndarray | None = None,
        roi_policy: str = "full",
        roi_truncation: str = "none",
    ) -> ReferenceInfo:
        total_start = time.perf_counter()
        frame = _as_uint8_gray(image)
        height, width = frame.shape
        subset_stride = getattr(self, "subset_stride", self.subset_size)
        grid_width, grid_height = subset_grid_shape(
            width,
            height,
            subset_size=self.subset_size,
            subset_stride=subset_stride,
        )
        if grid_width < 1 or grid_height < 1:
            raise ValueError(
                f"Reference dimensions must each be at least {self.subset_size} pixels."
            )
        config = _validated_config(iterations, tolerance, zncc_min)
        mask = validate_roi(roi_mask, roi_policy, frame.shape, self.motion_model, roi_truncation)
        roi = prepare_roi(
            mask, self.subset_size, subset_stride, policy=roi_policy, truncation=roi_truncation
        )
        reference_start = time.perf_counter()
        reference, inverse, metadata = _reference_arrays(
            frame,
            interpolation=self.interpolation,
            motion_model=self.motion_model,
            subset_size=self.subset_size,
            subset_stride=subset_stride,
            spatial_weighting=self.spatial_weighting,
            gaussian_sigma=self.gaussian_sigma,
        )
        reference_wall_ms = (time.perf_counter() - reference_start) * 1_000.0
        if roi is not None and roi.partial_indices is not None and roi.partial_indices.size:
            roi = prepare_partial_reference(
                reference,
                inverse,
                metadata,
                roi,
                width=width,
                height=height,
                size=self.subset_size,
                stride=subset_stride,
                motion_model=self.motion_model,
                weighting=self.spatial_weighting,
                sigma=self.gaussian_sigma,
            )
            reference_wall_ms = (time.perf_counter() - reference_start) * 1_000.0
        metadata_rows = metadata.reshape(-1, 28 if self.motion_model == "quadratic" else 16)
        reference_valid = immutable(
            ((metadata_rows[:, 1] > 0.5) | (metadata_rows[:, 1] == -4)).reshape(
                grid_height, grid_width
            )
        )
        if roi is not None and roi.effective:
            metadata_rows[~roi.selected.ravel(), 1] = -3.0
            # Native-only tail: one vec4 per subset. Shared WGSL consumers
            # leave ROI_ENABLED=0 and retain the historical buffer layout.
            links = np.zeros((roi.selected.size, 4), dtype=np.float32)
            links[:, 0] = roi.neighbour_bits.ravel()
            metadata = np.concatenate((metadata, links.ravel()))
            if roi.partial_indices is not None and roi.partial_indices.size:
                boxes = np.zeros((roi.selected.size, 4), np.float32)
                boxes[roi.partial_indices] = roi.support_boxes
                metadata = np.concatenate((metadata, boxes.ravel()))

        # All CPU validation is complete. From this point onward a failure may
        # have replaced GPU resources, so the previous reference must no
        # longer be advertised as usable.
        self._reference_ready = False
        self._close_pyramid_engines()
        self.height = height
        self.width = width
        self.grid_width = grid_width
        self.grid_height = grid_height
        self._config = config
        self.roi = roi
        self.reference_valid = reference_valid
        self._pyramid_config = {
            "pyramidLevels": self.pyramid_levels,
            "pyramidStrategy": "selective" if self.pyramid_levels > 1 else "single",
        }
        self._roi_seed_permissions = None
        runtime_start = time.perf_counter()
        self._select_roi_solver_pipelines()
        self._median_filter = self._default_median_filter
        if roi is not None and roi.effective and self.outlier_policy != "disabled":
            self._median_filter = self.device.create_compute_pipeline(
                layout=self._pipeline_layout,
                compute={
                    "module": self._shader,
                    "entry_point": "median_filter",
                    "constants": {
                        "MOTION_MODEL": MOTION_MODEL_PIPELINE_CONSTANTS[self.motion_model],
                        "OUTLIER_POLICY": OUTLIER_POLICY_PIPELINE_CONSTANTS[self.outlier_policy],
                        "SUBSET_SIZE": self.subset_size,
                        "SUBSET_STRIDE": subset_stride,
                        "ROI_ENABLED": 1,
                    },
                },
            )
        self._allocate_runtime(reference, inverse, metadata)
        self._update_config()
        admissible = reference_valid if roi is None else reference_valid & roi.selected
        # A geometric selection without texture is a valid empty measurement,
        # not evidence that either solver failed its numerical qualification.
        if self.interpolation == "bspline" and admissible.any():
            self._autotune_solver(frame)
        self._prepare_pyramid(frame, zncc_min=zncc_min)
        runtime_wall_ms = (time.perf_counter() - runtime_start) * 1_000.0
        self._reference_ready = True
        total_wall_ms = (time.perf_counter() - total_start) * 1_000.0
        return ReferenceInfo(
            preparation_wall_ms=reference_wall_ms + runtime_wall_ms,
            runtime_build_wall_ms=runtime_wall_ms,
            reference_build_wall_ms=reference_wall_ms,
            encoding_wall_ms=0.0,
            total_wall_ms=total_wall_ms,
            config={
                **self._config,
                **(
                    {"roi": {**roi.summary(), "selectedPerLevel": self._roi_level_counts}}
                    if roi is not None
                    else {}
                ),
                "interpolation": getattr(self, "interpolation", "bspline"),
                "motionModel": getattr(self, "motion_model", "affine"),
                "outlierPolicy": getattr(self, "outlier_policy", "replace"),
                "subsetStride": subset_stride,
                "spatialWeighting": getattr(self, "spatial_weighting", "uniform"),
                "gaussianSigma": getattr(
                    self,
                    "gaussian_sigma",
                    0.25 * self.subset_size,
                ),
            },
            device=self.device_name,
            grid=(self.grid_width, self.grid_height),
            width=self.width,
            height=self.height,
            subset_size=self.subset_size,
            subset_stride=subset_stride,
            roi=roi,
            reference_valid=reference_valid,
        )

    def _empty_roi_result(self, started: float, config: dict) -> RegistrationResult:
        """Produce a finite dense empty field without upload, solve or readback."""
        packed = np.zeros(
            (self.grid_height, self.grid_width, state_float_count(self.motion_model)),
            dtype=np.float32,
        )
        packed[..., parameter_count(self.motion_model)] = -3.0
        result = self._result_from_raw(packed, timing={"queueSubmissions": 0.0}, config=config)
        elapsed = (time.perf_counter() - started) * 1000
        result.timing = {**result.timing, "totalWallMs": elapsed, "resultAgeWallMs": elapsed}
        return result

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
        motion_model = getattr(self, "motion_model", "affine")
        component_count = parameter_count(motion_model)
        state_stride = state_float_count(motion_model)
        subset_stride = getattr(self, "subset_stride", self.subset_size)
        packed = np.frombuffer(raw, dtype=np.float32).reshape(
            self.grid_height,
            self.grid_width,
            state_stride,
        )
        parameters = packed[..., :component_count]
        status = packed[..., component_count]
        iteration_field = packed[..., component_count + 1]
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
            config={
                **config,
                **getattr(self, "_pyramid_config", {}),
                "width": self.width,
                "height": self.height,
                "subsetSize": self.subset_size,
                **(
                    {"roi": {**self.roi.summary(), "selectedPerLevel": self._roi_level_counts}}
                    if getattr(self, "roi", None) is not None
                    else {}
                ),
                "deviceType": getattr(self, "device_type", "auto"),
                "executionDeviceType": getattr(self, "execution_device_type", "unknown"),
                "interpolation": getattr(self, "interpolation", "bspline"),
                "motionModel": getattr(self, "motion_model", "affine"),
                "outlierPolicy": getattr(self, "outlier_policy", "replace"),
                "subsetStride": subset_stride,
                "spatialWeighting": getattr(self, "spatial_weighting", "uniform"),
                "gaussianSigma": getattr(
                    self,
                    "gaussian_sigma",
                    0.25 * self.subset_size,
                ),
            },
            width=self.width,
            height=self.height,
            subset_size=self.subset_size,
            subset_stride=subset_stride,
            roi=getattr(self, "roi", None),
            reference_valid=getattr(self, "reference_valid", None),
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
        if self.roi is not None and self.roi.empty:
            return self._empty_roi_result(total_start, self._config)
        self._update_config()

        upload_wall_ms = self._upload_frame(self._pipeline_slots[0], frame)

        execution_start = time.perf_counter()
        raw = self._execute_registration(self._solver)
        execution_readback_wall_ms = (time.perf_counter() - execution_start) * 1_000.0
        result = self._result_from_raw(
            raw,
            timing={
                "uploadWallMs": upload_wall_ms,
                "executeReadbackWallMs": execution_readback_wall_ms,
            },
            config=self._config,
        )
        result.timing = {
            **result.timing,
            "totalWallMs": (time.perf_counter() - total_start) * 1_000.0,
        }
        if self.pyramid_levels > 1:
            result = self._recover_pyramid(self._pipeline_slots[0], frame, result)
            result.timing = {
                **result.timing,
                "totalWallMs": (time.perf_counter() - total_start) * 1_000.0,
            }
        return result

    def _submit_slot(
        self,
        slot: _RegistrationSlot,
        frame: npt.NDArray[np.uint8],
        config: dict[str, float | int],
    ) -> _PendingRegistration:
        submitted_at = time.perf_counter()
        upload_wall_ms = self._upload_frame(slot, frame)
        retained_frame = np.array(frame, copy=True) if self.pyramid_levels > 1 else frame
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
            frame=retained_frame,
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
        timing["totalWallMs"] = (time.perf_counter() - pending.submitted_at) * 1_000.0
        if self.pyramid_levels > 1:
            result = self._recover_pyramid(pending.slot, pending.frame, result)
        # Streaming latency starts immediately before upload, after target
        # normalization, and includes readback, result construction and recovery.
        result.timing = {
            **result.timing,
            "totalWallMs": (time.perf_counter() - pending.submitted_at) * 1_000.0,
        }
        return result

    def register_stream(
        self,
        targets: Iterable[ImageInput],
        *,
        pipeline_depth: int = DEFAULT_IN_FLIGHT_FRAMES,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> Iterator[RegistrationResult]:
        """Register a sequence with bounded GPU work and readback in flight.

        ``pipeline_depth`` is the maximum number of images in flight, not an
        image-pyramid depth. Its default of two lets the GPU execute the next
        target while Python converts the previous packed readback into result
        arrays. Results are yielded in input order. The extra in-flight frame
        improves steady-state throughput at the cost of approximately one
        processed frame of latency.
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
        if self.roi is not None and self.roi.empty:
            for target in targets:
                started = time.perf_counter()
                self._prepare_target(target)
                yield self._empty_roi_result(started, self._config)
            return
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
        pipeline_depth: int = DEFAULT_IN_FLIGHT_FRAMES,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> list[RegistrationResult]:
        """Return all results from :meth:`register_stream` as a list.

        The default keeps at most two images in flight, matching
        :meth:`register_stream`.
        """

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
        self._close_pyramid_engines()
        self._destroy_runtime()

    def __enter__(self) -> ICGNWGPU:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
