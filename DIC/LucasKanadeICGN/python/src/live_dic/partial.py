"""Double-precision preparation for fixed, partial translation/affine supports.

No image-dependent quantity here is recomputed in the live target loop.
Complete subsets retain their original reference, Hessian and metadata.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .roi import (
    PARTIAL_MAX_NORMALIZED_CONDITION,
    PARTIAL_MIN_WEIGHTED_CONTRAST,
    RoiPlan,
    immutable,
    unpack_support,
)

# Hessian condition uses local coordinates divided by the nominal subset size.
MAX_NORMALIZED_CONDITION = PARTIAL_MAX_NORMALIZED_CONDITION
MIN_WEIGHTED_CONTRAST = PARTIAL_MIN_WEIGHTED_CONTRAST


def prepare_partial_reference(
    reference: np.ndarray,
    inverse: np.ndarray,
    metadata: np.ndarray,
    roi: RoiPlan,
    *,
    width: int,
    height: int,
    size: int,
    stride: int,
    motion_model: str,
    weighting: str,
    sigma: float,
) -> RoiPlan:
    """Update only partial windows, using exactly the GPU's stored gradients.

    Metadata validity -4 is internal: the full-window solver skips it and the
    compact partial solver replaces it with its actual numerical solve status.
    Invalid partial references retain validity 0 and finite zero inverses.
    """
    pixels = reference.reshape(height, width, 4)
    rows = metadata.reshape(-1, 16)
    inverses = inverse.reshape(-1, 6, 6)
    grid_width = roi.selected.shape[1]
    yy, xx = np.indices((size, size), dtype=np.float64)
    xx = xx.ravel() - (size - 1) / 2
    yy = yy.ravel() - (size - 1) / 2
    spatial = (
        np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        if weighting == "gaussian"
        else np.ones(size * size)
    )
    active = [0, 3] if motion_model == "translation" else list(range(6))
    normalization = np.array([1, 1 / size, 1 / size, 1, 1 / size, 1 / size])[active]
    conditions = np.zeros(roi.selected.shape, np.float64)
    ranks = np.zeros(roi.selected.shape, np.uint32)
    contrasts = np.zeros(roi.selected.shape, np.float64)
    for ordinal, index in enumerate(roi.partial_indices):
        gy, gx = divmod(int(index), grid_width)
        support = unpack_support(roi.support_words[ordinal], size).ravel()
        weights = spatial * support
        weight_sum = weights.sum(dtype=np.float64)
        tile = (
            pixels[gy * stride : gy * stride + size, gx * stride : gx * stride + size]
            .reshape(-1, 4)
            .astype(np.float64)
        )
        mean = np.dot(weights, tile[:, 0]) / weight_sum
        f = tile[:, 0] - mean
        gx_values, gy_values = tile[:, 1], tile[:, 2]
        jacobian = np.stack(
            (gx_values, gx_values * xx, gx_values * yy, gy_values, gy_values * xx, gy_values * yy),
            axis=1,
        )
        j = jacobian[:, active]
        hessian = j.T @ (weights[:, None] * j)
        normalized = hessian * normalization[:, None] * normalization[None, :]
        eigenvalues = np.linalg.eigvalsh(normalized)
        largest = float(eigenvalues[-1])
        cutoff = np.finfo(np.float64).eps * len(active) * max(largest, 0.0)
        rank = int(np.count_nonzero(eigenvalues > cutoff))
        contrast = float(np.sqrt(np.dot(weights, f * f) / weight_sum))
        condition = largest / eigenvalues[0] if eigenvalues[0] > 0 else 0.0
        conditions[gy, gx] = condition if np.isfinite(condition) else 0.0
        ranks[gy, gx] = rank
        contrasts[gy, gx] = contrast
        valid = (
            rank == len(active)
            and condition <= MAX_NORMALIZED_CONDITION
            and contrast >= MIN_WEIGHTED_CONTRAST
            and support.sum() > len(active)
        )
        inv = np.zeros((6, 6), np.float64)
        if valid:
            if motion_model == "translation":
                determinant = hessian[0, 0] * hessian[1, 1] - hessian[0, 1] ** 2
                block = (
                    np.array([[hessian[1, 1], -hessian[0, 1]], [-hessian[0, 1], hessian[0, 0]]])
                    / determinant
                )
            else:
                block = np.linalg.inv(hessian)
            valid = bool(
                np.isfinite(block).all() and np.max(np.abs(block)) <= np.finfo(np.float32).max
            )
            if valid:
                inv[np.ix_(active, active)] = block
        inverses[index] = inv
        rows[index] = 0
        rows[index, 0] = np.sqrt(np.dot(weights, f * f))
        rows[index, 1] = -4.0 if valid else 0.0
        rows[index, 2:8] = jacobian.T @ (weights * f)
        rows[index, 8:14] = jacobian.T @ weights
        # For contiguous subsets, pixel intensities are already centred with
        # the original full-window mean. This is the remaining local offset.
        rows[index, 14] = mean
        rows[index, 15] = weight_sum
    return replace(
        roi,
        support_condition=immutable(conditions),
        support_rank=immutable(ranks),
        support_contrast=immutable(contrasts),
    )


def filter_and_support_data(weights: np.ndarray, roi: RoiPlan | None) -> np.ndarray:
    """One shared storage buffer: 31 FIR float bits, count, indices, support bits.

    The shader declares uint32 storage and bitcasts only the finite FIR values.
    Mask words are integers throughout; no NaN payload encodes geometry.
    """
    values = weights.view(np.uint32)
    if roi is None or roi.partial_indices is None or roi.partial_indices.size == 0:
        return values
    return np.concatenate(
        (
            values,
            np.asarray([roi.partial_indices.size], np.uint32),
            roi.partial_indices,
            roi.support_words.ravel(),
        )
    )
