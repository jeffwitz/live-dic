"""Validation shared by the Python hosts for square DIC subsets."""

from __future__ import annotations

import operator

DEFAULT_SUBSET_SIZE = 16
SUPPORTED_SUBSET_SIZES = (8, 16, 32)


def normalize_subset_size(value: int) -> int:
    """Return a supported square subset width.

    The portable kernels keep a fixed 16-lane workgroup and distribute the
    selected number of pixels over those lanes. Restricting the public option
    to powers of two keeps the work balanced and leaves the validated 16x16
    specialization unchanged.
    """

    try:
        subset_size = operator.index(value)
    except TypeError as error:
        raise ValueError("subset_size must be one of 8, 16, or 32") from error
    if isinstance(value, bool) or subset_size not in SUPPORTED_SUBSET_SIZES:
        raise ValueError("subset_size must be one of 8, 16, or 32")
    return subset_size


def normalize_subset_stride(value: int | None, *, subset_size: int) -> int:
    """Return a positive grid spacing no larger than the subset width."""

    if value is None:
        return subset_size
    try:
        subset_stride = operator.index(value)
    except TypeError as error:
        raise ValueError("subset_stride must be a positive integer") from error
    if isinstance(value, bool) or not 1 <= subset_stride <= subset_size:
        raise ValueError("subset_stride must be between 1 and subset_size")
    return subset_stride


def subset_grid_shape(
    width: int,
    height: int,
    *,
    subset_size: int,
    subset_stride: int,
) -> tuple[int, int]:
    """Return the fully contained sliding-window grid dimensions."""

    if width < subset_size or height < subset_size:
        return (0, 0)
    return (
        1 + (width - subset_size) // subset_stride,
        1 + (height - subset_size) // subset_stride,
    )
