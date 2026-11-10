"""Validation shared by the Python hosts for square DIC subsets."""

from __future__ import annotations

import operator

DEFAULT_SUBSET_SIZE = 16
# A 4 x 4 support provides at least twelve samples for the quadratic model.
MIN_SUBSET_SIZE = 4
# The WGSL solver indexes a flattened subset with signed 32-bit arithmetic.
# Keep S*S and S*S+31 representable; actual image and adapter limits are lower.
MAX_SUBSET_SIZE = 46_340


def normalize_subset_size(value: int) -> int:
    """Return a valid square subset width for the portable WGSL solver.

    The flattened pixel loop distributes the square over the configured
    workgroup. Its 32-bit pixel indexing limits the theoretical width to 46,340;
    image dimensions, adapter limits, and available memory impose tighter
    practical limits at reference preparation.
    """

    try:
        subset_size = operator.index(value)
    except TypeError as error:
        raise ValueError(
            f"subset_size must be an integer from {MIN_SUBSET_SIZE} to {MAX_SUBSET_SIZE}"
        ) from error
    if isinstance(value, bool) or not MIN_SUBSET_SIZE <= subset_size <= MAX_SUBSET_SIZE:
        raise ValueError(
            f"subset_size must be an integer from {MIN_SUBSET_SIZE} to {MAX_SUBSET_SIZE}"
        )
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
