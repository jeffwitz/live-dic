"""Validation shared by the Python hosts for square DIC subsets."""

from __future__ import annotations

import operator

DEFAULT_SUBSET_SIZE = 16
SUPPORTED_SUBSET_SIZES = (8, 15, 16, 17, 31, 32, 33)
_SUPPORTED_SUBSET_SIZES_TEXT = ", ".join(str(value) for value in SUPPORTED_SUBSET_SIZES)


def normalize_subset_size(value: int) -> int:
    """Return a supported square subset width.

    The portable kernels keep a fixed 16-lane workgroup and distribute the
    selected number of pixels over those lanes. Odd widths expose an exact
    centre pixel, while the established power-of-two widths remain available.
    """

    try:
        subset_size = operator.index(value)
    except TypeError as error:
        raise ValueError(f"subset_size must be one of {_SUPPORTED_SUBSET_SIZES_TEXT}") from error
    if isinstance(value, bool) or subset_size not in SUPPORTED_SUBSET_SIZES:
        raise ValueError(f"subset_size must be one of {_SUPPORTED_SUBSET_SIZES_TEXT}")
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
