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
