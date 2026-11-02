"""Validation for the displacement outlier-detection policy."""

from __future__ import annotations

import operator
from typing import Final

OUTLIER_POLICIES: Final = ("replace", "flag", "disabled", "median")
OUTLIER_POLICY_PIPELINE_CONSTANTS: Final = {
    "flag": 0,
    "replace": 1,
    "disabled": 2,
    "median": 3,
}

_POLICY_ERROR = "outlier_policy must be 'replace', 'flag', 'disabled', or 'median'"
MEDIAN_FILTER_SIZES: Final = (3, 5, 7, 9)
_MEDIAN_SIZE_ERROR = "median_filter_size must be one of 3, 5, 7, or 9 subsets"


def normalize_outlier_policy(value: str) -> str:
    """Return a validated, lower-case outlier policy."""

    if not isinstance(value, str):
        raise ValueError(_POLICY_ERROR)
    normalized = value.strip().lower()
    if normalized not in OUTLIER_POLICIES:
        raise ValueError(_POLICY_ERROR)
    return normalized


def normalize_median_filter_size(value: int) -> int:
    """Validate an odd square neighborhood width on the subset grid."""

    try:
        size = operator.index(value)
    except TypeError as error:
        raise ValueError(_MEDIAN_SIZE_ERROR) from error
    if isinstance(value, bool) or size not in MEDIAN_FILTER_SIZES:
        raise ValueError(_MEDIAN_SIZE_ERROR)
    return size
