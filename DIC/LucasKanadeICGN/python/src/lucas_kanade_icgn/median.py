"""Median-filter scope validation and GPU specialization constants."""

from __future__ import annotations

from typing import Final

MEDIAN_SCOPES: Final = ("displacement", "all")
MEDIAN_SCOPE_PIPELINE_CONSTANTS: Final = {
    "displacement": 0,
    "all": 1,
}
MEDIAN_REPLACEMENTS: Final = ("flagged", "all")
MEDIAN_REPLACEMENT_PIPELINE_CONSTANTS: Final = {
    "flagged": 0,
    "all": 1,
}


def normalize_median_scope(value: str) -> str:
    """Return a validated, lower-case median-filter scope."""

    if not isinstance(value, str):
        raise ValueError("median_scope must be 'displacement' or 'all'")
    normalized = value.strip().lower()
    if normalized not in MEDIAN_SCOPES:
        raise ValueError("median_scope must be 'displacement' or 'all'")
    return normalized


def normalize_median_replacement(value: str) -> str:
    """Return a validated, lower-case median replacement policy."""

    if not isinstance(value, str):
        raise ValueError("median_replacement must be 'flagged' or 'all'")
    normalized = value.strip().lower()
    if normalized not in MEDIAN_REPLACEMENTS:
        raise ValueError("median_replacement must be 'flagged' or 'all'")
    return normalized
