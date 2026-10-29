"""Validation for the displacement outlier-detection policy."""

from __future__ import annotations

from typing import Final

OUTLIER_POLICIES: Final = ("replace", "flag", "disabled")
OUTLIER_POLICY_PIPELINE_CONSTANTS: Final = {
    "flag": 0,
    "replace": 1,
    "disabled": 2,
}


def normalize_outlier_policy(value: str) -> str:
    """Return a validated, lower-case outlier policy."""

    if not isinstance(value, str):
        raise ValueError("outlier_policy must be 'replace', 'flag', or 'disabled'")
    normalized = value.strip().lower()
    if normalized not in OUTLIER_POLICIES:
        raise ValueError("outlier_policy must be 'replace', 'flag', or 'disabled'")
    return normalized
