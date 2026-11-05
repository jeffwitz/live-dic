"""Interpolation mode names shared by the public Python backends."""

from __future__ import annotations

INTERPOLATION_MODES = ("bspline", "linear", "cubic")
"""Supported target reconstruction modes.

``bspline`` is the historical Gaussian plus cubic B-spline reconstruction.
``linear`` and ``cubic`` interpolate the unfiltered image directly.
"""

INTERPOLATION_PIPELINE_CONSTANTS = {
    "bspline": 0.0,
    "linear": 1.0,
    "cubic": 2.0,
}


def normalize_interpolation(value: str) -> str:
    """Return a validated, lower-case interpolation mode."""

    if not isinstance(value, str):
        raise ValueError("interpolation must be 'bspline', 'linear', or 'cubic'")
    normalized = value.strip().lower()
    if normalized not in INTERPOLATION_MODES:
        raise ValueError("interpolation must be 'bspline', 'linear', or 'cubic'")
    return normalized
