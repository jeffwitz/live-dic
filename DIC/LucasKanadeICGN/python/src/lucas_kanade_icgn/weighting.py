"""Spatial integration weights for subset-based Lucas-Kanade solvers."""

from __future__ import annotations

import math

SPATIAL_WEIGHTING_PIPELINE_CONSTANTS = {
    "uniform": 0,
    "gaussian": 1,
}


def normalize_spatial_weighting(value: str) -> str:
    """Return a supported spatial integration weighting name."""

    weighting = str(value).strip().lower()
    if weighting not in SPATIAL_WEIGHTING_PIPELINE_CONSTANTS:
        raise ValueError("spatial_weighting must be 'uniform' or 'gaussian'")
    return weighting


def normalize_gaussian_sigma(value: float | None, *, subset_size: int) -> float:
    """Return a finite positive Gaussian sigma in pixels.

    A quarter of the subset width provides a useful starting point while
    keeping the parameter explicit for metrological qualification.
    """

    sigma = 0.25 * subset_size if value is None else float(value)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("gaussian_sigma must be finite and greater than zero")
    return sigma
