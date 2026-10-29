"""Motion-model validation and GPU specialization constants."""

from __future__ import annotations

from typing import Final

MOTION_MODELS: Final = ("affine", "translation")
MOTION_MODEL_PIPELINE_CONSTANTS: Final = {
    "affine": 0,
    "translation": 1,
}


def normalize_motion_model(value: str) -> str:
    """Return a validated, lower-case motion model."""

    if not isinstance(value, str):
        raise ValueError("motion_model must be 'affine' or 'translation'")
    normalized = value.strip().lower()
    if normalized not in MOTION_MODELS:
        raise ValueError("motion_model must be 'affine' or 'translation'")
    return normalized
