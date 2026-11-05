"""Motion-model validation and GPU specialization constants."""

from __future__ import annotations

from typing import Final

MOTION_MODELS: Final = ("affine", "translation", "quadratic")
MOTION_MODEL_PIPELINE_CONSTANTS: Final = {
    "affine": 0,
    "translation": 1,
    "quadratic": 2,
}

AFFINE_PARAMETER_ORDER: Final = ("tx", "du_dx", "du_dy", "ty", "dv_dx", "dv_dy")
QUADRATIC_PARAMETER_ORDER: Final = (
    *AFFINE_PARAMETER_ORDER,
    "d2u_dx2",
    "d2u_dxdy",
    "d2u_dy2",
    "d2v_dx2",
    "d2v_dxdy",
    "d2v_dy2",
)


def normalize_motion_model(value: str) -> str:
    """Return a validated, lower-case motion model."""

    if not isinstance(value, str):
        raise ValueError("motion_model must be 'affine', 'translation', or 'quadratic'")
    normalized = value.strip().lower()
    if normalized not in MOTION_MODELS:
        raise ValueError("motion_model must be 'affine', 'translation', or 'quadratic'")
    return normalized


def parameter_order(motion_model: str) -> tuple[str, ...]:
    """Return the public parameter order for one validated motion model."""

    return (
        QUADRATIC_PARAMETER_ORDER
        if normalize_motion_model(motion_model) == "quadratic"
        else AFFINE_PARAMETER_ORDER
    )


def parameter_count(motion_model: str) -> int:
    """Return the number of reported warp parameters per subset."""

    return len(parameter_order(motion_model))


def state_float_count(motion_model: str) -> int:
    """Return the padded GPU-state stride in float32 values."""

    return 16 if normalize_motion_model(motion_model) == "quadratic" else 8
