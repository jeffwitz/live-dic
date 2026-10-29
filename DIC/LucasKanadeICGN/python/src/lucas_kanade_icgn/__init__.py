"""Headless Python interface for GPU-accelerated IC-GN solvers."""

from .engine import (
    PARAMETER_ORDER,
    BenchmarkResult,
    EngineInfo,
    ICGNError,
    ICGNWebGL,
    ImageInput,
    ReferenceInfo,
    RegistrationResult,
    register_pair,
)
from .full_pyramid_backend import DEFAULT_FULL_PYRAMID_LEVELS, ICGNWGPUFullPyramid
from .motion import AFFINE_PARAMETER_ORDER, MOTION_MODELS, QUADRATIC_PARAMETER_ORDER
from .outlier import OUTLIER_POLICIES
from .wgpu_backend import DEFAULT_IN_FLIGHT_FRAMES, DEFAULT_PYRAMID_LEVELS, ICGNWGPU

__all__ = [
    "BenchmarkResult",
    "AFFINE_PARAMETER_ORDER",
    "DEFAULT_IN_FLIGHT_FRAMES",
    "DEFAULT_FULL_PYRAMID_LEVELS",
    "DEFAULT_PYRAMID_LEVELS",
    "EngineInfo",
    "ImageInput",
    "ICGNError",
    "ICGNWebGL",
    "ICGNWGPU",
    "ICGNWGPUFullPyramid",
    "MOTION_MODELS",
    "OUTLIER_POLICIES",
    "PARAMETER_ORDER",
    "QUADRATIC_PARAMETER_ORDER",
    "ReferenceInfo",
    "RegistrationResult",
    "register_pair",
]
