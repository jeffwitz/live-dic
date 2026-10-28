"""Headless Python interface for GPU-accelerated affine IC-GN solvers."""

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
from .wgpu_backend import DEFAULT_IN_FLIGHT_FRAMES, ICGNWGPU

__all__ = [
    "BenchmarkResult",
    "DEFAULT_IN_FLIGHT_FRAMES",
    "EngineInfo",
    "ImageInput",
    "ICGNError",
    "ICGNWebGL",
    "ICGNWGPU",
    "PARAMETER_ORDER",
    "ReferenceInfo",
    "RegistrationResult",
    "register_pair",
]
