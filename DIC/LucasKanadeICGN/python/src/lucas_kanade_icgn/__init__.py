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
from .wgpu_backend import ICGNWGPU

__all__ = [
    "BenchmarkResult",
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
