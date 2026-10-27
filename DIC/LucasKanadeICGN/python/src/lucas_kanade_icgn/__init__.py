"""Headless Python interface for the affine IC-GN WebGL2 solver."""

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

__all__ = [
    "BenchmarkResult",
    "EngineInfo",
    "ImageInput",
    "ICGNError",
    "ICGNWebGL",
    "PARAMETER_ORDER",
    "ReferenceInfo",
    "RegistrationResult",
    "register_pair",
]
