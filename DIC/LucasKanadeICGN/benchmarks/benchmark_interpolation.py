"""Compare DIC-RT interpolation modes on analytic fractional translations."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
from lucas_kanade_icgn import ICGNWGPU, ICGNWebGL, ICGNWGPUFullPyramid


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--translation", type=float, nargs=2, default=(0.5, -0.375))
    parser.add_argument("--levels", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument(
        "--backend",
        choices=("native", "webgpu-wgsl", "three-tsl", "three-webgl"),
        default="native",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations-per-level", type=int, nargs="+")
    parser.add_argument("--tolerance-per-level", type=float, nargs="+")
    parser.add_argument(
        "--motion-model",
        choices=("affine", "translation", "quadratic"),
        default="affine",
    )
    parser.add_argument(
        "--interpolation",
        nargs="+",
        choices=("bspline", "linear", "cubic"),
        default=("bspline", "linear", "cubic"),
    )
    return parser.parse_args()


def _analytic_texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a deterministic continuous texture without resampling bias."""

    values = (
        127.0
        + 34.0 * np.sin(0.173 * x + 0.071 * y)
        + 29.0 * np.cos(0.043 * x - 0.191 * y)
        + 23.0 * np.sin(0.311 * x - 0.137 * y + 0.4)
        + 17.0 * np.cos(0.229 * x + 0.257 * y - 0.8)
    )
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def fractional_pair(
    width: int,
    height: int,
    translation: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference/target images sampled from one continuous translated field."""

    y, x = np.mgrid[:height, :width].astype(np.float64)
    dx, dy = translation
    reference = _analytic_texture(x, y)
    target = _analytic_texture(x - dx, y - dy)
    return np.ascontiguousarray(reference), np.ascontiguousarray(target)


def _measure(
    engine: Any,
    target: np.ndarray,
    truth: tuple[float, float],
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        engine.register(target)
    samples = []
    gpu_samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = engine.register(target)
        samples.append((time.perf_counter() - started) * 1_000.0)
        gpu_ms = result.timing.get("gpuMs")
        if gpu_ms is not None:
            gpu_samples.append(float(gpu_ms))
    assert result is not None
    median_ms = float(np.median(samples))
    return {
        "median_ms": median_ms,
        "p95_ms": float(np.percentile(samples, 95)),
        "fps_from_median": 1_000.0 / median_ms,
        "gpu_median_ms": float(np.median(gpu_samples)) if gpu_samples else None,
        "gpu_p95_ms": float(np.percentile(gpu_samples, 95)) if gpu_samples else None,
        "accepted": result.accepted,
        "replaced": result.replaced,
        "truth_rmse_px": result.rmse(truth),
        "mean_translation": result.mean_translation,
        "mean_iterations": result.mean_iterations,
    }


def main() -> int:
    args = _arguments()
    if args.width < 64 or args.height < 64:
        raise ValueError("width and height must each be at least 64 pixels")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    for values, name in (
        (args.iterations_per_level, "iterations-per-level"),
        (args.tolerance_per_level, "tolerance-per-level"),
    ):
        if values is not None and len(values) != args.levels:
            raise ValueError(f"{name} must contain one value per pyramid level")
    truth = tuple(float(value) for value in args.translation)
    reference, target = fractional_pair(args.width, args.height, truth)
    measurements = {}
    device = None
    for interpolation in args.interpolation:
        options = {"interpolation": interpolation, "motion_model": args.motion_model}
        reference_options = {}
        if args.backend == "native":
            engine_type = ICGNWGPU if args.levels == 1 else ICGNWGPUFullPyramid
            if args.levels == 1:
                options.update(prefer_subgroups=False, pyramid_levels=1)
                if args.iterations_per_level is not None:
                    reference_options["iterations"] = args.iterations_per_level[0]
                if args.tolerance_per_level is not None:
                    reference_options["tolerance"] = args.tolerance_per_level[0]
            else:
                options.update(pyramid_levels=args.levels)
                if args.iterations_per_level is not None:
                    options["iterations_per_level"] = tuple(args.iterations_per_level)
                if args.tolerance_per_level is not None:
                    options["tolerance_per_level"] = tuple(args.tolerance_per_level)
        else:
            engine_type = ICGNWebGL
            if args.backend == "webgpu-wgsl":
                options.update(backend="webgpu-wgsl", pyramid_levels=args.levels)
                if args.iterations_per_level is not None:
                    options["iterations_per_level"] = tuple(args.iterations_per_level)
                if args.tolerance_per_level is not None:
                    options["tolerance_per_level"] = tuple(args.tolerance_per_level)
            else:
                if args.levels != 1:
                    raise ValueError(
                        "The Three.js backend currently supports one scale"
                    )
                options.update(
                    backend="three-tsl",
                    force_three_webgl=args.backend == "three-webgl",
                )
                if args.iterations_per_level is not None:
                    reference_options["iterations"] = args.iterations_per_level[0]
                if args.tolerance_per_level is not None:
                    reference_options["tolerance"] = args.tolerance_per_level[0]
        with engine_type(**options) as engine:
            preparation = engine.set_reference(reference, **reference_options)
            device = preparation.device
            measurements[interpolation] = {
                "preparation_ms": preparation.total_wall_ms,
                "config": dict(preparation.config),
                **_measure(engine, target, truth, args.warmups, args.repeats),
            }
    print(
        json.dumps(
            {
                "device": device,
                "resolution": [args.width, args.height],
                "truth": list(truth),
                "pyramid_levels": args.levels,
                "backend": args.backend,
                "motion_model": args.motion_model,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "measurements": measurements,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
