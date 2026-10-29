"""Command-line interface for headless affine IC-GN registration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .engine import (
    PARAMETER_ORDER,
    BenchmarkResult,
    EngineInfo,
    ICGNError,
    ICGNWebGL,
    ReferenceInfo,
    RegistrationResult,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icgn-webgl",
        description="Run affine IC-GN registration through a headless browser GPU backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("reference", type=Path, help="reference image")
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        help="one or more target images with the same dimensions as the reference",
    )
    parser.add_argument("--iterations", type=int, help="maximum IC-GN iteration count")
    parser.add_argument("--tolerance", type=float, help="convergence tolerance in pixels")
    parser.add_argument("--zncc-min", type=float, help="minimum accepted ZNCC")
    parser.add_argument(
        "--truth",
        nargs=2,
        type=float,
        metavar=("DX", "DY"),
        help="constant true translation used to compute RMSE",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="benchmark one target instead of running the sequence interface",
    )
    parser.add_argument("--repeats", type=int, default=30, help="measured benchmark runs")
    parser.add_argument("--warmups", type=int, default=3, help="benchmark warm-up runs")
    parser.add_argument("--output", type=Path, help="write complete numeric fields to .npz")
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="optional runtime asset directory or tests_gpuio repository root",
    )
    parser.add_argument("--chromium", type=Path, help="Chromium or Chrome executable")
    parser.add_argument(
        "--backend",
        choices=("gpuio", "three-tsl", "webgpu-wgsl"),
        default="three-tsl",
        help="browser compute backend",
    )
    parser.add_argument(
        "--interpolation",
        choices=("bspline", "linear", "cubic"),
        default="bspline",
        help="target-image interpolation (bspline is the historical default)",
    )
    parser.add_argument(
        "--motion-model",
        choices=("affine", "translation"),
        default="affine",
        help="estimated warp model (translation uses an analytic 2x2 inverse Hessian)",
    )
    parser.add_argument(
        "--force-three-webgl",
        action="store_true",
        help="force the Three.js backend to use its WebGL2 compatibility path",
    )
    parser.add_argument(
        "--pyramid-levels",
        type=int,
        help="full-pyramid level count for the webgpu-wgsl backend (1 to 3)",
    )
    parser.add_argument(
        "--iterations-per-level",
        nargs="+",
        type=int,
        metavar="COUNT",
        help="coarse-to-fine iteration ceilings for webgpu-wgsl",
    )
    parser.add_argument(
        "--tolerance-per-level",
        nargs="+",
        type=float,
        metavar="PIXELS",
        help="coarse-to-fine convergence tolerances for webgpu-wgsl",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=20.0,
        help="browser startup timeout in seconds",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=120.0,
        help="per-call timeout in seconds",
    )
    parser.add_argument(
        "--enable-chromium-sandbox",
        action="store_true",
        help="do not pass --no-sandbox to Chromium",
    )
    parser.add_argument(
        "--chromium-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="additional Chromium argument; may be specified more than once",
    )
    return parser


def _algorithm_options(args: argparse.Namespace) -> dict[str, float | int | None]:
    return {
        "iterations": args.iterations,
        "tolerance": args.tolerance,
        "zncc_min": args.zncc_min,
    }


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _registration_summary(
    result: RegistrationResult,
    *,
    target: Path,
    truth: Sequence[float] | None,
) -> dict[str, Any]:
    unique, counts = np.unique(result.status.astype(np.int32), return_counts=True)
    summary: dict[str, Any] = {
        "target": str(target),
        "grid": list(result.status.shape[::-1]),
        "parameter_order": list(PARAMETER_ORDER),
        "accepted": result.accepted,
        "replaced": result.replaced,
        "status_counts": {
            str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)
        },
        "mean_translation": (
            None if result.mean_translation is None else list(result.mean_translation)
        ),
        "mean_iterations": _finite_or_none(result.mean_iterations),
        "timing_ms": dict(result.timing),
        "config": dict(result.config),
    }
    if truth is not None:
        summary["truth"] = [float(truth[0]), float(truth[1])]
        summary["translation_rmse_px"] = _finite_or_none(result.rmse(truth))
    return summary


def _reference_summary(reference: ReferenceInfo) -> dict[str, Any]:
    """Return separately labelled reference-construction timing boundaries."""

    return {
        "width": reference.width,
        "height": reference.height,
        "grid": list(reference.grid),
        "runtime_build_wall_ms": reference.runtime_build_wall_ms,
        "reference_build_wall_ms": reference.reference_build_wall_ms,
        "browser_preparation_wall_ms": reference.preparation_wall_ms,
        "encoding_wall_ms": reference.encoding_wall_ms,
        "python_call_wall_ms": reference.total_wall_ms,
    }


def _save_registrations(
    path: Path,
    targets: Sequence[Path],
    results: Sequence[RegistrationResult],
) -> None:
    """Save sequence outputs in one array per documented result field."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        targets=np.asarray([str(target) for target in targets]),
        parameter_order=np.asarray(PARAMETER_ORDER),
        parameters=np.stack([result.parameters for result in results]),
        displacement=np.stack([result.displacement for result in results]),
        status=np.stack([result.status for result in results]),
        iterations=np.stack([result.iterations for result in results]),
        accepted=np.asarray([result.accepted for result in results]),
        replaced=np.asarray([result.replaced for result in results]),
        mean_translation=np.asarray(
            [result.mean_translation or (np.nan, np.nan) for result in results],
            dtype=np.float64,
        ),
    )


def _save_benchmark(path: Path, target: Path, result: BenchmarkResult) -> None:
    """Save the timing samples and the final registration to a NumPy archive."""

    registration = result.registration
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target=np.asarray(str(target)),
        parameter_order=np.asarray(PARAMETER_ORDER),
        parameters=registration.parameters,
        displacement=registration.displacement,
        status=registration.status,
        iterations=registration.iterations,
        accepted=np.asarray(registration.accepted),
        replaced=np.asarray(registration.replaced),
        mean_translation=np.asarray(
            registration.mean_translation or (np.nan, np.nan), dtype=np.float64
        ),
        gpu_times_ms=result.gpu_times_ms,
        wall_times_ms=result.wall_times_ms,
    )


def _run_benchmark(
    engine: ICGNWebGL,
    info: EngineInfo,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target = args.targets[0]
    options = _algorithm_options(args)
    reference_info = engine.set_reference(args.reference, **options)
    benchmark = engine.benchmark_target(
        target,
        repeats=args.repeats,
        warmups=args.warmups,
        **options,
    )
    payload: dict[str, Any] = {
        "backend": args.backend,
        "device": benchmark.device,
        "timer_query": info.timer_query,
        "reference_preparation": _reference_summary(reference_info),
        "benchmark": {
            "repeats": benchmark.repeats,
            "warmups": benchmark.warmups,
            "gpu_median_ms": _finite_or_none(benchmark.gpu_median_ms),
            "gpu_p95_ms": _finite_or_none(benchmark.gpu_p95_ms),
            "wall_median_ms": benchmark.wall_median_ms,
            "wall_p95_ms": benchmark.wall_p95_ms,
        },
        "registration": _registration_summary(
            benchmark.registration,
            target=target,
            truth=args.truth,
        ),
    }
    if args.output:
        _save_benchmark(args.output, target, benchmark)
        payload["output"] = str(args.output)
    return payload


def _run_sequence(
    engine: ICGNWebGL,
    info: EngineInfo,
    args: argparse.Namespace,
) -> dict[str, Any]:
    options = _algorithm_options(args)
    reference_info = engine.set_reference(args.reference, **options)
    results = [engine.register(target, **options) for target in args.targets]
    payload: dict[str, Any] = {
        "backend": args.backend,
        "device": reference_info.device,
        "timer_query": info.timer_query,
        "reference": str(args.reference),
        "reference_preparation_ms": reference_info.preparation_wall_ms,
        "reference_preparation": _reference_summary(reference_info),
        "results": [
            _registration_summary(result, target=target, truth=args.truth)
            for target, result in zip(args.targets, results, strict=True)
        ],
    }
    if args.output:
        _save_registrations(args.output, args.targets, results)
        payload["output"] = str(args.output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.benchmark and len(args.targets) != 1:
        parser.error("--benchmark requires exactly one target image")

    try:
        with ICGNWebGL(
            repo_root=args.repo_root,
            chromium=args.chromium,
            startup_timeout=args.startup_timeout,
            call_timeout=args.call_timeout,
            no_sandbox=not args.enable_chromium_sandbox,
            chromium_args=args.chromium_arg,
            backend=args.backend,
            force_three_webgl=args.force_three_webgl,
            pyramid_levels=args.pyramid_levels,
            iterations_per_level=args.iterations_per_level,
            tolerance_per_level=args.tolerance_per_level,
            interpolation=args.interpolation,
            motion_model=args.motion_model,
        ) as engine:
            info = engine.info()
            payload = (
                _run_benchmark(engine, info, args)
                if args.benchmark
                else _run_sequence(engine, info, args)
            )
    except (ICGNError, OSError, ValueError) as error:
        print(f"icgn-webgl: {error}", file=sys.stderr)
        return 2

    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
