#!/usr/bin/env python3
"""Measure the public NumPy-image to NumPy-field boundary on changing frames.

Reference preparation, engine startup, and fixture construction are setup work
and are excluded. Every measured call starts with an already acquired uint8
NumPy image and ends only after the complete RegistrationResult field is
available in Python. No GPU-only duration is used as the primary result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWebGL, ICGNWGPUFullPyramid

DEFAULT_BACKENDS = ("native-pyramid", "webgpu-wgsl", "three-tsl")
TARGET_FIXTURES = (
    ("frame_001.png", (0.2, -0.1)),
    ("frame_005.png", (1.0, -0.5)),
    ("frame_010.png", (2.0, -1.0)),
    ("frame_020.png", (4.0, -2.0)),
)


def _arguments() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        type=Path,
        default=module_root / "assets",
        help="directory containing the qualified frame_*.png fixtures",
    )
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--campaigns", type=int, default=1)
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=1,
        help="native full-pyramid targets in flight; browser backends require 1",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("native-pyramid", "webgpu-wgsl", "three-tsl", "three-webgl"),
        dest="backends",
        help="backend to run; repeat the option to run several",
    )
    parser.add_argument("--pyramid-levels", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--subset-size",
        type=int,
        choices=(8, 15, 16, 17, 31, 32, 33),
        default=16,
    )
    parser.add_argument(
        "--subset-stride",
        type=int,
        help="subset-centre spacing (defaults to --subset-size)",
    )
    parser.add_argument(
        "--interpolation",
        choices=("bspline", "linear", "cubic"),
        default="bspline",
    )
    parser.add_argument(
        "--motion-model",
        choices=("affine", "translation", "quadratic"),
        default="affine",
    )
    parser.add_argument(
        "--median-scope",
        choices=("displacement", "all"),
        default="all",
    )
    parser.add_argument(
        "--spatial-weighting",
        choices=("uniform", "gaussian"),
        default="uniform",
        help="native subset integration weights",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        help="centred Gaussian sigma in pixels (default: subset width / 4)",
    )
    args = parser.parse_args()
    if args.width < 64 or args.height < 64:
        parser.error("--width and --height must each be at least 64")
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.campaigns < 1:
        parser.error("--campaigns must be positive")
    if args.pipeline_depth < 1:
        parser.error("--pipeline-depth must be positive")
    args.backends = tuple(args.backends or DEFAULT_BACKENDS)
    if args.pipeline_depth > 1 and any(
        backend != "native-pyramid" for backend in args.backends
    ):
        parser.error(
            "--pipeline-depth greater than 1 currently requires --backend native-pyramid"
        )
    if args.spatial_weighting != "uniform" and any(
        backend != "native-pyramid" for backend in args.backends
    ):
        parser.error(
            "Gaussian spatial weighting currently requires --backend native-pyramid"
        )
    return args


def _tile_gray(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        texture = np.asarray(image.convert("L"), dtype=np.uint8)
    repeat_y = (height + texture.shape[0] - 1) // texture.shape[0]
    repeat_x = (width + texture.shape[1] - 1) // texture.shape[1]
    return np.ascontiguousarray(np.tile(texture, (repeat_y, repeat_x))[:height, :width])


def _fixtures(
    assets: Path,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple[float, float], str]]]:
    reference = _tile_gray(assets / "frame_000.png", width, height)
    targets = []
    for filename, truth in TARGET_FIXTURES:
        frame = _tile_gray(assets / filename, width, height)
        digest = hashlib.sha256(frame).hexdigest()
        targets.append((frame, truth, digest))
    if len({digest for _, _, digest in targets}) != len(targets):
        raise RuntimeError("stream fixtures must contain distinct image bytes")
    return reference, targets


def _engine(
    backend: str,
    *,
    pyramid_levels: int,
    interpolation: str,
    motion_model: str,
    median_scope: str,
    subset_size: int,
    subset_stride: int | None,
    spatial_weighting: str,
    gaussian_sigma: float | None,
) -> AbstractContextManager[Any]:
    if backend == "native-pyramid":
        return ICGNWGPUFullPyramid(
            pyramid_levels=pyramid_levels,
            interpolation=interpolation,
            motion_model=motion_model,
            median_scope=median_scope,
            subset_size=subset_size,
            subset_stride=subset_stride,
            spatial_weighting=spatial_weighting,
            gaussian_sigma=gaussian_sigma,
        )
    return ICGNWebGL(
        backend="webgpu-wgsl" if backend == "webgpu-wgsl" else "three-tsl",
        pyramid_levels=pyramid_levels if backend == "webgpu-wgsl" else None,
        force_three_webgl=backend == "three-webgl",
        interpolation=interpolation,
        motion_model=motion_model,
        median_scope=median_scope,
        subset_size=subset_size,
        subset_stride=subset_stride,
        startup_timeout=120,
        call_timeout=600,
    )


def _numeric_timing_medians(results: list[Any]) -> dict[str, float]:
    keys = sorted(set().union(*(result.timing for result in results)))
    medians = {}
    for key in keys:
        values = [result.timing.get(key) for result in results]
        numeric = [float(value) for value in values if value is not None]
        if len(numeric) == len(values):
            medians[key] = float(np.median(numeric))
    return medians


def _measure(
    engine: Any,
    targets: list[tuple[np.ndarray, tuple[float, float], str]],
    *,
    frames: int,
    warmups: int,
    pipeline_depth: int,
    use_stream: bool,
) -> dict[str, Any]:
    for index in range(warmups):
        engine.register(targets[index % len(targets)][0])

    samples = []
    completion_times = []
    results = []
    truths = []
    selected = [targets[index % len(targets)] for index in range(frames)]
    stream_started = time.perf_counter()
    if not use_stream:
        for target, truth, _ in selected:
            started = time.perf_counter()
            result = engine.register(target)
            completed = time.perf_counter()
            samples.append((completed - started) * 1_000.0)
            completion_times.append(completed)
            results.append(result)
            truths.append(truth)
        execution = (
            "synchronous persistent engine; one result completed before the next input"
        )
    else:
        target_images = (target for target, _, _ in selected)
        for result, (_, truth, _) in zip(
            engine.register_stream(target_images, pipeline_depth=pipeline_depth),
            selected,
            strict=True,
        ):
            completed = time.perf_counter()
            samples.append(float(result.timing["resultAgeWallMs"]))
            completion_times.append(completed)
            results.append(result)
            truths.append(truth)
        execution = (
            "bounded native full-pyramid stream; "
            f"{pipeline_depth} independent target(s) in flight"
        )
    stream_wall_ms = (time.perf_counter() - stream_started) * 1_000.0

    shape = results[0].parameters.shape
    if any(result.parameters.shape != shape for result in results):
        raise RuntimeError("result field shape changed inside the stream")
    if not all(np.isfinite(result.parameters).all() for result in results):
        raise RuntimeError("a result field contains a non-finite parameter")

    periods = np.diff(np.asarray(completion_times, dtype=np.float64)) * 1_000.0
    rmses = [result.rmse(truth) for result, truth in zip(results, truths, strict=True)]
    accepted = [result.accepted for result in results]
    return {
        "boundary": "resident uint8 NumPy image to materialized NumPy parameter field",
        "execution": execution,
        "pipeline_depth": pipeline_depth,
        "frames": frames,
        "warmups": warmups,
        "latency_median_ms": float(np.median(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "stream_period_ms": stream_wall_ms / frames,
        "throughput_fps": frames * 1_000.0 / stream_wall_ms,
        "completion_period_median_ms": (
            float(np.median(periods)) if periods.size else float(samples[0])
        ),
        "field_shape": list(shape),
        "field_bytes": int(results[0].parameters.nbytes),
        "accepted_min": min(accepted),
        "accepted_max": max(accepted),
        "rmse_median_px": float(np.median(rmses)),
        "rmse_max_px": float(np.max(rmses)),
        "component_timing_medians_ms": _numeric_timing_medians(results),
    }


def _aggregate_campaigns(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    """Report medians across campaigns while retaining concise raw summaries."""

    if len(campaigns) == 1:
        return {**campaigns[0], "campaigns": 1}
    result = dict(campaigns[0])
    median_keys = (
        "latency_median_ms",
        "latency_p95_ms",
        "stream_period_ms",
        "throughput_fps",
        "completion_period_median_ms",
        "rmse_median_px",
    )
    for key in median_keys:
        result[key] = float(np.median([campaign[key] for campaign in campaigns]))
    result["accepted_min"] = min(campaign["accepted_min"] for campaign in campaigns)
    result["accepted_max"] = max(campaign["accepted_max"] for campaign in campaigns)
    result["rmse_max_px"] = max(campaign["rmse_max_px"] for campaign in campaigns)
    component_keys = campaigns[0]["component_timing_medians_ms"]
    result["component_timing_medians_ms"] = {
        key: float(
            np.median(
                [campaign["component_timing_medians_ms"][key] for campaign in campaigns]
            )
        )
        for key in component_keys
    }
    result["campaigns"] = len(campaigns)
    result["campaign_summaries"] = [
        {
            key: campaign[key]
            for key in (
                "latency_median_ms",
                "latency_p95_ms",
                "stream_period_ms",
                "throughput_fps",
            )
        }
        for campaign in campaigns
    ]
    return result


def main() -> int:
    args = _arguments()
    reference, targets = _fixtures(args.assets, args.width, args.height)
    measurements = {}
    for backend in args.backends:
        with _engine(
            backend,
            pyramid_levels=args.pyramid_levels,
            interpolation=args.interpolation,
            motion_model=args.motion_model,
            median_scope=args.median_scope,
            subset_size=args.subset_size,
            subset_stride=args.subset_stride,
            spatial_weighting=args.spatial_weighting,
            gaussian_sigma=args.gaussian_sigma,
        ) as engine:
            preparation = engine.set_reference(reference)
            campaigns = [
                _measure(
                    engine,
                    targets,
                    frames=args.frames,
                    warmups=args.warmups,
                    pipeline_depth=args.pipeline_depth,
                    use_stream=backend == "native-pyramid",
                )
                for _ in range(args.campaigns)
            ]
            measurements[backend] = {
                "device": preparation.device,
                "preparation_ms": preparation.total_wall_ms,
                **_aggregate_campaigns(campaigns),
            }

    print(
        json.dumps(
            {
                "resolution": [args.width, args.height],
                "input": "pre-acquired contiguous uint8 NumPy grayscale frames",
                "reference_preparation_included": False,
                "camera_or_decoder_included": False,
                "fixture_hashes": {
                    filename: digest
                    for (filename, _), (_, _, digest) in zip(
                        TARGET_FIXTURES, targets, strict=True
                    )
                },
                "pyramid_levels": args.pyramid_levels,
                "subset_size": args.subset_size,
                "subset_stride": args.subset_stride or args.subset_size,
                "interpolation": args.interpolation,
                "motion_model": args.motion_model,
                "median_scope": args.median_scope,
                "spatial_weighting": args.spatial_weighting,
                "gaussian_sigma": (
                    args.gaussian_sigma
                    if args.gaussian_sigma is not None
                    else 0.25 * args.subset_size
                ),
                "pipeline_depth": args.pipeline_depth,
                "campaigns": args.campaigns,
                "measurements": measurements,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
