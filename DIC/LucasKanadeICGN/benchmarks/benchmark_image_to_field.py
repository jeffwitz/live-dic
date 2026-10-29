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
    parser.add_argument(
        "--backend",
        action="append",
        choices=("native-pyramid", "webgpu-wgsl", "three-tsl", "three-webgl"),
        dest="backends",
        help="backend to run; repeat the option to run several",
    )
    parser.add_argument("--pyramid-levels", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--interpolation",
        choices=("bspline", "linear", "cubic"),
        default="bspline",
    )
    parser.add_argument(
        "--motion-model",
        choices=("affine", "translation"),
        default="affine",
    )
    args = parser.parse_args()
    if args.width < 64 or args.height < 64:
        parser.error("--width and --height must each be at least 64")
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    args.backends = tuple(args.backends or DEFAULT_BACKENDS)
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
) -> AbstractContextManager[Any]:
    if backend == "native-pyramid":
        return ICGNWGPUFullPyramid(
            pyramid_levels=pyramid_levels,
            interpolation=interpolation,
            motion_model=motion_model,
        )
    return ICGNWebGL(
        backend="webgpu-wgsl" if backend == "webgpu-wgsl" else "three-tsl",
        pyramid_levels=pyramid_levels if backend == "webgpu-wgsl" else None,
        force_three_webgl=backend == "three-webgl",
        interpolation=interpolation,
        motion_model=motion_model,
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
) -> dict[str, Any]:
    for index in range(warmups):
        engine.register(targets[index % len(targets)][0])

    samples = []
    completion_times = []
    results = []
    truths = []
    stream_started = time.perf_counter()
    for index in range(frames):
        target, truth, _ = targets[index % len(targets)]
        started = time.perf_counter()
        result = engine.register(target)
        completed = time.perf_counter()
        samples.append((completed - started) * 1_000.0)
        completion_times.append(completed)
        results.append(result)
        truths.append(truth)
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
        "execution": "synchronous persistent engine; one result completed before the next input",
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
        ) as engine:
            preparation = engine.set_reference(reference)
            measurements[backend] = {
                "device": preparation.device,
                "preparation_ms": preparation.total_wall_ms,
                **_measure(
                    engine,
                    targets,
                    frames=args.frames,
                    warmups=args.warmups,
                ),
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
                "interpolation": args.interpolation,
                "motion_model": args.motion_model,
                "measurements": measurements,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
