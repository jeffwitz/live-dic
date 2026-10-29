"""Benchmark the GPU-resident full-pyramid backend against one-scale IC-GN."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWGPU, ICGNWGPUFullPyramid

TRUTH = (4.0, -2.0)


def _arguments() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--texture",
        type=Path,
        default=module_root / "assets/frame_000.png",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--levels", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--motion-model",
        choices=("affine", "translation", "quadratic"),
        default="affine",
    )
    return parser.parse_args()


def _pair(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        texture = np.asarray(image.convert("L"), dtype=np.uint8)
    repeat_y = (height + texture.shape[0] - 1) // texture.shape[0]
    repeat_x = (width + texture.shape[1] - 1) // texture.shape[1]
    reference = np.tile(texture, (repeat_y, repeat_x))[:height, :width]
    target = np.roll(reference, shift=(-2, 4), axis=(0, 1))
    return np.ascontiguousarray(reference), np.ascontiguousarray(target)


def _measure(
    engine: Any, target: np.ndarray, warmups: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmups):
        engine.register(target)
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = engine.register(target)
        samples.append((time.perf_counter() - started) * 1_000.0)
    assert result is not None
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "fps_from_median": 1_000.0 / float(np.median(samples)),
        "accepted": result.accepted,
        "replaced": result.replaced,
        "truth_rmse_px": result.rmse(TRUTH),
        "mean_translation": result.mean_translation,
        "mean_iterations": result.mean_iterations,
        "queue_submissions": result.timing.get("queueSubmissions"),
    }


def main() -> int:
    args = _arguments()
    if args.width < 64 or args.height < 64:
        raise ValueError("width and height must each be at least 64 pixels")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    if any(level not in (2, 3) for level in args.levels):
        raise ValueError("full-pyramid levels must be 2 or 3")
    reference, target = _pair(args.texture, args.width, args.height)
    measurements: dict[str, Any] = {}

    with ICGNWGPU(
        prefer_subgroups=False,
        pyramid_levels=1,
        motion_model=args.motion_model,
    ) as engine:
        preparation = engine.set_reference(reference)
        measurements["one-scale"] = {
            "preparation_ms": preparation.total_wall_ms,
            **_measure(engine, target, args.warmups, args.repeats),
        }
        device = engine.device_name

    for levels in args.levels:
        with ICGNWGPUFullPyramid(
            pyramid_levels=levels,
            motion_model=args.motion_model,
        ) as engine:
            preparation = engine.set_reference(reference)
            measurements[f"full-pyramid-{levels}"] = {
                "preparation_ms": preparation.total_wall_ms,
                "iterations_coarse_to_fine": list(engine.iterations_per_level),
                "tolerances_coarse_to_fine": list(engine.tolerance_per_level),
                **_measure(engine, target, args.warmups, args.repeats),
            }

    print(
        json.dumps(
            {
                "device": device,
                "resolution": [args.width, args.height],
                "truth": list(TRUTH),
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
