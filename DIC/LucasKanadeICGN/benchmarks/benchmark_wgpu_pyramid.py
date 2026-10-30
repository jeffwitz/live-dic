"""Measure selective native pyramid recovery and the one-scale fast path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from lucas_kanade_icgn import ICGNWGPU
from PIL import Image

TRUTH = (4.0, -2.0)


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture", type=Path, default=root / "assets/frame_000.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--levels", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def _pair(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        texture = np.asarray(image.convert("L"), dtype=np.uint8)
    repeat_y = (height + texture.shape[0] - 1) // texture.shape[0]
    repeat_x = (width + texture.shape[1] - 1) // texture.shape[1]
    reference = np.tile(texture, (repeat_y, repeat_x))[:height, :width]
    target = np.roll(reference, shift=(-2, 4), axis=(0, 1))
    return reference, target


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def main() -> int:
    args = _arguments()
    if args.width < 64 or args.height < 64:
        raise ValueError("width and height must each be at least 64 pixels")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    reference, target = _pair(args.texture, args.width, args.height)
    measurements: dict[str, Any] = {}
    device = ""
    for levels in args.levels:
        with ICGNWGPU(prefer_subgroups=False, pyramid_levels=levels) as engine:
            preparation = engine.set_reference(reference)
            for _ in range(args.warmups):
                engine.register(target)
            samples = []
            result = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = engine.register(target)
                samples.append((time.perf_counter() - started) * 1_000.0)
            assert result is not None
            device = engine.device_name
            measurements[str(levels)] = {
                "preparation_wall_ms": preparation.total_wall_ms,
                "registration_median_ms": _percentile(samples, 50),
                "registration_p95_ms": _percentile(samples, 95),
                "accepted": result.accepted,
                "replaced": result.replaced,
                "truth_rmse_px": result.rmse(TRUTH),
                "pyramid_failed_before": result.timing.get("pyramidFailedBefore", 0.0),
                "pyramid_recovered": result.timing.get("pyramidRecovered", 0.0),
            }
    print(
        json.dumps(
            {
                "device": device,
                "resolution": [args.width, args.height],
                "truth": list(TRUTH),
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
