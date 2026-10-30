"""Measure synchronous and bounded in-flight wgpu-py sequence processing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from lucas_kanade_icgn import ICGNWGPU, RegistrationResult
from PIL import Image

UHD_WIDTH = 3840
UHD_HEIGHT = 2160
TRUTH = (2.0, -1.0)


def _arguments() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--texture",
        type=Path,
        default=module_root / "assets/frame_000.png",
        help="texture tiled to construct the offline UHD reference",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--campaigns", type=int, default=4)
    parser.add_argument(
        "--pyramid-levels",
        type=int,
        default=1,
        help="total image scales; one measures the unchanged fast path",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=(1, 2, 3),
        help="bounded in-flight depths to compare with synchronous register()",
    )
    return parser.parse_args()


def _make_pair(texture_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(texture_path) as image:
        texture = np.asarray(image.convert("L"), dtype=np.uint8)
    repeat_y = (UHD_HEIGHT + texture.shape[0] - 1) // texture.shape[0]
    repeat_x = (UHD_WIDTH + texture.shape[1] - 1) // texture.shape[1]
    reference = np.tile(texture, (repeat_y, repeat_x))[:UHD_HEIGHT, :UHD_WIDTH]
    target = np.roll(reference, shift=(-1, 2), axis=(0, 1))
    return reference, target


def _percentile(samples: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), percentile))


def _measure_sync(
    engine: ICGNWGPU,
    target: np.ndarray,
    repeats: int,
) -> tuple[dict[str, Any], RegistrationResult]:
    latencies = []
    started = time.perf_counter()
    result: RegistrationResult | None = None
    for _ in range(repeats):
        frame_started = time.perf_counter()
        result = engine.register(target)
        latencies.append((time.perf_counter() - frame_started) * 1_000.0)
    sequence_ms = (time.perf_counter() - started) * 1_000.0
    assert result is not None
    return (
        {
            "sequence_ms": sequence_ms,
            "period_ms": sequence_ms / repeats,
            "throughput_fps": repeats * 1_000.0 / sequence_ms,
            "latency_median_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
        },
        result,
    )


def _measure_pipeline(
    engine: ICGNWGPU,
    target: np.ndarray,
    repeats: int,
    depth: int,
) -> tuple[dict[str, Any], RegistrationResult]:
    started = time.perf_counter()
    results = list(
        engine.register_stream((target for _ in range(repeats)), pipeline_depth=depth)
    )
    sequence_ms = (time.perf_counter() - started) * 1_000.0
    latencies = [result.timing["totalWallMs"] for result in results]
    return (
        {
            "sequence_ms": sequence_ms,
            "period_ms": sequence_ms / repeats,
            "throughput_fps": repeats * 1_000.0 / sequence_ms,
            "latency_median_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
        },
        results[-1],
    )


def main() -> int:
    args = _arguments()
    if args.repeats < 1 or args.warmups < 0 or args.campaigns < 1:
        raise ValueError(
            "repeats and campaigns must be positive; warmups must be non-negative"
        )
    if any(depth < 1 for depth in args.depths):
        raise ValueError("all pipeline depths must be positive")
    reference, target = _make_pair(args.texture)
    methods = ("sync", *(f"pipeline-{depth}" for depth in args.depths))
    campaigns: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    with ICGNWGPU(
        prefer_subgroups=False,
        pyramid_levels=args.pyramid_levels,
    ) as engine:
        preparation = engine.set_reference(reference)
        for _ in range(args.warmups):
            engine.register(target)
        for depth in args.depths:
            list(
                engine.register_stream(
                    (target for _ in range(args.warmups)), pipeline_depth=depth
                )
            )

        final_results: dict[str, RegistrationResult] = {}
        for campaign in range(args.campaigns):
            offset = campaign % len(methods)
            order = (*methods[offset:], *methods[:offset])
            for method in order:
                if method == "sync":
                    measurement, result = _measure_sync(engine, target, args.repeats)
                else:
                    depth = int(method.rsplit("-", 1)[1])
                    measurement, result = _measure_pipeline(
                        engine,
                        target,
                        args.repeats,
                        depth,
                    )
                measurement["campaign"] = campaign + 1
                campaigns[method].append(measurement)
                final_results[method] = result

    summary = {}
    for method, samples in campaigns.items():
        result = final_results[method]
        summary[method] = {
            "period_median_ms": _percentile(
                [sample["period_ms"] for sample in samples], 50
            ),
            "period_p95_ms": _percentile(
                [sample["period_ms"] for sample in samples], 95
            ),
            "throughput_median_fps": _percentile(
                [sample["throughput_fps"] for sample in samples],
                50,
            ),
            "latency_median_ms": _percentile(
                [sample["latency_median_ms"] for sample in samples],
                50,
            ),
            "latency_p95_ms": _percentile(
                [sample["latency_p95_ms"] for sample in samples],
                50,
            ),
            "accepted": result.accepted,
            "truth_rmse_px": result.rmse(TRUTH),
        }

    print(
        json.dumps(
            {
                "device": engine.device_name,
                "resolution": [UHD_WIDTH, UHD_HEIGHT],
                "grid": [engine.grid_width, engine.grid_height],
                "repeats": args.repeats,
                "warmups": args.warmups,
                "campaigns": args.campaigns,
                "pyramid_levels": args.pyramid_levels,
                "preparation_wall_ms": preparation.total_wall_ms,
                "summary": summary,
                "campaign_samples": campaigns,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
