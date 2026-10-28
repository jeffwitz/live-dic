"""Validate and benchmark a GPU backend on a synthetic UHD image pair.

The shipped 512-pixel reference texture is tiled to 3840 x 2160, then shifted
by exactly ``(2, -1)`` pixels. Reference construction and repeated optical-flow
registration are reported as separate phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWebGL, ICGNWGPU


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
    parser.add_argument(
        "--backend",
        choices=("three-tsl", "gpuio", "wgpu-native"),
        default="three-tsl",
        help="GPU implementation to measure",
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


def _browser_report(
    backend: str,
    reference: np.ndarray,
    target: np.ndarray,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    with ICGNWebGL(backend=backend, startup_timeout=120, call_timeout=600) as engine:
        preparation = engine.set_reference(reference)
        runtime = engine.info()
        benchmark = engine.benchmark_target(
            target,
            repeats=repeats,
            warmups=warmups,
        )

    registration = benchmark.registration
    return {
        "backend": backend,
        "device": benchmark.device,
        "solver": runtime.solver,
        "subgroup_width": runtime.subgroup_width,
        "resolution": [preparation.width, preparation.height],
        "grid": list(preparation.grid),
        "truth": list(TRUTH),
        "preparation": {
            "runtime_build_wall_ms": preparation.runtime_build_wall_ms,
            "reference_build_wall_ms": preparation.reference_build_wall_ms,
            "browser_preparation_wall_ms": preparation.preparation_wall_ms,
            "encoding_wall_ms": preparation.encoding_wall_ms,
            "python_call_wall_ms": preparation.total_wall_ms,
        },
        "optical_flow": {
            "repeats": benchmark.repeats,
            "warmups": benchmark.warmups,
            "gpu_median_ms": benchmark.gpu_median_ms,
            "gpu_p95_ms": benchmark.gpu_p95_ms,
            "browser_wall_median_ms": benchmark.wall_median_ms,
            "browser_wall_p95_ms": benchmark.wall_p95_ms,
            "accepted": registration.accepted,
            "replaced": registration.replaced,
            "mean_translation": registration.mean_translation,
            "mean_iterations": registration.mean_iterations,
            "translation_rmse_px": registration.rmse(TRUTH),
        },
    }


def _native_report(
    reference: np.ndarray,
    target: np.ndarray,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    with ICGNWGPU() as engine:
        preparation = engine.set_reference(reference)
        for _ in range(warmups):
            engine.register(target)
        results = [engine.register(target) for _ in range(repeats)]
        device = engine.device_name

    registration = results[-1]
    total_times = np.asarray(
        [result.timing["totalWallMs"] for result in results], dtype=np.float64
    )
    upload_times = np.asarray(
        [result.timing["uploadWallMs"] for result in results], dtype=np.float64
    )
    execute_times = np.asarray(
        [result.timing["executeReadbackWallMs"] for result in results], dtype=np.float64
    )
    return {
        "backend": "wgpu-native",
        "device": device,
        "solver": "portable-workgroup",
        "subgroup_width": None,
        "resolution": [preparation.width, preparation.height],
        "grid": list(preparation.grid),
        "truth": list(TRUTH),
        "preparation": {
            "runtime_build_wall_ms": preparation.runtime_build_wall_ms,
            "reference_build_wall_ms": preparation.reference_build_wall_ms,
            "backend_preparation_wall_ms": preparation.preparation_wall_ms,
            "encoding_wall_ms": preparation.encoding_wall_ms,
            "python_call_wall_ms": preparation.total_wall_ms,
        },
        "optical_flow": {
            "repeats": repeats,
            "warmups": warmups,
            "full_call_median_ms": float(np.median(total_times)),
            "full_call_p95_ms": float(np.percentile(total_times, 95)),
            "upload_median_ms": float(np.median(upload_times)),
            "execute_readback_median_ms": float(np.median(execute_times)),
            "accepted": registration.accepted,
            "replaced": registration.replaced,
            "mean_translation": registration.mean_translation,
            "mean_iterations": registration.mean_iterations,
            "translation_rmse_px": registration.rmse(TRUTH),
        },
    }


def main() -> int:
    args = _arguments()
    reference, target = _make_pair(args.texture)
    if args.backend == "wgpu-native":
        report = _native_report(reference, target, args.repeats, args.warmups)
    else:
        report = _browser_report(
            args.backend, reference, target, args.repeats, args.warmups
        )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
