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
DEFAULT_TRUTH = (2, -1)


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
        "--translation",
        nargs=2,
        type=int,
        default=DEFAULT_TRUTH,
        metavar=("DX", "DY"),
        help="exact integer target translation in pixels",
    )
    parser.add_argument(
        "--backend",
        choices=("three-tsl", "webgpu-wgsl", "gpuio", "wgpu-native"),
        default="three-tsl",
        help="GPU implementation to measure",
    )
    parser.add_argument(
        "--native-solver",
        choices=("auto", "portable", "subgroup"),
        default="auto",
        help="native solver selection policy",
    )
    parser.add_argument(
        "--pyramid-levels",
        type=int,
        choices=(1, 2, 3),
        help="full-pyramid depth for the webgpu-wgsl browser backend",
    )
    parser.add_argument(
        "--force-three-webgl",
        action="store_true",
        help="force Three.js to use its WebGL2 compatibility backend",
    )
    args = parser.parse_args()
    if args.pyramid_levels is not None and args.backend != "webgpu-wgsl":
        parser.error("--pyramid-levels requires --backend webgpu-wgsl")
    if args.force_three_webgl and args.backend != "three-tsl":
        parser.error("--force-three-webgl requires --backend three-tsl")
    return args


def _make_pair(
    texture_path: Path, truth: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(texture_path) as image:
        texture = np.asarray(image.convert("L"), dtype=np.uint8)
    repeat_y = (UHD_HEIGHT + texture.shape[0] - 1) // texture.shape[0]
    repeat_x = (UHD_WIDTH + texture.shape[1] - 1) // texture.shape[1]
    reference = np.tile(texture, (repeat_y, repeat_x))[:UHD_HEIGHT, :UHD_WIDTH]
    target = np.roll(reference, shift=(truth[1], truth[0]), axis=(0, 1))
    return reference, target


def _browser_report(
    backend: str,
    reference: np.ndarray,
    target: np.ndarray,
    repeats: int,
    warmups: int,
    pyramid_levels: int | None,
    force_three_webgl: bool,
    truth: tuple[int, int],
) -> dict[str, Any]:
    with ICGNWebGL(
        backend=backend,
        pyramid_levels=pyramid_levels,
        force_three_webgl=force_three_webgl,
        startup_timeout=120,
        call_timeout=600,
    ) as engine:
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
        "truth": list(truth),
        "config": preparation.config,
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
            "translation_rmse_px": registration.rmse(truth),
        },
    }


def _native_report(
    reference: np.ndarray,
    target: np.ndarray,
    repeats: int,
    warmups: int,
    *,
    prefer_subgroups: bool | None,
    truth: tuple[int, int],
) -> dict[str, Any]:
    with ICGNWGPU(prefer_subgroups=prefer_subgroups) as engine:
        preparation = engine.set_reference(reference)
        for _ in range(warmups):
            engine.register(target)
        results = [engine.register(target) for _ in range(repeats)]
        device = engine.device_name
        solver = engine.solver
        subgroup_width = engine.subgroup_width
        solver_tuning_ms = engine.solver_tuning_ms

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
        "solver": solver,
        "subgroup_width": subgroup_width or None,
        "solver_tuning_ms": solver_tuning_ms,
        "resolution": [preparation.width, preparation.height],
        "grid": list(preparation.grid),
        "truth": list(truth),
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
            "translation_rmse_px": registration.rmse(truth),
        },
    }


def main() -> int:
    args = _arguments()
    truth = tuple(args.translation)
    reference, target = _make_pair(args.texture, truth)
    if args.backend == "wgpu-native":
        report = _native_report(
            reference,
            target,
            args.repeats,
            args.warmups,
            prefer_subgroups={"auto": None, "portable": False, "subgroup": True}[
                args.native_solver
            ],
            truth=truth,
        )
    else:
        report = _browser_report(
            args.backend,
            reference,
            target,
            args.repeats,
            args.warmups,
            args.pyramid_levels,
            args.force_three_webgl,
            truth,
        )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
