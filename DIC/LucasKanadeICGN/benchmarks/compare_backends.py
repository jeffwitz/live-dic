"""Compare GPU-IO and Three.js/TSL on the same affine IC-GN image pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lucas_kanade_icgn import ICGNWebGL


def _arguments() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=module_root / "assets/frame_000.png")
    parser.add_argument("--target", type=Path, default=module_root / "assets/frame_010.png")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    return parser.parse_args()


def _run_backend(
    backend: str,
    reference: Path,
    target: Path,
    repeats: int,
    warmups: int,
) -> tuple[dict[str, Any], Any]:
    with ICGNWebGL(backend=backend, startup_timeout=120, call_timeout=600) as engine:
        info = engine.info()
        benchmark = engine.benchmark(
            reference,
            target,
            repeats=repeats,
            warmups=warmups,
        )
    result = benchmark.registration
    summary = {
        "device": info.device,
        "gpu_median_ms": benchmark.gpu_median_ms,
        "gpu_p95_ms": benchmark.gpu_p95_ms,
        "wall_median_ms": benchmark.wall_median_ms,
        "wall_p95_ms": benchmark.wall_p95_ms,
        "accepted": result.accepted,
        "replaced": result.replaced,
        "mean_translation": result.mean_translation,
        "mean_iterations": result.mean_iterations,
        "rmse_truth": result.rmse((2.0, -1.0)),
    }
    return summary, result


def main() -> int:
    args = _arguments()
    report: dict[str, Any] = {}
    results = {}
    for backend in ("gpuio", "three-tsl"):
        report[backend], results[backend] = _run_backend(
            backend,
            args.reference,
            args.target,
            args.repeats,
            args.warmups,
        )

    gpuio = results["gpuio"]
    three = results["three-tsl"]
    jointly_accepted = (gpuio.status >= 0.5) & (three.status >= 0.5)
    difference = three.parameters - gpuio.parameters
    translation_difference = difference[..., (0, 3)][jointly_accepted]
    parameter_difference = difference[jointly_accepted]
    report["agreement"] = {
        "jointly_accepted": int(jointly_accepted.sum()),
        "status_equal": int(np.count_nonzero(gpuio.status == three.status)),
        "translation_rmse_px": float(np.sqrt(np.mean(translation_difference**2))),
        "translation_max_abs_px": float(np.max(np.abs(translation_difference))),
        "parameter_rmse": [
            float(np.sqrt(np.mean(parameter_difference[:, index] ** 2))) for index in range(6)
        ],
    }
    report["three_to_gpuio_gpu_ratio"] = (
        report["three-tsl"]["gpu_median_ms"] / report["gpuio"]["gpu_median_ms"]
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
