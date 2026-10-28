"""Compare native wgpu-py with the browser backends on preloaded NumPy frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWebGL, ICGNWGPU, RegistrationResult


def _arguments() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=module_root / "assets/frame_000.png"
    )
    parser.add_argument(
        "--target", type=Path, default=module_root / "assets/frame_010.png"
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="skip Chromium backends",
    )
    parser.add_argument(
        "--native-solver",
        choices=("auto", "portable", "subgroup"),
        default="auto",
        help="native solver selection policy",
    )
    return parser.parse_args()


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(image.convert("L"), dtype=np.uint8)


def _percentile(samples: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), percentile))


def _result_summary(
    results: list[RegistrationResult], total_key: str
) -> dict[str, Any]:
    final = results[-1]
    totals = [result.timing[total_key] for result in results]
    summary: dict[str, Any] = {
        "full_call_median_ms": _percentile(totals, 50),
        "full_call_p95_ms": _percentile(totals, 95),
        "accepted": final.accepted,
        "replaced": final.replaced,
        "mean_translation": final.mean_translation,
        "mean_iterations": final.mean_iterations,
        "truth_rmse_px": final.rmse((2.0, -1.0)),
    }
    gpu_samples = [result.timing.get("gpuMs") for result in results]
    if all(value is not None for value in gpu_samples):
        summary["gpu_median_ms"] = _percentile(
            [float(value) for value in gpu_samples if value is not None], 50
        )
    return summary


def _agreement(left: RegistrationResult, right: RegistrationResult) -> dict[str, Any]:
    joint = left.accepted_mask & right.accepted_mask
    difference = left.displacement[joint] - right.displacement[joint]
    return {
        "jointly_accepted": int(np.count_nonzero(joint)),
        "equal_statuses": int(np.count_nonzero(left.status == right.status)),
        "translation_rmse_px": float(np.sqrt(np.mean(difference * difference))),
        "translation_max_abs_px": float(np.max(np.abs(difference))),
    }


def main() -> int:
    args = _arguments()
    reference = _load_gray(args.reference)
    target = _load_gray(args.target)
    report: dict[str, Any] = {
        "resolution": [int(reference.shape[1]), int(reference.shape[0])],
        "repeats": args.repeats,
        "warmups": args.warmups,
    }

    solver_preference = {"auto": None, "portable": False, "subgroup": True}[
        args.native_solver
    ]
    with ICGNWGPU(prefer_subgroups=solver_preference) as engine:
        preparation = engine.set_reference(reference)
        for _ in range(args.warmups):
            engine.register(target)
        native_results = [engine.register(target) for _ in range(args.repeats)]
        native = native_results[-1]
        native_summary = _result_summary(native_results, "totalWallMs")
        native_summary.update(
            {
                "device": engine.device_name,
                "solver": engine.solver,
                "subgroup_width": engine.subgroup_width or None,
                "solver_tuning_ms": engine.solver_tuning_ms,
                "preparation_wall_ms": preparation.total_wall_ms,
                "upload_median_ms": _percentile(
                    [result.timing["uploadWallMs"] for result in native_results], 50
                ),
                "execute_readback_median_ms": _percentile(
                    [
                        result.timing["executeReadbackWallMs"]
                        for result in native_results
                    ],
                    50,
                ),
            }
        )
        report["wgpu-native"] = native_summary

    if not args.native_only:
        browser_results: dict[str, RegistrationResult] = {}
        for backend in ("three-tsl", "webgpu-wgsl", "gpuio"):
            with ICGNWebGL(
                backend=backend, startup_timeout=120, call_timeout=600
            ) as engine:
                preparation = engine.set_reference(reference)
                for _ in range(args.warmups):
                    engine.register(target)
                results = [engine.register(target) for _ in range(args.repeats)]
                browser_results[backend] = results[-1]
                summary = _result_summary(results, "pythonCallWallMs")
                summary.update(
                    {
                        "device": preparation.device,
                        "preparation_wall_ms": preparation.total_wall_ms,
                    }
                )
                report[backend] = summary
        report["native_vs_three_agreement"] = _agreement(
            native, browser_results["three-tsl"]
        )
        report["native_vs_shared_wgsl_agreement"] = _agreement(
            native, browser_results["webgpu-wgsl"]
        )
        report["native_vs_gpuio_agreement"] = _agreement(
            native, browser_results["gpuio"]
        )

    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
