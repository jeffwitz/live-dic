"""Compare native wgpu-py with the browser backends on preloaded NumPy frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from lucas_kanade_icgn import ICGNWGPU, ICGNWebGL, RegistrationResult
from PIL import Image


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
    parser.add_argument("--output", type=Path, help="write the complete JSON report")
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
    parser.add_argument(
        "--webgpu-wgsl-pyramid-levels",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help=(
            "raw browser WebGPU pyramid depth; one scale is the default here "
            "to preserve the historical like-for-like comparison"
        ),
    )
    return parser.parse_args()


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(image.convert("L"), dtype=np.uint8)


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _git_commit(repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _git_dirty(repository: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
        "full_call_samples_ms": totals,
        "accepted": final.accepted,
        "replaced": final.replaced,
        "configuration": dict(final.config),
        "mean_translation": final.mean_translation,
        "mean_iterations": final.mean_iterations,
        "truth_rmse_px": final.rmse((2.0, -1.0)),
    }
    gpu_samples = [result.timing.get("gpuMs") for result in results]
    if all(value is not None for value in gpu_samples):
        finite_gpu_samples = [
            float(value) for value in gpu_samples if value is not None
        ]
        summary["gpu_median_ms"] = _percentile(finite_gpu_samples, 50)
        summary["gpu_samples_ms"] = finite_gpu_samples
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
    module_root = Path(__file__).resolve().parents[1]
    repository_root = module_root.parents[1]
    reference = _load_gray(args.reference)
    target = _load_gray(args.target)
    report: dict[str, Any] = {
        "provenance": {
            "git_commit": _git_commit(repository_root),
            "git_dirty": _git_dirty(repository_root),
            "command": [sys.executable, *sys.argv],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "wgpu": _installed_version("wgpu"),
            "dic_rt": _installed_version("lucas-kanade-icgn-webgl"),
        },
        "inputs": {
            "reference": str(args.reference.resolve()),
            "reference_sha256": _sha256(args.reference),
            "target": str(args.target.resolve()),
            "target_sha256": _sha256(args.target),
        },
        "resolution": [int(reference.shape[1]), int(reference.shape[0])],
        "repeats": args.repeats,
        "warmups": args.warmups,
        "webgpu_wgsl_pyramid_levels": args.webgpu_wgsl_pyramid_levels,
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
                "solver_validation": engine.solver_validation,
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
            backend_options = (
                {"pyramid_levels": args.webgpu_wgsl_pyramid_levels}
                if backend == "webgpu-wgsl"
                else {}
            )
            with ICGNWebGL(
                backend=backend,
                startup_timeout=120,
                call_timeout=600,
                **backend_options,
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

    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
