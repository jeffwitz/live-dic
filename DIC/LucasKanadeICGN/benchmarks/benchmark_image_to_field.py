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
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
from lucas_kanade_icgn import ICGNWebGL, ICGNWGPUFullPyramid
from PIL import Image

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
    parser.add_argument(
        "--fixture-set",
        choices=("timing", "uhd-tensile"),
        default="timing",
        help="timing translations or original UHD reference/frame_040 affine truth",
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--campaigns", type=int, default=1)
    parser.add_argument(
        "--roi-mask",
        type=Path,
        help="boolean .npy reference mask; no implicit thresholding",
    )
    parser.add_argument("--roi-policy", choices=("full", "partial"), default="full")
    parser.add_argument(
        "--roi-truncation", choices=("none", "visibility"), default="none"
    )
    parser.add_argument(
        "--device-type",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="native adapter type; cpu requires an installed software WebGPU driver",
    )
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
        "--outlier-policy",
        choices=("replace", "flag", "disabled", "median"),
        default="replace",
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
    if args.roi_mask is not None and any(
        backend != "native-pyramid" for backend in args.backends
    ):
        parser.error("ROI requires --backend native-pyramid")
    if args.roi_truncation != "none" and (
        args.roi_mask is None or args.roi_policy != "partial"
    ):
        parser.error("--roi-truncation requires --roi-policy partial and --roi-mask")
    if args.device_type != "auto" and any(
        backend != "native-pyramid" for backend in args.backends
    ):
        parser.error("--device-type requires --backend native-pyramid")
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
    outlier_policy: str,
    subset_size: int,
    subset_stride: int | None,
    spatial_weighting: str,
    gaussian_sigma: float | None,
    device_type: str = "auto",
) -> AbstractContextManager[Any]:
    if backend == "native-pyramid":
        return ICGNWGPUFullPyramid(
            device_type=device_type,
            pyramid_levels=pyramid_levels,
            interpolation=interpolation,
            motion_model=motion_model,
            outlier_policy=outlier_policy,
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
        outlier_policy=outlier_policy,
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


def _support_quality(results: list[Any], truths: list[Any]) -> dict:
    """Evaluate geometry populations after timing, never in the live solver."""
    first = results[0]
    roi = getattr(first, "roi", None)
    selected = np.ones(first.status.shape, bool) if roi is None else roi.selected
    partial = np.zeros(first.status.shape, bool)
    indices = None if roi is None else getattr(roi, "partial_indices", None)
    if indices is not None:
        partial.ravel()[indices] = True
    output = {}
    for name, population in (("complete", selected & ~partial), ("partial", partial)):
        counts, rmses, p95s, gradient_errors = [], [], [], []
        for result, truth in zip(results, truths, strict=True):
            mask = result.accepted_mask & population
            counts.append(int(mask.sum()))
            if not mask.any():
                continue
            if isinstance(truth, dict):
                y, x = np.indices(result.status.shape)
                x = x * result.subset_stride + (result.subset_size - 1) / 2
                y = y * result.subset_stride + (result.subset_size - 1) / 2
                expected = np.stack(
                    (truth["ex"] * (x - 1920), truth["ey"] * (y - 1080)), axis=-1
                )[mask]
                gradients = [truth["ex"], 0.0, 0.0, truth["ey"]]
            else:
                expected, gradients = np.asarray(truth), [0.0, 0.0, 0.0, 0.0]
            distance = np.linalg.norm(result.displacement[mask] - expected, axis=-1)
            rmses.append(float(np.sqrt(np.mean(distance**2))))
            p95s.append(float(np.percentile(distance, 95)))
            if result.config.get("motionModel", "affine") == "affine":
                error = result.parameters[..., (1, 2, 4, 5)][mask] - gradients
                gradient_errors.append(float(np.sqrt(np.mean(error**2))))
        count = int(population.sum())
        output[name] = {
            "selected": count,
            "accepted_min": min(counts),
            "accepted_fraction_min": min(counts) / count if count else None,
            "displacement_rmse_max_px": max(rmses) if rmses else None,
            "displacement_p95_max_px": max(p95s) if p95s else None,
            "gradient_component_rmse_max": max(gradient_errors)
            if gradient_errors
            else None,
        }
    return output


def _measure(
    engine: Any,
    targets: list[tuple[np.ndarray, tuple[float, float], str]],
    *,
    frames: int,
    warmups: int,
    pipeline_depth: int,
    use_stream: bool,
) -> dict[str, Any]:
    if use_stream:
        for _ in engine.register_stream(
            (targets[index % len(targets)][0] for index in range(warmups)),
            pipeline_depth=pipeline_depth,
        ):
            pass
    else:
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
            samples.append(
                float(
                    result.timing.get("resultAgeWallMs", result.timing["totalWallMs"])
                )
            )
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
    rmses = []
    for result, truth in zip(results, truths, strict=True):
        if isinstance(truth, dict):
            y, x = np.indices(result.status.shape)
            x = x * result.subset_stride + (result.subset_size - 1) / 2
            y = y * result.subset_stride + (result.subset_size - 1) / 2
            expected = np.stack(
                (truth["ex"] * (x - 1920), truth["ey"] * (y - 1080)), axis=-1
            )
            error = (
                result.displacement[result.accepted_mask]
                - expected[result.accepted_mask]
            )
            rmses.append(
                float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))
                if error.size
                else None
            )
        else:
            rmses.append(result.rmse(truth))
    accepted = [result.accepted for result in results]
    finite_rmses = [
        value for value in rmses if value is not None and np.isfinite(value)
    ]
    return {
        "boundary": "resident uint8 NumPy image to materialized NumPy parameter field",
        "execution": execution,
        "pipeline_depth": pipeline_depth,
        "frames": frames,
        "warmups": warmups,
        "latency_median_ms": float(np.median(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_p99_ms": float(np.percentile(samples, 99)),
        "latency_samples_ms": samples,
        "completion_period_samples_ms": periods.tolist(),
        "populations": getattr(results[0], "populations", None),
        "config": dict(results[0].config),
        "support_quality": _support_quality(results, truths),
        "iterations_histogram": {
            str(int(key)): int(count)
            for key, count in zip(
                *np.unique(
                    np.concatenate(
                        [result.iterations[result.accepted_mask] for result in results]
                    ),
                    return_counts=True,
                ),
                strict=True,
            )
        },
        "stream_period_ms": stream_wall_ms / frames,
        "throughput_fps": frames * 1_000.0 / stream_wall_ms,
        "completion_period_median_ms": (
            float(np.median(periods)) if periods.size else float(samples[0])
        ),
        "field_shape": list(shape),
        "field_bytes": int(results[0].parameters.nbytes),
        "accepted_min": min(accepted),
        "accepted_max": max(accepted),
        "rmse_median_px": float(np.median(finite_rmses)) if finite_rmses else None,
        "rmse_max_px": float(np.max(finite_rmses)) if finite_rmses else None,
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
        "latency_p99_ms",
        "stream_period_ms",
        "throughput_fps",
        "completion_period_median_ms",
        "rmse_median_px",
    )
    for key in median_keys:
        values = [campaign[key] for campaign in campaigns if campaign[key] is not None]
        result[key] = float(np.median(values)) if values else None
    result["accepted_min"] = min(campaign["accepted_min"] for campaign in campaigns)
    result["accepted_max"] = max(campaign["accepted_max"] for campaign in campaigns)
    errors = [
        campaign["rmse_max_px"]
        for campaign in campaigns
        if campaign["rmse_max_px"] is not None
    ]
    result["rmse_max_px"] = max(errors) if errors else None
    result["raw_campaigns"] = campaigns
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
    fixture_names = [filename for filename, _ in TARGET_FIXTURES]
    if args.fixture_set == "timing":
        reference, targets = _fixtures(args.assets, args.width, args.height)
    else:
        import csv

        if (args.width, args.height) != (3840, 2160):
            raise ValueError(
                "Original UHD tensile images require 3840x2160; no resizing is applied"
            )
        assets = args.assets / "uhd_tensile"
        with (assets / "machine_truth.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            state = next(
                row for row in csv.DictReader(stream) if int(row["frame"]) == 40
            )
        with Image.open(assets / "reference.png") as image:
            reference = np.array(image, dtype=np.uint8)
        with Image.open(assets / "frame_040.png") as image:
            target = np.array(image, dtype=np.uint8)
        targets = [
            (reference, (0.0, 0.0), hashlib.sha256(reference).hexdigest()),
            (
                target,
                {
                    "ex": float(state["stretch_x"]) - 1,
                    "ey": float(state["stretch_y"]) - 1,
                },
                hashlib.sha256(target).hexdigest(),
            ),
        ]
        fixture_names = ["reference.png", "frame_040.png"]
    mask = None if args.roi_mask is None else np.load(args.roi_mask, allow_pickle=False)
    measurements = {}
    for backend in args.backends:
        with _engine(
            backend,
            pyramid_levels=args.pyramid_levels,
            interpolation=args.interpolation,
            motion_model=args.motion_model,
            outlier_policy=args.outlier_policy,
            subset_size=args.subset_size,
            subset_stride=args.subset_stride,
            spatial_weighting=args.spatial_weighting,
            gaussian_sigma=args.gaussian_sigma,
            device_type=args.device_type,
        ) as engine:
            preparation = (
                engine.set_reference(
                    reference,
                    roi_mask=mask,
                    roi_policy=args.roi_policy,
                    **(
                        {"roi_truncation": args.roi_truncation}
                        if args.roi_truncation != "none"
                        else {}
                    ),
                )
                if mask is not None
                else engine.set_reference(reference)
            )
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
                "adapter_info": dict(getattr(engine, "adapter_info", {})),
                "execution_device_type": getattr(
                    engine, "execution_device_type", "browser"
                ),
                "preparation_ms": preparation.total_wall_ms,
                **_aggregate_campaigns(campaigns),
            }

    import lucas_kanade_icgn

    source_dir = Path(lucas_kanade_icgn.__file__).parent
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source_dir,
            text=True,
        ).strip()
    )
    print(
        json.dumps(
            {
                "commit": revision,
                "source_hashes": {
                    str(path.relative_to(source_dir)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(
                        [*source_dir.glob("*.py"), *source_dir.glob("shaders/*.wgsl")]
                    )
                },
                "wgpu_version": importlib.metadata.version("wgpu"),
                "reference_hash": hashlib.sha256(reference).hexdigest(),
                "git_dirty": dirty,
                "command": sys.argv,
                "python": sys.version,
                "platform": platform.platform(),
                "roi_mask_file": None if args.roi_mask is None else str(args.roi_mask),
                "roi_policy": args.roi_policy,
                "roi_truncation": args.roi_truncation,
                "resolution": [args.width, args.height],
                "input": "pre-acquired contiguous uint8 NumPy grayscale frames",
                "reference_preparation_included": False,
                "camera_or_decoder_included": False,
                "fixture_hashes": {
                    filename: digest
                    for filename, (_, _, digest) in zip(
                        fixture_names, targets, strict=True
                    )
                },
                "fixture_set": args.fixture_set,
                "pyramid_levels": args.pyramid_levels,
                "subset_size": args.subset_size,
                "subset_stride": args.subset_stride or args.subset_size,
                "interpolation": args.interpolation,
                "motion_model": args.motion_model,
                "outlier_policy": args.outlier_policy,
                "spatial_weighting": args.spatial_weighting,
                "gaussian_sigma": (
                    args.gaussian_sigma
                    if args.gaussian_sigma is not None
                    else 0.25 * args.subset_size
                ),
                "pipeline_depth": args.pipeline_depth,
                "device_type": args.device_type,
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
