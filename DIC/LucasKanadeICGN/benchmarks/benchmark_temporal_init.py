"""Native UHD temporal capture range and resident NumPy image-to-field timing.

Sequence rendering and scientific scoring never execute in timed loops.
Every frame traverses every configured pyramid level, without conditional retry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/tests"))
from temporal_cases import FIXTURES, tensile_stretches, warp

VARIANTS = {
    "zero-2": {"initialization": "zero", "pyramid_levels": 2},
    "temporal-1": {"initialization": "temporal", "pyramid_levels": 1},
    "temporal-2": {"initialization": "temporal", "pyramid_levels": 2},
}


def engine_for(variant, model, outlier):
    return ICGNWGPUFullPyramid(
        device_type="gpu",
        motion_model=model,
        outlier_policy=outlier,
        **VARIANTS[variant],
    )


def scores(result, stretch, translation):
    y, x = np.indices(result.status.shape) * 16 + 7.5
    fixed = (x >= 224) & (x < 3616) & (y >= 64) & (y < 2096)
    truth_d = np.stack(
        (
            (x - 1920) * (stretch[0] - 1) + translation[0],
            (y - 1080) * (stretch[1] - 1) + translation[1],
        ),
        -1,
    )
    strain = np.stack(
        (
            result.parameters[..., 1],
            result.parameters[..., 5],
            0.5 * (result.parameters[..., 2] + result.parameters[..., 4]),
        ),
        -1,
    )
    truth_e = np.array((stretch[0] - 1, stretch[1] - 1, 0))
    output = {}
    for name, population in (("fixed", fixed), ("own", np.ones_like(fixed))):
        accepted = population & result.accepted_mask
        n = int(accepted.sum())
        requested = int(population.sum())
        output.update(
            {
                f"{name}_population": requested,
                f"{name}_accepted": n,
                f"{name}_coverage": n / requested,
                f"{name}_displacement_rmse_px": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (result.displacement[accepted] - truth_d[accepted]) ** 2,
                                axis=-1,
                            )
                        )
                    )
                )
                if n
                else None,
                f"{name}_strain_rmse": float(np.sqrt(np.mean((strain[accepted] - truth_e) ** 2)))
                if n
                else None,
            }
        )
    return output


def capture(reference, targets, stretches, outlier):
    rows = []
    for variant in VARIANTS:
        print(f"Capture {variant}: {len(targets)} native UHD frames", flush=True)
        with engine_for(variant, "affine", outlier) as engine:
            preparation = engine.set_reference(reference)
            fields = list(engine.register_stream(targets, pipeline_depth=2))
        for i, result in enumerate(fields):
            rows.append(
                {
                    "variant": variant,
                    "frame": i,
                    "stretch_x": float(stretches[i, 0]),
                    "stretch_y": float(stretches[i, 1]),
                    "mean_iterations": result.mean_iterations,
                    "temporal_seeded_coarse": result.timing.get("temporallySeededCoarseSubsets", 0),
                    "preparation_ms": preparation.total_wall_ms,
                    **scores(result, stretches[i], (0, 0)),
                }
            )
    return rows


def timing(reference, targets, model, outlier, campaigns):
    rows = []
    for repeat in range(campaigns):
        for variant in list(VARIANTS)[:: (-1 if repeat % 2 else 1)]:
            for depth in (1, 2):
                print(
                    f"Timing {model}/{variant}/depth{depth}, campaign {repeat + 1}",
                    flush=True,
                )
                with engine_for(variant, model, outlier) as engine:
                    preparation = engine.set_reference(reference)
                    adapter = dict(engine.adapter_info)
                    # Warm every real slot/path, then restart the chronological
                    # trajectory from its beginning before measuring frame 20+.
                    list(engine.register_stream(targets[:20], pipeline_depth=depth))
                    engine.reset_temporal_state()
                    list(engine.register_stream(targets[:20], pipeline_depth=depth))
                    fields, completed = [], []
                    start = time.perf_counter()
                    for result in engine.register_stream(targets[20:], pipeline_depth=depth):
                        fields.append(result)
                        completed.append(time.perf_counter())
                    wall_ms = (time.perf_counter() - start) * 1000
                # All reductions, scoring and serialization happen after timing.
                ages = [r.timing["resultAgeWallMs"] for r in fields]
                iterations = np.concatenate([r.iterations[r.accepted_mask] for r in fields])
                stretches = tensile_stretches() if model == "affine" else None
                quality = [
                    {
                        "frame": i,
                        **scores(
                            r,
                            stretches[i] if stretches is not None else (1, 1),
                            (0, 0) if model == "affine" else (i * 0.2, -i * 0.1),
                        ),
                    }
                    for i, r in enumerate(fields, start=20)
                ]
                row = {
                    "model": model,
                    "variant": variant,
                    "pipeline_depth": depth,
                    "campaign": repeat,
                    "frames": len(fields),
                    "warmups": 20,
                    "preparation_ms": preparation.total_wall_ms,
                    "throughput_fps": len(fields) * 1000 / wall_ms,
                    "age_median_ms": float(np.median(ages)),
                    "age_p95_ms": float(np.percentile(ages, 95)),
                    "age_p99_ms": float(np.percentile(ages, 99)),
                    "age_samples_ms": ages,
                    "completion_period_samples_ms": (np.diff(completed) * 1000).tolist(),
                    "mean_iterations": float(np.mean(iterations)) if iterations.size else None,
                    "iteration_histogram": np.bincount(
                        iterations.astype(int), minlength=31
                    ).tolist(),
                    "accepted_per_frame": [r.accepted for r in fields],
                    "adapter": adapter,
                    "quality_per_frame": quality,
                }
                print(
                    f"  {row['throughput_fps']:.1f} FPS, age p95 {row['age_p95_ms']:.2f} ms",
                    flush=True,
                )
                rows.append(row)
    return rows


def corruption(reference, targets, stretches, baseline, outlier):
    index = 100
    with engine_for("temporal-2", "affine", outlier) as engine:
        engine.set_reference(reference)
        list(engine.register_stream(targets[:index], pipeline_depth=2))
        sequence = [np.zeros_like(reference), *targets[index + 1 : index + 16]]
        fields = list(engine.register_stream(sequence, pipeline_depth=2))
    rows = [
        {
            "frame": index + i,
            "accepted_updates": r.timing["temporalAcceptedUpdates"],
            **scores(r, stretches[index + i], (0, 0)),
        }
        for i, r in enumerate(fields)
    ]
    control = {r["frame"]: r for r in baseline if r["variant"] == "temporal-2"}
    recovery = None
    for row in rows[1:]:
        expected = control[row["frame"]]
        if (
            row["fixed_coverage"] >= expected["fixed_coverage"] - 0.01
            and row["fixed_displacement_rmse_px"] is not None
            and row["fixed_displacement_rmse_px"] < 0.05
        ):
            recovery = row["frame"] - index
            break
    return {
        "injected_frame": index,
        "frames_until_recovery": recovery,
        "frames": rows,
        "criterion": "coverage within 1 percentage point of control and displacement RMSE <0.05 px",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/temporal-init-20261030",
    )
    parser.add_argument("--campaigns", type=int, default=3)
    parser.add_argument(
        "--outlier-policy", choices=("median", "replace", "disabled"), default="median"
    )
    args = parser.parse_args()
    if args.campaigns < 1:
        parser.error("campaigns must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with Image.open(FIXTURES / "reference.png") as image:
        reference = np.array(image, dtype=np.uint8)
    assert reference.shape == (2160, 3840)
    stretches = tensile_stretches()
    print(
        "Rendering 200 native UHD tensile images with float64 inverse bilinear sampling",
        flush=True,
    )
    targets = [warp(reference, stretch, center=(1920, 1080)) for stretch in stretches]
    baseline = capture(reference, targets, stretches, args.outlier_policy)
    timing_rows = timing(reference, targets, "affine", args.outlier_policy, args.campaigns)
    robust = corruption(reference, targets, stretches, baseline, args.outlier_policy)
    target_hashes = [hashlib.sha256(t).hexdigest() for t in targets]
    # Translation is qualified on pure translations, never on tensile strain.
    del targets
    print("Rendering 200 pure-translation native UHD images", flush=True)
    targets = [warp(reference, translation=(i * 0.2, -i * 0.1)) for i in range(200)]
    timing_rows.extend(
        timing(reference, targets, "translation", args.outlier_policy, args.campaigns)
    )
    with engine_for("temporal-2", "translation", args.outlier_policy) as engine:
        engine.set_reference(reference)
        fields = list(engine.register_stream(targets, pipeline_depth=2))
    translation_quality = [
        {"frame": i, **scores(r, (1, 1), (i * 0.2, -i * 0.1))} for i, r in enumerate(fields)
    ]
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        ),
        "command": [sys.executable, *sys.argv],
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "wgpu", "Pillow")
        },
        "reference_sha256": hashlib.sha256((FIXTURES / "reference.png").read_bytes()).hexdigest(),
        "truth_sha256": hashlib.sha256((FIXTURES / "machine_truth.csv").read_bytes()).hexdigest(),
        "tensile_target_pixel_sha256": target_hashes,
        "translation_target_pixel_sha256": [hashlib.sha256(t).hexdigest() for t in targets],
        "configuration": {
            "subset_size": 16,
            "subset_stride": 16,
            "interpolation": "bspline",
            "spatial_weighting": "uniform",
            "outlier_policy": args.outlier_policy,
            "variants": VARIANTS,
            "iteration_schedules": {"1": [30], "2": [8, 16]},
        },
        "timing_boundary": "resident uint8 NumPy target -> materialized NumPy field and statistics; no rendering/scoring",
        "formation": warp.__doc__,
        "capture": baseline,
        "timing": timing_rows,
        "corruption": robust,
        "translation_quality": translation_quality,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(baseline[0]))
        writer.writeheader()
        writer.writerows(baseline)
    print(f"Archived {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
