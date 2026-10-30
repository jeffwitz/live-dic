"""Native-UHD plate: ROI comparison against the original SYSCL analytic truth.

Only images and the optional reference mask enter registration. Mechanics is
used exclusively for offline scoring. Plotting is a separate, subsequent step.
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

import lucas_kanade_icgn
from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/tests"))
from plate_hole_reference import (
    CX,
    CY,
    RADIUS,
    displacement,
    grid,
    hole_mask,
    strain,
    strain_from_parameters,
    von_mises_strain,
)

VARIANTS = ("none", "full", "partial", "visibility")


def image(path):
    with Image.open(path) as data:
        result = np.array(data, dtype=np.uint8)
    if result.shape != (2160, 3840):
        raise ValueError(f"Native UHD required: {path}")
    return result


def score(result, truth, population):
    """Scores retain the fixed population and report its missing measurements."""
    selected = (
        np.ones(population.shape, bool)
        if result.roi_selected is None
        else result.roi_selected
    )
    accepted = population & selected & result.accepted_mask
    count, requested, valid = (
        int(population.sum()),
        int((population & selected).sum()),
        int(accepted.sum()),
    )
    output = {
        "population": count,
        "requested": requested,
        "accepted": valid,
        "coverage_physical": valid / count if count else None,
        "coverage_requested": valid / requested if requested else None,
    }
    if not valid:
        return {
            **output,
            **dict.fromkeys(
                (
                    "displacement_rmse_px",
                    "strain_rmse_ue",
                    "vm_rmse_ue",
                    "vm_bias_ue",
                    "vm_p95_abs_ue",
                    "vm_max_ue",
                    "truth_vm_max_on_accepted_ue",
                )
            ),
        }
    d = result.displacement[accepted] - truth["displacement"][accepted]
    measured = strain_from_parameters(result.parameters)[accepted]
    error = (von_mises_strain(measured) - truth["vm"][accepted]) * 1e6
    return {
        **output,
        "displacement_rmse_px": float(np.sqrt(np.mean(np.sum(d * d, axis=-1)))),
        "strain_rmse_ue": float(
            np.sqrt(np.mean((measured - truth["strain"][accepted]) ** 2)) * 1e6
        ),
        "vm_rmse_ue": float(np.sqrt(np.mean(error**2))),
        "vm_bias_ue": float(np.mean(error)),
        "vm_p95_abs_ue": float(np.percentile(np.abs(error), 95)),
        "vm_max_ue": float(np.max(von_mises_strain(measured)) * 1e6),
        "truth_vm_max_on_accepted_ue": float(truth["vm"][accepted].max() * 1e6),
    }


def measure_stream(engine, targets, frames, campaigns, depth):
    """Image-to-field stream timing, with no mechanics/figure work in its window."""
    output = []
    for _ in range(campaigns):
        for _ in engine.register_stream(
            (targets[k % len(targets)] for k in range(12)), pipeline_depth=depth
        ):
            pass
        ages, completed = [], []
        started = time.perf_counter()
        for result in engine.register_stream(
            (targets[k % len(targets)] for k in range(frames)), pipeline_depth=depth
        ):
            completed.append(time.perf_counter())
            ages.append(float(result.timing["resultAgeWallMs"]))
        wall_ms = (time.perf_counter() - started) * 1000
        output.append(
            {
                "frames": frames,
                "warmups": 12,
                "pipeline_depth": depth,
                "throughput_fps": frames * 1000 / wall_ms,
                "age_median_ms": float(np.median(ages)),
                "age_p95_ms": float(np.percentile(ages, 95)),
                "age_p99_ms": float(np.percentile(ages, 99)),
                "age_samples_ms": ages,
                "completion_period_samples_ms": (np.diff(completed) * 1000).tolist(),
            }
        )
    return {
        "campaigns": output,
        **{
            key: float(np.median([c[key] for c in output]))
            for key in ("throughput_fps", "age_median_ms", "age_p95_ms", "age_p99_ms")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/plate-hole-roi-20261030",
    )
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--campaigns", type=int, default=3)
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    if min(args.frames, args.campaigns, args.pipeline_depth) < 1:
        parser.error("frames, campaigns and depth must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = args.output / "fields"
    fields.mkdir(exist_ok=True)
    assets = ROOT / "assets/uhd_plate_hole"
    manifest = json.loads((assets / "manifest.json").read_text())
    for row in manifest["images"]:
        if (
            hashlib.sha256((assets / row["file"]).read_bytes()).hexdigest()
            != row["file_sha256"]
        ):
            raise ValueError(f"Fixture hash mismatch: {row['file']}")
    reference = image(assets / "reference.png")
    mask = hole_mask()
    np.savez_compressed(args.output / "reference_roi.npz", roi_mask=mask)
    x, y = grid()
    radius = np.hypot(x - CX, y - CY)
    physical = radius >= RADIUS
    interior = physical & (x >= 32) & (x < 3840 - 32) & (y >= 32) & (y < 2160 - 32)
    regions = {
        "physical_interior": interior,
        "near_hole_0_176": interior & (radius < 288),
    }
    for low, high in ((0, 16), (16, 32), (32, 64), (64, 176)):
        regions[f"edge_{low}_{high}"] = (
            interior & (radius >= RADIUS + low) & (radius < RADIUS + high)
        )
    targets, truths = {}, {}
    for row in manifest["images"][1:]:
        frame, p, h = row["frame"], row["primary_MPa"], row["secondary_MPa"]
        targets[frame] = image(assets / row["file"])
        u, e = np.zeros(x.shape + (2,)), np.zeros(x.shape + (3,))
        u[physical] = displacement(x[physical], y[physical], p, h)
        e[physical] = strain(x[physical], y[physical], p, h)
        truths[frame] = {"displacement": u, "strain": e, "vm": von_mises_strain(e)}
        np.savez_compressed(
            args.output / f"truth_{frame:03d}.npz",
            x=x,
            y=y,
            physical=physical,
            **truths[frame],
        )
    records, results = [], {}
    for policy in ("disabled", "median"):
        for variant in VARIANTS:
            with ICGNWGPUFullPyramid(
                device_type="gpu",
                motion_model="affine",
                subset_size=16,
                subset_stride=16,
                interpolation="bspline",
                outlier_policy=policy,
            ) as engine:
                kwargs = (
                    {}
                    if variant == "none"
                    else {
                        "roi_mask": mask,
                        "roi_policy": "full" if variant == "full" else "partial",
                        "roi_truncation": "visibility"
                        if variant == "visibility"
                        else "none",
                    }
                )
                info = engine.set_reference(reference, **kwargs)
                snapshots = []
                for frame, target in targets.items():
                    result = engine.register(target)
                    if not np.isfinite(result.parameters).all():
                        raise RuntimeError("Non-finite field")
                    result.save_npz(fields / f"{frame:03d}-{policy}-{variant}.npz")
                    results[(frame, policy, variant)] = result
                    snapshots.append(
                        {
                            "frame": frame,
                            "populations": result.populations,
                            "regions": {
                                name: score(result, truths[frame], pop)
                                for name, pop in regions.items()
                            },
                        }
                    )
                timing = (
                    None
                    if args.skip_timing
                    else measure_stream(
                        engine,
                        [reference, *targets.values()],
                        args.frames,
                        args.campaigns,
                        args.pipeline_depth,
                    )
                )
                records.append(
                    {
                        "mask": variant,
                        "outlier_policy": policy,
                        "adapter": dict(engine.adapter_info),
                        "preparation_ms": info.total_wall_ms,
                        "config": results[(95, policy, variant)].config,
                        "snapshots": snapshots,
                        "timing": timing,
                    }
                )
            print(f"Completed {policy}/{variant}", flush=True)
    # Common accepted support is fixed across all eight mask/filter variants.
    # New partial centres are assessed separately; full legitimately excludes them.
    full_selection = results[(95, "median", "full")].roi_selected
    partial_selection = results[(95, "median", "partial")].roi_selected
    added = partial_selection & ~full_selection & physical
    rows = []
    for frame in targets:
        common = interior.copy()
        for (f, _, _), result in results.items():
            if f == frame:
                common &= result.accepted_mask
        for record in records:
            policy, variant = record["outlier_policy"], record["mask"]
            result = results[(frame, policy, variant)]
            snapshot = next(s for s in record["snapshots"] if s["frame"] == frame)
            snapshot["regions"]["common_near_hole"] = score(
                result, truths[frame], common & (radius < 288)
            )
            snapshot["regions"]["added_partial_centres"] = score(
                result, truths[frame], added
            )
            for name, metrics in snapshot["regions"].items():
                rows.append(
                    {
                        "frame": frame,
                        "filter": policy,
                        "mask": variant,
                        "region": name,
                        **metrics,
                    }
                )
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source = Path(lucas_kanade_icgn.__file__).parent
    report = {
        "command": sys.argv,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "wgpu_version": importlib.metadata.version("wgpu"),
        "resolution": [3840, 2160],
        "equivalent": "strain, ezz=0; not stress",
        "coordinates": "reference pixel centres at integers; live-dic subsets at 7.5+16*i",
        "manifest": manifest,
        "records": records,
        "added_partial_centres": int(added.sum()),
        "source_hashes": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*source.glob("*.py"), *source.glob("shaders/*.wgsl")])
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "truth_helper_sha256": hashlib.sha256(
            (ROOT / "python/tests/plate_hole_reference.py").read_bytes()
        ).hexdigest(),
        "timing_boundary": "resident NumPy image to complete NumPy field; VM/strain scoring and plotting excluded",
        "notes": [
            "Pointwise analytical strain differs from the historical SYSCL 5x5 displacement-regression operator.",
            "Own-support maps and fixed common-support errors are both reported. Exclusions are never filled for plotting.",
            "No material law, load or analytical field enters registration.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
