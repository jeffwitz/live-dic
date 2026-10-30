"""Numerical-only check of rejected policies against the final runtime snapshot."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import lucas_kanade_icgn
from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "benchmarks"))
    sys.path.insert(0, str(ROOT / "python/tests"))
    from temporal_cases import tensile_stretches

    from benchmark_latency_reduction import options_for
    from benchmark_temporal_init import scores

    reference = np.array(Image.open(ROOT / "assets/uhd_tensile/reference.png"))
    cache = np.load(args.target_cache, mmap_mode="r", allow_pickle=False)
    targets = [np.array(frame) for frame in cache]
    manifest = json.loads(
        (ROOT / "benchmarks/results/temporal-init-20261030/report.json").read_text()
    )
    if [hashlib.sha256(t).hexdigest() for t in targets] != manifest["tensile_target_pixel_sha256"]:
        raise ValueError("The cache differs from the qualified native-UHD trajectory")
    source = Path(lucas_kanade_icgn.__file__).resolve().parent
    report = dict(
        numerical_validation_only=True,
        performance_claim=False,
        evaluated_quality_bounds=["fixed coverage", "displacement RMSE", "gradient RMSE"],
        scope="Any violation rejects the policy. No positive full/p95 qualification is claimed.",
        runtime_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "full_pyramid_backend.py",
                "wgpu_backend.py",
                "shaders/icgn.wgsl",
                "shaders/full_pyramid.wgsl",
            )
        },
        cases=[],
    )
    for variant in ("adaptive", "cubic-only"):
        options = options_for(variant)
        with ICGNWGPUFullPyramid(**options) as engine:
            engine.set_reference(reference)
            fields = list(engine.register_stream(targets, pipeline_depth=1))
        rows = [
            {"frame": index, **scores(field, stretch, (0, 0))}
            for index, (field, stretch) in enumerate(zip(fields, tensile_stretches(), strict=True))
        ]
        worst = {
            key: float((min if "coverage" in key else max)(row[key] for row in rows))
            for key in ("fixed_coverage", "fixed_displacement_rmse_px", "fixed_strain_rmse")
        }
        passes = all(
            r["fixed_coverage"] >= 0.9
            and r["fixed_displacement_rmse_px"] < 0.05
            and r["fixed_strain_rmse"] < 0.002
            for r in rows
        )
        report["cases"].append(
            dict(
                variant=variant, options=options, quality_pass=passes, worst_fixed=worst, rows=rows
            )
        )
        print(variant, passes, worst, flush=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
