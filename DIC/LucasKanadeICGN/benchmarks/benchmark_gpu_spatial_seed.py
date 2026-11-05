"""Qualify the GPU spatial-seed research host on unchanged analytic UHD inputs."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gpu_spatial_seed import GpuSpatialPyramid
from live_dic import ICGNWGPUFullPyramid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.warmup) < 1:
        parser.error("Counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    root = project.parents[1]
    assets = project / "assets/uhd_tensile"
    paths = [assets / "reference.png", assets / "frame_040.png"]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.array(image))
    reference, target = images
    scripts = [
        Path(__file__),
        Path(__file__).with_name("gpu_spatial_seed.py"),
        *[
            project / "python/src/lucas_kanade_icgn/shaders" / name
            for name in ("icgn.wgsl", "spatial_seed.wgsl", "full_pyramid.wgsl")
        ],
    ]
    options = dict(
        subset_size=32, subset_stride=16, device_type="gpu", tolerance_per_level=(0.005, 0.005)
    )
    report = dict(
        sourceCommit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        sourceDirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        command=f"python3 {Path(__file__).relative_to(root)} --output {args.output} --rounds {args.rounds} --repeats {args.repeats} --warmup {args.warmup}",
        inputSha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scriptSha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        platform=platform.platform(),
        versions={key: importlib.metadata.version(key) for key in ("numpy", "wgpu")},
        options=options,
        shape=reference.shape,
        iterations=[8, 16],
        znccMin=0.8,
        localModel="translation",
        outlierPolicy="disabled",
        interpolation="bspline",
        roi="none/full image",
        pipelineDepth=1,
        minScoreGain=1e-4,
        minSeedDisplacement=1e-3,
        criteria=dict(
            minimumInteriorCoverage=0.995,
            maximumVectorRmse=0.01518230373115895,
            maximumP95Error=0.02722860762111737,
        ),
        timingBoundary="resident NumPy target -> materialized NumPy field/statistics; reference prepared once; no decode/display/storage",
        diagnosticReadback="outside timing, once per case; no intermediate CPU decision or readback during registration",
        warmup=args.warmup,
        rounds=args.rounds,
        repeats=args.repeats,
        rows=[],
    )
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    report["cpuGovernor"] = governor.read_text().strip() if governor.exists() else None
    with (assets / "machine_truth.csv").open() as stream:
        truth = next(row for row in csv.DictReader(stream) if int(row["frame"]) == 40)

    def quality(result):
        if not np.isfinite(result.parameters).all() or np.any(
            result.parameters[..., (1, 2, 4, 5)] != 0
        ):
            raise RuntimeError("Nonfinite translation field or nonzero local gradients")
        y, x = np.indices(result.status.shape)
        x, y = x * 16 + 15.5, y * 16 + 15.5
        population = (x >= 32) & (x < 3840 - 32) & (y >= 32) & (y < 2160 - 32)
        accepted = population & (result.status > 0.5)
        expected = np.stack(
            (
                (float(truth["stretch_x"]) - 1) * (x - 1920),
                (float(truth["stretch_y"]) - 1) * (y - 1080),
            ),
            axis=-1,
        )
        error = np.linalg.norm(
            result.parameters[..., [0, 3]][accepted] - expected[accepted], axis=-1
        )
        return dict(
            accepted=result.accepted,
            population=int(population.sum()),
            coverage=float(accepted.sum() / population.sum()),
            vectorRmse=float(np.sqrt(np.mean(error**2))),
            p95Error=float(np.percentile(error, 95)),
        )

    def summarize(row):
        values = np.array([v for block in row["samplesByRoundMs"] for v in block])
        row.update(
            medianMs=float(np.median(values)),
            p95Ms=float(np.percentile(values, 95)),
            p99Ms=float(np.percentile(values, 99)),
            fps=float(1000 * len(values) / values.sum()),
        )

    # Establish direct parity with the production schedule before timing.
    with ICGNWGPUFullPyramid(
        **options, motion_model="translation", outlier_policy="disabled"
    ) as native:
        native.set_reference(reference)
        original = native.register(target)
    groups = [
        (
            "all-full",
            dict(sample_step=1, failed_only=False),
            [("baseline", 2), ("median", 2), ("seed-median", 2), ("spatial", 1), ("spatial", 2)],
        ),
        (
            "failed-full",
            dict(sample_step=1, failed_only=True),
            [("baseline", 2), ("spatial", 1), ("spatial", 2)],
        ),
        ("all-sampled", dict(sample_step=4, failed_only=False), [("baseline", 2), ("spatial", 2)]),
    ]
    for group_name, controls, cases in groups:
        t = time.perf_counter()
        with GpuSpatialPyramid(**options, **controls) as engine:
            engine.prepare_research(reference)
            preparation = (time.perf_counter() - t) * 1000
            report["adapter"] = engine.adapter_info
            rows = []
            for strategy, passes in cases:
                engine.strategy, engine.passes = strategy, passes
                for _ in range(args.warmup):
                    result = engine.register(target)
                if strategy == "baseline":
                    np.testing.assert_array_equal(result.parameters, original.parameters)
                    np.testing.assert_array_equal(result.status, original.status)
                    np.testing.assert_array_equal(result.iterations, original.iterations)
                counts = []
                if strategy == "spatial":
                    engine.capture_counts = True
                    engine.register(target)
                    counts = engine.read_diagnostics()
                    engine.capture_counts = False
                q = quality(result)
                criteria = report["criteria"]
                if (
                    q["coverage"] < criteria["minimumInteriorCoverage"]
                    or q["vectorRmse"] > criteria["maximumVectorRmse"]
                    or q["p95Error"] > criteria["maximumP95Error"]
                ):
                    raise RuntimeError(f"Analytic quality gate failed: {q}")
                rows.append(
                    dict(
                        group=group_name,
                        strategy=strategy,
                        passes=passes,
                        **controls,
                        preparationWallMs=preparation,
                        quality=q,
                        samplesByRoundMs=[],
                        diagnosticCounts=counts,
                    )
                )
            for round_index in range(args.rounds):
                order = rows if round_index % 2 == 0 else rows[::-1]
                for row in order:
                    engine.strategy, engine.passes = row["strategy"], row["passes"]
                    samples = []
                    for _ in range(args.repeats):
                        t = time.perf_counter()
                        result = engine.register(target)
                        samples.append((time.perf_counter() - t) * 1000)
                        if result.accepted != row["quality"]["accepted"]:
                            raise RuntimeError("Acceptance changed during repeated-pair timing")
                    row["samplesByRoundMs"].append(samples)
                    print(
                        group_name,
                        row["strategy"],
                        row["passes"],
                        round_index,
                        round(float(np.median(samples)), 3),
                        flush=True,
                    )
            for row in rows:
                summarize(row)
                report["rows"].append(row)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in ("group", "strategy", "passes", "medianMs", "p95Ms", "fps")
                }
                for row in report["rows"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
