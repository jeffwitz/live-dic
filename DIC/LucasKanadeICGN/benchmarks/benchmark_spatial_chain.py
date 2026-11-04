"""Alternate GPU propagation with exact empty-work termination on original UHD data."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image

from gpu_spatial_seed import GpuSpatialPyramid


def prepare_shared_variants(engine):
    """Cold calibration helper: both finalizers share the same prepared device."""
    module = engine.device.create_shader_module(
        code=files("lucas_kanade_icgn").joinpath("shaders/spatial_seed.wgsl").read_text()
    )
    legacy = engine.device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "finalize"}
    )
    for stage in engine.research:
        legacy_group = engine.device.create_bind_group(
            layout=legacy.get_bind_group_layout(0),
            entries=[engine._buffer_resource(6, stage["work"])],
        )
        stage["finalizer_variants"] = (
            (legacy, legacy_group),
            (stage["finalize"], stage["finalize_group"]),
        )


def select_variant(engine, stop):
    """Switch prebuilt references outside the measured registration call."""
    engine.stop_when_unchanged = stop
    for stage in engine.research:
        stage["finalize"], stage["finalize_group"] = stage["finalizer_variants"][int(stop)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--control-only", action="store_true", help="Measure reference against reference"
    )
    parser.add_argument("--candidate-first", action="store_true", help="Reverse preparation order")
    parser.add_argument(
        "--one-device", action="store_true", help="Reuse prepared resources between variants"
    )
    parser.add_argument(
        "--interleave", action="store_true", help="Alternate variants after each frame"
    )
    args = parser.parse_args()
    if args.control_only and args.candidate_first:
        parser.error("Choose control-only or candidate-first")
    if min(args.rounds, args.repeats, args.warmup) < 1:
        parser.error("Counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    root = project.parents[1]
    paths = [project / "assets/uhd_tensile" / name for name in ("reference.png", "frame_040.png")]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.array(image))
    with (paths[0].parent / "machine_truth.csv").open() as stream:
        truth = next(row for row in csv.DictReader(stream) if int(row["frame"]) == 40)
    options = dict(
        subset_size=32,
        subset_stride=16,
        device_type="gpu",
        failed_only=True,
        strategy="spatial",
        passes=2,
        tolerance_per_level=(0.005, 0.005),
    )
    report = dict(
        createdUtc=datetime.now(UTC).isoformat(),
        command=[sys.executable, *sys.argv],
        sourceCommit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        sourceDirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        platform=platform.platform(),
        versions={k: importlib.metadata.version(k) for k in ("numpy", "wgpu")},
        inputSha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        sourceSha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("gpu_spatial_seed.py"),
                project / "python/src/lucas_kanade_icgn/shaders/spatial_seed.wgsl",
            )
        },
        options=options,
        rounds=args.rounds,
        repeats=args.repeats,
        warmup=args.warmup,
        oneDevice=args.one_device,
        interleaved=args.interleave,
        boundary="resident NumPy target to materialized NumPy field/statistics; no decode/display/storage/preparation",
        comparison="Same candidates, scoring, acceptance and compact LK; skip later scoring only after an empty work list proves the state unchanged",
        rows=[],
    )
    engines = []
    try:
        expected = None
        cases = (
            (False, False)
            if args.control_only
            else ((True, False) if args.candidate_first else (False, True))
        )
        for stop in cases:
            start = time.perf_counter()
            if args.one_device and engines:
                engine = engines[0]
            else:
                engine = GpuSpatialPyramid(**options, stop_when_unchanged=args.one_device or stop)
                engine.prepare_research(images[0])
                if args.one_device:
                    prepare_shared_variants(engine)
            engines.append(engine)
            preparation = (time.perf_counter() - start) * 1000
            if args.one_device:
                select_variant(engine, stop)
            for _ in range(args.warmup):
                result = engine.register(images[1])
            if expected is None:
                expected = result
            else:
                for attribute in ("parameters", "status", "iterations"):
                    np.testing.assert_array_equal(
                        getattr(result, attribute), getattr(expected, attribute)
                    )
            y, x = np.indices(result.status.shape)
            x, y = x * 16 + 15.5, y * 16 + 15.5
            population = (x >= 32) & (x < 3840 - 32) & (y >= 32) & (y < 2160 - 32)
            accepted = population & (result.status > 0.5)
            displacement = np.stack(
                (
                    (float(truth["stretch_x"]) - 1) * (x - 1920),
                    (float(truth["stretch_y"]) - 1) * (y - 1080),
                ),
                axis=-1,
            )
            errors = np.linalg.norm(
                result.parameters[..., [0, 3]][accepted] - displacement[accepted], axis=-1
            )
            quality = dict(
                coverage=float(accepted.sum() / population.sum()),
                vectorRmse=float(np.sqrt(np.mean(errors**2))),
                p95Error=float(np.percentile(errors, 95)),
                accepted=result.accepted,
            )
            assert quality["coverage"] >= 0.995
            assert quality["vectorRmse"] < 0.01518230373115895
            assert quality["p95Error"] < 0.02722860762111737
            assert np.isfinite(result.parameters).all()
            assert not np.any(result.parameters[..., [1, 2, 4, 5]])
            report["adapter"] = engine.adapter_info
            report["rows"].append(
                dict(
                    stopWhenUnchanged=stop,
                    configuration=dict(result.config),
                    preparationWallMs=preparation
                    if not args.one_device or len(engines) == 1
                    else None,
                    quality=quality,
                    samplesByRoundMs=[],
                )
            )
        for round_index in range(args.rounds):
            order = list(range(len(engines)))
            if round_index % 2:
                order.reverse()
            if args.interleave:
                blocks = [[] for _ in engines]
                for index in order:
                    if args.one_device:
                        select_variant(engines[index], cases[index])
                    for _ in range(args.warmup):
                        engines[index].register(images[1])
                for iteration in range(args.repeats):
                    for index in order if iteration % 2 == 0 else order[::-1]:
                        if args.one_device:
                            select_variant(engines[index], cases[index])
                        start = time.perf_counter()
                        result = engines[index].register(images[1])
                        blocks[index].append((time.perf_counter() - start) * 1000)
                        assert result.accepted == report["rows"][index]["quality"]["accepted"]
                for index in order:
                    report["rows"][index]["samplesByRoundMs"].append(blocks[index])
                    print(round_index, index, float(np.median(blocks[index])), flush=True)
                continue
            for index in order:
                engine = engines[index]
                if args.one_device:
                    select_variant(engine, cases[index])
                for _ in range(args.warmup):
                    engine.register(images[1])
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    result = engine.register(images[1])
                    samples.append((time.perf_counter() - start) * 1000)
                    assert result.accepted == report["rows"][index]["quality"]["accepted"]
                report["rows"][index]["samplesByRoundMs"].append(samples)
                print(round_index, index, float(np.median(samples)), flush=True)
        for row in report["rows"]:
            samples = np.array(row["samplesByRoundMs"]).ravel()
            row.update(
                medianMs=float(np.median(samples)),
                p95Ms=float(np.percentile(samples, 95)),
                p99Ms=float(np.percentile(samples, 99)),
                fps=float(1000 / np.mean(samples)),
            )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    finally:
        for engine in engines:
            engine.close()


if __name__ == "__main__":
    main()
