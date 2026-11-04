"""Bounded UHD propagation soak: paired timings, field parity and process RSS.

This measures a fixed resident NumPy pair, not a camera, rendering or storage
pipeline. Numeric timing records stream to disk; computed fields are not kept.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_spatial_chain import power_context, prepare_shared_variants, select_variant
from gpu_spatial_seed import GpuSpatialPyramid


def rss_bytes():
    """Linux process RSS, not physical GPU memory; unavailable elsewhere."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def statistics(samples):
    return dict(
        medianMs=float(np.median(samples)),
        p95Ms=float(np.percentile(samples, 95)),
        p99Ms=float(np.percentile(samples, 99)),
        engineFieldsPerSecond=float(1000 / np.mean(samples)),
        over60HzBudgetFraction=float(np.mean(samples > 1000 / 60)),
    )


def assert_field_parity(field, expected):
    """Exact work/score reuse preserves arrays, including rejected states."""
    for name in ("parameters", "status", "iterations"):
        np.testing.assert_array_equal(getattr(field, name), getattr(expected, name))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--duration-seconds", type=float)
    duration.add_argument("--until-utc", help="ISO-8601 UTC deadline, e.g. 2026-11-04T06:35:00Z")
    parser.add_argument("--checkpoint-pairs", type=int, default=1024)
    parser.add_argument("--score-cache", action="store_true", help="Compare exact score reuse")
    args = parser.parse_args()
    seconds = args.duration_seconds
    if args.until_utc:
        end = datetime.fromisoformat(args.until_utc)
        if end.tzinfo is None:
            parser.error("until-utc must include a timezone")
        seconds = (end - datetime.now(UTC)).total_seconds()
    if not np.isfinite(seconds) or seconds <= 0 or args.checkpoint_pairs < 1:
        parser.error("A future deadline/positive duration and checkpoint size are required")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.perf_counter() + seconds
    project = Path(__file__).resolve().parents[1]
    root = project.parents[1]
    assets = project / "assets/uhd_tensile"
    paths = [assets / name for name in ("reference.png", "frame_040.png")]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.array(image))
    if any(image.shape != (2160, 3840) or image.dtype != np.uint8 for image in images):
        raise ValueError("This protocol requires the original uint8 grayscale UHD pair")
    with (assets / "machine_truth.csv").open() as stream:
        truth = next(row for row in csv.DictReader(stream) if int(row["frame"]) == 40)
    report = dict(
        createdUtc=datetime.now(UTC).isoformat(),
        command=[sys.executable, *sys.argv],
        sourceCommit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        sourceDirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        versions={key: importlib.metadata.version(key) for key in ("numpy", "wgpu")},
        platform=platform.platform(),
        inputSha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        sourceSha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("gpu_spatial_seed.py"),
                Path(__file__).with_name("benchmark_spatial_chain.py"),
                project / "python/src/lucas_kanade_icgn/shaders/spatial_seed.wgsl",
                project / "python/src/lucas_kanade_icgn/shaders/icgn.wgsl",
                project / "python/src/lucas_kanade_icgn/shaders/full_pyramid.wgsl",
            )
        },
        state="preparing",
        pairs=0,
        processId=os.getpid(),
        checkpoints=[],
        boundary="resident NumPy target to materialized field/statistics; excludes decode, preparation, display, durable fields and diagnostic logging",
        schedule="One prepared hardware device, alternating per-frame AB/BA; control and candidate each have two failed-only passes",
        comparison="score-cache" if args.score_cache else "empty-work",
        rssNote="Process RSS includes CPU/driver mappings, not physical/free VRAM. Only the last fields and a bounded timing batch are retained.",
        timingSnapshotNote="Checkpoint last-pair wall timers are descriptive samples, not stage percentile distributions or GPU timestamps. Some timers overlap and must not be summed.",
        rawFormat=dict(dtype="<f8", columns=["pairElapsedMs", "controlMs", "candidateMs"]),
    )
    report.update(power_context())
    atomic_json(args.output / "progress.json", report)
    engine = None
    pending = []
    completed = False
    started = time.perf_counter()
    try:
        engine = GpuSpatialPyramid(
            strategy="spatial",
            failed_only=True,
            passes=2,
            stop_when_unchanged=True,
            cache_scores=args.score_cache,
            device_type="gpu",
            subset_size=32,
            subset_stride=16,
            tolerance_per_level=(0.005, 0.005),
        )
        engine.prepare_research(images[0])
        if not args.score_cache:
            prepare_shared_variants(engine)
        report["adapter"] = engine.adapter_info
        report["deviceLimits"] = dict(engine.device.limits)
        expected = None
        configurations = {}
        for variant in (False, True):
            select_variant(engine, variant, score_cache=args.score_cache)
            for _ in range(16):
                field = engine.register(images[1])
            configurations["candidate" if variant else "control"] = dict(field.config)
            if expected is None:
                expected = field
            else:
                assert_field_parity(field, expected)
        y, x = np.indices(expected.status.shape)
        x, y = x * 16 + 15.5, y * 16 + 15.5
        population = (x >= 32) & (x < 3808) & (y >= 32) & (y < 2128)
        valid = population & expected.accepted_mask
        actual = expected.parameters[..., [0, 3]][valid]
        truth_displacement = np.stack(
            (
                (float(truth["stretch_x"]) - 1) * (x - 1920),
                (float(truth["stretch_y"]) - 1) * (y - 1080),
            ),
            axis=-1,
        )[valid]
        errors = np.linalg.norm(actual - truth_displacement, axis=-1)
        quality = dict(
            population=int(population.sum()),
            accepted=expected.accepted,
            coverage=float(valid.sum() / population.sum()),
            vectorRmse=float(np.sqrt(np.mean(errors**2))),
            p95Error=float(np.percentile(errors, 95)),
        )
        assert quality["coverage"] >= 0.995 and quality["vectorRmse"] < 0.01518230373115895
        assert quality["p95Error"] < 0.02722860762111737
        assert np.isfinite(expected.parameters).all() and not np.any(
            expected.parameters[..., [1, 2, 4, 5]]
        )
        report.update(
            quality=quality,
            configurations=configurations,
            preparationAndWarmupWallMs=(time.perf_counter() - started) * 1000,
            initialRssBytes=rss_bytes(),
            state="running",
        )
        start_measurement = time.perf_counter()
        report["measurementStartedEpochSeconds"] = time.time()
        if args.duration_seconds is not None:
            deadline = start_measurement + args.duration_seconds
        if time.perf_counter() >= deadline:
            raise RuntimeError("Deadline elapsed during preparation; no steady-state run")
        latest = {}
        with (args.output / "timings.bin").open("xb") as stream:
            while time.perf_counter() < deadline:
                elapsed = (time.perf_counter() - start_measurement) * 1000
                times = [0.0, 0.0]
                for index in (0, 1) if report["pairs"] % 2 == 0 else (1, 0):
                    select_variant(engine, bool(index), score_cache=args.score_cache)
                    start = time.perf_counter()
                    field = engine.register(images[1])
                    times[index] = (time.perf_counter() - start) * 1000
                    if field.accepted != quality["accepted"]:
                        raise RuntimeError("Acceptance changed during fixed-pair soak")
                    latest[index] = field
                pending.append([elapsed, *times])
                report["pairs"] += 1
                if len(pending) >= args.checkpoint_pairs:
                    for field in latest.values():
                        assert_field_parity(field, expected)
                    block = np.array(pending, dtype="<f8")
                    stream.write(block.tobytes())
                    stream.flush()
                    pending.clear()
                    report["checkpoints"].append(
                        dict(
                            pairs=report["pairs"],
                            elapsedSeconds=(time.perf_counter() - start_measurement),
                            rssBytes=rss_bytes(),
                            control=statistics(block[:, 1]),
                            candidate=statistics(block[:, 2]),
                            pairedMedianSavingMs=float(np.median(block[:, 1] - block[:, 2])),
                            fieldParity=True,
                            lastPairTiming={
                                "control": dict(latest[0].timing),
                                "candidate": dict(latest[1].timing),
                            },
                        )
                    )
                    atomic_json(args.output / "progress.json", report)
                    print(
                        report["pairs"],
                        report["checkpoints"][-1]["pairedMedianSavingMs"],
                        flush=True,
                    )
            for field in latest.values():
                assert_field_parity(field, expected)
            if pending:
                stream.write(np.array(pending, dtype="<f8").tobytes())
                stream.flush()
                pending.clear()
        completed = True
        report["measurementWallSeconds"] = time.perf_counter() - start_measurement
        report["measurementFinishedEpochSeconds"] = time.time()
        report["finalRssBytesBeforeStatistics"] = rss_bytes()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            engine.close()
        report["state"] = "complete" if completed else "incomplete"
        report["finishedUtc"] = datetime.now(UTC).isoformat()
        raw = args.output / "timings.bin"
        if pending:
            with raw.open("ab") as stream:
                stream.write(np.array(pending, dtype="<f8").tobytes())
            pending.clear()
        if raw.exists() and raw.stat().st_size:
            samples = np.fromfile(raw, dtype="<f8").reshape(-1, 3)
            np.savez_compressed(args.output / "timings.npz", timings=samples)
            report["recordedPairs"] = len(samples)
            report["control"] = statistics(samples[:, 1])
            report["candidate"] = statistics(samples[:, 2])
            report["pairedMedianSavingMs"] = float(np.median(samples[:, 1] - samples[:, 2]))
            report["rawSha256"] = hashlib.sha256(
                (args.output / "timings.npz").read_bytes()
            ).hexdigest()
        if completed:
            report["combinedObservedFieldsPerSecond"] = (
                2 * report["pairs"] / report["measurementWallSeconds"]
            )
        atomic_json(args.output / "report.json", report)
        atomic_json(args.output / "progress.json", report)


if __name__ == "__main__":
    main()
