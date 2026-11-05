"""Time the CPU spatial-seed oracle against native UHD image-to-field controls.

No engine defaults change. Images are resident NumPy arrays, references are
prepared once, final fields/statistics are materialized, and file I/O is outside
the timed boundary. OpenCV is an optional research dependency.
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
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from live_dic import ICGNWGPU, ICGNWGPUFullPyramid
from live_dic.roi import prepare_roi
from live_dic.wgpu_backend import _reduce_half, _transfer_pyramid_seed
from spatial_seed_reference import select_translation_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--native-repeats", type=int, default=20)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.native_repeats) < 1:
        parser.error("Repetition counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    assets = Path(__file__).resolve().parents[1] / "assets/uhd_tensile"
    paths = [assets / "reference.png", assets / "frame_040.png"]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.array(image))
    reference, target = images
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    options = dict(
        motion_model="translation",
        subset_size=32,
        subset_stride=16,
        outlier_policy="disabled",
        device_type="gpu",
    )
    report = dict(
        command=sys.argv,
        sourceCommit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        sourceDirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        scriptSha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("spatial_seed_reference.py"))
        },
        inputSha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        shape=reference.shape,
        options=options,
        iterations=[8, 16],
        tolerance=[0.005, 0.005],
        znccMin=0.8,
        pipelineDepth=1,
        opencvThreads=cv2.getNumThreads(),
        opencl=cv2.ocl.useOpenCL(),
        warmupCallsPerCase=1,
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "wgpu", "opencv-python-headless")
        },
        cpuGovernor=Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .read_text()
        .strip()
        if Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists()
        else None,
        boundary="resident NumPy target -> materialized NumPy field and statistics; no disk I/O, decode or display",
        notes=[
            "CPU research oracle, not a production GPU propagation implementation.",
            "Target reduction, scoring, uploads, all re-solves/readbacks and statistics are timed.",
            "The independent-level control also transfers seeds on CPU; native full pyramid keeps them on GPU.",
            "Repeated frame-40 pair, zero seeds each call, no temporal reuse.",
            "Short campaign; tail quantiles with few samples are descriptive, not latency guarantees.",
        ],
        preparationWallMs={},
        samples=[],
        quality={},
    )
    with ExitStack() as stack:
        start = time.perf_counter()
        native = stack.enter_context(
            ICGNWGPUFullPyramid(
                **options, iterations_per_level=(8, 16), tolerance_per_level=(0.005, 0.005)
            )
        )
        native.set_reference(reference)
        report["preparationWallMs"]["nativeFullPyramid"] = (time.perf_counter() - start) * 1000
        report["adapter"] = native.adapter_info
        start = time.perf_counter()
        coarse_reference = _reduce_half(reference)
        fine = stack.enter_context(ICGNWGPU(**options, prefer_subgroups=False))
        coarse = stack.enter_context(ICGNWGPU(**options, prefer_subgroups=False))
        fine.set_reference(reference, iterations=16)
        coarse.set_reference(coarse_reference, iterations=8)
        fine_plan = prepare_roi(np.ones(reference.shape, bool), 32, 16)
        coarse_plan = prepare_roi(np.ones(coarse_reference.shape, bool), 32, 16)
        fine_active = fine_plan.selected.astype(np.uint32)
        coarse_active = coarse_plan.selected.astype(np.uint32)
        fine_zero = np.zeros((*fine_active.shape, 8), np.float32)
        coarse_zero = np.zeros((*coarse_active.shape, 8), np.float32)
        report["preparationWallMs"]["independentLevelsAndCpuPlans"] = (
            time.perf_counter() - start
        ) * 1000

        def run(case):
            stages = {}

            def timed(name, operation):
                t = time.perf_counter()
                value = operation()
                stages[name] = stages.get(name, 0) + (time.perf_counter() - t) * 1000
                return value

            if case == "native":
                result = timed("nativeSolve", lambda: native.register(target))
            else:
                mode, count = case.split("-")
                count = int(count)
                reduced = timed("targetReductionCpu", lambda: _reduce_half(target))
                cr = timed(
                    "coarseSolves",
                    lambda: coarse._register_seeded_frame(reduced, coarse_zero, coarse_active),
                )

                def candidates(f, g, result, plan, fallback, label):
                    if mode == "spatial":
                        return timed(
                            label,
                            lambda: select_translation_candidates(
                                f, g, result.parameters, plan, fallback=fallback
                            ),
                        )[0]
                    seeds = np.zeros((*result.status.shape, 8), np.float32)
                    seeds[..., :6] = result.parameters
                    return seeds

                for _ in range(count):
                    seeds = candidates(
                        coarse_reference, reduced, cr, coarse_plan, coarse_zero, "coarseScoringCpu"
                    )
                    cr = timed(
                        "coarseSolves",
                        lambda seeds=seeds: coarse._register_seeded_frame(
                            reduced, seeds, coarse_active
                        ),
                    )
                seeds, _ = timed(
                    "seedTransferCpu",
                    lambda: _transfer_pyramid_seed(
                        cr,
                        source_available=coarse_plan.selected,
                        source_scale=2,
                        destination_scale=1,
                        destination_grid=(fine.grid_width, fine.grid_height),
                        requested=fine_plan.selected,
                        subset_size=32,
                        subset_stride=16,
                    ),
                )
                result = timed(
                    "fineSolves", lambda: fine._register_seeded_frame(target, seeds, fine_active)
                )
                for _ in range(count):
                    seeds = candidates(
                        reference, target, result, fine_plan, fine_zero, "fineScoringCpu"
                    )
                    result = timed(
                        "fineSolves",
                        lambda seeds=seeds: fine._register_seeded_frame(target, seeds, fine_active),
                    )
            return result, stages

        with (assets / "machine_truth.csv").open() as stream:
            truth = next(row for row in csv.DictReader(stream) if int(row["frame"]) == 40)
        y, x = np.indices(fine_active.shape)
        x, y = x * 16 + 15.5, y * 16 + 15.5
        population = (x >= 32) & (x < 3840 - 32) & (y >= 32) & (y < 2160 - 32)
        expected = np.stack(
            (
                (float(truth["stretch_x"]) - 1) * (x - 1920),
                (float(truth["stretch_y"]) - 1) * (y - 1080),
            ),
            axis=-1,
        )

        def quality(result):
            accepted = population & (result.status > 0.5)
            error = np.linalg.norm(
                result.parameters[..., [0, 3]][accepted] - expected[accepted], axis=-1
            )
            return dict(
                accepted=result.accepted,
                population=int(population.sum()),
                interiorCoverage=float(accepted.sum() / population.sum()),
                vectorRmse=float(np.sqrt(np.mean(error**2))),
                p95Error=float(np.percentile(error, 95)),
            )

        cases = ["native", "continue-0", "continue-2", "spatial-1", "spatial-2"]
        for case in cases:
            result, _ = run(case)
            report["quality"][case] = quality(result)
            print("WARM", case, flush=True)
        for round_index in range(args.rounds):
            order = cases if round_index % 2 == 0 else cases[::-1]
            for case in order:
                repeats = args.native_repeats if case == "native" else args.repeats
                for repeat in range(repeats):
                    t = time.perf_counter()
                    result, stages = run(case)
                    elapsed = (time.perf_counter() - t) * 1000
                    report["samples"].append(
                        dict(
                            case=case,
                            round=round_index,
                            repeat=repeat,
                            wallMs=elapsed,
                            stages=stages,
                            accepted=result.accepted,
                        )
                    )
                print("MEASURED", round_index, case, round(elapsed, 2), "ms", flush=True)
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
    report["summary"] = {}
    for case in cases:
        samples = [s for s in report["samples"] if s["case"] == case]
        values = np.array([s["wallMs"] for s in samples])
        report["summary"][case] = dict(
            count=len(values),
            medianMs=float(np.median(values)),
            p95Ms=float(np.percentile(values, 95)),
            fps=float(1000 * len(values) / values.sum()),
            stageMedianMs={
                key: float(np.median([s["stages"][key] for s in samples]))
                for key in samples[0]["stages"]
            },
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
