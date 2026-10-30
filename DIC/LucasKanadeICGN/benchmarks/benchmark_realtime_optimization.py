"""Matched native-UHD tensile stream control for scheduling optimizations.

An input cache must match every pixel hash of the existing 200-frame temporal
qualification. Targets are copied to resident NumPy arrays before timing.
Run baseline and candidate processes serially, alternating their order.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

import lucas_kanade_icgn
from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/tests"))
from temporal_cases import tensile_stretches, warp

from benchmark_temporal_init import scores


def profile():
    return subprocess.check_output(["powerprofilesctl", "get"], text=True).strip()


def iteration_statistics(result):
    accepted = result.accepted_mask
    return {
        "accepted": result.accepted,
        "nonconverged": int(np.count_nonzero(result.status == 0)),
        "mean_all_iterations": float(np.mean(result.iterations)),
        "mean_accepted_iterations": result.mean_iterations,
        "histogram_all": np.bincount(result.iterations.astype(int).ravel(), minlength=31).tolist(),
        "histogram_accepted": np.bincount(
            result.iterations[accepted].astype(int), minlength=31
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Create and hash-check the native-UHD trajectory outside timing, then exit",
    )
    parser.add_argument("--temporal-fine-correction", action="store_true")
    parser.add_argument("--pipeline-depth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--iteration-diagnostics", action="store_true")
    args = parser.parse_args()
    with Image.open(ROOT / "assets/uhd_tensile/reference.png") as image:
        reference = np.array(image, dtype=np.uint8)
    manifest = json.loads(
        (ROOT / "benchmarks/results/temporal-init-20261030/report.json").read_text()
    )
    if args.prepare_cache:
        if args.target_cache.exists():
            parser.error("The target cache already exists; use a new path")
        args.target_cache.parent.mkdir(parents=True, exist_ok=True)
        cache = np.lib.format.open_memmap(
            args.target_cache, mode="w+", dtype=np.uint8, shape=(200, 2160, 3840)
        )
        for index, stretch in enumerate(tensile_stretches()):
            target = warp(reference, stretch)
            expected = manifest["tensile_target_pixel_sha256"][index]
            if hashlib.sha256(target).hexdigest() != expected:
                raise ValueError(f"Frame {index} differs from the qualified trajectory")
            cache[index] = target
        cache.flush()
        print(f"Created and verified 200 native-UHD frames: {args.target_cache}")
        return
    if args.output is None:
        parser.error("--output is required when measuring")
    cache = np.load(args.target_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (200, 2160, 3840) or cache.dtype != np.uint8:
        raise ValueError("The cache must contain exactly 200 native-UHD uint8 images")
    targets = [np.array(frame, copy=True) for frame in cache]
    del cache
    hashes = [hashlib.sha256(t).hexdigest() for t in targets]
    if hashes != manifest["tensile_target_pixel_sha256"]:
        raise ValueError("Cached targets do not match the qualified trajectory")
    options = dict(
        initialization="temporal",
        pyramid_levels=2,
        device_type="gpu",
        motion_model="affine",
        outlier_policy="median",
    )
    if args.temporal_fine_correction:
        if "temporal_fine_correction" not in inspect.signature(ICGNWGPUFullPyramid).parameters:
            raise ValueError("This revision does not implement temporal fine correction")
        options["temporal_fine_correction"] = True
    profile_at_start = profile()
    source = Path(lucas_kanade_icgn.__file__).resolve().parent
    report = {
        "package_source": str(source),
        "source_hashes": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "full_pyramid_backend.py",
                "wgpu_backend.py",
                "shaders/icgn.wgsl",
                "shaders/full_pyramid.wgsl",
            )
        },
        "command": [sys.executable, *sys.argv],
        "platform": platform.platform(),
        "versions": {name: metadata.version(name) for name in ("numpy", "wgpu", "Pillow")},
        "profile_at_start": profile_at_start,
        "options": options,
        "pipeline_depth": args.pipeline_depth,
        "target_hashes": hashes,
        "boundary": "resident uint8 NumPy image -> materialized NumPy field and statistics",
    }
    with ICGNWGPUFullPyramid(**options) as engine:
        prepared = engine.set_reference(reference)
        list(engine.register_stream(targets[:20], pipeline_depth=args.pipeline_depth))
        engine.reset_temporal_state()
        list(engine.register_stream(targets[:20], pipeline_depth=args.pipeline_depth))
        fields, completed = [], []
        start = time.perf_counter()
        for result in engine.register_stream(targets[20:], pipeline_depth=args.pipeline_depth):
            fields.append(result)
            completed.append(time.perf_counter())
        elapsed = time.perf_counter() - start
        ages = [r.timing["resultAgeWallMs"] for r in fields]
        stretches = tensile_stretches()
        quality = []
        y, x = np.indices(fields[0].status.shape) * 16 + 7.5
        population = (x >= 224) & (x < 3616) & (y >= 64) & (y < 2096)
        for frame, result in enumerate(fields, start=20):
            d = np.stack(
                ((x - 1920) * (stretches[frame, 0] - 1), (y - 1080) * (stretches[frame, 1] - 1)), -1
            )
            mask = population & result.accepted_mask
            error = np.linalg.norm(result.displacement[mask] - d[mask], axis=-1)
            quality.append(
                {
                    "frame": frame,
                    **scores(result, stretches[frame], (0, 0)),
                    "fixed_displacement_p95_px": float(np.percentile(error, 95))
                    if error.size
                    else None,
                    "fixed_gradient_component_rmse": np.sqrt(
                        np.mean(
                            (
                                result.parameters[..., [1, 2, 4, 5]][mask]
                                - np.array((stretches[frame, 0] - 1, 0, 0, stretches[frame, 1] - 1))
                            )
                            ** 2,
                            axis=0,
                        )
                    ).tolist(),
                    **iteration_statistics(result),
                }
            )
        report.update(
            frames=len(fields),
            warmups=20,
            preparation_ms=prepared.total_wall_ms,
            adapter=engine.adapter_info,
            throughput_fps=len(fields) / elapsed,
            age_median_ms=float(np.median(ages)),
            age_p95_ms=float(np.percentile(ages, 95)),
            age_p99_ms=float(np.percentile(ages, 99)),
            age_samples_ms=ages,
            completion_period_samples_ms=(np.diff(completed) * 1000).tolist(),
            component_timings=[r.timing for r in fields],
            quality=quality,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields[-1].save_npz(args.output.with_suffix(".npz"))
        if args.iteration_diagnostics:
            # Independent untimed replay: extra coarse readbacks never enter
            # the public image-to-field timings reported above.
            engine.reset_temporal_state()
            rows = []
            for frame, target in enumerate(targets):
                result = engine.register(target)
                coarse = engine._levels[-1]
                slot = coarse._pipeline_slots[0]
                raw = engine.device.queue.read_buffer(coarse._output_buffer(slot))
                coarse_result = coarse._result_from_raw(raw, timing={}, config=coarse._config)
                rows.append(
                    {
                        "frame": frame,
                        "fine": iteration_statistics(result),
                        "coarse": iteration_statistics(coarse_result),
                    }
                )
            report["untimed_iteration_diagnostics"] = rows
    report["profile_at_end"] = profile()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"{source}: correction={args.temporal_fine_correction}, depth={args.pipeline_depth}, "
        f"{report['throughput_fps']:.2f} FPS, age {report['age_median_ms']:.3f} / "
        f"{report['age_p95_ms']:.3f} ms (median/p95)",
        flush=True,
    )


if __name__ == "__main__":
    main()
