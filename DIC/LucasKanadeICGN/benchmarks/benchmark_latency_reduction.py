"""Native-UHD latency ablation with quality and untimed same-seed abort audits.

Input formation, scoring and diagnostic readbacks are outside the timed stream.
The optional campaign driver alternates process order and retains every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from temporal_cases import tensile_stretches

from benchmark_temporal_init import scores

VARIANTS = (
    "baseline",
    "velocity",
    "abort",
    "adaptive",
    "cubic",
    "cubic-fixed",
    "abort-fast",
    "cubic-only",
)


def options_for(variant):
    options = dict(
        initialization="temporal",
        temporal_fine_correction=True,
        pyramid_levels=2,
        motion_model="affine",
        outlier_policy="median",
        device_type="gpu",
    )
    if variant not in ("baseline", "cubic-only"):
        options["temporal_prediction"] = "velocity"
    if variant in ("abort", "adaptive", "cubic", "cubic-fixed", "abort-fast"):
        options.update(divergence_patience=3, divergence_min_iterations=4)
    if variant == "abort-fast":
        options.update(divergence_patience=1, divergence_min_iterations=2)
    if variant in ("adaptive", "cubic"):
        options.update(pyramid_policy="adaptive", adaptive_acceptance=0.95)
    if variant in ("cubic", "cubic-fixed", "cubic-only"):
        options["interpolation"] = "cubic"
    return options


def profile():
    return subprocess.check_output(["powerprofilesctl", "get"], text=True).strip()


def iterations(result):
    return dict(
        mean_all=float(np.mean(result.iterations)),
        mean_accepted=result.mean_iterations,
        histogram_all=np.bincount(result.iterations.astype(int).ravel(), minlength=31).tolist(),
        accepted=result.accepted,
        nonconverged=int(np.count_nonzero(result.status == 0)),
    )


def read_raw(level):
    slot = level._pipeline_slots[0]
    raw = level.device.queue.read_buffer(slot.buffers["state"])
    return level._result_from_raw(raw, timing={}, config=level._config)


def audit_iterations(engine, targets):
    """Replay sequentially; extra reads/reruns never enter production timings.

    The no-abort shadow uses the identical retained seed and current image.
    It executes after temporal histories were merged, never updates them, and
    therefore measures local late-convergence loss without changing the next
    production frame's initial history. It is not a second production retry.
    """
    control = {}
    if getattr(engine, "divergence_patience", None) is not None:
        for index, level in enumerate(engine._levels):
            constants = {**level._solver_constants(), "ROI_ENABLED": 0, "DIVERGENCE_PATIENCE": 0}
            control[index] = engine.device.create_compute_pipeline(
                layout=engine.device.create_pipeline_layout(
                    bind_group_layouts=[level._seeded_bind_group_layout]
                ),
                compute={
                    "module": level._shader,
                    "entry_point": "solve_seeded",
                    "constants": constants,
                },
            )
    engine.reset_temporal_state()
    rows = []
    for frame, target in enumerate(targets):
        result = engine.register(target)
        row = {"frame": frame, "levels_used": int(result.timing["pyramidLevels"])}
        for index, name in ((0, "fine"), (1, "coarse")):
            if index >= row["levels_used"]:
                row[name] = {"executed": False, "mean_all": 0.0}
                continue
            level = engine._levels[index]
            raw = read_raw(level)
            entry = {"executed": True, **iterations(raw)}
            if index in control:
                slot = level._pipeline_slots[0]
                encoder = engine.device.create_command_encoder()
                level._encode_registration_passes(
                    encoder, slot, control[index], solver_bind_group=slot.seeded_bind_group
                )
                engine.device.queue.submit([encoder.finish()])
                shadow = read_raw(level)
                cap = engine.iterations_per_level[-index - 1]
                aborted = (raw.status == 0) & (raw.iterations < cap)
                lost = aborted & (shadow.status == 1)
                entry.update(
                    aborted=int(np.count_nonzero(aborted)),
                    would_converge_late=int(np.count_nonzero(lost)),
                    shadow_mean_all=float(np.mean(shadow.iterations)),
                )
                if index == 0:
                    y, x = np.indices(raw.status.shape) * 16 + 7.5
                    fixed = (x >= 224) & (x < 3616) & (y >= 64) & (y < 2096)
                    entry["would_converge_late_fixed"] = int(np.count_nonzero(lost & fixed))
            row[name] = entry
        rows.append(row)
    return rows


def run_campaign(args):
    for repeat in range(3):
        variants = args.variants if repeat % 2 == 0 else tuple(reversed(args.variants))
        for variant in variants:
            for depth in (1, 2):
                output = args.output / f"{variant}-d{depth}-r{repeat}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--target-cache",
                    str(args.target_cache),
                    "--variant",
                    variant,
                    "--pipeline-depth",
                    str(depth),
                    "--output",
                    str(output),
                ]
                if repeat == 0 and depth == 1:
                    command.append("--diagnostics")
                if args.solver_workgroup_size is not None:
                    command.extend(("--solver-workgroup-size", str(args.solver_workgroup_size)))
                environment = os.environ.copy()
                if variant == "baseline" and args.baseline_source is not None:
                    environment["PYTHONPATH"] = str(args.baseline_source)
                print(f"Starting {variant}, depth {depth}, repeat {repeat}", flush=True)
                subprocess.run(command, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--pipeline-depth", choices=(1, 2), type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--solver-workgroup-size", type=int, choices=(16, 32, 128))
    args = parser.parse_args()
    if args.campaign:
        run_campaign(args)
        return
    reference = np.array(Image.open(ROOT / "assets/uhd_tensile/reference.png"))
    cache = np.load(args.target_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (200, 2160, 3840) or cache.dtype != np.uint8:
        raise ValueError("Expected the qualified native-UHD uint8 trajectory")
    targets = [np.array(frame, copy=True) for frame in cache]
    hashes = [hashlib.sha256(frame).hexdigest() for frame in targets]
    manifest = json.loads(
        (ROOT / "benchmarks/results/temporal-init-20261030/report.json").read_text()
    )
    if hashes != manifest["tensile_target_pixel_sha256"]:
        raise ValueError("Target pixels differ from the qualified trajectory")
    source = Path(lucas_kanade_icgn.__file__).resolve().parent

    def git(*command):
        return subprocess.check_output(["git", "-C", str(source), *command], text=True).strip()

    options = options_for(args.variant)
    if args.solver_workgroup_size is not None:
        options["solver_workgroup_size"] = args.solver_workgroup_size
    report = dict(
        variant=args.variant,
        requested_depth=args.pipeline_depth,
        options=options,
        boundary="resident uint8 NumPy image -> materialized NumPy field and statistics",
        command=[sys.executable, *sys.argv],
        source=str(source),
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        source_hashes={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "full_pyramid_backend.py",
                "wgpu_backend.py",
                "shaders/icgn.wgsl",
                "shaders/full_pyramid.wgsl",
            )
        },
        platform=platform.platform(),
        versions={name: metadata.version(name) for name in ("numpy", "wgpu", "Pillow")},
        profile_before=profile(),
        target_pixel_sha256=hashes,
        warmups=20,
    )
    with ICGNWGPUFullPyramid(**report["options"]) as engine:
        prepared = engine.set_reference(reference)
        list(engine.register_stream(targets[:20], pipeline_depth=args.pipeline_depth))
        engine.reset_temporal_state()
        list(engine.register_stream(targets[:20], pipeline_depth=args.pipeline_depth))
        fields = []
        completed = []
        start = time.perf_counter()
        for result in engine.register_stream(targets[20:], pipeline_depth=args.pipeline_depth):
            fields.append(result)
            completed.append(time.perf_counter())
        elapsed = time.perf_counter() - start
        ages = [r.timing["resultAgeWallMs"] for r in fields]
        stretches = tensile_stretches()
        y, x = np.indices(fields[0].status.shape) * 16 + 7.5
        population = (x >= 224) & (x < 3616) & (y >= 64) & (y < 2096)
        quality = []
        for frame, result in enumerate(fields, 20):
            lx, ly = stretches[frame]
            truth = np.stack(((lx - 1) * (x - 1920), (ly - 1) * (y - 1080)), -1)
            accepted = population & result.accepted_mask
            error = np.linalg.norm(result.displacement[accepted] - truth[accepted], axis=-1)
            gradients = result.parameters[..., [1, 2, 4, 5]][accepted] - (lx - 1, 0, 0, ly - 1)
            quality.append(
                dict(
                    frame=frame,
                    **scores(result, stretches[frame], (0, 0)),
                    fixed_displacement_p95_px=float(np.percentile(error, 95))
                    if error.size
                    else None,
                    fixed_gradient_component_rmse=np.sqrt(np.mean(gradients**2, axis=0)).tolist()
                    if error.size
                    else None,
                    levels_used=int(result.timing["pyramidLevels"]),
                    fine_iterations=iterations(result),
                )
            )
        report.update(
            adapter=engine.adapter_info,
            preparation_ms=prepared.total_wall_ms,
            frames=len(fields),
            throughput_fps=len(fields) / elapsed,
            age_median_ms=float(np.median(ages)),
            age_p95_ms=float(np.percentile(ages, 95)),
            age_p99_ms=float(np.percentile(ages, 99)),
            age_samples_ms=ages,
            completion_period_samples_ms=(np.diff(completed) * 1000).tolist(),
            component_timings=[r.timing for r in fields],
            quality=quality,
            effective_solver_config=fields[-1].config,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields[-1].save_npz(args.output.with_suffix(".npz"))
        if args.diagnostics:
            report["untimed_diagnostics"] = audit_iterations(engine, targets)
    report["profile_after"] = profile()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"{args.variant}, depth {args.pipeline_depth}: {report['throughput_fps']:.2f} FPS, "
        f"age {report['age_median_ms']:.3f}/{report['age_p95_ms']:.3f}/{report['age_p99_ms']:.3f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
