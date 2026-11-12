"""Measure 4K seeded/frontier boundaries and isolated WebGPU buffer transfers.

Run alone on the target device. The isolated transfers are diagnostic, not
component-level attribution of the compute calls or pure GPU timestamps.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import wgpu

from live_dic import ICGNWGPU


def median_range(values):
    return {
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "samples_ms": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup < 1:
        parser.error("use at least three repetitions and one warmup")
    reference = np.random.default_rng(714).integers(0, 256, (2160, 3840), dtype=np.uint8)
    target = np.ascontiguousarray(np.roll(reference, (1, 2), axis=(0, 1)))
    result = {
        "configuration": {
            "width": 3840,
            "height": 2160,
            "subset": 64,
            "stride": 64,
            "interpolation": "linear",
            "reference_sha256": hashlib.sha256(reference).hexdigest(),
            "target_sha256": hashlib.sha256(target).hexdigest(),
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "python": platform.python_version(),
        },
        "samples": [],
    }
    package_root = Path(__file__).resolve().parents[1] / "python/src/live_dic"
    result["source"] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files_sha256": {
            str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                package_root / "wgpu_backend.py",
                package_root / "engine.py",
                package_root / "shaders/affine_zncc.wgsl",
                package_root / "shaders/spatial_seed.wgsl",
                package_root / "shaders/icgn.wgsl",
            )
        },
        "wgpu_version": wgpu.__version__,
        "numpy_version": np.__version__,
        "timing_note": "Upload buffer timing includes a four-byte readback ordering barrier; no GPU timestamp queries.",
    }
    with ICGNWGPU(
        device_type="gpu",
        subset_size=64,
        subset_stride=64,
        interpolation="linear",
        outlier_policy="disabled",
        prefer_subgroups=False,
    ) as engine:
        started = time.perf_counter()
        engine.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
        result["reference_setup_wall_ms"] = (time.perf_counter() - started) * 1000
        result["adapter"] = dict(engine.adapter_info)
        shape = (engine.grid_height, engine.grid_width)
        count = int(np.prod(shape))
        state = np.zeros((*shape, 6), np.float32)
        state[..., 0], state[..., 3] = 2, 1
        known = np.zeros(shape, bool)
        known[1:-1:4, 1:-1:4] = True
        result["configuration"].update(
            {
                "grid": list(shape),
                "known_seeds": int(known.sum()),
                "target_bytes": int(target.nbytes),
                "state_buffer_bytes": count * 32,
                "mask_buffer_bytes": count * 4,
            }
        )
        # Explicit isolated transfer roundtrips are measured on buffers, not
        # textures; the call boundaries above remain the authoritative pipeline timings.
        payloads = {
            "frame_sized_buffer": np.ascontiguousarray(target).ravel(),
            "state_buffer": np.zeros(count * 8, np.float32),
        }
        transfers = {}
        for name, data in payloads.items():
            buffer = engine.device.create_buffer(
                size=data.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            )
            uploads, downloads = [], []
            try:
                for iteration in range(args.warmup + args.repetitions):
                    start = time.perf_counter()
                    engine.device.queue.write_buffer(buffer, 0, data)
                    # A four-byte readback is a portable ordering barrier. Its
                    # own readback overhead is part of this upload measurement.
                    engine.device.queue.read_buffer(buffer, size=4)
                    up = (time.perf_counter() - start) * 1000
                    start = time.perf_counter()
                    received = engine.device.queue.read_buffer(buffer)
                    down = (time.perf_counter() - start) * 1000
                    if len(received) != data.nbytes:
                        raise RuntimeError("short buffer readback")
                    if iteration >= args.warmup:
                        uploads.append(up)
                        downloads.append(down)
            finally:
                buffer.destroy()
            transfers[name] = {
                "bytes": data.nbytes,
                "upload_synchronized": median_range(uploads),
                "download_synchronized": median_range(downloads),
            }
        result["isolated_buffer_transfers"] = transfers
        for iteration in range(args.warmup + args.repetitions):
            start = time.perf_counter()
            seeds, active = engine.affine_frontier_seeds(state, known)
            frontier_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            solved = engine.register_seeded(target, seeds, active, include_zncc=True)
            solve_ms = (time.perf_counter() - start) * 1000
            if iteration >= args.warmup:
                result["samples"].append(
                    {
                        "frontier_wall_ms": frontier_ms,
                        "solve_wall_ms": solve_ms,
                        "target_upload_wall_ms": solved.timing["uploadWallMs"],
                        "solve_and_readback_wall_ms": solved.timing["executeReadbackWallMs"],
                        "final_zncc_wall_ms": solved.timing["finalZnccWallMs"],
                        "active": int(active.sum()),
                        "accepted": solved.accepted,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
