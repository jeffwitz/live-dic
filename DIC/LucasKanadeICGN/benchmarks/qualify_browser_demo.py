"""Replay decoded demo pixels natively, retaining all accepted truth errors.

FFmpeg is a developer-only diagnostic tool here, never a package dependency.
This is not a performance campaign: decoding and registration are sequential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("This optional diagnostic requires a developer-installed ffmpeg")
    demo = ROOT / "web-app/demo"
    manifest = json.loads((demo / "manifest.json").read_text())
    width, height = manifest["width"], manifest["height"]
    reference = np.array(Image.open(demo / "reference.png").convert("L"))
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(demo / "tensile.webm"),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    options = dict(
        initialization="temporal",
        temporal_prediction="velocity",
        temporal_fine_correction=True,
        divergence_patience=3,
        divergence_min_iterations=4,
        outlier_policy="median",
        device_type="gpu",
    )
    rows = []
    with ICGNWGPUFullPyramid(**options) as engine:
        engine.set_reference(reference)
        with subprocess.Popen(command, stdout=subprocess.PIPE) as decoder:
            try:
                for index, (sx, sy) in enumerate(manifest["stretch"]):
                    pixels = decoder.stdout.read(width * height)
                    if len(pixels) != width * height:
                        raise RuntimeError(f"Incomplete decoded frame {index}")
                    frame = np.frombuffer(pixels, np.uint8).reshape(height, width)
                    result = engine.register(frame)
                    gy, gx = np.indices(result.status.shape)
                    cx, cy = gx * 16 + 7.5, gy * 16 + 7.5
                    truth = np.stack(
                        ((sx - 1) * (cx - width / 2), (sy - 1) * (cy - height / 2)), axis=-1
                    )
                    errors = np.linalg.norm(result.displacement - truth, axis=-1)
                    accepted = result.accepted_mask
                    worst = np.unravel_index(
                        np.argmax(np.where(accepted, errors, -1)), errors.shape
                    )
                    row = dict(
                        frame=index,
                        accepted=int(accepted.sum()),
                        decoded_sha256=hashlib.sha256(pixels).hexdigest(),
                        rmse=float(np.sqrt(np.mean(errors[accepted] ** 2)))
                        if accepted.any()
                        else None,
                        max=float(errors[accepted].max()) if accepted.any() else None,
                        bad=int(np.count_nonzero(accepted & (errors > 0.1))),
                        worst=dict(
                            center=[float(cx[worst]), float(cy[worst])],
                            truth=truth[worst].tolist(),
                            measured=result.displacement[worst].tolist(),
                            status=int(result.status[worst]),
                        ),
                    )
                    rows.append(row)
                if decoder.stdout.read(1) or decoder.wait() != 0:
                    raise RuntimeError("Unexpected demo frame count or decoder failure")
            except BaseException:
                decoder.kill()
                raise
        adapter = engine.adapter_info
    report = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        command=command,
        configuration=options,
        adapter=adapter,
        versions={name: metadata.version(name) for name in ("wgpu", "numpy", "Pillow")},
        demo_sha256=hashlib.sha256((demo / "tensile.webm").read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((demo / "manifest.json").read_bytes()).hexdigest(),
        population="Every accepted subset, including points whose analytic target lies outside the image; no interior crop",
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"frames": len(rows), "max_frame_rmse": max(row["rmse"] or 0 for row in rows)})
    )


if __name__ == "__main__":
    main()
