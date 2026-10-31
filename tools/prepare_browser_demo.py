"""Create derived 1080p demo assets; never write to the archived UHD fixtures.

Pillow/NumPy form images; optional developer tool ffmpeg encodes VP9 for the browser demonstration.
The shipped app and Python installation never invoke an encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "DIC/LucasKanadeICGN"
sys.path.insert(0, str(PROJECT / "python/tests"))
from temporal_cases import tensile_stretches, warp  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "web-app/demo")
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("Asset regeneration needs ffmpeg with libvpx-vp9; runtime does not")
    source = PROJECT / "assets/uhd_tensile/reference.png"
    truth_file = source.parent / "machine_truth.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    reference = np.array(Image.open(source).resize((1920, 1080), Image.Resampling.BOX))
    Image.fromarray(reference).save(args.output / "reference.png")
    movie = args.output / "tensile.webm"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        "1920x1080",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "22",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=in_range=full:out_range=full",
        "-color_range",
        "pc",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        "-deadline",
        "good",
        "-cpu-used",
        "6",
        "-row-mt",
        "1",
        "-threads",
        "4",
        str(movie),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    hashes = []
    stretches = tensile_stretches()
    try:
        for index, stretch in enumerate(stretches):
            frame = warp(reference, stretch=stretch)
            hashes.append(hashlib.sha256(frame).hexdigest())
            process.stdin.write(frame.tobytes())
            if index % 20 == 0:
                print("formed", index, flush=True)
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Demo encoding failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    manifest = dict(
        width=1920,
        height=1080,
        frame_rate=30,
        frames=len(stretches),
        description="Analytic tensile replay from a BOX-downsampled archived reference. Float64 bilinear inverse warp, reflected boundaries, nearest uint8. Derived images, not original native-UHD pixels.",
        purpose="demonstration, education and field screening; not metrology qualification",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        truth_sha256=hashlib.sha256(truth_file.read_bytes()).hexdigest(),
        encoding="VP9 CRF 22, full-range grayscale YUV420, BT.709; compressed demonstration, not a numerical regression fixture",
        encoder=subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        command=command,
        pixel_sha256=hashes,
        stretch=stretches.tolist(),
        files={
            name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for name in ("reference.png", "tensile.webm")
        },
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
