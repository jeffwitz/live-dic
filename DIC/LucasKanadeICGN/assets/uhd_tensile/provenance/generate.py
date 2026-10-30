#!/usr/bin/env python3
"""Generate one deterministic 4K tensile sequence and its exact machine curve."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
W, H, COUNT = 3840, 2160, 200
E, NU, YIELD, K, N = 210000.0, 0.30, 235.0, 450.0, 0.25
A0 = 25.0 * 2.0  # mm²
SEED = 20261012


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state(t: int) -> dict:
    yield_eng = math.expm1(YIELD / E)
    eng = yield_eng * t / 40 if t <= 40 else yield_eng + (0.10 - yield_eng) * (t - 40) / 159
    eta = math.log1p(eng)
    if eta <= YIELD / E + 1e-14:
        stress, plastic = E * eta, 0.0
    else:
        plastic = brentq(lambda p: (YIELD + K * p**N) / E + p - eta,
                         0.0, eta, xtol=1e-15)
        stress = YIELD + K * plastic**N
    transverse = -NU * stress / E - plastic / 2
    lx, ly = math.exp(eta), math.exp(transverse)
    return dict(frame=t, time_s=t / 30, engineering_axial=eng,
                true_axial=eta, plastic_true=plastic, stress_true_MPa=stress,
                stretch_x=lx, stretch_y=ly, engineering_transverse=ly - 1,
                machine_force_N=stress * A0 * ly * ly)


def texture() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    # Stationary, band-limited synthetic speckles; all frames share this master.
    noise = rng.standard_normal((H, W), dtype=np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    binary = (smooth > np.quantile(smooth, 0.58)).astype(np.float32)
    blurred = cv2.GaussianBlur(binary, (0, 0), 0.65)
    return np.clip(np.rint(205 - 160 * blurred), 16, 235).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "inputs")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = [state(t) for t in range(COUNT)]
    with (out / "machine_truth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    base = texture()
    cv2.imwrite(str(out / "anchor_preview.png"), base)
    x = np.arange(W, dtype=np.float32)[None, :]
    y = np.arange(H, dtype=np.float32)[:, None]
    u = bytes([128]) * (W * H // 4)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pixel_format", "yuv420p", "-video_size", f"{W}x{H}", "-framerate", "30",
               "-i", "pipe:0", "-frames:v", str(COUNT), "-an", "-c:v", "libx264",
               "-preset", "medium", "-pix_fmt", "yuv420p", "-qp", "24", "-g", "240",
               "-bf", "0", "-x264-params",
               "ref=1:scenecut=0:open-gop=0:rc-lookahead=0:mbtree=0:repeat-headers=1",
               "-f", "h264", str(out / "tensile.h264")]
    env = {key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"}
    raw_hash = hashlib.sha256()
    with (out / "ffmpeg_stderr.txt").open("wb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors, env=env)
        try:
            for s in rows:
                # OpenCV map addresses pixel centres by their integer pixel indices.
                mx = np.broadcast_to((x - W / 2) / s["stretch_x"] + W / 2, (H, W)).copy()
                my = np.broadcast_to((y - H / 2) / s["stretch_y"] + H / 2, (H, W)).copy()
                image = cv2.remap(base, mx, my, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
                if s["frame"] in (40, 100, 199):
                    cv2.imwrite(str(out / f"frame_{s['frame']:03d}_preview.png"), image)
                payload = image.tobytes()
                for part in (payload, u, u):
                    raw_hash.update(part)
                    process.stdin.write(part)
                if s["frame"] % 20 == 0:
                    print(f"rendered {s['frame'] + 1}/{COUNT}", flush=True)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError((out / "ffmpeg_stderr.txt").read_text()[-3000:])
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,pix_fmt,refs,has_b_frames", "-show_entries",
        "frame=pict_type,key_frame", "-of", "json", str(out / "tensile.h264")], text=True, env=env))
    frames = probe["frames"]
    if len(frames) != COUNT or [r["pict_type"] for r in frames] != ["I"] + ["P"] * (COUNT - 1):
        raise AssertionError("unexpected frame count or GOP")
    manifest = dict(material=dict(E_MPa=E, nu=NU, yield_MPa=YIELD, Ludwik_K_MPa=K,
                                  Ludwik_n=N, area0_mm2=A0), dimensions=[W, H], frames=COUNT,
                    seed=SEED, master_yuv420_sha256=raw_hash.hexdigest(),
                    bitstream_sha256=sha(out / "tensile.h264"),
                    bitstream_bytes=(out / "tensile.h264").stat().st_size,
                    command=command, ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True, env=env).splitlines()[0],
                    opencv=cv2.__version__, numpy=np.__version__, probe=probe["streams"][0],
                    first_last_truth=[rows[0], rows[-1]])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"bitstream": str(out / "tensile.h264"), "sha256": manifest["bitstream_sha256"]}))


if __name__ == "__main__":
    main()
