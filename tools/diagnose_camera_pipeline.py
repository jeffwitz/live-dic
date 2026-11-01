"""Measure a clearly simulated camera pipeline; never open a physical camera.

A static native-size crop of the checked-in UHD tensile reference is delivered
through Chromium's fake Y4M camera at 60 Hz. These controls separate reference
preparation from solving and recording; they do not qualify a real sensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from qualify_local_sources import wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]


def median(rows, key):
    return float(np.median([row[key] for row in rows])) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", choices=["1920x1080", "3840x2160"], default="1920x1080")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.duration <= 3600 or not 0 <= args.warmup <= 60:
        parser.error("duration must be 1..3600 seconds; warmup must be 0..60 seconds")
    site = args.site.resolve()
    source = ROOT / "DIC/LucasKanadeICGN/assets/uhd_tensile/reference.png"
    width, height = map(int, args.resolution.split("x"))
    x, y = (900, 500) if width == 1920 else (0, 0)
    pixels = np.asarray(Image.open(source).convert("L"))[y : y + height, x : x + width]
    assert pixels.shape == (height, width)
    flags = (
        (
            "--use-angle=swiftshader",
            "--enable-unsafe-webgpu",
            "--use-webgpu-adapter=swiftshader",
            "--use-vulkan=swiftshader",
        )
        if args.software
        else ()
    )
    server = _StaticServer(site.parent)
    server.start()
    visible = Path.home() / "snap/chromium/common"
    report = {
        "build": json.loads((site / "build.json").read_text()),
        "cases": [],
        "kind": "Static simulated camera control; no physical camera or general real-time qualification",
        "source": {
            "file": str(source.relative_to(ROOT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "crop": [x, y, width, height],
            "format": "Y4M, original luma, neutral chroma",
            "frameRate": 60,
        },
        "duration_seconds": args.duration,
        "warmup_seconds": args.warmup,
        "software_requested": args.software,
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="live-dic-camera-diagnosis-", dir=visible if visible.is_dir() else None
        ) as temp:
            capture = Path(temp) / "static.y4m"
            with capture.open("wb") as file:
                file.write(f"YUV4MPEG2 W{width} H{height} F60:1 Ip A1:1 C420jpeg\n".encode())
                for _ in range(2):
                    file.write(b"FRAME\n" + pixels.tobytes() + bytes([128]) * (width * height // 2))
            with ICGNWebGL(
                backend="webgpu-wgsl",
                startup_timeout=60,
                call_timeout=240,
                chromium_args=(
                    *flags,
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    f"--use-file-for-fake-video-capture={capture}",
                ),
            ) as browser:
                url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1"
                browser._evaluate(f"location.assign({json.dumps(url)});true")
                wait(browser, "globalThis.liveDicStatus?.completed>=2")
                browser._evaluate(
                    f"document.getElementById('resolution').value={json.dumps(args.resolution)};document.getElementById('camera').onclick().then(()=>true)"
                )
                wait(
                    browser,
                    "liveDicSourceState().source==='camera'&&!liveDicSourceState().loadingSource&&liveDicStatus.completed>=5",
                )
                for mode, recording in [("fixed", False), ("fixed", True), ("consecutive", False)]:
                    browser._evaluate(
                        "(async()=>{if(!liveDicSourceState().paused)await document.getElementById('pause').onclick();while(liveDicSourceState().inFlight)await new Promise(r=>setTimeout(r,20));return true;})()"
                    )
                    browser._evaluate(
                        f"document.getElementById('comparison').value={json.dumps(mode)};document.getElementById('comparison').onchange().then(()=>true)"
                    )
                    browser._evaluate(
                        f"document.getElementById('record-fields').checked={json.dumps(recording)};document.getElementById('record-fields').onchange().then(()=>true)"
                    )
                    browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                    time.sleep(args.warmup)
                    start = browser._evaluate("({completed:liveDicStatus.completed})")
                    previous = browser._evaluate("liveDicReport()")
                    worker_start = (
                        previous["samples"][-1]["completedMs"] if previous["samples"] else -1
                    )
                    started = time.monotonic()
                    time.sleep(args.duration)
                    stop = browser._evaluate(
                        "({completed:liveDicStatus.completed,fps:document.getElementById('fps').textContent,age:document.getElementById('age').textContent,state:liveDicSourceState(),fields:liveDicWorkbench.report().count,pipeline:document.getElementById('pipeline-timing').textContent,errors:liveDicStatus.errors})"
                    )
                    elapsed = time.monotonic() - started
                    assert not stop["errors"], stop["errors"]
                    worker = browser._evaluate("liveDicReport()")
                    rows = [
                        r
                        for r in worker["samples"]
                        if r["completedMs"] > worker_start and r["comparison"]["mode"] == mode
                    ]
                    case = {
                        "mode": mode,
                        "recording": recording,
                        "elapsed_seconds": elapsed,
                        "completed_fps": (stop["completed"] - start["completed"]) / elapsed,
                        "median_preparation_ms": median(rows, "referencePreparationMs"),
                        "median_observable_solver_ms": median(rows, "gpuFieldObservableMs"),
                        "start": start,
                        "stop": stop,
                        "samples": rows,
                    }
                    report["cases"].append(case)
                    report["capabilities"] = worker["capabilities"]
                    report["user_agent"] = browser._evaluate("navigator.userAgent")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(
                        json.dumps(
                            {
                                k: case[k]
                                for k in [
                                    "mode",
                                    "recording",
                                    "completed_fps",
                                    "median_preparation_ms",
                                    "median_observable_solver_ms",
                                ]
                            }
                        ),
                        flush=True,
                    )
    finally:
        server.close()


if __name__ == "__main__":
    main()
