"""Exercise camera ownership and UHD stream controls using a clearly labelled fake device.

No physical camera is opened. The temporary Y4M luma planes come from the
unmodified archived UHD plate images; chroma is neutral. This is a browser
acquisition/lifecycle test, not new analytical fixtures or an FPS qualification.
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
from qualify_local_sources import export, wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]


def write_capture(path):
    originals = ROOT / "DIC/LucasKanadeICGN/assets/uhd_plate_hole"
    records = []
    with path.open("wb") as file:
        file.write(b"YUV4MPEG2 W3840 H2160 F10:1 Ip A1:1 C420jpeg\n")
        chroma = bytes([128]) * (3840 * 2160 // 2)
        for name in ["reference.png", "frame_095.png", "frame_225.png"]:
            source = originals / name
            image = np.asarray(Image.open(source).convert("L"))
            assert image.shape == (2160, 3840)
            file.write(b"FRAME\n")
            file.write(image.tobytes())
            file.write(chroma)
            records.append(dict(file=name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument(
        "--fallback", action="store_true", help="also force the video callback path"
    )
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.duration <= 3600:
        parser.error("--duration must be between 1 and 3600 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    site = args.site.resolve()
    server = _StaticServer(site.parent)
    server.start()
    visible = Path.home() / "snap/chromium/common"
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
    try:
        with tempfile.TemporaryDirectory(
            prefix="live-dic-camera-", dir=visible if visible.is_dir() else None
        ) as temp:
            capture = Path(temp) / "plate.y4m"
            inputs = write_capture(capture)
            flags = (
                *flags,
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={capture}",
                "--window-size=1600,1200",
            )
            with ICGNWebGL(
                backend="webgpu-wgsl", startup_timeout=60, call_timeout=240, chromium_args=flags
            ) as browser:
                url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate"
                browser._evaluate(f"location.assign({json.dumps(url)});true")
                wait(browser, "globalThis.liveDicStatus?.completed>=1")
                browser._evaluate("""(()=>{const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                    globalThis.testCameraTracks=[]; globalThis.testCameraAcquire=async options=>{
                        const stream=await original(options); testCameraTracks.push(...stream.getTracks());return stream;};
                    navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Simulated denied camera permission','NotAllowedError');};
                    document.getElementById('resolution').value='3840x2160';return true;})()""")
                browser._evaluate("document.getElementById('camera').onclick().then(()=>true)")
                assert "Simulated denied" in browser._evaluate(
                    "document.getElementById('message').textContent"
                )
                assert browser._evaluate(
                    "liveDicSourceState().source===null&&testCameraTracks.length===0"
                )
                browser._evaluate(
                    "navigator.mediaDevices.getUserMedia=testCameraAcquire;document.getElementById('camera').onclick().then(()=>true)"
                )
                wait(
                    browser,
                    "liveDicSourceState().source==='camera'&&!liveDicSourceState().loadingSource&&liveDicStatus.completed>=3",
                )
                initial = browser._evaluate("liveDicSourceState()")
                actual = initial["settings"]["actual"]
                assert [actual["width"], actual["height"]] == [3840, 2160], initial
                assert initial["settings"]["requested"] == dict(
                    width=3840, height=2160, frameRate=60
                ), initial
                assert "Actual 3840" in browser._evaluate(
                    "document.getElementById('camera-summary').textContent"
                )
                start = time.monotonic()
                samples = []
                while time.monotonic() - start < args.duration:
                    time.sleep(min(3, max(0.1, args.duration - (time.monotonic() - start))))
                    sample = browser._evaluate(
                        "({completed:liveDicStatus.completed,errors:liveDicStatus.errors,state:liveDicSourceState(),dropped:Number(document.getElementById('drops').textContent)})"
                    )
                    assert not sample["errors"], sample
                    assert sample["state"]["inFlight"] <= 1, sample
                    samples.append(dict(elapsed_seconds=time.monotonic() - start, **sample))
                assert samples[-1]["completed"] > samples[0]["completed"], samples
                browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                wait(browser, "liveDicSourceState().paused&&liveDicSourceState().inFlight===0")
                fixed = export(browser)
                assert fixed["parameters"].shape == (32400, 6)
                assert np.isfinite(fixed["parameters"]).all()
                assert np.count_nonzero(fixed["status"] > 0) > 20000
                assert fixed["metadata"]["acquisition"]["requested"]["frameRate"] == 60
                browser._evaluate(
                    "document.getElementById('comparison').value='consecutive';document.getElementById('comparison').onchange().then(()=>true)"
                )
                before = browser._evaluate("liveDicStatus.completed")
                browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                wait(browser, f"liveDicStatus.completed>={before + 2}")
                browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                wait(browser, "liveDicSourceState().inFlight===0")
                pair = export(browser)
                comparison = pair["metadata"]["comparison"]
                assert comparison["mode"] == "consecutive", comparison
                assert comparison["target"]["ordinal"] - comparison["reference"]["ordinal"] == 1, (
                    comparison
                )
                assert (
                    comparison["target"]["timestampUs"] > comparison["reference"]["timestampUs"]
                ), comparison
                assert np.isfinite(pair["parameters"]).all()
                browser._evaluate(
                    "document.getElementById('record-fields').checked=true;document.getElementById('record-fields').onchange();document.getElementById('pause').onclick();true"
                )
                wait(browser, "liveDicWorkbench.report().count>=2")
                browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                wait(browser, "liveDicSourceState().inFlight===0")
                browser._evaluate(
                    "document.getElementById('record-fields').checked=false;document.getElementById('record-fields').onchange();true"
                )
                assert browser._evaluate("document.getElementById('saved-fields').value===''")
                browser._evaluate(
                    "const saved=document.getElementById('saved-fields');saved.value=saved.options[saved.options.length-1].value;saved.onchange();true"
                )
                wait(
                    browser,
                    "liveDicWorkbench.report().reviewing&&liveDicWorkbench.report().recipe!=null",
                )
                recording_label = browser._evaluate(
                    "document.getElementById('result-title').textContent"
                )
                assert "Input " in recording_label, recording_label
                browser._evaluate(
                    "const stopped=testCameraTracks.at(-1);stopped.stop();stopped.dispatchEvent(new Event('ended'));true"
                )
                wait(
                    browser,
                    "liveDicSourceState().settings.deliveryState==='stopped'&&liveDicSourceState().paused",
                )
                assert browser._evaluate(
                    "document.getElementById('pause').textContent==='Restart camera'"
                )
                assert browser._evaluate(
                    "document.getElementById('run-state').textContent.includes('Saved field')"
                )
                browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                wait(
                    browser,
                    "!liveDicSourceState().loadingSource&&liveDicSourceState().settings.deliveryState==='running'",
                )
                assert browser._evaluate(
                    "testCameraTracks[0].readyState==='ended'&&testCameraTracks.at(-1).readyState==='live'"
                )
                fallback = None
                if args.fallback:
                    browser._evaluate(
                        "globalThis.MediaStreamTrackProcessor=undefined;document.getElementById('comparison').value='fixed';document.getElementById('comparison').onchange().then(()=>true)"
                    )
                    # Restart the chosen camera through the actual selection control.
                    browser._evaluate(
                        "const select=document.getElementById('camera-device');select.value=select.options[1].value;select.onchange();true"
                    )
                    wait(
                        browser, "!liveDicSourceState().loadingSource&&!liveDicSourceState().paused"
                    )
                    before = browser._evaluate("liveDicStatus.completed")
                    wait(browser, f"liveDicStatus.completed>={before + 3}")
                    fallback = browser._evaluate("liveDicSourceState()")
                    assert (
                        fallback["settings"]["frameDelivery"] == "Video-frame callback fallback"
                    ), fallback
                    assert browser._evaluate("testCameraTracks[0].readyState==='ended'"), (
                        "Old camera track was not stopped"
                    )
                browser._evaluate("document.getElementById('plate').onclick().then(()=>true)")
                wait(
                    browser,
                    "liveDicSourceState().source==='plate'&&!liveDicSourceState().loadingSource",
                )
                assert browser._evaluate(
                    "testCameraTracks.every(track=>track.readyState==='ended')"
                ), "Camera retained after source replacement"
                report = dict(
                    build=json.loads((site / "build.json").read_text()),
                    software=args.software,
                    user_agent=browser._evaluate("navigator.userAgent"),
                    kind="simulated camera correctness; not physical camera or FPS qualification",
                    temporary_capture=dict(
                        format="Y4M 4:2:0; original luma; neutral chroma",
                        frame_rate=10,
                        images=inputs,
                    ),
                    initial=initial,
                    soak=samples,
                    fixed_accepted=int(np.count_nonzero(fixed["status"] > 0)),
                    comparison=comparison,
                    fallback=fallback,
                    all_tracks_closed=True,
                    ended_restart=True,
                    recording_label=recording_label,
                    ended_event_kind="synthetic ended event after stopping the fake track; not physical disconnect",
                    errors=browser._evaluate("liveDicStatus.errors"),
                )
                assert not report["errors"], report["errors"]
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                print(
                    json.dumps(
                        dict(
                            accepted=report["fixed_accepted"],
                            tracks_closed=True,
                            errors=report["errors"],
                        )
                    )
                )
    finally:
        server.close()


if __name__ == "__main__":
    main()
