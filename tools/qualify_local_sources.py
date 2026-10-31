"""Exercise real local file pickers, chronological pairs and whole-ROI failures."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWebGL, ICGNWGPUFullPyramid
from lucas_kanade_icgn.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "DIC/LucasKanadeICGN"


async def choose_files(browser, selector, files):
    import websockets

    async with websockets.connect(
        browser._websocket_url, origin=browser._devtools_origin
    ) as socket:
        serial = 0

        async def call(method, params):
            nonlocal serial
            serial += 1
            await socket.send(json.dumps(dict(id=serial, method=method, params=params)))
            while True:
                reply = json.loads(await socket.recv())
                if reply.get("id") == serial:
                    if "error" in reply:
                        raise RuntimeError(reply["error"])
                    return reply["result"]

        document = await call("DOM.getDocument", {})
        node = await call(
            "DOM.querySelector", dict(nodeId=document["root"]["nodeId"], selector=selector)
        )
        await call(
            "DOM.setFileInputFiles", dict(nodeId=node["nodeId"], files=[str(p) for p in files])
        )


def wait(browser, condition, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        errors = browser._evaluate("globalThis.liveDicStatus?.errors??[]")
        if errors:
            raise AssertionError(errors)
        if browser._evaluate(condition):
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"Timeout: {condition}")
        time.sleep(0.15)


def export(browser):
    result = browser._evaluate("""(async()=>{
      const original=URL.createObjectURL,blobs={};URL.createObjectURL=b=>{blobs[b.type]=b;return original(b);};
      try{await document.getElementById('export').onclick();return {csv:await blobs['text/csv'].text(),metadata:JSON.parse(await blobs['application/json'].text())};}
      finally{URL.createObjectURL=original;}
    })()""")
    rows = np.genfromtxt(io.StringIO(result.pop("csv")), delimiter=",", skip_header=1)
    assert np.isfinite(rows).all()
    result["status"] = rows[:, 2].astype(np.int32)
    result["parameters"] = rows[:, 4:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software", action="store_true")
    parser.add_argument(
        "--video", type=Path, help="optional UHD local video; defaults to the shipped 1080p video"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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
    site = args.site.resolve()
    server = _StaticServer(site.parent)
    server.start()
    rows = []
    try:
        # Snap Chromium has its own /tmp; stage user-selected files in its visible home.
        snap_common = Path.home() / "snap/chromium/common"
        with (
            tempfile.TemporaryDirectory(
                prefix="live-dic-local-input-", dir=snap_common if snap_common.is_dir() else None
            ) as temp,
            ICGNWebGL(
                backend="webgpu-wgsl",
                startup_timeout=60,
                chromium_args=(*flags, "--window-size=1600,1200"),
                call_timeout=240,
            ) as browser,
        ):
            folder = Path(temp) / "frames"
            folder.mkdir()
            originals = [
                PROJECT / "assets/uhd_plate_hole" / name
                for name in ["reference.png", "frame_095.png", "frame_225.png"]
            ]
            for i, path in enumerate(originals):
                shutil.copyfile(path, folder / f"frame_{i}.png")
            Image.fromarray(np.zeros((2160, 3840), np.uint8)).save(folder / "frame_3.png")
            shutil.copyfile(originals[0], folder / "frame_4.png")
            shutil.copyfile(originals[1], folder / "frame_5.png")
            Image.fromarray(np.zeros((32, 48), np.uint8)).save(folder / "frame_6.png")
            (folder / "README.txt").write_text("Ignored non-image negative control")
            url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1"
            browser._evaluate(f"location.assign({json.dumps(url)});true")
            wait(browser, "globalThis.liveDicStatus?.completed>=2")
            asyncio.run(choose_files(browser, "#image-folder", [folder]))
            wait(
                browser,
                "globalThis.liveDicSourceState && liveDicSourceState().source==='images' && !liveDicSourceState().loadingSource && !liveDicSourceState().localBusy",
            )
            state = browser._evaluate("liveDicSourceState()")
            assert state["fileCount"] == 7, state
            assert len(state["settings"]["ignored"]) == 1, state
            browser._evaluate("document.getElementById('draw').onclick().then(()=>true)")
            browser._evaluate(
                "(async()=>{const {platePreset}=await import('./ui/roi-geometry.js');liveDicRoi.reset(3840,2160,platePreset());return true;})()"
            )
            browser._evaluate("document.getElementById('apply-roi').onclick().then(()=>true)")
            browser._evaluate(
                "document.getElementById('local-seek').value=1;document.getElementById('local-seek').onchange().then(()=>true)"
            )

            def snapshot(
                label, reference_index, target_index, reference_path=None, target_path=None
            ):
                data = export(browser)
                metadata = data["metadata"]
                comparison = metadata["comparison"]
                assert comparison["reference"]["index"] == reference_index, comparison
                assert comparison["target"]["index"] == target_index, comparison
                selected = metadata["configuration"]["roi"]["selected"]
                assert selected == 32220, (label, selected)
                status = data["status"]
                assert np.count_nonzero(status == -3) == 32400 - selected
                record = dict(
                    label=label,
                    comparison=comparison,
                    selected=selected,
                    status_counts={
                        str(int(k)): int(v)
                        for k, v in zip(*np.unique(status, return_counts=True), strict=True)
                    },
                )
                if reference_path is not None:
                    reference = np.array(Image.open(reference_path))
                    target = np.array(Image.open(target_path))
                    y, x = np.ogrid[:2160, :3840]
                    mask = (x - 1920) ** 2 + (y - 1080) ** 2 >= 112**2
                    with ICGNWGPUFullPyramid(
                        device_type="cpu" if args.software else "gpu",
                        outlier_policy="median",
                        initialization="temporal",
                        temporal_prediction="velocity",
                        temporal_fine_correction=True,
                        divergence_patience=3,
                        divergence_min_iterations=4,
                    ) as native:
                        native.set_reference(reference, roi_mask=mask)
                        expected = native.register(target)
                    common = (status > 0.5) & expected.accepted_mask.ravel()
                    delta = data["parameters"][common] - expected.parameters.reshape(-1, 6)[common]
                    record.update(
                        parameter_rmse=float(np.sqrt(np.mean(delta**2))) if common.any() else None,
                        parameter_max=float(np.max(np.abs(delta))) if common.any() else None,
                        status_mismatches=int(np.count_nonzero(status != expected.status.ravel())),
                    )
                    assert record["status_mismatches"] / status.size <= 0.005, record
                    if common.any():
                        assert (
                            record["parameter_rmse"] <= 0.002 and record["parameter_max"] <= 0.02
                        ), record
                np.savez_compressed(
                    args.output / f"{label}.npz", parameters=data["parameters"], status=status
                )
                rows.append(record)
                print(record, flush=True)
                return record

            snapshot("fixed", 0, 1, originals[0], originals[1])
            browser._evaluate(
                "document.getElementById('comparison').value='consecutive';document.getElementById('comparison').onchange().then(()=>true)"
            )
            browser._evaluate(
                "document.getElementById('local-seek').value=2;document.getElementById('local-seek').onchange().then(()=>true)"
            )
            snapshot("consecutive", 1, 2, originals[1], originals[2])
            for index in (3, 4, 5):
                browser._evaluate(
                    f"document.getElementById('local-seek').value={index};document.getElementById('local-seek').onchange().then(()=>true)"
                )
                record = snapshot(f"failure-recovery-{index}", index - 1, index)
                if index == 4:
                    assert record["comparison"]["referenceAdmissible"] == 0, record
                    assert not any(int(k) > 0 for k in record["status_counts"]), record
                if index == 5:
                    assert (
                        sum(n for k, n in record["status_counts"].items() if int(k) > 0) > 30000
                    ), record
            browser._evaluate("document.getElementById('next-image').onclick().then(()=>true)")
            message = browser._evaluate("document.getElementById('message').textContent")
            assert "never resized" in message, message
            assert browser._evaluate("liveDicSourceState().index") == 5
            # New file picker replaces the sequence; current dimensions are retained, never resampled.
            video_file = (args.video or PROJECT / "web-app/demo/tensile.webm").resolve()
            asyncio.run(choose_files(browser, "#video-file", [video_file]))
            wait(
                browser,
                "liveDicSourceState().source==='video-file' && !liveDicSourceState().loadingSource",
            )
            # Fixed mode restores the user capture before explicitly seeking a local file.
            browser._evaluate(
                "document.getElementById('comparison').value='fixed';document.getElementById('comparison').onchange().then(()=>true)"
            )
            browser._evaluate(
                "document.getElementById('local-seek').value=.5;document.getElementById('local-seek').onchange().then(()=>true)"
            )
            data = export(browser)
            assert abs(data["metadata"]["comparison"]["target"]["timeSeconds"] - 0.5) < 0.001, data[
                "metadata"
            ]["comparison"]
            video_state = browser._evaluate("liveDicSourceState()")
            assert video_state["videoPaused"], video_state
            before = browser._evaluate("liveDicStatus.completed")
            browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
            wait(browser, f"liveDicStatus.completed>{before + 1}")
            browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
            wait(browser, "liveDicSourceState().inFlight===0")
            report = dict(
                build=json.loads((site / "build.json").read_text()),
                software=args.software,
                version=browser._evaluate("navigator.userAgent"),
                rows=rows,
                dimension_error=message,
                video=dict(
                    file=video_file.name,
                    sha256=hashlib.sha256(video_file.read_bytes()).hexdigest(),
                    state=video_state,
                    configuration=data["metadata"]["configuration"],
                ),
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            assert not report["errors"], report["errors"]
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
