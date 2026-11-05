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

from live_dic import ICGNBrowser
from lucas_kanade_icgn.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]


def median(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", choices=["1920x1080", "3840x2160"], default="1920x1080")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--software", action="store_true")
    parser.add_argument("--overlap-pipeline", action="store_true")
    parser.add_argument("--display-rate", type=int, choices=[0, 1, 5, 10, 15, 30, 60], default=30)
    parser.add_argument("--suspend-preview", action="store_true")
    parser.add_argument(
        "--solver-profile",
        choices=["app", "common-zero"],
        default="app",
        help="preserve application settings or explicitly match zero-initialized native/browser benchmarks",
    )
    parser.add_argument(
        "--storage-probe",
        action="store_true",
        help="measure identical storage admission boundaries on old/new sites; reference metadata must include workerArrivalEpochMs",
    )
    parser.add_argument("--preview-width", type=int, choices=[0, 1280, 1920], default=1920)
    parser.add_argument("--comparison", choices=["all", "fixed", "consecutive"], default="all")
    parser.add_argument("--recording", choices=["all", "on", "off"], default="all")
    parser.add_argument(
        "--gpu-reference", action="store_true", help="opt into persistent GPU reference preparation"
    )
    args = parser.parse_args()
    if args.suspend_preview and not args.overlap_pipeline:
        parser.error("--suspend-preview requires --overlap-pipeline")
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
        "gpu_reference_requested": args.gpu_reference,
        "overlap_pipeline_requested": args.overlap_pipeline,
        "display_rate_requested": args.display_rate,
        "preview_suspended_requested": args.suspend_preview,
        "preview_width_requested": args.preview_width,
        "comparison_requested": args.comparison,
        "recording_requested": args.recording,
        "solver_profile_requested": args.solver_profile,
        "storage_probe_requested": args.storage_probe,
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
            with ICGNBrowser(
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
                if args.gpu_reference:
                    browser._evaluate(
                        "document.getElementById('gpu-reference-preparation').checked=true;true"
                    )
                if args.solver_profile == "common-zero":
                    browser._evaluate("""(()=>{
                      document.getElementById('custom-solver').checked=true;
                      for(const [id,value] of Object.entries({'motion-model':'affine','subset-size':'16','subset-stride':'16','interpolation':'bspline','pyramid-levels':'2','coarse-iterations':'8','fine-iterations':'16','outlier-policy':'median'}))document.getElementById(id).value=value;
                      document.getElementById('temporal-start').checked=false;return true;
                    })()""")
                if args.storage_probe:
                    browser._evaluate("""(async()=>{
                      const {FieldStorage}=await import('./ui/field-storage.js');
                      globalThis.benchmarkAdmissions=[];const add=FieldStorage.prototype.addMany;
                      FieldStorage.prototype.addMany=async function(entries,...args){
                        const started=performance.now(),result=await add.call(this,entries,...args),finished=performance.now();
                        for(const entry of entries)benchmarkAdmissions.push({admittedEpochMs:performance.timeOrigin+finished,storageMs:finished-started,ordinal:entry.metadata.comparison?.target?.ordinal,workerToAdmissionMs:performance.timeOrigin+finished-entry.metadata.workerArrivalEpochMs,retained:!!result.failure});
                        return result;
                      };return true;
                    })()""")
                if args.overlap_pipeline:
                    display_value = "off" if args.suspend_preview else str(args.display_rate)
                    browser._evaluate(
                        f"document.getElementById('overlap-pipeline').checked=true;document.getElementById('display-rate').value={json.dumps(display_value)};document.getElementById('preview-width').value={json.dumps(str(args.preview_width))};true"
                    )
                browser._evaluate(
                    f"document.getElementById('resolution').value={json.dumps(args.resolution)};document.getElementById('camera').onclick().then(()=>true)"
                )
                browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                wait(
                    browser,
                    "liveDicSourceState().source==='camera'&&!liveDicSourceState().loadingSource&&liveDicStatus.completed>=5",
                )
                for mode, recording in [
                    ("fixed", False),
                    ("fixed", True),
                    ("consecutive", False),
                    ("consecutive", True),
                ]:
                    if args.comparison != "all" and mode != args.comparison:
                        continue
                    if args.recording != "all" and recording != (args.recording == "on"):
                        continue
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
                    start = browser._evaluate(
                        "({completed:liveDicStatus.completed,diskFields:liveDicWorkbench.report().storage.diskFields,epochMs:performance.timeOrigin+performance.now()})"
                    )
                    previous = browser._evaluate("liveDicReport()")
                    worker_start = (
                        previous["samples"][-1]["completedMs"] if previous["samples"] else -1
                    )
                    started = time.monotonic()
                    time.sleep(args.duration)
                    stop = browser._evaluate(
                        "({completed:liveDicStatus.completed,fps:document.getElementById('fps').textContent,age:document.getElementById('age').textContent,state:liveDicSourceState(),canvas:[document.getElementById('field').width,document.getElementById('field').height],fields:liveDicWorkbench.report().count,diskFields:liveDicWorkbench.report().storage.diskFields,storage:liveDicWorkbench.report().storage,epochMs:performance.timeOrigin+performance.now(),pipeline:document.getElementById('pipeline-timing').textContent,errors:liveDicStatus.errors})"
                    )
                    host_elapsed = time.monotonic() - started
                    elapsed = (stop["epochMs"] - start["epochMs"]) / 1000
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
                        "host_elapsed_seconds": host_elapsed,
                        "throughput_boundary": "Main-thread count snapshots on its monotonic clock; computed replies and committed disk fields reported separately",
                        "completed_fps": (stop["completed"] - start["completed"]) / elapsed,
                        "committed_fps": (stop["diskFields"] - start["diskFields"]) / elapsed,
                        "median_preparation_ms": median(rows, "referencePreparationMs"),
                        "median_observable_solver_ms": median(rows, "gpuFieldObservableMs"),
                        "start": start,
                        "stop": stop,
                        "samples": rows,
                    }
                    if previous.get("presentation") and worker.get("presentation"):
                        before, after = previous["presentation"], worker["presentation"]
                        display_elapsed = (
                            after["snapshotEpochMs"] - before["snapshotEpochMs"]
                        ) / 1000
                        case["presentation"] = {
                            "before": before,
                            "after": after,
                            "elapsed_seconds": display_elapsed,
                            "draws": after["draws"] - before["draws"],
                            "bitmaps": after["bitmaps"] - before["bitmaps"],
                            "draw_fps": (after["draws"] - before["draws"]) / display_elapsed,
                            "draw_submission_wall_ms": after["drawWallMs"] - before["drawWallMs"],
                            "bitmap_wall_ms": after["bitmapWallMs"] - before["bitmapWallMs"],
                            "timing_boundary": "Worker presentation counter snapshots; CPU/API wall times, not GPU timestamps",
                        }
                    case["pipeline"] = browser._evaluate(
                        "globalThis.liveDicPipelineReport?.()??null"
                    )
                    pipeline = case["pipeline"]
                    samples = (
                        (
                            pipeline["recording"]["samples"]
                            if args.overlap_pipeline
                            else pipeline["synchronousAdmissions"]
                        )
                        if pipeline
                        else []
                    )
                    if args.storage_probe:
                        samples = browser._evaluate("benchmarkAdmissions")
                    case["admission_samples"] = [
                        sample
                        for sample in samples
                        if start["epochMs"] <= sample["admittedEpochMs"] <= stop["epochMs"]
                    ]
                    # Finish outstanding work outside the timed window, then
                    # score the actual last committed field of the static scene.
                    browser._evaluate("""(async()=>{
                      if(!liveDicSourceState().paused)await document.getElementById('pause').onclick();
                      while(liveDicSourceState().inFlight)await new Promise(resolve=>setTimeout(resolve,10));
                      await liveDicWorkbench.flush();return true;
                    })()""")
                    case["final_state"] = browser._evaluate(
                        "({source:liveDicSourceState(),storage:liveDicWorkbench.report().storage,pipeline:globalThis.liveDicPipelineReport?.()??null})"
                    )
                    if recording:
                        case["quality"] = browser._evaluate("""(async()=>{
                          const store=liveDicWorkbench.storage;if(!store.backend||!store.session)return null;
                          const saved=await store.backend.read(store.session.id,store.session.nextId-1);if(!saved)return null;
                          const field=saved.field;let count=0,sum=0,max=0;
                          for(let i=0;i<field.status.length;i++)if(field.status[i]>=.5){
                            const u=field.parameters[6*i],v=field.parameters[6*i+3],error=Math.hypot(u,v);
                            sum+=error*error;max=Math.max(max,error);count++;
                          }
                          return {width:field.width,height:field.height,grid:field.grid,config:field.config,
                            finite:field.parameters.every(Number.isFinite),accepted:count,total:field.status.length,
                            displacementRmse:count?Math.sqrt(sum/count):null,maxDisplacementError:count?max:null,
                            parameterType:field.parameters.constructor.name,truth:'Identical static fake-camera frames; displacement (0,0)'};
                        })()""")
                    case["adjacent_pairs"] = (
                        all(
                            row["comparison"]["target"]["ordinal"]
                            - row["comparison"]["reference"].get(
                                "ordinal", row["comparison"]["target"]["ordinal"] - 1
                            )
                            == 1
                            for row in rows
                        )
                        if mode == "consecutive"
                        else None
                    )
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
                                    "committed_fps",
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
