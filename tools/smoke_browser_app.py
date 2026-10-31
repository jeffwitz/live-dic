"""Exercise the assembled PWA under a repository subpath using the Python harness."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from urllib.parse import quote

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--software", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--export",
        action="store_true",
        help="also validate explicit CSV/JSON downloads after timing",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    site = args.site.resolve()
    server = _StaticServer(site.parent)
    server.start()
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
        browser = ICGNWebGL(backend="webgpu-wgsl", chromium_args=flags, call_timeout=180)
        with browser:
            url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1"
            if args.software:
                url += "&rate=0.0625"
            if args.offline:
                url += "&offline=1"
            browser._evaluate(f"location.assign({json.dumps(url)}); true")
            started = time.monotonic()
            deadline = started + max(120, args.duration + 60)
            processing_started = None
            while True:
                snapshot = browser._evaluate(
                    "globalThis.liveDicStatus ? {ready:liveDicStatus.ready, completed:liveDicStatus.completed, errors:liveDicStatus.errors} : null"
                )
                if snapshot and snapshot["errors"]:
                    raise RuntimeError(snapshot["errors"])
                if snapshot and snapshot["completed"] and processing_started is None:
                    processing_started = time.monotonic()
                if (
                    snapshot
                    and snapshot["completed"] >= args.frames
                    and processing_started is not None
                    and time.monotonic() - processing_started >= args.duration
                ):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"App did not produce enough results: {snapshot}")
                time.sleep(0.5)
            snapshot = browser._evaluate("JSON.parse(JSON.stringify(liveDicStatus))")
            worker_report = browser._evaluate("liveDicReport()")
            exported = None
            if args.export:
                exported = browser._evaluate("""(async()=>{
                  const original=URL.createObjectURL, blobs={};
                  URL.createObjectURL=(blob)=>{blobs[blob.type]=blob;return original(blob);};
                  try {
                    await document.getElementById('export').onclick();
                    const csv=await blobs['text/csv'].text();
                    const metadata=JSON.parse(await blobs['application/json'].text());
                    const lines=csv.trim().split('\\n');
                    const finite=lines.slice(1).every(line=>line.split(',').every(v=>Number.isFinite(Number(v))));
                    return {rows:lines.length-1, finite, grid:metadata.grid,
                      readbackWallMs:metadata.readbackWallMs, configuration:metadata.configuration,
                      adapter:metadata.capabilities.adapter, roi:metadata.roi,
                      timingBoundary:'Explicit export request while paused -> full field mapped and decoded; acquisition age excluded'};
                  } finally {URL.createObjectURL=original;}
                })()""")
                if (
                    not exported["finite"]
                    or exported["rows"] != exported["grid"][0] * exported["grid"][1]
                ):
                    raise AssertionError(f"Invalid CSV export: {exported}")
            if args.offline:
                browser._evaluate("navigator.serviceWorker.ready.then(()=>true)")
                server.close()
                browser._evaluate("location.reload(); true")
                offline_deadline = time.monotonic() + 120
                while True:
                    offline = browser._evaluate(
                        "globalThis.liveDicStatus ? JSON.parse(JSON.stringify(liveDicStatus)) : null"
                    )
                    if offline and offline["errors"]:
                        raise RuntimeError(offline["errors"])
                    if offline and offline["completed"] >= 10:
                        break
                    if time.monotonic() > offline_deadline:
                        raise RuntimeError("Offline reload did not complete ten fields")
                    time.sleep(0.5)
            visual = browser._evaluate("""(async()=>{
              const image=await createImageBitmap(document.getElementById('field'));
              const canvas=new OffscreenCanvas(32,18), context=canvas.getContext('2d');
              context.drawImage(image,0,0,32,18);image.close();
              const rgba=context.getImageData(0,0,32,18).data;
              const values=Array.from({length:32*18},(_,i)=>rgba[i*4]);
              const mean=values.reduce((a,b)=>a+b,0)/values.length;
              return Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/values.length);
            })()""")
            if visual < 3:
                raise AssertionError(f"Rendered canvas is blank: standard deviation {visual}")
            results = snapshot["results"]
            if not results or any(not isinstance(row.get("accepted"), int) for row in results):
                raise AssertionError("Missing finite field diagnostics")
            acceptance = [row["accepted"] / row["selected"] for row in results if row["selected"]]
            if min(acceptance) < 0.70:
                raise AssertionError(
                    f"Demo acceptance below predeclared smoke threshold: {min(acceptance)}"
                )
            report = dict(
                url=url,
                elapsed_seconds=time.monotonic() - started,
                snapshot=snapshot,
                worker_report=worker_report,
                processing_span_seconds=(
                    worker_report["samples"][-1]["completedMs"]
                    - worker_report["samples"][0]["workerAvailableMs"]
                )
                / 1000,
                details=browser._evaluate('document.getElementById("details").textContent'),
                version=browser._evaluate("navigator.userAgent"),
                timing_boundary="Worker frame available -> observable GPU field and small HUD diagnostics; full field not copied",
                minimum_acceptance=min(acceptance),
                canvas_std=visual,
                offline_reload=args.offline,
                console_errors=snapshot["errors"],
                export=exported,
            )
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if args.screenshot:
                import websockets

                async def screenshot():
                    async with websockets.connect(
                        browser._websocket_url, origin=browser._devtools_origin
                    ) as socket:
                        await socket.send(
                            json.dumps(
                                dict(
                                    id=1,
                                    method="Page.captureScreenshot",
                                    params=dict(format="png", captureBeyondViewport=True),
                                )
                            )
                        )
                        while True:
                            response = json.loads(await socket.recv())
                            if response.get("id") == 1:
                                args.screenshot.write_bytes(
                                    base64.b64decode(response["result"]["data"])
                                )
                                return

                asyncio.run(screenshot())
            print(
                json.dumps(
                    dict(
                        completed=snapshot["completed"],
                        duration=report["elapsed_seconds"],
                        minimum_acceptance=min(acceptance),
                    )
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
