"""Verify UHD saved-field analysis with navigator.gpu unavailable.

The Chromium harness bootstraps normally, then a new page simulates a missing
WebGPU API before importing application code. No correlation runs on that page.
The input is an actual NPZ from the usual UHD workbench qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from qualify_local_sources import choose_files, wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer


async def review_page(browser, url, no_worker=False):
    import websockets

    async with websockets.connect(
        browser._websocket_url, origin=browser._devtools_origin
    ) as socket:
        serial = 0

        async def call(method, **params):
            nonlocal serial
            serial += 1
            await socket.send(json.dumps(dict(id=serial, method=method, params=params)))
            while True:
                result = json.loads(await socket.recv())
                if result.get("id") == serial:
                    return result

        await call("Page.enable")
        await call(
            "Page.addScriptToEvaluateOnNewDocument",
            source="Object.defineProperty(navigator,'gpu',{value:undefined,configurable:true});globalThis.OffscreenCanvas=undefined;globalThis.reviewOnlySimulation=true;"
            + ("globalThis.Worker=undefined;" if no_worker else ""),
        )
        await call("Page.navigate", url=url)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            reply = await call(
                "Runtime.evaluate",
                expression="globalThis.reviewOnlySimulation&&document.getElementById('message')?.textContent.includes('WebGPU is unavailable for correlation')",
                returnByValue=True,
            )
            if reply.get("result", {}).get("result", {}).get("value"):
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("Review-only page did not initialize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument(
        "--field",
        type=Path,
        required=True,
        help="UHD processed-field.npz produced by qualify_browser_workbench.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--software", action="store_true", help="bootstrap the harness on SwiftShader"
    )
    parser.add_argument(
        "--no-worker", action="store_true", help="also exercise main-thread CPU analysis"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.field, allow_pickle=False) as archive:
        assert archive["parameters"].shape == (135, 240, 6)
        expected = archive["processed_vonMises"][archive["strain_valid"] != 0]
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
        with (
            tempfile.TemporaryDirectory(
                prefix="live-dic-review-", dir=visible if visible.is_dir() else None
            ) as temp,
            ICGNWebGL(
                backend="webgpu-wgsl",
                startup_timeout=60,
                call_timeout=180,
                chromium_args=(*flags, "--window-size=1600,1200"),
            ) as browser,
        ):
            url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1"
            asyncio.run(review_page(browser, url, args.no_worker))
            assert browser._evaluate(
                "navigator.gpu===undefined&&document.getElementById('pane-source').hidden&&!document.getElementById('pane-view').hidden"
            )
            assert browser._evaluate(
                "['camera','demo','plate','open-video','open-folder','open-images'].every(id=>document.getElementById(id).disabled)"
            )
            source = Path(temp) / "uhd-field.npz"
            shutil.copyfile(args.field, source)
            asyncio.run(choose_files(browser, "#result-file", [source]))
            wait(
                browser,
                "liveDicWorkbench.report().count===1&&liveDicWorkbench.report().recipe!=null",
            )
            browser._evaluate(
                "document.getElementById('review-component').value='vonMises';document.getElementById('review-component').onchange();true"
            )
            report = browser._evaluate("liveDicWorkbench.report()")
            assert report["statistics"]["count"] == expected.size
            np.testing.assert_allclose(
                [
                    report["statistics"]["mean"],
                    report["statistics"]["min"],
                    report["statistics"]["max"],
                ],
                [expected.mean(), expected.min(), expected.max()],
                atol=1e-12,
                rtol=1e-10,
            )
            assert browser._evaluate("liveDicStatus.completed===0&&liveDicStatus.errors.length===0")
            output = dict(
                build=json.loads((site / "build.json").read_text()),
                kind="WebGPU API absence simulated in application page; CPU saved-field analysis only",
                no_worker=args.no_worker,
                software_bootstrap=args.software,
                source_sha256=hashlib.sha256(args.field.read_bytes()).hexdigest(),
                field=report,
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            (args.output / "report.json").write_text(
                json.dumps(output, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        count=report["statistics"]["count"], correlations=0, errors=output["errors"]
                    )
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
