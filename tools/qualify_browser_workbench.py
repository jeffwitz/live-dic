"""Exercise the actual UHD results UI, extraction tools and NumPy archive interchange."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
from qualify_browser_editor import protocol
from qualify_local_sources import choose_files, wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer


def click(browser, selector, x, y):
    rect = browser._evaluate(
        f"(()=>{{const r=document.querySelector({json.dumps(selector)}).getBoundingClientRect();return {{x:r.x,y:r.y,width:r.width,height:r.height}};}})()"
    )
    px, py = rect["x"] + x / 3840 * rect["width"], rect["y"] + y / 2160 * rect["height"]
    asyncio.run(
        protocol(
            browser,
            [
                (
                    "Input.dispatchMouseEvent",
                    dict(type="mousePressed", x=px, y=py, button="left", clickCount=1),
                ),
                (
                    "Input.dispatchMouseEvent",
                    dict(type="mouseReleased", x=px, y=py, button="left", clickCount=1),
                ),
            ],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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
        with ICGNWebGL(
            backend="webgpu-wgsl",
            startup_timeout=60,
            chromium_args=(*flags, "--window-size=1600,1200"),
            call_timeout=240,
        ) as browser:
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.completed>=1")
            browser._evaluate("document.getElementById('help').focus();true")
            asyncio.run(
                protocol(
                    browser,
                    [
                        (
                            "Input.dispatchKeyEvent",
                            dict(type="keyDown", key=" ", code="Space", windowsVirtualKeyCode=32),
                        ),
                        (
                            "Input.dispatchKeyEvent",
                            dict(type="keyUp", key=" ", code="Space", windowsVirtualKeyCode=32),
                        ),
                    ],
                )
            )
            assert browser._evaluate("document.getElementById('quick-guide').open"), (
                "Space on a focused button must activate that button"
            )
            browser._evaluate("document.getElementById('quick-guide').close();true")
            browser._evaluate(
                "document.querySelector('[data-pane=view]').click();document.getElementById('freeze-field').onclick().then(()=>true)"
            )
            wait(browser, "globalThis.liveDicWorkbench?.report().recipe!=null")
            baseline = browser._evaluate("liveDicWorkbench.report()")
            assert baseline["count"] == 1 and baseline["reviewing"], baseline
            assert baseline["statistics"]["count"] > 30000, baseline
            assert baseline["unsavedProject"], baseline
            assert not browser._evaluate(
                "window.dispatchEvent(new Event('beforeunload',{cancelable:true}))"
            )
            for key, value in [
                ("review-component", "vonMises"),
                ("post-smoothing", "gaussian"),
                ("strain-measure", "green"),
            ]:
                browser._evaluate(
                    f"document.getElementById({json.dumps(key)}).value={json.dumps(value)};document.getElementById({json.dumps(key)}).onchange();true"
                )
            wait(browser, "liveDicWorkbench.report().recipe?.strainMeasure==='green'")
            processed = browser._evaluate("liveDicWorkbench.report()")
            assert processed["recipe"]["strainSource"] == "vsg", processed
            assert processed["recipe"]["smoothing"] == "gaussian", processed
            browser._evaluate("document.querySelector('[data-inspect=line]').click();true")
            click(browser, "#inspection", 800, 900)
            click(browser, "#inspection", 1400, 900)
            profile = browser._evaluate("liveDicWorkbench.report().extraction")
            assert len(profile) > 20 and all(row["value"] is not None for row in profile), profile
            browser._evaluate("document.querySelector('[data-inspect=gauge]').click();true")
            click(browser, "#inspection", 800, 900)
            click(browser, "#inspection", 1400, 900)
            gauge = browser._evaluate("liveDicWorkbench.report().extraction")
            assert len(gauge) == 1 and np.isfinite(gauge[0]["engineeringStrain"]), gauge
            browser._evaluate("document.querySelector('[data-inspect=region]').click();true")
            click(browser, "#inspection", 800, 800)
            click(browser, "#inspection", 1400, 1100)
            region = browser._evaluate("liveDicWorkbench.report().extraction")[0]
            assert region["valid"] > 500, region
            # Capture the real download Blob; NumPy independently verifies shapes and operators.
            encoded = browser._evaluate("""(async()=>{
              const original=URL.createObjectURL;let saved;
              URL.createObjectURL=b=>{if(b.type==='application/octet-stream')saved=b;return original(b);};
              try{document.getElementById('save-npz').onclick();const bytes=new Uint8Array(await saved.arrayBuffer());
                const parts=[];for(let i=0;i<bytes.length;i+=32768)parts.push(String.fromCharCode(...bytes.subarray(i,i+32768)));
                return btoa(parts.join(''));}finally{URL.createObjectURL=original;}
            })()""")
            payload = base64.b64decode(encoded)
            archive_path = args.output / "processed-field.npz"
            archive_path.write_bytes(payload)
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                assert data["parameters"].shape == (135, 240, 6)
                assert data["roi_mask"].shape == (2160, 3840)
                recipe = json.loads(str(data["recipe_json"]))
                assert recipe == processed["recipe"], recipe
                y, x, radius = 56, 62, 2
                gy, gx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
                weights = np.exp(-(gx**2 + gy**2) / 2)

                def smooth(row, column, component):
                    values = data["parameters"][
                        row - radius : row + radius + 1,
                        column - radius : column + radius + 1,
                        component,
                    ]
                    return np.sum(weights * values) / weights.sum()

                coords = np.column_stack([np.ones(gx.size), gx.ravel() * 16, gy.ravel() * 16])
                uv = np.array(
                    [
                        [smooth(y + dy, x + dx, 0), smooth(y + dy, x + dx, 3)]
                        for dy, dx in zip(gy.ravel(), gx.ravel(), strict=True)
                    ]
                )
                slopes = np.linalg.lstsq(coords, uv, rcond=None)[0][1:]
                f = np.eye(2) + slopes.T
                strain = (f.T @ f - np.eye(2)) / 2
                expected = [strain[0, 0], strain[1, 1], strain[0, 1]]
                actual = [data[f"processed_{k}"][y, x] for k in ["exx", "eyy", "exy"]]
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
                assert np.all(data["analysis_valid"][data["status"] < 0.5] == 0)
                numeric_error = float(np.max(np.abs(np.asarray(actual) - expected)))
                yy, xx = np.mgrid[:135, :240]
                rx, ry = xx * 16 + 7.5, yy * 16 + 7.5
                chosen = (
                    (rx >= region["x0"])
                    & (rx <= region["x1"])
                    & (ry >= region["y0"])
                    & (ry <= region["y1"])
                    & (data["strain_valid"] != 0)
                )
                expected_region = data["processed_vonMises"][chosen]
                assert region["valid"] == int(chosen.sum()), region
                np.testing.assert_allclose(
                    [region["mean"], region["standardDeviation"]],
                    [expected_region.mean(), expected_region.std()],
                    atol=1e-12,
                    rtol=1e-10,
                )
            # Comparing computed data is temporary; returning restores the explicit processing recipe.
            browser._evaluate("document.getElementById('compare-computed').onclick();true")
            wait(browser, "liveDicWorkbench.report().recipe?.smoothing==='none'")
            browser._evaluate("document.getElementById('compare-computed').onclick();true")
            wait(browser, "liveDicWorkbench.report().recipe?.smoothing==='gaussian'")
            screenshots = {}
            for label, target in [("field", ".viewer"), ("analysis", "#results-workbench")]:
                browser._evaluate(
                    f"document.querySelector({json.dumps(target)}).scrollIntoView({{block:'start'}});true"
                )
                reply = asyncio.run(
                    protocol(
                        browser,
                        [
                            (
                                "Page.captureScreenshot",
                                dict(format="png", captureBeyondViewport=False),
                            )
                        ],
                    )
                )[0]
                path = args.output / f"{label}.png"
                path.write_bytes(base64.b64decode(reply["data"]))
                screenshots[label] = hashlib.sha256(path.read_bytes()).hexdigest()
            mobile = asyncio.run(
                protocol(
                    browser,
                    [
                        (
                            "Emulation.setDeviceMetricsOverride",
                            dict(width=390, height=844, deviceScaleFactor=1, mobile=True),
                        ),
                        (
                            "Runtime.evaluate",
                            dict(
                                expression="(async()=>{await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));document.getElementById('results-workbench').scrollIntoView({block:'start'});return {width:innerWidth,pageWidth:document.documentElement.scrollWidth,chartWidth:document.getElementById('profile-chart').clientWidth};})()",
                                awaitPromise=True,
                                returnByValue=True,
                            ),
                        ),
                        ("Page.captureScreenshot", dict(format="png", captureBeyondViewport=False)),
                        (
                            "Runtime.evaluate",
                            dict(
                                expression="(async()=>{document.getElementById('profile-chart').scrollIntoView({block:'center'});await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));return true;})()",
                                awaitPromise=True,
                            ),
                        ),
                        ("Page.captureScreenshot", dict(format="png", captureBeyondViewport=False)),
                        ("Emulation.clearDeviceMetricsOverride", {}),
                    ],
                )
            )
            viewport = mobile[1]["result"]["value"]
            assert viewport["width"] == 390 and viewport["pageWidth"] <= 390, viewport
            (args.output / "mobile-analysis.png").write_bytes(base64.b64decode(mobile[2]["data"]))
            (args.output / "mobile-chart.png").write_bytes(base64.b64decode(mobile[4]["data"]))
            browser._evaluate(
                "document.getElementById('return-live').onclick();document.getElementById('record-fields').checked=true;document.getElementById('record-fields').onchange();true"
            )
            browser._evaluate(
                "document.getElementById('plate-load').value='225';document.getElementById('plate-load').onchange().then(()=>true)"
            )
            wait(browser, "liveDicWorkbench.report().count===2")
            browser._evaluate(
                "document.getElementById('record-fields').checked=false;document.getElementById('record-fields').onchange();true"
            )
            # Upload through an actual file input, including Snap's separate /tmp namespace.
            visible = Path.home() / "snap/chromium/common"
            with tempfile.TemporaryDirectory(
                prefix="live-dic-npz-", dir=visible if visible.is_dir() else None
            ) as temp:
                local = Path(temp) / "field.npz"
                local.write_bytes(payload)
                asyncio.run(choose_files(browser, "#result-file", [local]))
                wait(
                    browser,
                    "liveDicWorkbench.report().count===3 && liveDicWorkbench.report().recipe?.smoothing==='gaussian'",
                )
            browser._evaluate(
                "document.querySelector('.viewer').scrollIntoView({block:'start'});document.querySelector('[data-inspect=gauge]').click();true"
            )
            click(browser, "#inspection", 800, 900)
            click(browser, "#inspection", 1400, 900)
            browser._evaluate("document.getElementById('extract-history').onclick().then(()=>true)")
            history = browser._evaluate("liveDicWorkbench.report().extraction")
            assert len(history) == 3 and all(row["value"] is not None for row in history), history
            assert abs(history[0]["value"] - history[2]["value"]) < 1e-12, history
            # Native CLI-style compressed sequence: the UI must request absent image extent and frame.
            with tempfile.TemporaryDirectory(
                prefix="live-dic-native-npz-", dir=visible if visible.is_dir() else None
            ) as temp:
                native_file = Path(temp) / "native-sequence.npz"
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    config = json.loads(str(data["config_json"]))
                    config.pop("roi", None)
                    np.savez_compressed(
                        native_file,
                        parameters=np.stack([data["parameters"], data["parameters"]]),
                        status=np.stack([data["status"], data["status"]]),
                        iterations=np.stack([data["iterations"], data["iterations"]]),
                        config_json=np.asarray(json.dumps(config)),
                    )
                asyncio.run(choose_files(browser, "#result-file", [native_file]))
                wait(browser, "document.getElementById('import-geometry').open")
                browser._evaluate(
                    "document.getElementById('import-width').value=3840;document.getElementById('import-height').value=2160;document.getElementById('import-frame').value=2;document.getElementById('import-geometry-form').requestSubmit();true"
                )
                wait(
                    browser,
                    "liveDicWorkbench.report().count===4 && liveDicWorkbench.report().recipe!=null",
                )
                geometry = browser._evaluate("liveDicWorkbench.report().provenance.importGeometry")
                assert (
                    geometry["frameIndex"] == 1
                    and geometry["width"] == 3840
                    and geometry["height"] == 2160
                ), geometry
            # Export immediately after an edit: the archive must await that analysis recipe.
            browser._evaluate(
                "document.getElementById('post-window').value='9';document.getElementById('post-window').onchange();true"
            )
            project_encoded = browser._evaluate("""(async()=>{
              const original=URL.createObjectURL;let saved;URL.createObjectURL=b=>{if(b.type==='application/zip')saved=b;return original(b);};
              try{await document.getElementById('save-project').onclick();const bytes=new Uint8Array(await saved.arrayBuffer()),parts=[];
                for(let i=0;i<bytes.length;i+=32768)parts.push(String.fromCharCode(...bytes.subarray(i,i+32768)));return btoa(parts.join(''));}
              finally{URL.createObjectURL=original;}
            })()""")
            assert not browser._evaluate("liveDicWorkbench.report().unsavedProject")
            assert browser._evaluate(
                "window.dispatchEvent(new Event('beforeunload',{cancelable:true}))"
            )
            project_payload = base64.b64decode(project_encoded)
            with zipfile.ZipFile(io.BytesIO(project_payload)) as project_zip:
                manifest = json.loads(project_zip.read("manifest.json"))
                selected_recipe = next(
                    field["recipe"]
                    for field in manifest["fields"]
                    if field["originalId"] == manifest["settings"]["selectedId"]
                )
                assert selected_recipe["window"] == 9, selected_recipe
            with tempfile.TemporaryDirectory(
                prefix="live-dic-project-", dir=visible if visible.is_dir() else None
            ) as temp:
                project_file = Path(temp) / "experiment.dicproject"
                project_file.write_bytes(project_payload)
                asyncio.run(choose_files(browser, "#project-file", [project_file]))
                wait(
                    browser,
                    "liveDicWorkbench.report().count===8 && liveDicWorkbench.report().recipe!=null",
                )
            browser._evaluate(
                "const select=document.getElementById('saved-fields');select.value=select.options[4].value;select.onchange();true"
            )
            wait(browser, "liveDicWorkbench.report().recipe?.smoothing==='gaussian'")
            restored_region = browser._evaluate("liveDicWorkbench.report().extraction")[0]
            assert restored_region == region, (restored_region, region)
            browser._evaluate("document.getElementById('apply-recipe-analysis').onclick();true")
            assert "6 matching" in browser._evaluate(
                "document.getElementById('processing-batch-state').textContent"
            )
            start_selected = browser._evaluate("liveDicWorkbench.report().selected")
            browser._evaluate("document.getElementById('replay-results').onclick();true")
            wait(
                browser,
                f"liveDicWorkbench.report().selected!=={start_selected}&&liveDicWorkbench.report().recipe?.smoothing==='gaussian'",
            )
            browser._evaluate(
                "if(liveDicWorkbench.report().replaying)document.getElementById('replay-results').onclick();true"
            )
            browser._evaluate(
                "document.getElementById('result-seek').value=0;document.getElementById('result-seek').oninput();true"
            )
            wait(
                browser,
                "liveDicWorkbench.report().selected===1&&liveDicWorkbench.report().recipe?.smoothing==='gaussian'",
            )
            report = dict(
                build=json.loads((site / "build.json").read_text()),
                software=args.software,
                user_agent=browser._evaluate("navigator.userAgent"),
                baseline=baseline,
                processed=processed,
                profile=profile,
                history=history,
                native_import_geometry=geometry,
                gauge=gauge,
                region=region,
                operator_max_error=numeric_error,
                archive_sha256=hashlib.sha256(payload).hexdigest(),
                project_sha256=hashlib.sha256(project_payload).hexdigest(),
                project_bytes=len(project_payload),
                screenshots=screenshots,
                final=browser._evaluate("liveDicWorkbench.report()"),
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            assert not report["errors"], report["errors"]
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "fields": report["final"]["count"],
                        "operator_max_error": numeric_error,
                        "errors": report["errors"],
                    }
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
