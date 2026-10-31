"""Qualify full-resolution mask-image input, vector clipping and PNG round trips."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from qualify_browser_editor import protocol
from qualify_local_sources import choose_files, export, wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer


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
    y, x = np.ogrid[:2160, :3840]
    mask = ((x - 1920) ** 2 + (y - 1080) ** 2 >= 112**2).astype(np.uint8)

    def digest(values):
        return hashlib.sha256(values.tobytes()).hexdigest()

    visible = Path.home() / "snap/chromium/common"
    try:
        with (
            tempfile.TemporaryDirectory(
                prefix="live-dic-mask-images-", dir=visible if visible.is_dir() else None
            ) as temp,
            ICGNWebGL(
                backend="webgpu-wgsl",
                startup_timeout=60,
                chromium_args=(*flags, "--window-size=1600,1200"),
                call_timeout=240,
            ) as browser,
        ):
            folder = Path(temp)
            binary = folder / "material.png"
            Image.fromarray(mask * 255).save(binary)
            wrong = folder / "wrong.png"
            Image.fromarray(np.zeros((32, 48), np.uint8)).save(wrong)
            gray = folder / "gray.png"
            pixels = mask * 255
            pixels[1080, 1920] = 127
            Image.fromarray(pixels).save(gray)
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.completed>=1")

            def choose(path):
                asyncio.run(choose_files(browser, "#mask-file", [path]))
                wait(browser, "!liveDicSourceState().loadingSource")

            def mask_hash():
                return browser._evaluate(
                    "(async()=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',liveDicRoi.mask()))].map(v=>v.toString(16).padStart(2,'0')).join(''))()"
                )

            choose(binary)
            assert not browser._evaluate("document.getElementById('roi-editor').hidden")
            assert mask_hash() == digest(mask)
            browser._evaluate("document.getElementById('apply-roi').onclick().then(()=>true)")
            initial = export(browser)
            assert initial["metadata"]["configuration"]["roi"]["selected"] == 32220
            assert initial["metadata"]["roiGeometry"]["baseImage"]["threshold"] is None
            for path, expected in [(wrong, "never resized"), (gray, "not opaque black/white")]:
                choose(path)
                message = browser._evaluate("document.getElementById('message').textContent")
                assert expected in message, message
                assert mask_hash() == digest(mask)
            browser._evaluate(
                "document.getElementById('mask-conversion').value='threshold';document.getElementById('mask-conversion').onchange();true"
            )
            choose(gray)
            assert mask_hash() == digest(mask)
            browser._evaluate("document.getElementById('apply-roi').onclick().then(()=>true)")
            threshold = export(browser)
            assert threshold["metadata"]["roiGeometry"]["baseImage"]["threshold"] == 128
            np.testing.assert_array_equal(initial["status"], threshold["status"])
            np.testing.assert_allclose(
                initial["parameters"], threshold["parameters"], atol=1e-7, rtol=0
            )
            browser._evaluate("document.getElementById('draw').onclick().then(()=>true)")
            browser._evaluate("document.getElementById('shape').value='circle';true")
            rect = browser._evaluate(
                "(()=>{const r=document.getElementById('roi').getBoundingClientRect();return{x:r.x,y:r.y,w:r.width,h:r.height};})()"
            )

            def point(px, py):
                return dict(
                    x=rect["x"] + px / 3840 * rect["w"], y=rect["y"] + py / 2160 * rect["h"]
                )

            asyncio.run(
                protocol(
                    browser,
                    [
                        (
                            "Input.dispatchMouseEvent",
                            dict(
                                type="mousePressed", button="left", clickCount=1, **point(480, 480)
                            ),
                        ),
                        (
                            "Input.dispatchMouseEvent",
                            dict(type="mouseMoved", button="left", buttons=1, **point(560, 560)),
                        ),
                        (
                            "Input.dispatchMouseEvent",
                            dict(
                                type="mouseReleased", button="left", clickCount=1, **point(560, 560)
                            ),
                        ),
                    ],
                )
            )
            shapes = browser._evaluate("liveDicRoi.shapes()")
            assert len(shapes) == 2, shapes
            circle = shapes[-1]
            cx = circle["x"] + circle["width"] / 2
            cy = circle["y"] + circle["height"] / 2
            expected = mask & (
                ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) >= (circle["width"] / 2) ** 2
            )
            assert mask_hash() == digest(expected.astype(np.uint8))
            encoded = browser._evaluate("""(async()=>{const original=URL.createObjectURL;let saved;URL.createObjectURL=b=>{if(b.type==='image/png')saved=b;return original(b);};
              try{await document.getElementById('save-mask').onclick();const data=new Uint8Array(await saved.arrayBuffer()),parts=[];for(let i=0;i<data.length;i+=32768)parts.push(String.fromCharCode(...data.subarray(i,i+32768)));return btoa(parts.join(''));}finally{URL.createObjectURL=original;}})()""")
            png = base64.b64decode(encoded)
            (args.output / "edited-mask.png").write_bytes(png)
            decoded = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            np.testing.assert_array_equal(
                decoded, np.repeat((expected * 255)[..., None], 3, axis=2)
            )
            browser._evaluate("document.getElementById('apply-roi').onclick().then(()=>true)")
            result = export(browser)
            selected = expected.reshape(135, 16, 240, 16).all(axis=(1, 3))
            np.testing.assert_array_equal(result["status"].reshape(135, 240) != -3, selected)
            assert result["metadata"]["configuration"]["roi"]["selected"] == int(selected.sum())
            browser._evaluate("document.getElementById('draw').onclick().then(()=>true)")
            browser._evaluate("document.getElementById('undo-roi').onclick();true")
            assert mask_hash() == digest(mask)
            # An unapplied undo must not alter the existing numerical result or its provenance.
            unapplied = export(browser)
            np.testing.assert_array_equal(result["status"], unapplied["status"])
            assert len(unapplied["metadata"]["roiGeometry"]["shapes"]) == 2
            shot = asyncio.run(
                protocol(
                    browser,
                    [("Page.captureScreenshot", dict(format="png", captureBeyondViewport=False))],
                )
            )[0]
            (args.output / "editor.png").write_bytes(base64.b64decode(shot["data"]))
            browser._evaluate("document.getElementById('cancel-roi').onclick().then(()=>true)")
            assert mask_hash() == digest(expected.astype(np.uint8))
            browser._evaluate("document.getElementById('draw').onclick().then(()=>true)")
            browser._evaluate(
                "document.getElementById('shape').value='circle';document.getElementById('add-centred').onclick();true"
            )
            assert mask_hash() != digest(expected.astype(np.uint8))

            async def check_touch():
                # Emulation belongs to the CDP connection. Keep it attached through
                # layout, touch input and screenshot rather than testing desktop twice.
                import websockets

                async with websockets.connect(
                    browser._websocket_url,
                    origin=browser._devtools_origin,
                    max_size=32 * 1024 * 1024,
                ) as socket:
                    serial = 0

                    async def call(method, **params):
                        nonlocal serial
                        serial += 1
                        await socket.send(json.dumps(dict(id=serial, method=method, params=params)))
                        while True:
                            response = json.loads(await socket.recv())
                            if response.get("id") == serial:
                                if "error" in response:
                                    raise RuntimeError(response["error"])
                                return response.get("result", {})

                    async def evaluate(expression):
                        reply = await call(
                            "Runtime.evaluate",
                            expression=expression,
                            returnByValue=True,
                            awaitPromise=True,
                        )
                        if "exceptionDetails" in reply:
                            raise RuntimeError(reply["exceptionDetails"])
                        return reply["result"].get("value")

                    await call(
                        "Emulation.setDeviceMetricsOverride",
                        width=390,
                        height=844,
                        deviceScaleFactor=1,
                        mobile=True,
                    )
                    await call("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=2)
                    geometry = await evaluate("""(async()=>{
                        await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
                        document.querySelector('.viewer').scrollIntoView({block:'start'});
                        const r=document.getElementById('roi').getBoundingClientRect();
                        return {x:r.x,y:r.y,w:r.width,h:r.height,viewport:innerWidth,
                            pageWidth:document.documentElement.scrollWidth};})()""")
                    assert geometry["viewport"] == 390, geometry
                    assert geometry["pageWidth"] <= 390, geometry
                    before = await evaluate("liveDicRoi.shapes().at(-1)")
                    tx = geometry["x"] + (before["x"] + before["width"]) / 3840 * geometry["w"]
                    ty = geometry["y"] + (before["y"] + before["height"]) / 2160 * geometry["h"]
                    await call(
                        "Input.dispatchTouchEvent",
                        type="touchStart",
                        touchPoints=[dict(x=tx, y=ty)],
                    )
                    await call(
                        "Input.dispatchTouchEvent",
                        type="touchMove",
                        touchPoints=[dict(x=tx + 8, y=ty + 4)],
                    )
                    await call("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])
                    after = await evaluate("liveDicRoi.shapes().at(-1)")
                    assert after["width"] != before["width"], (before, after)
                    shot = await call(
                        "Page.captureScreenshot", format="png", captureBeyondViewport=False
                    )
                    (args.output / "mobile-editor.png").write_bytes(base64.b64decode(shot["data"]))
                    return before, after, geometry

            touch_before, touch_after, touch_geometry = asyncio.run(check_touch())
            browser._evaluate("document.getElementById('cancel-roi').onclick().then(()=>true)")
            assert mask_hash() == digest(expected.astype(np.uint8))
            report = dict(
                build=json.loads((site / "build.json").read_text()),
                software=args.software,
                user_agent=browser._evaluate("navigator.userAgent"),
                touch_resize=dict(before=touch_before, after=touch_after),
                emulated_viewport=touch_geometry,
                original_selected=32220,
                clipped_selected=int(selected.sum()),
                mask_sha256=digest(mask),
                clipped_sha256=digest(expected.astype(np.uint8)),
                exported_png_sha256=hashlib.sha256(png).hexdigest(),
                geometry=result["metadata"]["roiGeometry"],
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            assert not report["errors"], report["errors"]
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(dict(selected=report["clipped_selected"], errors=report["errors"])))
    finally:
        server.close()


if __name__ == "__main__":
    main()
