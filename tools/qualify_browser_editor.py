"""Exercise real ROI pointer editing, UHD plate loads and GPU display oracles."""

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


async def protocol(browser, commands):
    import websockets

    results = []
    async with websockets.connect(
        browser._websocket_url, origin=browser._devtools_origin, max_size=32 * 1024 * 1024
    ) as socket:
        for index, (method, params) in enumerate(commands, 1):
            await socket.send(json.dumps(dict(id=index, method=method, params=params)))
            while True:
                response = json.loads(await socket.recv())
                if response.get("id") == index:
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    results.append(response.get("result", {}))
                    break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
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
            chromium_args=(*flags, "--window-size=1600,1200"),
            call_timeout=180,
        ) as browser:
            url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate"
            browser._evaluate(f"location.assign({json.dumps(url)});true")
            deadline = time.monotonic() + 180
            while True:
                state = browser._evaluate(
                    "globalThis.liveDicStatus ? {completed:liveDicStatus.completed,errors:liveDicStatus.errors} : null"
                )
                if state and state["errors"]:
                    raise RuntimeError(state["errors"])
                if state and state["completed"]:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Plate did not initialize")
                time.sleep(0.25)
            # A constant synthetic field tests actual shader output, not just a copied formula.
            renderer = browser._evaluate("""(async()=>{
              const {FieldRenderer}=await import('./render/field.js');
              const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice();
              const canvas=new OffscreenCanvas(128,128),renderer=new FieldRenderer(device,canvas);
              const values=new Float32Array(64*8);for(let i=0;i<64;i++)values.set([4,.003,.002,2,.002,-.001,1,3],i*8);
              const buffer=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(buffer,0,values);
              const texture=device.createTexture({size:[128,128],format:'rgba8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
              const engine={device,fieldBuffer:buffer,imageTexture:texture};
              const result={width:128,height:128,grid:[8,8],subsetSize:16,subsetStride:16,config:{motionModel:'affine'}};
              async function pixels(arrows){renderer.draw(engine,result,{mode:7,scale:.01,opacity:1,arrows,arrowGain:3,arrowStep:2});await device.queue.onSubmittedWorkDone();
                const bitmap=canvas.transferToImageBitmap(),read=new OffscreenCanvas(128,128),ctx=read.getContext('2d');ctx.drawImage(bitmap,0,0);bitmap.close();return ctx.getImageData(0,0,128,128).data;}
              const plain=await pixels(false),arrows=await pixels(true),pixel=[...plain.slice((64*128+64)*4,(64*128+64)*4+3)];
              const changed=arrows.filter((v,i)=>v!==plain[i]).length;
              const read=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,read,0,values.byteLength);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
              const unchanged=new Float32Array(read.getMappedRange()).every((v,i)=>v===values[i]);read.unmap();
              for(let i=0;i<64;i++)values[i*8+6]=-3;device.queue.writeBuffer(buffer,0,values);
              const invalid0=await pixels(false),invalid1=await pixels(true),invalidEqual=invalid0.every((v,i)=>v===invalid1[i]);
              renderer.destroy();buffer.destroy();read.destroy();texture.destroy();device.destroy();
              return {pixel,arrowChangedChannels:changed,unchanged,invalidEqual};
            })()""")
            # eVM = 1/300, fixed scale .01, white -> red linear colour map.
            expected = [238, 182, 177]
            assert max(abs(a - b) for a, b in zip(renderer["pixel"], expected, strict=True)) <= 2, (
                renderer
            )
            assert (
                renderer["arrowChangedChannels"] > 100
                and renderer["unchanged"]
                and renderer["invalidEqual"]
            ), renderer
            original = browser._evaluate("liveDicStatus.results.at(-1).plateScore")
            assert original["coverage"] >= 0.90 and original["displacementRmsePx"] < 0.05, original
            browser._evaluate("document.getElementById('draw').onclick().then(()=>true)")

            def pointer(points, tool):
                browser._evaluate(
                    f"document.getElementById('shape').value={json.dumps(tool)};document.getElementById('shape').onchange();true"
                )
                box = browser._evaluate(
                    "(()=>{const b=document.getElementById('roi').getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,height:b.height};})()"
                )
                coords = [
                    (box["x"] + x / 3840 * box["width"], box["y"] + y / 2160 * box["height"])
                    for x, y in points
                ]
                commands = []
                if tool == "polygon":
                    for x, y in coords:
                        commands.extend(
                            [
                                (
                                    "Input.dispatchMouseEvent",
                                    dict(
                                        type="mousePressed",
                                        x=x,
                                        y=y,
                                        button="left",
                                        buttons=1,
                                        clickCount=1,
                                    ),
                                ),
                                (
                                    "Input.dispatchMouseEvent",
                                    dict(
                                        type="mouseReleased",
                                        x=x,
                                        y=y,
                                        button="left",
                                        buttons=0,
                                        clickCount=1,
                                    ),
                                ),
                            ]
                        )
                else:
                    x, y = coords[0]
                    commands.append(
                        (
                            "Input.dispatchMouseEvent",
                            dict(
                                type="mousePressed",
                                x=x,
                                y=y,
                                button="left",
                                buttons=1,
                                clickCount=1,
                            ),
                        )
                    )
                    for x, y in coords[1:]:
                        commands.append(
                            (
                                "Input.dispatchMouseEvent",
                                dict(type="mouseMoved", x=x, y=y, buttons=1),
                            )
                        )
                    commands.append(
                        (
                            "Input.dispatchMouseEvent",
                            dict(
                                type="mouseReleased",
                                x=x,
                                y=y,
                                button="left",
                                buttons=0,
                                clickCount=1,
                            ),
                        )
                    )
                asyncio.run(protocol(browser, commands))
                if tool == "polygon":
                    browser._evaluate("document.getElementById('finish-shape').onclick();true")

            pointer([(450, 400), (1000, 800)], "rectangle")
            pointer([(550, 500), (900, 700)], "ellipse")
            pointer([(2500, 800), (2750, 1050)], "circle")
            browser._evaluate(
                "document.getElementById('roi-width').value=150;document.getElementById('update-shape').onclick();true"
            )
            pointer([(2575, 875), (2600, 875)], "select")
            pointer([(1200, 1500), (1450, 1500), (1350, 1750)], "polygon")
            before = browser._evaluate("liveDicRoi.shapes()")
            assert len(before) == 6, before
            browser._evaluate(
                "document.getElementById('delete-roi').onclick();document.getElementById('undo-roi').onclick();true"
            )
            assert browser._evaluate("liveDicRoi.shapes()") == before
            circle = next(s for s in before if s["type"] == "circle" and s["id"] != "hole")
            assert abs(circle["x"] - 2525) < 1 and circle["width"] == circle["height"] == 150, (
                circle
            )

            def screenshot(name, full=True):
                data = asyncio.run(
                    protocol(
                        browser,
                        [
                            (
                                "Page.captureScreenshot",
                                dict(format="png", captureBeyondViewport=full),
                            )
                        ],
                    )
                )[0]["data"]
                (args.screenshots / name).write_bytes(base64.b64decode(data))

            screenshot("editor.png")
            browser._evaluate("document.getElementById('apply-roi').onclick().then(()=>true)")
            edited = browser._evaluate("liveDicStatus.results.at(-1).plateScore")
            assert edited["coverage"] >= 0.90 and edited["displacementRmsePx"] < 0.05, edited
            browser._evaluate(
                "(async()=>{const c=document.getElementById('plate-load');c.value='225';await c.onchange();return true;})()"
            )
            second = browser._evaluate("liveDicStatus.results.at(-1).plateScore")
            assert second["coverage"] >= 0.90 and second["displacementRmsePx"] < 0.05, second
            browser._evaluate(
                "document.getElementById('zoom').value=4;document.getElementById('zoom').onchange();document.getElementById('centre-view').onclick();true"
            )
            browser._evaluate("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
            zoom = browser._evaluate(
                "(()=>{const v=document.getElementById('image-viewport');return {width:v.clientWidth,height:v.clientHeight,scrollWidth:v.scrollWidth,scrollHeight:v.scrollHeight,left:v.scrollLeft,top:v.scrollTop,canvas:document.getElementById('field').getBoundingClientRect().toJSON()};})()"
            )
            zoom["centrePixel"] = browser._evaluate(
                "(async()=>{const b=await createImageBitmap(document.getElementById('field')),c=new OffscreenCanvas(1,1),ctx=c.getContext('2d');ctx.drawImage(b,1920,1080,1,1,0,0,1,1);b.close();return [...ctx.getImageData(0,0,1,1).data];})()"
            )
            data = asyncio.run(
                protocol(
                    browser,
                    [("Page.captureScreenshot", dict(format="png", captureBeyondViewport=False))],
                )
            )[0]["data"]
            (args.screenshots / "viewport.png").write_bytes(base64.b64decode(data))
            screenshot("von-mises-arrows.png", full=False)
            assert all(abs(v - 122) <= 1 for v in zoom["centrePixel"][:3]), zoom
            errors = browser._evaluate("liveDicStatus.errors")
            assert not errors, errors
            report = dict(
                url=url,
                software=args.software,
                version=browser._evaluate("navigator.userAgent"),
                build=json.loads((site / "build.json").read_text()),
                renderer=renderer,
                zoom=zoom,
                original=original,
                edited=edited,
                second_load=second,
                shapes=before,
                details=browser._evaluate("document.getElementById('details').textContent"),
                errors=errors,
            )
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(
                json.dumps(
                    dict(renderer=renderer, original=original, edited=edited, second_load=second)
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
