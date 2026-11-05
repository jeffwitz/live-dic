"""Qualify real TIFF image/mask import and PNG-equivalent fields on original UHD data."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from qualify_local_sources import choose_files, export, wait

from live_dic import ICGNBrowser
from live_dic.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]


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
    server_stopped = False
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
        visible = Path.home() / "snap/chromium/common"
        with (
            tempfile.TemporaryDirectory(dir=visible if visible.is_dir() else None) as temporary,
            ICGNBrowser(backend="webgpu-wgsl", chromium_args=flags, call_timeout=300) as browser,
        ):
            paths, pixel_hashes = {}, {}
            original = ROOT / "DIC/LucasKanadeICGN/assets/uhd_plate_hole"
            for depth in [8, 16]:
                paths[depth] = []
                for i, name in enumerate(["reference.png", "frame_095.png"]):
                    pixels = np.asarray(Image.open(original / name).convert("L"))
                    values = pixels if depth == 8 else pixels.astype(np.uint16) * 257
                    path = Path(temporary) / f"image_{i:03}_{depth}.tiff"
                    Image.fromarray(values).save(
                        path, compression="tiff_lzw" if depth == 8 else "tiff_adobe_deflate"
                    )
                    paths[depth].append(path)
                    rgba = np.empty((*pixels.shape, 4), np.uint8)
                    rgba[..., :3] = pixels[..., None]
                    rgba[..., 3] = 255
                    pixel_hashes[path.name] = hashlib.sha256(rgba.tobytes()).hexdigest()
            mask = np.ones((2160, 3840), np.uint8) * 255
            mask[:128] = 0
            mask[800:1200, 1600:2000] = 0
            mask_path = Path(temporary) / "mask.tif"
            Image.fromarray(mask).save(mask_path, compression="packbits")
            corrupt_path = Path(temporary) / "corrupt_deflate.tif"
            corrupt = bytearray(paths[16][0].read_bytes())
            with Image.open(paths[16][0]) as image:
                corrupt[image.tag_v2[273][0]] ^= 255
            corrupt_path.write_bytes(corrupt)
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=manual&offline=1')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.ready")
            # Actual File input and Worker decoding, before feeding the engine.
            browser._evaluate(
                "const picker=document.createElement('input');picker.type='file';picker.id='tiff-test-files';picker.multiple=true;document.body.append(picker);true"
            )
            await_files = [*paths[8], *paths[16], mask_path, corrupt_path]
            asyncio.run(choose_files(browser, "#tiff-test-files", await_files))
            decoded = browser._evaluate("""(async()=>{
              const media=await import('./ui/local-media.js'),masks=await import('./ui/mask-image.js'),files=[...document.getElementById('tiff-test-files').files];
              const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join(''),rows=[];
              for(const file of files.slice(0,4)){const bitmap=await media.decodeImage(file,[3840,2160]);
                try{const c=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=c.getContext('2d');ctx.drawImage(bitmap,0,0);
                  rows.push({name:file.name,width:bitmap.width,height:bitmap.height,sha256:await hash(ctx.getImageData(0,0,c.width,c.height).data),metadata:media.fileInfo(file).decoding});}
                finally{bitmap.close();}}
              const mask=await masks.decodeMaskImage(files[4],3840,2160),abort=new AbortController();
              const pending=media.decodeImage(files[0],null,abort.signal).catch(e=>e.name);abort.abort();
              const mismatch=await media.decodeImage(files[0],[32,32]).catch(e=>e.message);
              const corrupt=await media.decodeImage(files[5]).then(bitmap=>{bitmap.close();return null;},e=>e.message);
              const WorkerClass=globalThis.Worker;let noWorker;
              try{globalThis.Worker=undefined;noWorker=await media.decodeImage(files[0]).catch(e=>e.message);}finally{globalThis.Worker=WorkerClass;}
              return {rows,maskHash:await hash(mask),cancelled:await pending,mismatch,noWorker,corrupt};
            })()""")
            for row in decoded["rows"]:
                assert row["sha256"] == pixel_hashes[row["name"]], row
            assert decoded["maskHash"] == hashlib.sha256((mask > 0).tobytes()).hexdigest()
            assert decoded["cancelled"] == "AbortError"
            assert "never resized" in decoded["mismatch"]
            assert "requires browser Workers" in decoded["noWorker"]
            assert "Cannot decode corrupt_deflate.tif" in decoded["corrupt"]
            # Baseline and TIFF targets go through the real source and Solve UI.
            runs = []
            for name, inputs in [
                ("png", [original / "reference.png", original / "frame_095.png"]),
                ("tiff8", paths[8]),
                ("tiff16", paths[16]),
            ]:
                # Snap Chromium can only open files in its visible home directory.
                if name == "png":
                    import shutil

                    copies = []
                    for i, source in enumerate(inputs):
                        copy = Path(temporary) / f"png_{i:03}.png"
                        shutil.copyfile(source, copy)
                        copies.append(copy)
                    inputs = copies
                asyncio.run(choose_files(browser, "#image-files", inputs))
                wait(
                    browser,
                    "liveDicSourceState().source==='images'&&!liveDicSourceState().loadingSource",
                )
                browser._evaluate(
                    "document.getElementById('record-fields').checked=false;document.getElementById('solve-range').value='single';document.getElementById('pause').onclick().then(()=>true)"
                )
                wait(browser, "liveDicSourceState().paused&&!liveDicSourceState().localBusy")
                result = export(browser)
                runs.append((name, result))
            baseline = runs[0][1]
            comparisons = []
            for name, result in runs[1:]:
                assert np.array_equal(result["status"], baseline["status"])
                assert np.array_equal(result["parameters"], baseline["parameters"])
                comparisons.append(dict(format=name, exact_status=True, exact_parameters=True))
            # A new TIFF Worker must load its decoder with the origin stopped.
            browser._evaluate("""(async()=>{await navigator.serviceWorker.ready;
              if(!navigator.serviceWorker.controller)await new Promise((resolve,reject)=>{
                const timer=setTimeout(()=>reject(new Error('Service worker did not claim the page')),15000);
                navigator.serviceWorker.addEventListener('controllerchange',()=>{clearTimeout(timer);resolve();},{once:true});});return true;})()""")
            server.close()
            server_stopped = True
            offline = browser._evaluate("""(async()=>{
              const {decodeImage}=await import('./ui/local-media.js'),file=document.getElementById('tiff-test-files').files[3];
              const bitmap=await decodeImage(file,[3840,2160]);try{
                const c=new OffscreenCanvas(3840,2160),ctx=c.getContext('2d');ctx.drawImage(bitmap,0,0);
                const hash=await crypto.subtle.digest('SHA-256',ctx.getImageData(0,0,3840,2160).data);
                return Array.from(new Uint8Array(hash),x=>x.toString(16).padStart(2,'0')).join('');
              }finally{bitmap.close();}})()""")
            assert offline == pixel_hashes[paths[16][1].name]
            report = dict(
                build=json.loads((site / "build.json").read_text()),
                software=args.software,
                user_agent=browser._evaluate("navigator.userAgent"),
                decoded=decoded,
                comparisons=comparisons,
                offline_pixel_sha256=offline,
                source=browser._evaluate("liveDicSourceState()"),
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            assert not report["errors"], report["errors"]
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(dict(comparisons=comparisons, errors=report["errors"])))
    finally:
        if not server_stopped:
            server.close()


if __name__ == "__main__":
    main()
