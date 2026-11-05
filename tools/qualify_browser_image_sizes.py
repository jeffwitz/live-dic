"""Exercise full-resolution image imports beyond baseline WebGPU device limits."""

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
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=manual')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.ready")
            records = []
            for width, height, suffix in [
                (4112, 3008, "tif"),
                (4113, 3009, "png"),
                (9001, 1201, "png"),
            ]:
                paths = []
                for i, (dx, dy) in enumerate([(0, 0), (2, -1)]):
                    x = np.arange(width, dtype=np.float64)[None, :] - dx
                    y = np.arange(height, dtype=np.float64)[:, None] - dy
                    pixels = np.rint(
                        128
                        + 40 * np.sin(0.21 * x + 0.03 * y)
                        + 30 * np.cos(0.08 * x - 0.23 * y)
                        + 20 * np.sin(0.31 * x + 0.27 * y)
                    ).astype(np.uint8)
                    path = Path(temporary) / f"{width}_{height}_{i}.{suffix}"
                    Image.fromarray(pixels).save(path)
                    paths.append(path)
                asyncio.run(choose_files(browser, "#image-files", paths))
                wait(
                    browser,
                    "liveDicSourceState().source==='images'&&!liveDicSourceState().loadingSource",
                )
                browser._evaluate(
                    "document.getElementById('record-fields').checked=false;document.getElementById('solve-range').value='single';document.getElementById('pause').onclick().then(()=>true)"
                )
                result = export(browser)
                status, parameters = result["status"], result["parameters"]
                gw, gh = 1 + (width - 16) // 16, 1 + (height - 16) // 16
                assert len(status) == gw * gh
                interior = np.ones((gh, gw), bool)
                interior[[0, -1], :] = False
                interior[:, [0, -1]] = False
                valid = (status > 0.5) & interior.ravel()
                errors = parameters[valid][:, [0, 3]] - [2, -1]
                rmse = float(np.sqrt(np.mean(errors**2)))
                coverage = float(valid.sum() / interior.sum())
                assert coverage > 0.90 and rmse < 0.05, (width, height, coverage, rmse)
                assert np.isfinite(parameters).all()
                records.append(
                    dict(
                        width=width,
                        height=height,
                        format=suffix,
                        grid=[gw, gh],
                        field_points=len(status),
                        reference_bytes=width * height * 16,
                        accepted_interior=int(valid.sum()),
                        interior_coverage=coverage,
                        displacement_rmse=rmse,
                        input_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                        metadata=result["metadata"],
                    )
                )
            # Geometry may be unknown until an ImageBitmap is prepared. Growing
            # a reference must upgrade an owning device but never an injected one.
            lifecycle = browser._evaluate("""(async()=>{
              const {DICEngine}=await import('./web/core/engine.js');
              const e=await DICEngine.create({pyramidLevels:1,outlierPolicy:'disabled'});
              let bitmap;try{
                const small=await createImageBitmap(new ImageData(64,64));await e.setReference(small);small.close();
                const first=e.device,oldLimit=first.limits.maxStorageBufferBindingSize;
                const c=new OffscreenCanvas(4112,3008),ctx=c.getContext('2d');ctx.fillStyle='#888';ctx.fillRect(0,0,c.width,c.height);
                bitmap=await createImageBitmap(c);const p=await e.setReference(bitmap);
                const result={replaced:first!==e.device,oldLimit,newLimit:e.device.limits.maxStorageBufferBindingSize,width:p.width,height:p.height};
                return result;
              }finally{bitmap?.close();e.destroy();}})()""")
            assert lifecycle["replaced"] and lifecycle["newLimit"] >= 4112 * 3008 * 16
            report = dict(
                build=json.loads((site / "build.json").read_text()),
                software=args.software,
                records=records,
                lifecycle=lifecycle,
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            assert not report["errors"], report["errors"]
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        records=[
                            {
                                k: r[k]
                                for k in (
                                    "width",
                                    "height",
                                    "interior_coverage",
                                    "displacement_rmse",
                                )
                            }
                            for r in records
                        ],
                        lifecycle=lifecycle,
                    )
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
