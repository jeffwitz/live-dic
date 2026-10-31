"""Round-trip actual native UHD quadratic results through the browser workbench.

This checks coefficient identity, units and separation from post-processing.
It does not establish new quadratic accuracy or performance claims.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from qualify_browser_editor import protocol
from qualify_local_sources import choose_files, wait

from lucas_kanade_icgn import ICGNWebGL, ICGNWGPUFullPyramid
from lucas_kanade_icgn.engine import _StaticServer
from lucas_kanade_icgn.motion import parameter_order

ROOT = Path(__file__).resolve().parents[1]


def native_field(destination, device):
    directory = ROOT / "DIC/LucasKanadeICGN/assets/uhd_tensile"
    reference = np.asarray(Image.open(directory / "reference.png"))
    target = np.asarray(Image.open(directory / "frame_040.png"))
    with ICGNWGPUFullPyramid(
        device_type=device, motion_model="quadratic", outlier_policy="disabled"
    ) as engine:
        info = engine.set_reference(reference)
        result = engine.register(target)
        assert result.parameters.shape == (135, 240, 12)
        assert np.isfinite(result.parameters).all()
        np.savez_compressed(
            destination,
            parameters=result.parameters,
            parameter_order=np.asarray(result.parameter_order),
            status=result.status,
            iterations=result.iterations,
            width=np.asarray(info.width),
            height=np.asarray(info.height),
            subset_size=np.asarray(info.subset_size),
            subset_stride=np.asarray(info.subset_stride),
            config_json=np.asarray(json.dumps(dict(result.config), allow_nan=False)),
        )
        return dict(
            device=info.device,
            configuration=dict(result.config),
            accepted=int(result.accepted_mask.sum()),
            inputs={
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ["reference.png", "frame_040.png"]
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--field",
        type=Path,
        help="reuse an already materialized native field instead of solving again",
    )
    parser.add_argument("--native-device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--software",
        action="store_true",
        help="use SwiftShader for the browser; independent of native-device",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.field or args.output / "native-quadratic.npz"
    native = (
        native_field(archive_path, args.native_device)
        if args.field is None
        else dict(reused_field=str(args.field))
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        parameters = archive["parameters"].copy()
        status = archive["status"].copy()
        iterations = archive["iterations"].copy()
        assert parameters.shape == (135, 240, 12)
    selected = status > 0.5
    values = parameters[..., 6][selected].astype(np.float64) / 0.01
    assert values.size > 20000, (
        "A representative finite native field is required for this UI round trip"
    )
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
    visible = Path.home() / "snap/chromium/common"
    try:
        with (
            tempfile.TemporaryDirectory(
                prefix="live-dic-coefficients-", dir=visible if visible.is_dir() else None
            ) as temp,
            ICGNWebGL(
                backend="webgpu-wgsl",
                startup_timeout=60,
                call_timeout=240,
                chromium_args=(*flags, "--window-size=1600,1200"),
            ) as browser,
        ):
            url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate"
            browser._evaluate(f"location.assign({json.dumps(url)});true")
            wait(browser, "globalThis.liveDicStatus?.completed>=1")
            file = Path(temp) / "native-quadratic.npz"
            shutil.copyfile(archive_path, file)
            asyncio.run(choose_files(browser, "#result-file", [file]))
            wait(
                browser,
                "liveDicWorkbench.report().count===1&&liveDicWorkbench.report().recipe!=null",
            )
            browser._evaluate(
                "document.getElementById('review-component').value='computed-p6';document.getElementById('review-component').onchange();document.getElementById('length-unit').value='mm';document.getElementById('length-scale').value='.01';document.getElementById('length-unit').onchange();true"
            )
            wait(browser, "liveDicWorkbench.report().recipe?.unit==='mm'")
            before = browser._evaluate("liveDicWorkbench.report()")
            assert before["statistics"]["count"] == values.size
            np.testing.assert_allclose(
                [before["statistics"][key] for key in ["mean", "min", "max"]],
                [values.mean(), values.min(), values.max()],
                atol=1e-12,
                rtol=1e-10,
            )
            assert "1/mm" in browser._evaluate(
                "document.getElementById('quantity-source').textContent"
            )
            browser._evaluate(
                "document.getElementById('post-smoothing').value='gaussian';document.getElementById('post-smoothing').onchange();true"
            )
            wait(browser, "liveDicWorkbench.report().recipe?.smoothing==='gaussian'")
            after = browser._evaluate("liveDicWorkbench.report()")
            assert before["statistics"] == after["statistics"], (
                "Computed coefficients must not change with displacement smoothing"
            )
            encoded = browser._evaluate(
                """(async()=>{const original=URL.createObjectURL;let saved;URL.createObjectURL=b=>{if(b.type==='application/octet-stream')saved=b;return original(b);};try{document.getElementById('save-npz').onclick();const data=new Uint8Array(await saved.arrayBuffer()),parts=[];for(let i=0;i<data.length;i+=32768)parts.push(String.fromCharCode(...data.subarray(i,i+32768)));return btoa(parts.join(''));}finally{URL.createObjectURL=original;}})()"""
            )
            with np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["parameters"], parameters)
                assert tuple(saved["parameter_order"]) == parameter_order("quadratic")
                assert json.loads(str(saved["metadata_json"]))["parameterUnits"][6] == "1/px"
                np.testing.assert_array_equal(saved["status"], status)
                np.testing.assert_array_equal(saved["iterations"], iterations)
                assert json.loads(str(saved["recipe_json"]))["smoothing"] == "gaussian"
            browser._evaluate(
                "document.querySelector('.viewer').scrollIntoView({block:'start'});true"
            )
            shot = asyncio.run(
                protocol(
                    browser,
                    [("Page.captureScreenshot", dict(format="png", captureBeyondViewport=False))],
                )
            )[0]
            (args.output / "computed-curvature.png").write_bytes(base64.b64decode(shot["data"]))
            report = dict(
                build=json.loads((site / "build.json").read_text()),
                kind="native UHD quadratic output UI round trip; not an accuracy/performance campaign",
                native=native,
                software_browser=args.software,
                native_field_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                before=before,
                after=after,
                raw_arrays_unchanged=True,
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            assert not report["errors"], report["errors"]
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    dict(count=values.size, raw_arrays_unchanged=True, errors=report["errors"])
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
