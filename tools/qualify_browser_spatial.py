"""Exercise Solve's spatial propagation on original UHD images and saved fields."""

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
from qualify_local_sources import choose_files, wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assets = ROOT / "DIC/LucasKanadeICGN/assets/uhd_tensile"
    with (assets / "machine_truth.csv").open() as stream:
        truth = next(row for row in csv.DictReader(stream) if int(row["frame"]) == 40)
    server = _StaticServer(args.site.resolve().parent)
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
        shared = Path.home() / "snap/chromium/common"
        with (
            tempfile.TemporaryDirectory(dir=shared if shared.is_dir() else None) as temporary,
            ICGNWebGL(
                backend="webgpu-wgsl", chromium_args=flags, startup_timeout=90, call_timeout=300
            ) as browser,
        ):
            paths = [Path(temporary) / "00-reference.png", Path(temporary) / "01-frame_040.png"]
            originals = [assets / "reference.png", assets / "frame_040.png"]
            for original, copy in zip(originals, paths, strict=True):
                shutil.copyfile(original, copy)
                assert original.read_bytes() == copy.read_bytes()
            url = f"http://127.0.0.1:{server.port}/{quote(args.site.name)}/?smoke=1&case=manual"
            browser._evaluate(f"location.assign({json.dumps(url)});true")
            wait(browser, "globalThis.liveDicStatus?.ready")
            browser._evaluate(
                "document.querySelector('button[data-workspace-mode=analysis]').click();true"
            )
            asyncio.run(choose_files(browser, "#image-files", paths))
            wait(
                browser,
                "liveDicSourceState().source==='images'&&!liveDicSourceState().loadingSource",
            )
            assert browser._evaluate("document.getElementById('spatial-propagation').value==='0'")
            browser._evaluate("""document.querySelector('[data-pane=solve]').click();
              document.getElementById('custom-solver').checked=true;
              document.getElementById('motion-model').value='translation';
              document.getElementById('subset-size').value='32';document.getElementById('subset-stride').value='16';
              document.getElementById('pyramid-levels').value='2';
              document.getElementById('outlier-policy').value='median';
              document.getElementById('global-affine-start').checked=true;
              document.getElementById('global-affine-levels').value='6';
              document.getElementById('temporal-start').checked=false;
              document.getElementById('fourier-start').checked=false;
              document.getElementById('solve-range').value='single';
              document.getElementById('custom-solver').onchange();true""")
            assert browser._evaluate(
                "JSON.parse(document.getElementById('details').textContent).reference.prepared===false"
            )
            records = []
            for passes in (0, 2):
                browser._evaluate(
                    f"document.querySelector('[data-pane=solve]').click();document.getElementById('spatial-propagation').value='{passes}';document.getElementById('spatial-propagation').onchange();true"
                )
                assert browser._evaluate(
                    "document.getElementById('solver-summary').dataset.pending==='true'"
                )
                browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
                count = len(records) + 1
                wait(
                    browser,
                    f"liveDicWorkbench.report().count==={count}&&liveDicWorkbench.report().recipe!==null",
                )
                record = browser._evaluate(f"""(async()=>{{
                  const session=liveDicWorkbench.storage.session;
                  const {{field,metadata}}=await liveDicWorkbench.storage.backend.read(session.id,{count});
                  if(field.config.motionModel!=='translation')throw new Error('Unexpected local model: '+JSON.stringify(field.config));
                  let population=0,valid=0,squared=0;const errors=[];
                  for(let i=0;i<field.status.length;i++){{
                    for(let k=0;k<6;k++)if(!Number.isFinite(field.parameters[i*6+k]))throw new Error('Nonfinite field');
                    for(const k of [1,2,4,5])if(field.parameters[i*6+k]!==0)throw new Error('Translation acquired gradients');
                    const x=(i%field.grid[0])*16+15.5,y=Math.floor(i/field.grid[0])*16+15.5;
                    if(x<32||x>=3840-32||y<32||y>=2160-32)continue;population++;
                    if(field.status[i]>.5){{valid++;const error=Math.hypot(field.parameters[i*6]-({float(truth["stretch_x"])}-1)*(x-1920),field.parameters[i*6+3]-({float(truth["stretch_y"])}-1)*(y-1080));errors.push(error);squared+=error*error;}}
                  }}
                  errors.sort((a,b)=>a-b);
                  return {{config:field.config,comparison:metadata.comparison,population,valid,coverage:valid/population,
                    vectorRmse:Math.sqrt(squared/valid),p95Error:errors[Math.floor(.95*(errors.length-1))],
                    applied:document.getElementById('applied-solver-summary').textContent}};
                }})()""")
                assert record["config"]["motionModel"] == "translation"
                assert record["config"]["initialization"] == "global-affine"
                assert record["config"].get("spatialPropagationPasses", 0) == passes
                assert record["comparison"]["reference"]["name"].endswith("00-reference.png")
                assert record["coverage"] >= 0.995
                assert record["vectorRmse"] < 0.01518230373115895
                assert record["p95Error"] < 0.02722860762111737
                if passes:
                    assert "GPU seed propagation: 2" in record["applied"]
                records.append(record)
                print(passes, record["coverage"], record["vectorRmse"], flush=True)
            # Stored disabled-propagation provenance survives later Solve edits.
            first = browser._evaluate("""(async()=>{const s=liveDicWorkbench.storage.session;
              const {field}=await liveDicWorkbench.storage.backend.read(s.id,1);return field.config;})()""")
            assert first == records[0]["config"]
            payload = browser._evaluate("""(async()=>{
              const create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;let blob;
              URL.createObjectURL=value=>{blob=value;return create(value);};HTMLAnchorElement.prototype.click=()=>{};
              try{await document.getElementById('save-npz').onclick();if(!blob)throw new Error('No NPZ export');
                const bytes=new Uint8Array(await blob.arrayBuffer()),parts=[];
                for(let i=0;i<bytes.length;i+=32768)parts.push(String.fromCharCode(...bytes.subarray(i,i+32768)));
                return btoa(parts.join(''));
              }finally{URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;}
            })()""")
            export_path = args.output / "propagation-field.npz"
            export_path.write_bytes(base64.b64decode(payload))
            with np.load(export_path, allow_pickle=False) as archive:
                exported_config = json.loads(str(archive["config_json"].ravel()[0]))
                assert exported_config["spatialPropagationPasses"] == 2
                assert np.isfinite(archive["parameters"]).all()
            report = dict(
                createdUtc=datetime.now(UTC).isoformat(),
                command=[sys.executable, *sys.argv],
                build=browser._evaluate("fetch('./build.json').then(r=>r.json())"),
                userAgent=browser._evaluate("navigator.userAgent"),
                software=args.software,
                criteria=dict(
                    minimumInteriorCoverage=0.995,
                    maximumVectorRmsePixels=0.01518230373115895,
                    maximumP95ErrorPixels=0.02722860762111737,
                    gradients="Exactly zero for local translation",
                    population="Subset centres >=32 pixels from all image edges",
                ),
                inputSha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals},
                records=records,
                capabilities=browser._evaluate(
                    "JSON.parse(document.getElementById('details').textContent).capabilities"
                ),
                npzSha256=hashlib.sha256(export_path.read_bytes()).hexdigest(),
                notes=[
                    "Actual UHD PNG file input and worker/session/storage path, not a throughput benchmark.",
                    "Global affine six-level initialization remains independent of local translation.",
                ],
            )
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
