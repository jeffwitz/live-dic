"""Check cold solver edits, automatic application and saved provenance on full-size image pairs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from qualify_local_sources import choose_files, wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.target):
        parser.error("Supply both --reference and --target")
    args.output.mkdir(parents=True, exist_ok=True)
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
        visible = Path.home() / "snap/chromium/common"
        with (
            tempfile.TemporaryDirectory(dir=visible if visible.is_dir() else None) as temporary,
            ICGNWebGL(backend="webgpu-wgsl", chromium_args=flags, call_timeout=300) as browser,
        ):
            paths = [Path(temporary) / "Ref.tiff", Path(temporary) / "Def.tiff"]
            if args.reference:
                for original, target in zip([args.reference, args.target], paths, strict=True):
                    # Preserve original bytes and format; this tool accepts TIFF real pairs.
                    shutil.copyfile(original, target)
            else:
                reference = np.array(
                    Image.open(ROOT / "DIC/LucasKanadeICGN/assets/uhd_tensile/reference.png")
                )
                assert reference.shape == (2160, 3840)
                target = np.zeros_like(reference)
                target[:-32, 64:] = reference[32:, :-64]
                for pixels, path in zip([reference, target], paths, strict=True):
                    Image.fromarray(pixels).save(path)
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(args.site.name)}/?smoke=1&case=manual')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.ready")
            asyncio.run(choose_files(browser, "#image-files", paths))
            wait(
                browser,
                "liveDicSourceState().source==='images'&&!liveDicSourceState().loadingSource",
            )
            browser._evaluate(
                "globalThis.knownPairTruth=" + ("false" if args.reference else "true") + ";true"
            )
            before = browser._evaluate(
                "JSON.parse(document.getElementById('details').textContent).reference.config"
            )
            assert before["initialization"] == "temporal"
            browser._evaluate(
                "document.getElementById('solve-range').value='single';document.getElementById('pause').onclick().then(()=>true)"
            )
            wait(
                browser,
                "liveDicWorkbench.report().count===1&&liveDicWorkbench.report().recipe!==null",
            )
            record_script = """(async()=>{
              const session=liveDicWorkbench.storage.session, backend=liveDicWorkbench.storage.backend, records=[];
              const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),x=>x.toString(16).padStart(2,'0')).join('');
              for(let i=1;i<session.nextId;i++){
                const {field,metadata}=await backend.read(session.id,i);let accepted=0;
                for(const s of field.status)accepted+=+(s>.5);
                let quality=null;
                if(globalThis.knownPairTruth){
                  const [dx,dy]=metadata.comparison.reference.name.endsWith('Ref.tiff')?[64,-32]:[-64,32];
                  let selected=0,valid=0,squared=0,accurate=0;
                  for(let k=0;k<field.status.length;k++){
                    const x=(k%field.grid[0])*field.subsetStride+(field.subsetSize-1)/2,y=Math.floor(k/field.grid[0])*field.subsetStride+(field.subsetSize-1)/2;
                    if(x<=192||x>=field.width-192||y<=192||y>=field.height-192||x+dx<=192||x+dx>=field.width-192||y+dy<=192||y+dy>=field.height-192||field.roiSelected&&!field.roiSelected[k])continue;
                    selected++;if(field.status[k]>.5){valid++;const e=Math.hypot(field.parameters[k*6]-dx,field.parameters[k*6+3]-dy);squared+=e*e;accurate+=+(e<.1);}
                  }
                  quality={selected,valid,coverage:valid/selected,rmse:valid?Math.sqrt(squared/valid):null,accurateCoverage:accurate/selected};
                }
                records.push({index:i,accepted,total:field.status.length,quality,config:field.config,analysisEpoch:metadata.analysisEpoch,
                  parametersSha256:await hash(field.parameters.buffer),statusSha256:await hash(field.status.buffer),comparison:metadata.comparison});
              }return records;})()"""
            first = browser._evaluate(record_script)[0]
            browser._evaluate("""document.querySelector('[data-pane=solve]').click();
              document.getElementById('fourier-start').checked=true;
              document.getElementById('custom-solver').checked=true;document.getElementById('subset-size').value='32';
              document.getElementById('subset-stride').value=16;document.getElementById('subset-size').onchange();true""")
            assert browser._evaluate(
                "document.getElementById('solver-summary').dataset.pending==='true'&&!document.getElementById('apply-solver-settings').disabled"
            )
            # No re-capture and no Apply button: Start must apply and use the original reference.
            browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
            wait(
                browser,
                "liveDicWorkbench.report().count===2&&liveDicWorkbench.report().recipe!==null",
            )
            records = browser._evaluate(record_script)
            assert records[0] == first
            good = records[1]
            assert (
                good["config"]["initialization"] == "fourier" and good["config"]["subsetSize"] == 32
            )
            assert good["comparison"]["reference"]["name"].endswith("Def.tiff")
            assert good["accepted"] / good["total"] > 0.90, good
            assert good["analysisEpoch"] != first["analysisEpoch"]
            assert browser._evaluate(
                "document.getElementById('saved-solver-summary').textContent.includes('initialization: fourier')"
            )
            # Replay the failed field: its metadata/arrays must stay temporal and unchanged.
            completed = browser._evaluate("liveDicStatus.completed")
            browser._evaluate(
                "document.getElementById('saved-fields').value=1;document.getElementById('saved-fields').onchange();true"
            )
            wait(
                browser,
                "liveDicWorkbench.report().selected===1&&liveDicWorkbench.report().recipe!==null",
            )
            assert browser._evaluate(
                "document.getElementById('saved-solver-summary').textContent.includes('initialization: temporal')"
            )
            assert browser._evaluate("liveDicStatus.completed") == completed
            # Immediate display changes do not rebuild the prepared reference.
            browser._evaluate(
                "document.querySelector('[data-pane=solve]').click();document.getElementById('display-rate').value='1';document.getElementById('display-rate').onchange();document.getElementById('local-seek').value=1;document.getElementById('local-seek').onchange().then(()=>true)"
            )
            assert browser._evaluate("document.getElementById('apply-solver-settings').disabled")
            browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
            wait(
                browser,
                "liveDicWorkbench.report().count===3&&liveDicWorkbench.report().recipe!==null",
            )
            records = browser._evaluate(record_script)
            assert records[2]["analysisEpoch"] == good["analysisEpoch"]
            assert records[2]["parametersSha256"] == good["parametersSha256"]
            # Applying a policy change prepares only; it must not create a measured field.
            browser._evaluate(
                "document.querySelector('[data-pane=solve]').click();document.getElementById('outlier-policy').value='disabled';document.getElementById('outlier-policy').onchange();document.getElementById('apply-solver-settings').onclick().then(()=>true)"
            )
            assert browser._evaluate("liveDicWorkbench.report().count") == 3
            assert browser._evaluate(
                "JSON.parse(document.getElementById('details').textContent).reference.config.outlierPolicy==='disabled'"
            )
            # Selecting Ref (last in natural order) must suggest Def, not self-correlation.
            browser._evaluate(
                "document.getElementById('image-reference').value=1;document.getElementById('use-image-reference').onclick().then(()=>true)"
            )
            state = browser._evaluate("liveDicSourceState()")
            assert state["index"] == 0 and state["settings"]["reference"]["name"].endswith(
                "Ref.tiff"
            )
            browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
            wait(
                browser,
                "liveDicWorkbench.report().count===4&&liveDicWorkbench.report().recipe!==null",
            )
            records = browser._evaluate(record_script)
            assert records[0] == first
            assert records[3]["accepted"] / records[3]["total"] > 0.90
            # A cold solver edit retains the applied geometry and image mask.
            browser._evaluate(
                """document.querySelector('[data-pane=solve]').click();document.getElementById('draw').onclick().then(()=>true)"""
            )
            roi = browser._evaluate("""(async()=>{
              const info=JSON.parse(document.getElementById('details').textContent).reference;
              liveDicRoi.reset(info.width,info.height,[{id:'outer',type:'rectangle',x:64,y:64,width:info.width-128,height:info.height-128},
                {id:'hole',type:'circle',x:info.width/2-80,y:info.height/2-80,width:160,height:160}]);
              await document.getElementById('apply-roi').onclick();
              return JSON.parse(document.getElementById('details').textContent).reference.config.roi;
            })()""")
            browser._evaluate(
                "document.getElementById('outlier-policy').value='median';document.getElementById('outlier-policy').onchange();document.getElementById('pause').onclick().then(()=>true)"
            )
            wait(
                browser,
                "liveDicWorkbench.report().count===5&&liveDicWorkbench.report().recipe!==null",
            )
            records = browser._evaluate(record_script)
            assert records[4]["config"]["roi"]["maskSha256"] == roi["maskSha256"]
            assert records[4]["config"]["roi"]["selected"] == roi["selected"]
            assert records[0] == first
            if not args.reference:
                for record in records[1:]:
                    assert (
                        record["quality"]["coverage"] >= 0.90 and record["quality"]["rmse"] < 0.05
                    ), record
            # Invalid settings stop before preparation/solving, preserving all records.
            browser._evaluate(
                "document.querySelector('[data-pane=solve]').click();document.getElementById('coarse-iterations').value=0;document.getElementById('coarse-iterations').onchange();document.getElementById('pause').onclick().then(()=>true)"
            )
            assert browser._evaluate("liveDicWorkbench.report().count") == 5
            assert "GN limits" in browser._evaluate(
                "document.getElementById('message').textContent"
            )
            assert not browser._evaluate("liveDicStatus.errors")
            report = dict(
                build=json.loads((args.site / "build.json").read_text()),
                software=args.software,
                source_sha256={
                    name: hashlib.sha256((args.site / name).read_bytes()).hexdigest()
                    for name in ["app.js", "index.html", "ui/workspace.js", "ui/workbench.js"]
                },
                qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input_kind="private supplied TIFF pair"
                if args.reference
                else "archived native UHD texture, known integer shift",
                input_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                initial_configuration=before,
                records=records,
                source=state,
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        fields=[
                            dict(
                                index=r["index"],
                                accepted=r["accepted"],
                                total=r["total"],
                                initialization=r["config"]["initialization"],
                            )
                            for r in records
                        ],
                        errors=report["errors"],
                    )
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
