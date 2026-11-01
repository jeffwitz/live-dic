"""Exercise bounded overlap/failure/drain with real native-UHD fields and IndexedDB.

This is a correctness qualification with deliberately held transactions, never
an FPS campaign. No physical camera or resized numerical input is used.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

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
    report = {
        "build": json.loads((site / "build.json").read_text()),
        "softwareRequested": args.software,
        "kind": __doc__,
        "phases": {},
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="live-dic-overlap-input-", dir=visible if visible.is_dir() else None
        ) as temp:
            inputs = []
            for index in range(7):
                original = (
                    ROOT
                    / "DIC/LucasKanadeICGN/assets/uhd_tensile"
                    / ("reference.png" if index % 2 == 0 else "frame_040.png")
                )
                target = Path(temp) / f"frame_{index:03}.png"
                shutil.copyfile(original, target)
                inputs.append(target)
            report["inputs"] = [
                {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in inputs
            ]
            with ICGNWebGL(
                backend="webgpu-wgsl", startup_timeout=60, call_timeout=240, chromium_args=flags
            ) as browser:
                try:
                    browser._evaluate(
                        f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate')});true"
                    )
                    wait(browser, "globalThis.liveDicStatus?.completed>=1", timeout=180)
                    browser._evaluate(
                        "document.getElementById('overlap-pipeline').checked=true;document.getElementById('gpu-reference-preparation').checked=true;true"
                    )
                    asyncio.run(choose_files(browser, "#image-files", inputs))
                    wait(
                        browser,
                        "liveDicSourceState().source==='images'&&!liveDicSourceState().loadingSource",
                        timeout=180,
                    )
                    # Preview target one before explicitly calculating file indices 1..6.
                    browser._evaluate(
                        "document.getElementById('local-seek').value='1';document.getElementById('local-seek').onchange().then(()=>true)"
                    )
                    wait(browser, "!liveDicSourceState().localBusy")
                    report["initial"] = browser._evaluate(
                        "({source:liveDicSourceState(),pipeline:liveDicPipelineReport(),canvas:[document.getElementById('field').width,document.getElementById('field').height]})"
                    )
                    assert report["initial"]["canvas"] == [1920, 1080], report["initial"]
                    browser._evaluate("""(async()=>{
                      const {IndexedFieldDatabase}=await import('./ui/field-storage.js');
                      const {RecordingQueue}=await import('./ui/recording-queue.js');
                      globalThis.pipelineFields=[];const submit=RecordingQueue.prototype.submit;
                      RecordingQueue.prototype.submit=function(ticket,record){pipelineFields.push({frameIndex:record.metadata.frameIndex,parameters:record.field.parameters.slice(),status:record.field.status.slice(),iterations:record.field.iterations.slice()});return submit.call(this,ticket,record);};
                      globalThis.pipelineDisk={original:IndexedFieldDatabase.prototype.write,entered:0,fail:false,release:null};
                      IndexedFieldDatabase.prototype.write=async function(...args){
                        pipelineDisk.entered++;await new Promise(resolve=>pipelineDisk.release=resolve);
                        if(pipelineDisk.fail)throw new DOMException('Injected overlap disk quota','QuotaExceededError');
                        return pipelineDisk.original.apply(this,args);
                      };
                      document.getElementById('image-rate').value='0';
                      document.getElementById('record-fields').checked=true;await document.getElementById('record-fields').onchange();
                      globalThis.pipelineCompletedStart=liveDicStatus.completed;
                      document.getElementById('pause').onclick();return true;
                    })()""")
                    wait(
                        browser, "liveDicPipelineReport().recording.pendingRecords===2", timeout=180
                    )
                    report["phases"]["held"] = browser._evaluate(
                        "({pipeline:liveDicPipelineReport(),library:liveDicWorkbench.report(),computed:liveDicStatus.completed-pipelineCompletedStart,source:liveDicSourceState()})"
                    )
                    assert report["phases"]["held"]["computed"] == 2, report["phases"]["held"]
                    assert report["phases"]["held"]["library"]["count"] == 0
                    assert report["phases"]["held"]["pipeline"]["recording"]["peakPending"] == 2
                    # Pause must wait for both writes and must not compute the next
                    # file that was waiting for capacity when Pause was requested.
                    browser._evaluate(
                        "globalThis.pipelinePauseDone=false;document.getElementById('pause').onclick().then(()=>pipelinePauseDone=true);true"
                    )
                    assert not browser._evaluate("pipelinePauseDone")
                    browser._evaluate("pipelineDisk.release();true")
                    wait(browser, "pipelineDisk.entered===2")
                    assert not browser._evaluate("pipelinePauseDone")
                    browser._evaluate("pipelineDisk.release();true")
                    wait(browser, "pipelinePauseDone", timeout=180)
                    report["phases"]["drained"] = browser._evaluate(
                        "({pipeline:liveDicPipelineReport(),library:liveDicWorkbench.report(),computed:liveDicStatus.completed-pipelineCompletedStart,source:liveDicSourceState()})"
                    )
                    assert report["phases"]["drained"]["computed"] == 2
                    assert report["phases"]["drained"]["library"]["count"] == 2
                    assert report["phases"]["drained"]["pipeline"]["recording"]["pending"] == 0
                    # Hold decoding of the next file until recording has failed.
                    # A late decode must not bypass Pause merely because the
                    # recording checkbox was turned off by the failure handler.
                    browser._evaluate("""(()=>{
                      const decode=globalThis.createImageBitmap;globalThis.pipelineDecodeWaiting=false;
                      globalThis.createImageBitmap=async(source,...args)=>{
                        const image=await decode(source,...args);
                        if(source instanceof File&&source.name==='frame_005.png'){
                          pipelineDecodeWaiting=true;globalThis.createImageBitmap=decode;await new Promise(resolve=>globalThis.pipelineDecodeRelease=resolve);
                        }return image;
                      };return true;
                    })()""")
                    # Fail one held write after a second field has already completed.
                    browser._evaluate(
                        "pipelineDisk.fail=true;document.getElementById('pause').onclick();true"
                    )
                    wait(
                        browser, "liveDicPipelineReport().recording.pendingRecords===2", timeout=180
                    )
                    wait(browser, "pipelineDecodeWaiting", timeout=180)
                    browser._evaluate("pipelineDisk.release();true")
                    wait(
                        browser,
                        "liveDicPipelineReport().recording.blocked&&liveDicPipelineReport().recording.pending===0",
                        timeout=180,
                    )
                    report["phases"]["failure"] = browser._evaluate(
                        "({pipeline:liveDicPipelineReport(),library:liveDicWorkbench.report(),source:liveDicSourceState(),message:document.getElementById('message').textContent})"
                    )
                    assert report["phases"]["failure"]["library"]["count"] == 4
                    assert report["phases"]["failure"]["library"]["storage"]["unsavedFields"] == 2
                    assert (
                        report["phases"]["failure"]["pipeline"]["recording"]["retainedAfterFailure"]
                        == 2
                    )
                    assert report["phases"]["failure"]["source"]["paused"]
                    browser._evaluate("pipelineDecodeRelease();true")
                    browser._evaluate("""(async()=>{
                      const {IndexedFieldDatabase}=await import('./ui/field-storage.js');IndexedFieldDatabase.prototype.write=pipelineDisk.original;
                      await document.getElementById('retry-storage').onclick();
                      document.getElementById('record-fields').checked=true;await document.getElementById('record-fields').onchange();return true;
                    })()""")
                    report["phases"]["recovery"] = browser._evaluate(
                        "({pipeline:liveDicPipelineReport(),library:liveDicWorkbench.report(),source:liveDicSourceState()})"
                    )
                    assert report["phases"]["recovery"]["library"]["storage"]["diskFields"] == 4
                    assert report["phases"]["recovery"]["library"]["storage"]["unsavedFields"] == 0
                    assert not report["phases"]["recovery"]["pipeline"]["recording"]["blocked"]
                    assert report["phases"]["recovery"]["source"]["index"] == 4, report["phases"][
                        "recovery"
                    ]
                    # A source replacement must wait for old-epoch fields still
                    # being saved; no old field may acquire the new source/ROI.
                    browser._evaluate("""(async()=>{
                      const {IndexedFieldDatabase}=await import('./ui/field-storage.js');pipelineDisk.entered=0;
                      IndexedFieldDatabase.prototype.write=async function(...args){pipelineDisk.entered++;await new Promise(resolve=>pipelineDisk.release=resolve);return pipelineDisk.original.apply(this,args);};
                      document.getElementById('pause').onclick();return true;
                    })()""")
                    wait(
                        browser, "liveDicPipelineReport().recording.pendingRecords===2", timeout=180
                    )
                    browser._evaluate(
                        "globalThis.pipelineSourceChanged=false;document.getElementById('plate').onclick().then(()=>pipelineSourceChanged=true);true"
                    )
                    assert not browser._evaluate("pipelineSourceChanged")
                    assert browser._evaluate("liveDicSourceState().source==='images'")
                    browser._evaluate("pipelineDisk.release();true")
                    wait(browser, "pipelineDisk.entered===2")
                    browser._evaluate("pipelineDisk.release();true")
                    wait(browser, "pipelineSourceChanged", timeout=180)
                    browser._evaluate(
                        "import('./ui/field-storage.js').then(({IndexedFieldDatabase})=>{IndexedFieldDatabase.prototype.write=pipelineDisk.original;return true;})"
                    )
                    report["phases"]["sourceChanged"] = browser._evaluate(
                        "({pipeline:liveDicPipelineReport(),library:liveDicWorkbench.report(),source:liveDicSourceState()})"
                    )
                    assert report["phases"]["sourceChanged"]["source"]["source"] == "plate"
                    assert (
                        report["phases"]["sourceChanged"]["pipeline"]["recording"]["pending"] == 0
                    )
                    assert (
                        report["phases"]["sourceChanged"]["library"]["storage"]["diskFields"] == 6
                    )
                    # A complete project must contain all six full-resolution fields,
                    # with their original order, configuration and byte arrays.
                    report["archive"] = browser._evaluate("""(async()=>{
                      const original=URL.createObjectURL;let blob;URL.createObjectURL=value=>{if(value instanceof Blob&&value.type==='application/zip')blob=value;return original(value);};
                      try {
                        await document.getElementById('save-project').onclick();if(!blob)throw Error('Missing project archive');
                        const {importProjectArchive}=await import('./ui/archive.js');const project=await importProjectArchive(await blob.arrayBuffer());
                        return {count:project.records.length,records:project.records.map((r,i)=>({exact:['parameters','status','iterations'].every(key=>{const a=new Uint8Array(r.field[key].buffer),b=new Uint8Array(pipelineFields[i][key].buffer);return a.length===b.length&&a.every((v,j)=>v===b[j]);}),width:r.field.width,height:r.field.height,grid:r.field.grid,frameIndex:r.metadata.frameIndex,supervision:r.metadata.supervision,finite:r.field.parameters.every(Number.isFinite),bytes:r.field.parameters.byteLength}))};
                      }finally{URL.createObjectURL=original;}
                    })()""")
                    assert report["archive"]["count"] == 6
                    assert [r["frameIndex"] for r in report["archive"]["records"]] == [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                    ]
                    assert all(
                        r["width"] == 3840
                        and r["height"] == 2160
                        and r["grid"] == [240, 135]
                        and r["finite"]
                        and r["exact"]
                        for r in report["archive"]["records"]
                    )
                    report["errors"] = browser._evaluate("liveDicStatus.errors")
                    assert not report["errors"], report["errors"]
                except Exception as error:
                    report["failure"] = str(error)
                    try:
                        report["failureState"] = browser._evaluate(
                            "({source:globalThis.liveDicSourceState?.(),pipeline:globalThis.liveDicPipelineReport?.(),status:globalThis.liveDicStatus?{completed:liveDicStatus.completed,errors:liveDicStatus.errors}:null})"
                        )
                    except Exception as inspectionError:
                        report["inspectionFailure"] = str(inspectionError)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
                    args.output.write_bytes(
                        gzip.compress(payload, mtime=0) if args.output.suffix == ".gz" else payload
                    )
                    raise
    finally:
        server.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    args.output.write_bytes(
        gzip.compress(payload, mtime=0) if args.output.suffix == ".gz" else payload
    )
    print(
        json.dumps(
            {
                "phases": list(report["phases"]),
                "fields": report["archive"]["count"],
                "errors": report["errors"],
            }
        )
    )


if __name__ == "__main__":
    main()
