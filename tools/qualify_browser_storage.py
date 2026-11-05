"""Qualify real IndexedDB eviction, recovery and portable exports of UHD fields.

The default stress replays distinct copies of one actually calculated UHD field:
it tests storage, not a new numerical trajectory or live FPS. --video additionally
records actual decoded/solved UHD frames until their payload exceeds 1 GiB.
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
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
from qualify_browser_editor import protocol
from qualify_local_sources import choose_files, wait

from live_dic import ICGNBrowser
from lucas_kanade_icgn.engine import _StaticServer

MIB = 1024**2


def evaluate(browser, source):
    return browser._evaluate(f"(async()=>{{{source}}})()")


def memory(browser):
    return evaluate(
        browser,
        "await document.getElementById('inspect-memory').onclick();"
        "return JSON.parse(document.getElementById('memory-report').textContent);",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software", action="store_true")
    parser.add_argument("--fields", type=int, default=200)
    parser.add_argument("--video", type=Path, help="optional Snap-visible original-size UHD movie")
    args = parser.parse_args()
    if args.fields < 160:
        parser.error("at least 160 fields are required to exceed the former 128 MiB limit")
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
    report = {"software": args.software, "build": json.loads((site / "build.json").read_text())}
    try:
        with ICGNBrowser(
            backend="webgpu-wgsl",
            startup_timeout=60,
            chromium_args=(*flags, "--window-size=1600,1200"),
            call_timeout=300,
        ) as browser:
            url = f"http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate"
            browser._evaluate(f"location.assign({json.dumps(url)});true")
            wait(browser, "globalThis.liveDicStatus?.completed>=1")
            evaluate(
                browser,
                "globalThis.confirm=()=>true;"
                "document.getElementById('results-memory-limit').value='128';"
                "document.getElementById('results-memory-limit').onchange();"
                "await document.getElementById('freeze-field').onclick();return true;",
            )
            wait(browser, "liveDicWorkbench.report().recipe!=null")
            baseline = browser._evaluate("liveDicWorkbench.report()")
            assert baseline["statistics"]["count"] > 30000, baseline
            report["gpuBefore"] = memory(browser)
            report["capabilities"] = evaluate(
                browser, "return (await liveDicReport()).capabilities;"
            )
            evaluate(
                browser,
                "globalThis.originalField=structuredClone([...liveDicWorkbench.storage.cache.values()][0]);"
                "globalThis.hashArrays=async entry=>{const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),v=>v.toString(16).padStart(2,'0')).join('');"
                "return {parameters:await hash(entry.field.parameters),status:await hash(entry.field.status),iterations:await hash(entry.field.iterations),mask:entry.mask?await hash(entry.mask):null};};"
                "globalThis.originalHashes=await hashArrays(originalField);"
                "liveDicWorkbench.returnLive();return true;",
            )
            report["originalHashes"] = browser._evaluate("originalHashes")
            print("Recording distinct snapshots of an actual UHD solution", flush=True)
            for start in range(1, args.fields, 25):
                evaluate(
                    browser,
                    f"for(let i={start};i<Math.min({args.fields},{start + 25});i++){{"
                    "const field=structuredClone(originalField.field),metadata=structuredClone(originalField.metadata);"
                    "metadata.storageQualification='Replay snapshot; not another GN solve';"
                    "metadata.comparison.target={...metadata.comparison.target,index:i};"
                    "await liveDicWorkbench.add(field,metadata,{mask:originalField.mask});}return true;",
                )
            stress = browser._evaluate("liveDicWorkbench.report()")
            assert stress["count"] == args.fields, stress
            assert stress["storage"]["diskFields"] == args.fields, stress
            assert stress["storage"]["residentBytes"] <= 128 * MIB, stress
            assert stress["storage"]["storedPayloadBytes"] > 128 * MIB, stress
            assert stress["storage"]["residentFields"] < args.fields, stress
            report["replayedSnapshots"] = stress
            report["gpuAfter"] = memory(browser)
            assert {k: report["gpuBefore"]["gpu"][k] for k in ("engine", "supervisor")} == {
                k: report["gpuAfter"]["gpu"][k] for k in ("engine", "supervisor")
            }, "Recording results changed live GPU resource allocation"
            # The first unpinned field must actually have left the cache.
            assert browser._evaluate("!liveDicWorkbench.storage.cache.has(2)")
            evaluate(
                browser, "await document.getElementById('save-session').onclick();return true;"
            )
            session = stress["storage"]["sessionId"]
            browser._evaluate("location.reload();true")
            wait(browser, "globalThis.liveDicStatus?.completed>=1")
            evaluate(
                browser,
                "globalThis.confirm=()=>true;await document.getElementById('refresh-sessions').onclick();"
                f"document.getElementById('stored-sessions').value={json.dumps(session)};"
                "await document.getElementById('open-session').onclick();return true;",
            )
            wait(browser, "liveDicWorkbench.report().recipe!=null")
            reopened = browser._evaluate("liveDicWorkbench.report()")
            assert reopened["count"] == args.fields and reopened["storage"]["readOnly"], reopened
            assert reopened["storage"]["residentFields"] <= 2, reopened
            # Independent reload verification: all raw arrays and the ROI hash must be exact.
            reloaded = evaluate(
                browser,
                "const {FieldStorage}=await import('./ui/field-storage.js');const store=new FieldStorage({memoryLimit:128*1024**2});"
                f"const saved=await store.openSession({json.dumps(session)});"
                "const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),v=>v.toString(16).padStart(2,'0')).join('');"
                "const records=[];for(const index of [0,Math.floor(saved.entries.length/2),saved.entries.length-1]){const e=await store.load(saved.entries[index]);"
                "records.push({id:e.id,index:e.metadata.comparison.target.index??0,parameters:await hash(e.field.parameters),status:await hash(e.field.status),iterations:await hash(e.field.iterations),mask:e.mask?await hash(e.mask):null});}"
                "await store.close();return records;",
            )
            expected = report["originalHashes"]
            for record in reloaded:
                assert {key: record[key] for key in expected} == expected, record
            report["reopened"] = reopened
            report["reloadHashes"] = reloaded
            report["history"] = evaluate(
                browser,
                "const {FieldStorage}=await import('./ui/field-storage.js'),{extractHistory}=await import('./ui/history.js');"
                "const s=new FieldStorage({memoryLimit:16*1024**2});"
                f"const saved=await s.openSession({json.dumps(session)});"
                "const rows=await extractHistory(saved.entries,{recipe:liveDicWorkbench.report().recipe,component:'u',points:[[1000,900]],tool:'probe',loadEntry:e=>s.load(e)});"
                "const result={count:rows.length,first:rows[0],last:rows.at(-1),exact:rows.every((r,i)=>r.order===i+1&&r.value===rows[0].value),peakFinal:s.residentBytes()};await s.close();return result;",
            )
            assert report["history"]["count"] == args.fields and report["history"]["exact"], report[
                "history"
            ]
            # Drive the actual multipart UI. Deny real downloads, but capture the exact Blobs.
            asyncio.run(protocol(browser, [("Browser.setDownloadBehavior", {"behavior": "deny"})]))
            evaluate(
                browser,
                "globalThis.savedBlobs=[];globalThis.oldObjectURL=URL.createObjectURL;"
                "URL.createObjectURL=b=>{savedBlobs.push(b);return oldObjectURL(b);};"
                "await document.getElementById('save-project').onclick();return true;",
            )
            assert browser._evaluate("document.getElementById('export-parts').open")
            parts = []
            while not browser._evaluate("document.getElementById('download-next-part').disabled"):
                encoded = evaluate(
                    browser,
                    "await document.getElementById('download-next-part').onclick();"
                    "const blob=savedBlobs.pop();if(!blob)throw Error('No archive part was requested');"
                    "const a=new Uint8Array(await blob.arrayBuffer()),chunks=[];for(let i=0;i<a.length;i+=32768)chunks.push(String.fromCharCode(...a.subarray(i,i+32768)));return btoa(chunks.join(''));",
                )
                payload = base64.b64decode(encoded)
                filename = f"live-dic-results-part-{len(parts) + 1:05}.dicproject"
                (args.output / filename).write_bytes(payload)
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                    # Read every NPY independently with NumPy, without pickle.
                    for name in archive.namelist():
                        if name.endswith(".npy"):
                            assert np.isfinite(
                                np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
                            ).all()
                    for field in manifest["fields"]:
                        for key in ("parameters", "status", "iterations"):
                            array = np.load(
                                io.BytesIO(archive.read(f"{field['prefix']}/{key}.npy")),
                                allow_pickle=False,
                            )
                            assert hashlib.sha256(array.tobytes()).hexdigest() == expected[key]
                        if field["maskFile"]:
                            array = np.load(
                                io.BytesIO(archive.read(field["maskFile"])), allow_pickle=False
                            )
                            assert hashlib.sha256(array.tobytes()).hexdigest() == expected["mask"]
                parts.append(
                    {
                        "name": filename,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "manifest": manifest,
                    }
                )
                print(f"Checked archive part {len(parts)}", flush=True)
            evaluate(
                browser,
                "URL.createObjectURL=oldObjectURL;document.getElementById('export-parts').close();return true;",
            )
            report["exports"] = parts
            assert sorted(f["originalId"] for p in parts for f in p["manifest"]["fields"]) == list(
                range(1, args.fields + 1)
            )
            # Re-import the actual numbered files through the browser's portable file picker.
            evaluate(
                browser, "await document.getElementById('clear-fields').onclick();return true;"
            )
            snap = Path.home() / "snap/chromium/common"
            with tempfile.TemporaryDirectory(
                prefix="live-dic-storage-import-", dir=snap if snap.is_dir() else None
            ) as folder:
                files = []
                for part in parts:
                    target = Path(folder) / part["name"]
                    shutil.copyfile(args.output / part["name"], target)
                    files.append(target)
                asyncio.run(choose_files(browser, "#project-file", files))
                wait(
                    browser,
                    f"liveDicWorkbench.report().count==={args.fields}&&liveDicWorkbench.report().recipe!=null",
                )
            report["importedParts"] = browser._evaluate("liveDicWorkbench.report()")
            assert report["importedParts"]["storage"]["diskFields"] == args.fields, report[
                "importedParts"
            ]
            # Inject one transaction failure through the real acquisition admission path.
            evaluate(
                browser,
                "await document.getElementById('clear-fields').onclick();"
                "globalThis.seed=structuredClone([...await (async()=>{const {FieldStorage}=await import('./ui/field-storage.js');const s=new FieldStorage();"
                f"const v=await s.openSession({json.dumps(session)});const e=await s.load(v.entries[0]);await s.close();return [e];"
                "})()][0]);const s=liveDicWorkbench.storage;await s.database();globalThis.writeOriginal=s.backend.write.bind(s.backend);"
                "s.backend.write=async()=>{throw new DOMException('Injected disk full','QuotaExceededError');};"
                "await liveDicWorkbench.add(structuredClone(seed.field),seed.metadata,{mask:seed.mask});return true;",
            )
            failure = browser._evaluate("liveDicWorkbench.report()")
            assert failure["storage"]["unsavedFields"] == 1, failure
            assert failure["storage"]["diskFields"] == 0, failure
            assert browser._evaluate(
                "liveDicSourceState().paused&&!document.getElementById('record-fields').checked"
            )
            assert browser._evaluate(
                "document.getElementById('record-fields').disabled&&document.getElementById('freeze-field').disabled"
            ), "A failed write must prevent repeated recording before recovery"
            evaluate(
                browser,
                "liveDicWorkbench.storage.backend.write=writeOriginal;await document.getElementById('retry-storage').onclick();return true;",
            )
            recovered = browser._evaluate("liveDicWorkbench.report()")
            assert (
                recovered["storage"]["diskFields"] == 1
                and recovered["storage"]["unsavedFields"] == 0
            ), recovered
            report["injectedQuotaFailure"] = {"beforeRetry": failure, "afterRetry": recovered}
            report["batchedRecovery"] = evaluate(
                browser,
                "const {FieldStorage,IndexedFieldDatabase}=await import('./ui/field-storage.js');"
                "const db=await IndexedFieldDatabase.open(),write=db.write.bind(db);let unavailable=true,attempt=0;"
                "db.write=async(...args)=>{if(unavailable||++attempt===3)throw new DOMException('Injected recovery quota','QuotaExceededError');return write(...args);};"
                "const s=new FieldStorage({memoryLimit:1024**2,openDatabase:async()=>db});"
                "for(let id=1;id<=12;id++)await s.addMany([{id,label:'Recovery '+id,field:structuredClone(seed.field),metadata:seed.metadata,mask:seed.mask}]);"
                "unavailable=false;let reason=null;try{await s.retry();}catch(e){reason=e.message;}const partial=s.report();"
                "db.write=write;await s.retry();const complete=s.report(),catalog=await db.catalog(s.session.id);"
                "await s.close();return {reason,partial,complete,rows:catalog.length};",
            )
            assert report["batchedRecovery"]["partial"]["diskFields"] == 2, report[
                "batchedRecovery"
            ]
            assert report["batchedRecovery"]["complete"]["unsavedFields"] == 0, report[
                "batchedRecovery"
            ]
            assert report["batchedRecovery"]["rows"] == 12, report["batchedRecovery"]
            report["concurrentSessionDeletion"] = evaluate(
                browser,
                "const {FieldStorage}=await import('./ui/field-storage.js');const s=new FieldStorage();"
                f"const saved=await s.openSession({json.dumps(session)});await s.backend.deleteSession({json.dumps(session)});"
                "let reason=null;try{await s.saveSettings(saved.entries,1);}catch(e){reason=e.message;}await s.close();return {reason};",
            )
            assert "deleted" in (report["concurrentSessionDeletion"]["reason"] or ""), report[
                "concurrentSessionDeletion"
            ]
            report["noDisk"] = evaluate(
                browser,
                "await document.getElementById('clear-fields').onclick();const s=liveDicWorkbench.storage;"
                "s.openDatabase=async()=>{throw new Error('Injected storage unavailable');};s.setLimit(12*1024**2);"
                "let n=0;do{await liveDicWorkbench.add(structuredClone(seed.field),seed.metadata,{mask:seed.mask});n++;}while(s.residentBytes()<=s.memoryLimit&&n<20);"
                "return {state:liveDicWorkbench.report(),allRetained:[...s.cache.values()].every(e=>e.field.parameters?.length),paused:liveDicSourceState().paused};",
            )
            assert report["noDisk"]["allRetained"] and report["noDisk"]["paused"], report["noDisk"]
            assert report["noDisk"]["state"]["storage"]["unsavedFields"] > 0, report["noDisk"]
            if args.video:
                assert not args.software, "Use hardware for the long real-video admission check"
                evaluate(
                    browser, "await document.getElementById('clear-fields').onclick();return true;"
                )
                asyncio.run(choose_files(browser, "#video-file", [args.video.resolve()]))
                wait(
                    browser,
                    "liveDicSourceState().source==='video-file'&&!liveDicSourceState().loadingSource",
                )
                evaluate(
                    browser,
                    "document.getElementById('local-seek').value=.5;await document.getElementById('local-seek').onchange();"
                    "document.getElementById('results-memory-limit').value='1024';document.getElementById('results-memory-limit').onchange();"
                    "document.getElementById('record-fields').checked=true;await document.getElementById('record-fields').onchange();"
                    "document.querySelector('video').loop=true;await document.getElementById('pause').onclick();return true;",
                )
                print("Recording actual UHD video beyond 1 GiB", flush=True)
                deadline = time.monotonic() + 600
                while True:
                    state = browser._evaluate("liveDicWorkbench.report()")
                    if state["storage"]["storedPayloadBytes"] > 1024 * MIB + 32 * MIB:
                        break
                    assert time.monotonic() < deadline, state
                    assert not state["storage"]["unsavedFields"], state
                    time.sleep(2)
                evaluate(
                    browser,
                    "await document.getElementById('pause').onclick();await liveDicWorkbench.flush();return true;",
                )
                wait(browser, "liveDicSourceState().inFlight===0")
                video = browser._evaluate("liveDicWorkbench.report()")
                assert video["storage"]["residentBytes"] <= 1024 * MIB, video
                assert video["storage"]["diskFields"] == video["count"], video
                report["actualUhdVideo"] = {
                    "source": str(args.video),
                    "sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
                    "state": video,
                    "gpu": memory(browser),
                }
                print(
                    f"Actual UHD video: {video['count']} fields, {video['storage']['storedPayloadBytes'] / MIB:.1f} MiB committed",
                    flush=True,
                )
            report["userAgent"] = browser._evaluate("navigator.userAgent")
            report["errors"] = browser._evaluate("liveDicStatus.errors")
            assert not report["errors"], report["errors"]
            report["scope"] = (
                "Storage correctness with real UHD geometry. Snapshot replay is explicitly separate from actual video solves. No FPS claim or physical/free VRAM measurement."
            )
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {"fields": args.fields, "parts": len(parts), "output": str(args.output)}
                ),
                flush=True,
            )
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        raise
    finally:
        server.close()


if __name__ == "__main__":
    main()
