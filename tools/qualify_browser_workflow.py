"""Exercise the actual prepare / solve / replay workflow on original UHD images."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from qualify_browser_editor import protocol
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
            ICGNWebGL(
                backend="webgpu-wgsl",
                chromium_args=(*flags, "--window-size=1600,1200"),
                call_timeout=300,
            ) as browser,
        ):
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=manual')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.ready")
            browser._evaluate("document.getElementById('plate').onclick().then(()=>true)")
            assert browser._evaluate("liveDicStatus.completed") == 0
            browser._evaluate(
                "document.getElementById('plate-load').value='225';document.getElementById('plate-load').onchange().then(()=>true)"
            )
            assert browser._evaluate("liveDicStatus.completed") == 0
            assert browser._evaluate("liveDicSourceState().viewingInput")
            assert browser._evaluate(
                "document.getElementById('pause').closest('#pane-solve')!==null"
            )
            browser._evaluate("document.getElementById('draw').onclick().then(()=>true)")
            assert browser._evaluate(
                "!liveDicSourceState().viewingInput&&!document.getElementById('field').hidden"
            )
            assert browser._evaluate("liveDicPipelineReport().presentation.referencePreview")
            assert browser._evaluate("liveDicStatus.completed") == 0
            browser._evaluate("document.getElementById('cancel-roi').onclick().then(()=>true)")
            assert browser._evaluate("liveDicSourceState().viewingInput")
            fixtures = ROOT / "DIC/LucasKanadeICGN/assets/uhd_plate_hole"
            files = []
            for index, name in enumerate(
                ["reference.png", "frame_095.png", "frame_225.png", "frame_095.png"]
            ):
                path = Path(temporary) / f"{index:03}.png"
                shutil.copyfile(fixtures / name, path)
                files.append(path)
            asyncio.run(choose_files(browser, "#image-files", files))
            wait(
                browser,
                "liveDicSourceState().source==='images'&&!liveDicSourceState().loadingSource",
            )
            assert browser._evaluate("liveDicStatus.completed") == 0
            browser._evaluate(
                "document.getElementById('local-seek').value=2;document.getElementById('local-seek').onchange().then(()=>true)"
            )
            assert browser._evaluate("liveDicStatus.completed") == 0
            browser._evaluate(
                "document.getElementById('local-seek').value=1;document.getElementById('local-seek').onchange().then(()=>true)"
            )
            before = browser._evaluate(
                "({source:liveDicSourceState(),completed:liveDicStatus.completed,library:liveDicWorkbench.report(),saving:document.getElementById('record-fields').checked})"
            )
            assert before["saving"] and before["source"]["viewingInput"], before
            browser._evaluate(
                "document.querySelector('[data-pane=solve]').click();document.getElementById('pause').onclick().then(()=>true)"
            )
            wait(
                browser,
                "liveDicWorkbench.report().count===3&&liveDicWorkbench.report().recipe!==null",
            )
            solved = browser._evaluate(
                "({source:liveDicSourceState(),completed:liveDicStatus.completed,library:liveDicWorkbench.report()})"
            )
            assert solved["completed"] == 3 and solved["source"]["paused"]
            assert solved["source"]["pane"] == "view" and solved["library"]["reviewing"]
            assert browser._evaluate("!document.getElementById('result-transport').hidden")
            browser._evaluate(
                "document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'r',bubbles:true}));true"
            )
            assert browser._evaluate(
                "!liveDicSourceState().loadingSource&&liveDicWorkbench.report().reviewing"
            )
            # Stored numerical arrays must survive every display/navigation action.
            digest = """(async()=>{const s=liveDicWorkbench.storage,hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a.buffer)),v=>v.toString(16).padStart(2,'0')).join('');
              const fields=[];for(let i=1;i<s.session.nextId;i++){const {field}=await s.backend.read(s.session.id,i);fields.push({size:[field.width,field.height],parameters:await hash(field.parameters),status:await hash(field.status),iterations:await hash(field.iterations)});}return fields;})()"""
            raw_before = browser._evaluate(digest)
            # Deliberately save a different presentation on a later frame.
            browser._evaluate(
                "document.getElementById('result-seek').value=1;document.getElementById('result-seek').oninput().then(()=>true)"
            )
            wait(
                browser,
                "liveDicWorkbench.report().selected===2&&liveDicWorkbench.report().recipe!==null",
            )
            browser._evaluate(
                "document.getElementById('review-component').value='u';document.getElementById('review-component').onchange();document.getElementById('result-seek').value=0;document.getElementById('result-seek').oninput().then(()=>true)"
            )
            wait(
                browser,
                "liveDicWorkbench.report().selected===1&&liveDicWorkbench.report().recipe!==null",
            )
            browser._evaluate(
                "document.getElementById('review-component').value='vonMises';document.getElementById('review-component').onchange();document.getElementById('replay-rate').value=1;document.getElementById('replay-results').onclick();true"
            )
            wait(
                browser,
                "liveDicWorkbench.report().selected===2&&liveDicWorkbench.report().recipe!==null",
            )
            assert (
                browser._evaluate("document.getElementById('review-component').value") == "vonMises"
            )
            # Quantity changes during playback must remain selected on the next frame.
            browser._evaluate(
                "document.getElementById('review-component').value='exx';document.getElementById('review-component').onchange();true"
            )
            wait(
                browser,
                "liveDicWorkbench.report().selected===3&&!liveDicWorkbench.report().replaying&&liveDicWorkbench.report().recipe!==null",
            )
            assert browser._evaluate("document.getElementById('review-component').value") == "exx"
            assert browser._evaluate("liveDicStatus.completed") == 3
            # Rapid seeks converge to the last requested saved result.
            browser._evaluate("""(()=>{
              const store=liveDicWorkbench.storage;globalThis.seekLoads=[];
              globalThis.originalLoad=store.load.bind(store);let first=true;
              store.load=async entry=>{seekLoads.push(entry.id);if(first){first=false;await new Promise(resolve=>globalThis.releaseSeek=resolve);}return originalLoad(entry);};
              return true;
            })()""")
            browser._evaluate(
                "for(const index of [0,2,1]){document.getElementById('result-seek').value=index;document.getElementById('result-seek').oninput();}true"
            )
            assert browser._evaluate("seekLoads") == [1]
            browser._evaluate("releaseSeek();true")
            wait(
                browser,
                "liveDicWorkbench.report().selected===2&&liveDicWorkbench.report().recipe!==null",
            )
            assert browser._evaluate("liveDicStatus.completed") == 3
            assert browser._evaluate("seekLoads") == [1, 2]
            browser._evaluate("liveDicWorkbench.storage.load=originalLoad;true")
            # A seek during an in-progress playback load must wait for it too.
            browser._evaluate("""(()=>{
              globalThis.playbackLoads=[];let first=true;
              liveDicWorkbench.storage.load=async entry=>{playbackLoads.push(entry.id);if(first){first=false;await new Promise(resolve=>globalThis.releasePlaybackLoad=resolve);}return originalLoad(entry);};
              document.getElementById('replay-rate').value=30;document.getElementById('replay-results').onclick();return true;
            })()""")
            wait(browser, "playbackLoads.length===1")
            browser._evaluate(
                "for(const index of [0,1]){document.getElementById('result-seek').value=index;document.getElementById('result-seek').oninput();}true"
            )
            assert browser._evaluate("playbackLoads") == [3]
            browser._evaluate("releasePlaybackLoad();true")
            wait(
                browser,
                "liveDicWorkbench.report().selected===2&&liveDicWorkbench.report().recipe!==null",
            )
            assert browser._evaluate("playbackLoads") == [3, 2]
            browser._evaluate("liveDicWorkbench.storage.load=originalLoad;true")
            assert browser._evaluate("liveDicStatus.completed") == 3
            raw_after = browser._evaluate(digest)
            assert raw_before == raw_after and all(row["size"] == [3840, 2160] for row in raw_after)
            pixels = asyncio.run(
                protocol(
                    browser,
                    [("Page.captureScreenshot", dict(format="png", captureBeyondViewport=False))],
                )
            )[0]["data"]
            (args.output / "results.png").write_bytes(base64.b64decode(pixels))
            report = dict(
                build=json.loads((site / "build.json").read_text()),
                software=args.software,
                input_hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],
                before=before,
                solved=solved,
                final=browser._evaluate("liveDicWorkbench.report()"),
                raw_fields=raw_after,
                correlations=browser._evaluate("liveDicStatus.completed"),
                coalesced_loads=browser._evaluate("seekLoads"),
                playback_loads=browser._evaluate("playbackLoads"),
                errors=browser._evaluate("liveDicStatus.errors"),
            )
            assert not report["errors"], report["errors"]
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        fields=report["final"]["count"],
                        correlations=report["correlations"],
                        quantity=report["final"]["quantity"],
                        errors=report["errors"],
                    )
                )
            )
    finally:
        server.close()


if __name__ == "__main__":
    main()
