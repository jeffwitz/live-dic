"""Controller timing contracts with a deterministic clock, without shader changes."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("node"), "Node is needed for browser timing contracts")
class BrowserTimingTests(unittest.TestCase):
    def test_comparison_switch_clears_old_age_and_fps_windows(self):
        # Reproduce the assembled module layout without copying images/vendor assets.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package.json").write_text('{"type":"module"}')
            shutil.copytree(PROJECT / "python/src/lucas_kanade_icgn/web/core", root / "web/core")
            (root / "ui").mkdir()
            (root / "render").mkdir()
            for name in ("session.js", "plate-truth.js"):
                shutil.copyfile(PROJECT / "web-app/ui" / name, root / "ui" / name)
            shutil.copyfile(PROJECT / "web-app/render/field.js", root / "render/field.js")
            script = f"""
              import {{DICEngine}} from {json.dumps((root / "web/core/engine.js").as_uri())};
              import {{createSession}} from {json.dumps((root / "ui/session.js").as_uri())};
              let clock=0,solveMs=100,preparationMs=0;
              globalThis.performance={{now:()=>clock}};
              globalThis.GPUTextureUsage={{COPY_DST:2,TEXTURE_BINDING:4,RENDER_ATTACHMENT:16}};
              const frame=()=>({{width:64,height:64,close(){{}},timestamp:1234}});
              globalThis.createImageBitmap=async()=>frame();
              const info={{referenceValid:new Uint8Array([1]),grid:[1,1],config:{{}}}};
              const engine={{device:{{lost:new Promise(()=>{{}}),addEventListener(){{}},
                  createTexture:()=>({{destroy(){{}}}}),queue:{{copyExternalImageToTexture(){{}}}}}},
                ring:{{idle:true}},resetTemporalState(){{}},setAnalyticTruth(){{}},destroy(){{}},
                setReference:async()=>{{clock+=preparationMs;return info;}},
                register:async()=>{{clock+=solveMs;return {{width:64,height:64,grid:[1,1],timing:{{wallMs:solveMs}},accepted:1,selected:1,meanIterations:4}};}}}};
              DICEngine.create=async()=>engine;
              const messages=[],session=createSession(message=>messages.push(message));
              await session.handle({{type:'reference',frame:frame(),options:{{profile:'portable'}}}});
              for(let i=0;i<3;i++){{clock+=10;await session.handle({{type:'frame',frame:frame(),comparison:{{mode:'fixed'}}}});}}
              const before=messages.filter(m=>m.type==='result').at(-1);
              await session.handle({{type:'comparison',mode:'consecutive'}});
              solveMs=5;preparationMs=50;
              await session.handle({{type:'frame',frame:frame(),referenceFrame:frame(),comparison:{{mode:'consecutive'}}}});
              const after=messages.filter(m=>m.type==='result').at(-1);
              console.log(JSON.stringify({{before,after,errors:messages.filter(m=>m.type==='error')}}));
            """
            result = json.loads(
                subprocess.check_output(
                    ["node", "--input-type=module"], input=script, text=True, encoding="utf-8"
                )
            )
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["before"]["ageMedian"], 100)
        self.assertGreater(result["before"]["fps"], 0)
        self.assertEqual(result["after"]["ageMedian"], 55)
        self.assertEqual(result["after"]["ageP95"], 55)
        self.assertEqual(result["after"]["fps"], 0)
        self.assertEqual(result["after"]["timing"]["pairReferencePreparationWallMs"], 50)


if __name__ == "__main__":
    unittest.main()
