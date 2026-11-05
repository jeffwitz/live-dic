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
    def test_preview_throttling_keeps_every_measurement_and_flushes_latest_pair(self):
        self._exercise_preview(True)

    def test_suspended_preview_records_every_field_and_flushes_only_on_request(self):
        self._exercise_preview(False)

    def _exercise_preview(self, preview_enabled):
        runtime = tempfile.TemporaryDirectory()
        self.addCleanup(runtime.cleanup)
        root = Path(runtime.name)
        (root / "package.json").write_text('{"type":"module"}')
        shutil.copytree(PROJECT / "python/src/live_dic/web/core", root / "web/core")
        shutil.copytree(PROJECT / "web-app/ui", root / "ui")
        shutil.copytree(PROJECT / "web-app/render", root / "render")
        script = f"""
          import {{DICEngine}} from {json.dumps((root / "web/core/engine.js").as_uri())};
          import {{LiveVectorRenderer as FieldRenderer}} from {json.dumps((root / "render/live-vectors.js").as_uri())};
          import {{createSession}} from {json.dumps((root / "ui/session.js").as_uri())};
          let clock=0,engineOptions,preparations=0,resets=0,solves=0;const timers=new Map();let timer=0;
          globalThis.performance={{now:()=>clock,timeOrigin:1000}};
          globalThis.setTimeout=fn=>{{timers.set(++timer,fn);return timer;}};
          globalThis.clearTimeout=id=>timers.delete(id);
          Object.defineProperty(globalThis,'navigator',{{configurable:true,value:{{gpu:{{requestAdapter:async()=>null,getPreferredCanvasFormat:()=> 'rgba8unorm'}}}}}});
          globalThis.isSecureContext=true;
          globalThis.GPUBufferUsage={{UNIFORM:1,COPY_DST:2}};
          globalThis.GPUShaderStage={{VERTEX:1}};
          globalThis.GPUTextureUsage={{COPY_DST:2,TEXTURE_BINDING:4,RENDER_ATTACHMENT:16}};
          const frame=()=>({{width:64,height:64,close(){{this.closed=true;}}}});
          globalThis.createImageBitmap=async()=>frame();
          const context={{configure(){{}},unconfigure(){{}}}},canvas={{width:64,height:64,getContext:()=>context}};
          const info={{width:64,height:64,referenceValid:new Uint8Array(16).fill(1),grid:[4,4],config:{{motionModel:'affine'}}}};
          const engine={{device:{{lost:new Promise(()=>{{}}),addEventListener(){{}},createShaderModule:()=>({{}}),createRenderPipeline:()=>({{}}),
             createBindGroupLayout:()=>({{}}),createPipelineLayout:()=>({{}}),
             createBuffer:()=>({{destroy(){{}}}}),createTexture:()=>({{createView:()=>({{}}),destroy(){{}}}}),queue:{{copyExternalImageToTexture(){{}}}}}},
             ring:{{idle:true}},resetTemporalState(){{resets++;}},setAnalyticTruth(){{}},destroy(){{}},setReference:async()=>{{preparations++;return info;}},
             register:async()=>{{solves++;return {{width:64,height:64,grid:[4,4],config:info.config,timing:{{wallMs:0}},accepted:4,selected:16}};}},
             exportField:async()=>({{width:64,height:64,parameters:new Float32Array(96),status:new Float32Array(16),iterations:new Float32Array(16)}})}};
          DICEngine.create=async options=>{{engineOptions=options;return engine;}};
          const draws=[];FieldRenderer.prototype.draw=function(engine,result,options){{
            const width=options.previewMaxWidth||64;this.canvas.width=width;this.canvas.height=width;draws.push(width);return 1;
          }};
          const messages=[],session=createSession(m=>messages.push(m));
          await session.handle({{type:'boot',canvas,measure:false}});
          await session.handle({{type:'reference',frame:frame(),options:{{profile:'portable',supervision:{{displayRate:10,previewMaxWidth:32,previewEnabled:{json.dumps(preview_enabled)}}}}}}});
          const targets=[];
          for(let i=1;i<=3;i++){{clock+={10 if preview_enabled else 1000};const target=frame();targets.push(target);await session.handle({{type:'frame',id:i,frame:target,recordRequested:true,comparison:{{mode:'fixed',target:{{ordinal:i}}}}}});}}
          const results=messages.filter(m=>m.type==='result');
          const before={{draws:[...draws],measurements:results.length,timers:timers.size,recordWidths:results.map(m=>m.record.field.width)}};
          const numericalBefore=[preparations,resets,solves];
          await session.handle({{type:'presentation',id:8,options:{{previewEnabled:false,displayRate:1,previewMaxWidth:32}}}});
          const presentationDidNotCalculate=JSON.stringify(numericalBefore)===JSON.stringify([preparations,resets,solves]);
          await session.handle({{type:'flush',id:9}});const flushed=messages.at(-1);
          await session.handle({{type:'show-reference',show:true}});const reference=messages.at(-1);
          clock+=10;const final=frame();targets.push(final);
          await session.handle({{type:'frame',id:4,frame:final,comparison:{{mode:'fixed',target:{{ordinal:4}}}}}});
          await session.handle({{type:'destroy'}});
          console.log(JSON.stringify({{before,flushed,reference,draws,presentationDidNotCalculate,preparations,errors:messages.filter(m=>m.type==='error'),timers:timers.size,closed:targets.every(t=>t.closed),numericalOptionsUnchanged:!('supervision' in engineOptions)}}));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(
            actual["before"],
            dict(draws=[32], measurements=3, timers=int(preview_enabled), recordWidths=[64] * 3),
        )
        self.assertEqual(actual["flushed"]["presentation"]["resultId"], 3)
        self.assertEqual(actual["flushed"]["presentation"]["width"], 32)
        self.assertTrue(actual["reference"]["presentation"]["referencePreview"])
        self.assertEqual(actual["reference"]["presentation"]["width"], 64)
        self.assertEqual(actual["draws"], [32, 32, 64])
        self.assertEqual(actual["timers"], 0)
        self.assertEqual(actual["errors"], [])
        self.assertTrue(actual["closed"])
        self.assertTrue(actual["numericalOptionsUnchanged"])
        self.assertTrue(actual["presentationDidNotCalculate"])
        self.assertEqual(actual["preparations"], 1)

    def test_comparison_switch_clears_old_age_and_fps_windows(self):
        # Reproduce the assembled module layout without copying images/vendor assets.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package.json").write_text('{"type":"module"}')
            shutil.copytree(PROJECT / "python/src/live_dic/web/core", root / "web/core")
            (root / "ui").mkdir()
            shutil.copytree(PROJECT / "web-app/render", root / "render")
            for name in ("session.js", "plate-truth.js", "supervision.js"):
                shutil.copyfile(PROJECT / "web-app/ui" / name, root / "ui" / name)
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
                  createTexture:()=>({{createView:()=>({{}}),destroy(){{}}}}),queue:{{copyExternalImageToTexture(){{}}}}}},
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
