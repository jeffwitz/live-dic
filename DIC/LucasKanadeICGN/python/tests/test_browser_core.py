"""Dependency-free browser core contracts and opt-in Chromium execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from importlib.resources import files
from pathlib import Path

import numpy as np

from lucas_kanade_icgn.roi import prepare_roi, reduce_mask, seed_permissions

CORE = Path(str(files("lucas_kanade_icgn").joinpath("web/core")))


@unittest.skipUnless(shutil.which("node"), "Node is needed to execute ES modules")
class BrowserGeometryTests(unittest.TestCase):
    def test_full_roi_geometry_matches_native_conservative_links(self):
        mask = np.ones((93, 111), bool)
        mask[11:77, 54] = False
        mask[28:39, 21:32] = False
        for size, stride in ((16, 16), (17, 8), (32, 16)):
            fine = prepare_roi(mask, size, stride)
            reduced = reduce_mask(mask)
            coarse = prepare_roi(reduced, size, stride)
            links = seed_permissions(
                coarse,
                fine,
                source_scale=2,
                destination_scale=1,
                size=size,
                stride=stride,
                reference_mask=mask,
            )
            script = f"""
              import {{ planRoi, reduceMask, seedPermissions }} from {json.dumps((CORE / "roi.js").as_uri())};
              const mask = new Uint8Array({json.dumps(mask.astype(int).ravel().tolist())});
              const fine = planRoi(mask, 111, 93, {size}, {stride});
              const r = reduceMask(mask, 111, 93);
              const coarse = planRoi(r.mask, r.width, r.height, {size}, {stride});
              console.log(JSON.stringify({{ selected: [...fine.selected], neighbours: [...fine.neighbours],
                reduced: [...r.mask], links: [...seedPermissions(coarse, fine, 2, 1, {size}, {stride}, mask, 111, 93)] }}));
            """
            output = subprocess.check_output(
                ["node", "--input-type=module"], input=script, text=True
            )
            actual = json.loads(output)
            np.testing.assert_array_equal(actual["selected"], fine.selected.ravel())
            np.testing.assert_array_equal(actual["neighbours"], fine.neighbour_bits.ravel())
            np.testing.assert_array_equal(actual["reduced"], reduced.ravel())
            np.testing.assert_array_equal(actual["links"], links.ravel())

    def test_slot_ring_starts_both_slots_and_delivers_in_order(self):
        script = f"""
          import {{ SlotRing }} from {json.dumps((CORE / "ring.js").as_uri())};
          const ring = new SlotRing(2), events = [];
          let done0, done1;
          const first = ring.run(i => {{ events.push(i); return new Promise(r => done0 = r); }});
          const second = ring.run(i => {{ events.push(i); return new Promise(r => done1 = r); }});
          const blocked = await ring.run(() => 3).then(() => false, () => true);
          await Promise.resolve();
          const concurrent = events.length === 2;
          done1(1); done0(0);
          const values = await Promise.all([first, second]);
          console.log(JSON.stringify({{concurrent, blocked, values, idle:ring.idle}}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(result, dict(concurrent=True, blocked=True, values=[0, 1], idle=True))

    def test_constructor_rejects_unported_or_invalid_options_without_a_gpu(self):
        script = f"""
          import {{DICEngine}} from {json.dumps((CORE / "engine.js").as_uri())};
          const cases=[{{subsetSize:'16'}},{{pyramidLevels:true}},{{initialization:'temporal',pyramidLevels:3,temporalFineCorrection:true}},
            {{spatialWeighting:'gaussian'}},{{roiPolicy:'partial'}},{{solverWorkgroupSize:32}},{{unknownOption:1}},{{roiPolicy:0}},{{readback:null}},{{spatialWeighting:null}},{{initialization:null}},{{pipelineDepth:null}}];
          console.log(JSON.stringify(cases.map(options=>{{try{{new DICEngine(options);return false;}}catch{{return true;}}}})));
        """
        results = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertTrue(all(results))


@unittest.skipUnless(
    os.environ.get("ICGN_RUN_BROWSER_TESTS") == "1", "set ICGN_RUN_BROWSER_TESTS=1"
)
class BrowserCoreExecutionTests(unittest.TestCase):
    def test_core_is_instantiable_without_page_globals_and_destroy_is_idempotent(self):
        from lucas_kanade_icgn import ICGNWebGL

        with ICGNWebGL(
            backend="webgpu-wgsl",
            startup_timeout=60,
            chromium_args=(
                (
                    "--use-angle=swiftshader",
                    "--enable-unsafe-webgpu",
                    "--use-webgpu-adapter=swiftshader",
                    "--use-vulkan=swiftshader",
                )
                if os.environ.get("ICGN_BROWSER_SOFTWARE") == "1"
                else ()
            ),
        ) as browser:
            value = browser._evaluate("""(async () => {
              const { DICEngine } = await import('../core/engine.js');
              const engine = await DICEngine.create({width: 64, height: 64});
              const image = Uint8Array.from({length: 4096}, (_, i) => (i*7+i*i)%251);
              await engine.setReference(image);
              const result = await engine.register(image);
              const finite = result.parameters.every(Number.isFinite);
              const storageLimit = engine.device.limits.maxStorageBuffersPerShaderStage;
              engine.destroy(); engine.destroy();
              return {finite, storageLimit, accepted: result.accepted};
            })()""")
        self.assertTrue(value["finite"])
        self.assertEqual(value["storageLimit"], 8)
        self.assertGreater(value["accepted"], 0)

    def test_integer_bt709_conversion_and_gpu_only_results(self):
        from lucas_kanade_icgn import ICGNWebGL

        with ICGNWebGL(
            backend="webgpu-wgsl",
            startup_timeout=60,
            chromium_args=(
                (
                    "--use-angle=swiftshader",
                    "--enable-unsafe-webgpu",
                    "--use-webgpu-adapter=swiftshader",
                    "--use-vulkan=swiftshader",
                )
                if os.environ.get("ICGN_BROWSER_SOFTWARE") == "1"
                else ()
            ),
        ) as browser:
            actual = browser._evaluate("""(async () => {
              const {createInputConverter} = await import('../core/input.js');
              const {DICEngine} = await import('../core/engine.js');
              const adapter = await navigator.gpu.requestAdapter();
              const device = await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:8}});
              const width=37,height=35, rgba=new Uint8ClampedArray(width*height*4), expected=[];
              for(let i=0;i<width*height;i++) {
                const r=(i*17)%256,g=(i*71)%256,b=(i*129)%256;
                rgba.set([r,g,b,255],4*i); expected.push(Math.floor((2126*r+7152*g+722*b+5000)/10000));
              }
              const bitmap=await createImageBitmap(new ImageData(rgba,width,height));
              const converter=await createInputConverter(device,width,height);
              const pixels=await converter.referencePixels(bitmap);
              const matches=pixels.every((v,i)=>v===expected[i]); converter.destroy();
              const engine=await DICEngine.create({width,height,device,pyramidLevels:1,readback:'none'});
              await engine.setReference(bitmap);
              const result=await engine.register(bitmap);
              const exported=await engine.exportField();
              const noField=!('parameters' in result), finite=exported.parameters.every(Number.isFinite);
              engine.destroy(); bitmap.close(); device.destroy();
              return {matches,noField,finite};
            })()""")
        self.assertEqual(actual, dict(matches=True, noField=True, finite=True))

    def test_full_roi_empty_and_missing_coarse_seed(self):
        from lucas_kanade_icgn import ICGNWebGL

        with ICGNWebGL(
            backend="webgpu-wgsl",
            startup_timeout=60,
            chromium_args=(
                (
                    "--use-angle=swiftshader",
                    "--enable-unsafe-webgpu",
                    "--use-webgpu-adapter=swiftshader",
                    "--use-vulkan=swiftshader",
                )
                if os.environ.get("ICGN_BROWSER_SOFTWARE") == "1"
                else ()
            ),
        ) as browser:
            actual = browser._evaluate("""(async () => {
              const {DICEngine}=await import('../core/engine.js');
              const pixels=Uint8Array.from({length:16384},(_,i)=>(i*7+i*i)%251);
              const engine=await DICEngine.create({width:128,height:128});
              await engine.setReference(pixels,{roiMask:new Uint8Array(16384)});
              const empty=await engine.register(pixels);
              const excluded=empty.status.every(v=>v===-3);
              const band=new Uint8Array(16384);band.fill(1,32*128,64*128);
              await engine.setReference(pixels,{roiMask:band});band.fill(0);
              const result=await engine.register(pixels);
              let partial=false;
              try{await engine.setReference(pixels,{roiPolicy:'partial'});}catch(error){partial=error.message.includes('not supported');}
              engine.destroy();
              return {excluded,accepted:result.accepted,partial};
            })()""")
        self.assertTrue(actual["excluded"])
        self.assertGreater(actual["accepted"], 0)
        self.assertTrue(actual["partial"])

    def test_decimated_readback_is_a_gpu_sample_of_the_complete_field(self):
        from lucas_kanade_icgn import ICGNWebGL

        flags = (
            (
                "--use-angle=swiftshader",
                "--enable-unsafe-webgpu",
                "--use-webgpu-adapter=swiftshader",
                "--use-vulkan=swiftshader",
            )
            if os.environ.get("ICGN_BROWSER_SOFTWARE") == "1"
            else ()
        )
        with ICGNWebGL(backend="webgpu-wgsl", chromium_args=flags) as browser:
            actual = browser._evaluate("""(async()=>{
              const {DICEngine}=await import('../core/engine.js');
              const width=640,height=480,pixels=Uint8Array.from({length:width*height},(_,i)=>(i*7+i*i)%251);
              const engine=await DICEngine.create({width,height,readback:'decimated'});
              await engine.setReference(pixels);const result=await engine.register(pixels);
              const full=await engine.exportField();let matches=true;
              result.sampleIndices.forEach((index,i)=>{
                for(let k=0;k<6;k++)if(result.packed[i*8+k]!==full.parameters[index*6+k])matches=false;
                if(result.packed[i*8+6]!==full.status[index]||result.packed[i*8+7]!==full.iterations[index])matches=false;
              });engine.destroy();return {matches,smaller:result.sampleIndices.length<full.status.length};
            })()""")
        self.assertEqual(actual, dict(matches=True, smaller=True))
