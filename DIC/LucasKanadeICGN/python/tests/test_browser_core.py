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
    def test_global_affine_initializer_recovers_masked_subpixel_affine_pair(self):
        initializer = CORE / "global-affine.js"
        script = f"""
          import {{ GlobalAffineInitializer }} from {json.dumps(initializer.as_uri())};
          const width=1280,height=1024,reference=new Uint8Array(width*height),target=new Uint8Array(width*height);
          const mask=new Uint8Array(width*height);
          for(let y=0;y<height;y++)for(let x=0;x<width;x++){{
            const i=y*width+x;
            reference[i]=Math.round(128+38*Math.sin(x*.17)+31*Math.sin(y*.13)+24*Math.sin((x+y)*.21)+16*Math.cos((2*x-y)*.09));
            mask[i]=+(x>=24&&x<width-24&&y>=24&&y<height-24&&!(x>=240&&x<248&&y>=180&&y<204));
          }}
          const [a,b,c,d,dx,dy]=[1.025,.008,-.006,.985,3.2,-2.3],cx=(width-1)/2,cy=(height-1)/2,det=a*d-b*c;
          for(let y=0;y<height;y++)for(let x=0;x<width;x++){{
            const X=x-cx-dx,Y=y-cy-dy,rx=(d*X-b*Y)/det+cx,ry=(-c*X+a*Y)/det+cy;
            const x0=Math.floor(rx),y0=Math.floor(ry);if(x0<0||x0>=width-1||y0<0||y0>=height-1)continue;
            const fx=rx-x0,fy=ry-y0,i=y0*width+x0;
            target[y*width+x]=Math.round((reference[i]*(1-fx)+reference[i+1]*fx)*(1-fy)
              +(reference[i+width]*(1-fx)+reference[i+width+1]*fx)*fy);
          }}
          const defaultEstimate=new GlobalAffineInitializer(reference,width,height,mask).estimate(target);
          const sixScaleEstimate=new GlobalAffineInitializer(reference,width,height,mask,6).estimate(target);
          console.log(JSON.stringify({{defaultEstimate,sixScaleEstimate}}));
        """
        estimates = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        estimate = estimates["defaultEstimate"]
        six_scale = estimates["sixScaleEstimate"]
        self.assertEqual(estimate["levels"], 5)
        self.assertGreater(estimate["score"], 0.98)
        self.assertAlmostEqual(estimate["dx"], 3.2, delta=0.2)
        self.assertAlmostEqual(estimate["dy"], -2.3, delta=0.25)
        np.testing.assert_allclose(estimate["linear"], [0.025, 0.008, -0.006, -0.015], atol=0.004)
        self.assertEqual(six_scale["levels"], 6)
        self.assertEqual(len(six_scale["diagnostics"]), 6)
        self.assertTrue(np.isfinite([six_scale["score"], six_scale["dx"], six_scale["dy"]]).all())
        self.assertGreaterEqual(six_scale["sampleCount"], 500)

    def test_global_affine_level_count_is_bounded(self):
        initializer = CORE / "global-affine.js"
        script = f"""
          import {{ GlobalAffineInitializer }} from {json.dumps(initializer.as_uri())};
          const reference=new Uint8Array(1024*768);
          const failures=[4,9].map(levels=>{{try{{new GlobalAffineInitializer(reference,1024,768,null,levels);return null;}}catch(error){{return error.message;}}}});
          console.log(JSON.stringify(failures));
        """
        failures = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertTrue(all("between 5 and 8" in message for message in failures))

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
    def test_device_negotiates_large_reference_and_preserves_injected_ownership(self):
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
        with ICGNWebGL(backend="webgpu-wgsl", chromium_args=flags, call_timeout=300) as browser:
            result = browser._evaluate("""(async()=>{
              const {DICEngine}=await import('../core/engine.js'),width=4112,height=3008,required=width*height*16;
              const adapter=await navigator.gpu.requestAdapter(),injected=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:8}});
              let refusal;try{await DICEngine.create({device:injected,width,height});}catch(e){refusal=e.message;}
              const probe=injected.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST});probe.destroy();injected.destroy();
              if(adapter.limits.maxStorageBufferBindingSize<required||adapter.limits.maxBufferSize<required){
                let reason;try{await DICEngine.create({width,height});}catch(e){reason=e.message;}
                return {supported:false,refusal,reason};
              }
              const e=await DICEngine.create({pyramidLevels:1,outlierPolicy:'disabled'});let small,large;
              try{
                small=await createImageBitmap(new ImageData(64,64));await e.setReference(small);
                const old=e.device,rgba=new Uint8ClampedArray(width*height*4);for(let i=3;i<rgba.length;i+=4)rgba[i]=255;
                large=await createImageBitmap(new ImageData(rgba,width,height));const info=await e.setReference(large),r=await e.register(large);
                return {supported:true,refusal,replaced:old!==e.device,limit:e.device.limits.maxStorageBufferBindingSize,
                  width:info.width,height:info.height,count:r.status.length,finite:r.parameters.every(Number.isFinite)};
              }finally{small?.close();large?.close();e.destroy();}
            })()""")
        self.assertIn("maxStorageBufferBindingSize", result["refusal"])
        if result["supported"]:
            self.assertTrue(result["replaced"])
            self.assertTrue(result["finite"])
            self.assertGreaterEqual(result["limit"], 4112 * 3008 * 16)
            self.assertEqual((result["width"], result["height"]), (4112, 3008))
            self.assertEqual(result["count"], 257 * 188)
        else:
            self.assertIn("GPU adapter maxStorageBufferBindingSize", result["reason"])

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

    def test_global_affine_seed_runs_through_the_gpu_pyramid(self):
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
        with ICGNWebGL(backend="webgpu-wgsl", chromium_args=flags, call_timeout=180) as browser:
            actual = browser._evaluate(
                """(async () => {
                  const {DICEngine}=await import('../core/engine.js');
                  const width=1024,height=768,reference=new Uint8Array(width*height),target=new Uint8Array(width*height),mask=new Uint8Array(width*height);
                  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
                    const i=y*width+x;
                    reference[i]=Math.round(128+38*Math.sin(x*.17)+31*Math.sin(y*.13)+24*Math.sin((x+y)*.21)+16*Math.cos((2*x-y)*.09));
                    mask[i]=+(x>=24&&x<width-24&&y>=24&&y<height-24&&!(x>=500&&x<508&&y>=360&&y<384));
                  }
                  const [a,b,c,d,dx,dy]=[1.025,.008,-.006,.985,3.2,-2.3],cx=(width-1)/2,cy=(height-1)/2,det=a*d-b*c;
                  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
                    const X=x-cx-dx,Y=y-cy-dy,rx=(d*X-b*Y)/det+cx,ry=(-c*X+a*Y)/det+cy;
                    const x0=Math.floor(rx),y0=Math.floor(ry);if(x0<0||x0>=width-1||y0<0||y0>=height-1)continue;
                    const fx=rx-x0,fy=ry-y0,i=y0*width+x0;
                    target[y*width+x]=Math.round((reference[i]*(1-fx)+reference[i+1]*fx)*(1-fy)
                      +(reference[i+width]*(1-fx)+reference[i+width+1]*fx)*fy);
                  }
                  const rgba=bytes=>{const data=new Uint8ClampedArray(bytes.length*4);for(let i=0;i<bytes.length;i++)data.set([bytes[i],bytes[i],bytes[i],255],i*4);return createImageBitmap(new ImageData(data,width,height));};
                  const referenceFrame=await rgba(reference),targetFrame=await rgba(target);
                  const engine=await DICEngine.create({width,height,motionModel:'affine',initialization:'global-affine',pyramidLevels:2,
                    iterationsPerLevel:[8,16],outlierPolicy:'disabled'});
                  try{
                    await engine.setReference(referenceFrame,{roiMask:mask});
                    const result=await engine.register(targetFrame,{sourceKind:'images'});
                    let cameraMessage='';try{await engine.register(targetFrame,{sourceKind:'camera'});}catch(error){cameraMessage=error.message;}
                    return {fit:result.config.globalAffineEstimate,accepted:result.accepted,finite:result.parameters.every(Number.isFinite),
                      timing:result.timing,cameraMessage};
                  }finally{engine.destroy();referenceFrame.close();targetFrame.close();}
                })()"""
            )
        self.assertEqual(actual["fit"]["levels"], 5)
        self.assertGreater(actual["fit"]["score"], 0.98)
        self.assertGreater(actual["accepted"], 0)
        self.assertTrue(actual["finite"])
        self.assertGreater(actual["timing"]["globalAffineWallMs"], 0)
        self.assertIn("loaded image or video file", actual["cameraMessage"])

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
