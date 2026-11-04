"""Dependency-free browser core contracts and opt-in Chromium execution."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import unittest
from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn.roi import prepare_roi, reduce_mask, seed_permissions

CORE = Path(str(files("lucas_kanade_icgn").joinpath("web/core")))
UHD_TENSILE = Path(__file__).resolve().parents[2] / "assets" / "uhd_tensile"


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
          const cases=[{{subsetSize:'16'}},{{subsetSize:3}},{{subsetSize:65}},{{pyramidLevels:true}},{{initialization:'temporal',pyramidLevels:3,temporalFineCorrection:true}},
            {{spatialWeighting:'gaussian'}},{{roiPolicy:'partial'}},{{solverWorkgroupSize:128}},{{unknownOption:1}},{{roiPolicy:0}},{{readback:null}},{{spatialWeighting:null}},{{initialization:null}},{{pipelineDepth:null}}];
          const rejected=cases.map(options=>{{try{{new DICEngine(options);return false;}}catch{{return true;}}}});
          let acceptsSix=false;try{{new DICEngine({{initialization:'global-affine',globalAffineLevels:6}});acceptsSix=true;}}catch{{}}
          let acceptsTranslationGlobalAffine=false;try{{new DICEngine({{initialization:'global-affine',globalAffineLevels:6,motionModel:'translation'}});acceptsTranslationGlobalAffine=true;}}catch{{}}
          const invalidScales=[4,9,5.5].map(value=>{{try{{new DICEngine({{initialization:'global-affine',globalAffineLevels:value}});return false;}}catch{{return true;}}}});
          let rejectsWithoutInitializer=false;try{{new DICEngine({{globalAffineLevels:6}});}}catch{{rejectsWithoutInitializer=true;}}
          const wide=new DICEngine({{solverWorkgroupSize:32}});
          const width=wide.host.configuration.solverWorkgroupSize;let widthFixed=false;
          try{{wide.host.configure({{solverWorkgroupSize:16}});}}catch{{widthFixed=wide.host.configuration.solverWorkgroupSize===32;}}wide.destroy();
          const acceptsSubsetSizes=[4,7,24,64].map(subsetSize=>{{try{{new DICEngine({{subsetSize}});return true;}}catch{{return false;}}}});
          console.log(JSON.stringify({{rejected,acceptsSix,acceptsTranslationGlobalAffine,invalidScales,rejectsWithoutInitializer,acceptsSubsetSizes,width,widthFixed}}));
        """
        results = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertTrue(all(results["rejected"]))
        self.assertTrue(results["acceptsSix"])
        self.assertTrue(results["acceptsTranslationGlobalAffine"])
        self.assertTrue(all(results["invalidScales"]))
        self.assertTrue(results["rejectsWithoutInitializer"])
        self.assertTrue(all(results["acceptsSubsetSizes"]))
        self.assertEqual(results["width"], 32)
        self.assertTrue(results["widthFixed"])


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
                  const run=async motionModel=>{
                    const referenceFrame=await rgba(reference),targetFrame=await rgba(target);
                    const engine=await DICEngine.create({width,height,motionModel,initialization:'global-affine',pyramidLevels:2,
                      iterationsPerLevel:[8,16],outlierPolicy:'disabled'});
                    try{
                      await engine.setReference(referenceFrame,
                        motionModel==='quadratic'?{}:{roiMask:mask});
                      const result=await engine.register(targetFrame,{sourceKind:'images'});
                      let cameraMessage='';
                      if(motionModel==='affine'){
                        const cameraFrame=await rgba(target);
                        try{await engine.register(cameraFrame,{sourceKind:'camera'});}catch(error){cameraMessage=error.message;}
                        finally{cameraFrame.close();}
                      }
                      let truthRmse=null,maxLocalGradient=null,maxCurvature=null;
                      if(motionModel==='translation'||motionModel==='quadratic'){
                        let squaredError=0,count=0;maxLocalGradient=0;
                        if(motionModel==='quadratic')maxCurvature=0;
                        const parameterStride=motionModel==='quadratic'?12:6;
                        for(let index=0;index<result.status.length;index++){
                          const offset=index*parameterStride;
                          maxLocalGradient=Math.max(maxLocalGradient,Math.abs(result.parameters[offset+1]),
                            Math.abs(result.parameters[offset+2]),Math.abs(result.parameters[offset+4]),
                            Math.abs(result.parameters[offset+5]));
                          if(result.status[index]<.5)continue;
                          if(motionModel==='quadratic')for(let k=6;k<12;k++)
                            maxCurvature=Math.max(maxCurvature,Math.abs(result.parameters[offset+k]));
                          const gx=index%result.grid[0],gy=Math.floor(index/result.grid[0]);
                          const x=gx*16+8,y=gy*16+8,X=x-cx,Y=y-cy;
                          const expectedU=dx+(a-1)*X+b*Y,expectedV=dy+c*X+(d-1)*Y;
                          squaredError+=(result.parameters[offset]-expectedU)**2
                            +(result.parameters[offset+3]-expectedV)**2;count++;
                        }
                        truthRmse=Math.sqrt(squaredError/(2*count));
                      }
                      return {model:result.config.motionModel,fit:result.config.globalAffineEstimate,accepted:result.accepted,
                        finite:result.parameters.every(Number.isFinite),timing:result.timing,cameraMessage,truthRmse,maxLocalGradient,maxCurvature};
                    }finally{engine.destroy();referenceFrame.close();targetFrame.close();}
                  };
                  return {affine:await run('affine'),translation:await run('translation'),quadratic:await run('quadratic')};
                })()"""
            )
        for model in ("affine", "translation", "quadratic"):
            with self.subTest(model=model):
                result = actual[model]
                self.assertEqual(result["model"], model)
                self.assertEqual(result["fit"]["levels"], 5)
                self.assertGreater(result["fit"]["score"], 0.98)
                self.assertGreater(result["accepted"], 0)
                self.assertTrue(result["finite"])
                self.assertGreater(result["timing"]["globalAffineWallMs"], 0)
        self.assertIn("loaded image or video file", actual["affine"]["cameraMessage"])
        self.assertLess(actual["translation"]["truthRmse"], 0.15)
        self.assertLess(actual["translation"]["maxLocalGradient"], 1e-8)
        self.assertGreater(actual["quadratic"]["accepted"], 1000)
        self.assertLess(actual["quadratic"]["truthRmse"], 0.15)
        self.assertLess(actual["quadratic"]["maxCurvature"], 0.02)

    def test_global_affine_initialization_seeds_uhd_translation_solver(self):
        from lucas_kanade_icgn import ICGNWebGL

        with Image.open(UHD_TENSILE / "reference.png") as image:
            reference = np.asarray(image, dtype=np.uint8)
        with Image.open(UHD_TENSILE / "frame_040.png") as image:
            target = np.asarray(image, dtype=np.uint8)
        with (UHD_TENSILE / "machine_truth.csv").open(newline="", encoding="utf-8") as stream:
            truth = next(row for row in csv.DictReader(stream) if int(row["frame"]) == 40)
        stretch_x = float(truth["stretch_x"])
        stretch_y = float(truth["stretch_y"])

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
        with ICGNWebGL(
            backend="webgpu-wgsl",
            motion_model="translation",
            initialization="global-affine",
            global_affine_levels=5,
            pyramid_levels=2,
            iterations_per_level=(8, 16),
            interpolation="bspline",
            outlier_policy="disabled",
            subset_size=16,
            subset_stride=16,
            chromium_args=flags,
            call_timeout=300,
        ) as browser:
            reference_info = browser.set_reference(reference)
            result = browser.register(target)

        self.assertEqual((reference_info.width, reference_info.height), (3840, 2160))
        self.assertEqual(result.config["motionModel"], "translation")
        self.assertEqual(result.config["initialization"], "global-affine")
        self.assertTrue(np.isfinite(result.parameters).all())
        fit = result.config["globalAffineEstimate"]
        self.assertGreater(fit["score"], 0.98)
        self.assertAlmostEqual(fit["linear"][0], stretch_x - 1, delta=5e-4)
        self.assertAlmostEqual(fit["linear"][3], stretch_y - 1, delta=5e-4)

        grid_y, grid_x = np.indices(result.status.shape)
        x = grid_x * result.subset_stride + (result.subset_size - 1) / 2
        y = grid_y * result.subset_stride + (result.subset_size - 1) / 2
        expected = np.stack(
            (
                (stretch_x - 1) * (x - 1920),
                (stretch_y - 1) * (y - 1080),
            ),
            axis=-1,
        )
        interior = (x >= 32) & (x < 3840 - 32) & (y >= 32) & (y < 2160 - 32)
        accepted = interior & result.accepted_mask
        self.assertGreater(int(accepted.sum()), 0)
        error = np.linalg.norm(result.displacement[accepted] - expected[accepted], axis=-1)
        self.assertGreaterEqual(float(accepted.sum() / interior.sum()), 0.90)
        self.assertLess(float(np.sqrt(np.mean(error**2))), 0.05)
        self.assertLess(float(np.percentile(error, 95)), 0.10)
        self.assertLess(float(np.max(np.abs(result.parameters[..., (1, 2, 4, 5)]))), 1e-8)

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
