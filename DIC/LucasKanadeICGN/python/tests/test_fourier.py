"""Global rigid seeds, independently refined by the unchanged GPU solver."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from importlib.resources import files

import numpy as np

from live_dic import ICGNBrowser, ICGNWGPUFullPyramid
from live_dic.fourier import FourierInitializer


def texture(size=256):
    return np.random.default_rng(481).integers(0, 256, (size, size), dtype=np.uint8)


def shift(image, dx, dy):
    h, w = image.shape
    result = np.zeros_like(image)
    result[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = image[
        max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
    ]
    return result


class FourierTests(unittest.TestCase):
    def test_signed_linear_shifts_cached_reference_and_window_ownership(self):
        image = texture()
        estimator = FourierInitializer(image)
        spectrum = estimator.spectrum.copy()
        for dx, dy in [(64, -32), (-64, 32), (0, 0), (32, 64)]:
            actual = estimator.estimate(shift(image, dx, dy))
            self.assertEqual((actual["dx"], actual["dy"]), (dx, dy))
            self.assertGreater(actual["peakScore"], 8)
            json.dumps(actual, allow_nan=False)
            actual["window"]["x"] = 100
            self.assertEqual(estimator.window["x"], 0)
        np.testing.assert_array_equal(estimator.spectrum, spectrum)

    def test_no_texture_unrelated_target_and_unavailable_roi_are_rejected(self):
        image = texture()
        estimator = FourierInitializer(image)
        for target in [
            np.zeros_like(image),
            np.random.default_rng(72).integers(0, 256, image.shape, dtype=np.uint8),
        ]:
            with self.assertRaisesRegex(ValueError, "texture|ambiguous"):
                estimator.estimate(target)
        for reference, mask in [
            (np.zeros_like(image), None),
            (image, np.zeros_like(image, dtype=bool)),
        ]:
            with self.assertRaisesRegex(ValueError, "textured"):
                FourierInitializer(reference, mask)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            estimator.estimate(image[:-1])

    def test_masked_reference_is_owned_and_not_applied_to_moved_target(self):
        image = texture(512)
        mask = np.ones_like(image, dtype=bool)
        mask[:80] = False
        estimator = FourierInitializer(image, mask)
        mask[:] = False
        actual = estimator.estimate(shift(image, 64, 32))
        self.assertEqual((actual["dx"], actual["dy"]), (64, 32))

    def test_public_modes_validate_before_acquiring_gpu_or_browser(self):
        for backend in ["three-tsl"]:
            with self.assertRaisesRegex(ValueError, "webgpu-wgsl"):
                ICGNBrowser(backend=backend, initialization="fourier")
        with self.assertRaisesRegex(ValueError, "translation and affine"):
            ICGNWGPUFullPyramid(initialization="fourier", motion_model="quadratic")
        browser = ICGNBrowser(backend="webgpu-wgsl", initialization="fourier")
        self.assertEqual(
            browser._options(iterations=None, tolerance=None, zncc_min=None)["initialization"],
            "fourier",
        )

    @unittest.skipUnless(shutil.which("node"), "Node is required")
    def test_javascript_matches_numpy_phase_correlation(self):
        image = texture()
        target = shift(image, 64, -32)
        expected = FourierInitializer(image).estimate(target)
        module = files("live_dic").joinpath("web/core/fourier.js")
        script = f"""import {{FourierInitializer}} from {json.dumps(module.as_uri())};
        const a=Uint8Array.from({json.dumps(image.ravel().tolist())}),b=Uint8Array.from({json.dumps(target.ravel().tolist())});
        const f=new FourierInitializer(a,256,256),estimate=f.estimate(b);let rejected=false;
        try{{f.estimate(new Uint8Array(a.length));}}catch{{rejected=true;}}
        console.log(JSON.stringify({{estimate,rejected,bytes:f.memoryBytes()}}));"""
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual((actual["estimate"]["dx"], actual["estimate"]["dy"]), (64, -32))
        self.assertAlmostEqual(actual["estimate"]["peakScore"], expected["peakScore"], places=6)
        self.assertTrue(actual["rejected"])
        self.assertGreater(actual["bytes"], 4 * 256 * 256 * 8)


@unittest.skipUnless(os.environ.get("ICGN_RUN_GPU_TESTS") == "1", "set ICGN_RUN_GPU_TESTS=1")
class FourierGpuTests(unittest.TestCase):
    def test_empty_roi_skips_fourier_and_returns_no_measurements(self):
        image = texture()
        with ICGNWGPUFullPyramid(initialization="fourier") as engine:
            engine.set_reference(image, roi_mask=np.zeros_like(image, dtype=bool))
            self.assertIsNone(engine._fourier)
            result = engine.register(np.zeros_like(image))
            self.assertEqual(result.accepted, 0)
            self.assertNotIn("globalTranslation", result.config)
            json.dumps(result.config, allow_nan=False)

    def test_ordered_stream_keeps_seeds_and_delivers_pending_before_rejection(self):
        image = texture(512)
        with ICGNWGPUFullPyramid(
            initialization="fourier", motion_model="translation", outlier_policy="disabled"
        ) as engine:
            engine.set_reference(image)
            targets = [shift(image, 64, -32), shift(image, -32, 16), np.zeros_like(image)]
            stream = engine.register_stream(targets, pipeline_depth=2)
            for truth in [(64, -32), (-32, 16)]:
                result = next(stream)
                self.assertEqual(
                    (
                        result.config["globalTranslation"]["dx"],
                        result.config["globalTranslation"]["dy"],
                    ),
                    truth,
                )
                gy, gx = np.indices(result.status.shape)
                valid = result.accepted_mask & (gx > 6) & (gx < 25) & (gy > 6) & (gy < 25)
                self.assertGreater(np.count_nonzero(valid), 250)
                np.testing.assert_allclose(
                    result.displacement[valid],
                    np.broadcast_to(truth, result.displacement[valid].shape),
                    atol=0.01,
                )
            with self.assertRaisesRegex(Exception, "texture"):
                next(stream)
            recovered = engine.register(image)
            self.assertEqual(recovered.config["globalTranslation"]["dx"], 0)


@unittest.skipUnless(
    os.environ.get("ICGN_RUN_BROWSER_TESTS") == "1", "set ICGN_RUN_BROWSER_TESTS=1"
)
class FourierBrowserTests(unittest.TestCase):
    def test_core_bytes_external_gpu_preparation_and_pairs(self):
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
        with ICGNBrowser(backend="webgpu-wgsl", chromium_args=flags, call_timeout=240) as browser:
            actual = browser._evaluate("""(async()=>{
              const {DICEngine}=await import('../core/engine.js');
              const n=512,a=new Uint8Array(n*n),b=new Uint8Array(n*n);let seed=1;
              for(let i=0;i<a.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;a[i]=seed>>>24;}
              for(let y=32;y<n;y++)for(let x=0;x<n-64;x++)b[(y-32)*n+x+64]=a[y*n+x];
              const bitmap=async bytes=>{const canvas=new OffscreenCanvas(n,n),ctx=canvas.getContext('2d'),im=ctx.createImageData(n,n);
                for(let i=0;i<bytes.length;i++){im.data[i*4]=im.data[i*4+1]=im.data[i*4+2]=bytes[i];im.data[i*4+3]=255;}ctx.putImageData(im,0,0);return createImageBitmap(canvas);};
              const ab=await bitmap(a),bb=await bitmap(b),records=[];
              try{for(const preparation of ['cpu','gpu']){
                const engine=await DICEngine.create({width:n,height:n,initialization:'fourier',referencePreparation:preparation,outlierPolicy:'disabled'});
                try{await engine.setReference(ab);for(const current of [b,bb]){
                  const result=await engine.register(current);if(result.config.globalTranslation.dx!==64||result.config.globalTranslation.dy!==-32)throw new Error('seed mismatch');
                  let count=0,max=0;for(let y=7;y<25;y++)for(let x=7;x<25;x++){
                    const k=y*32+x;if(result.status[k]>.5){count++;max=Math.max(max,Math.abs(result.parameters[k*6]-64),Math.abs(result.parameters[k*6+3]+32));}}
                  if(count<300||max>.01)throw new Error(`field ${preparation}: ${count} ${max}`);
                  records.push({preparation,count,max,seed:result.config.globalTranslation,memory:engine.memoryUsage()});
                }
                if(preparation==='gpu'){const pair=await engine.registerPair(ab,bb);if(pair.config.globalTranslation.dx!==64)throw new Error('pair seed');}
                let rejected=false;try{await engine.register(new Uint8Array(a.length));}catch(e){rejected=e.message.includes('texture');}
                if(!rejected)throw new Error('flat target accepted');const saved=await engine.exportField();if(saved.config.globalTranslation.dx!==64)throw new Error('failed target erased export seed');const recovered=await engine.register(a);
                if(recovered.config.globalTranslation.dx!==0)throw new Error('stale seed');
                }finally{engine.destroy();}
              }}finally{ab.close();bb.close();}
              const engine=await DICEngine.create({width:n,height:n,initialization:'fourier',pipelineDepth:2,motionModel:'translation',outlierPolicy:'disabled'});
              try{await engine.setReference(a);const [shifted,identity]=await Promise.all([engine.register(b),engine.register(a)]);
                if(shifted.config.globalTranslation.dx!==64||identity.config.globalTranslation.dx!==0)throw new Error('ring seed metadata mixed');
              }finally{engine.destroy();}
              return records;
            })()""")
        self.assertEqual(len(actual), 4)
        self.assertTrue(all(row["memory"]["cpuInitializationBytes"] > 0 for row in actual))
