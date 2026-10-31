"""Dependency-free browser core contracts and opt-in Chromium execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from lucas_kanade_icgn.roi import prepare_roi, reduce_mask, seed_permissions

CORE = Path(__file__).resolve().parents[1] / "src/lucas_kanade_icgn/web/core"


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
                ["node", "--input-type=module", "-e", script], text=True
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
            subprocess.check_output(["node", "--input-type=module", "-e", script], text=True)
        )
        self.assertEqual(result, dict(concurrent=True, blocked=True, values=[0, 1], idle=True))


@unittest.skipUnless(
    os.environ.get("ICGN_RUN_BROWSER_TESTS") == "1", "set ICGN_RUN_BROWSER_TESTS=1"
)
class BrowserCoreExecutionTests(unittest.TestCase):
    def test_core_is_instantiable_without_page_globals_and_destroy_is_idempotent(self):
        from lucas_kanade_icgn import ICGNWebGL

        with ICGNWebGL(backend="webgpu-wgsl") as browser:
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

        with ICGNWebGL(backend="webgpu-wgsl") as browser:
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
