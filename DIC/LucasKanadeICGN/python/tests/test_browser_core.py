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
