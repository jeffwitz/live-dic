"""Canvas-independent mask geometry and scientific display oracles."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np
from plate_hole_reference import displacement, strain, von_mises_strain

APP = Path(__file__).resolve().parents[2] / "web-app"


def javascript(script):
    return json.loads(
        subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
    )


@unittest.skipUnless(shutil.which("node"), "Node is needed for browser geometry contracts")
class BrowserRoiEditorTests(unittest.TestCase):
    def test_overlapping_outer_domain_does_not_fill_a_hole(self):
        shapes = [
            dict(id="a", type="rectangle", x=0, y=0, width=40, height=40),
            dict(id="b", type="rectangle", x=30, y=0, width=34, height=40),
            dict(id="h", type="circle", x=25.5, y=10.5, width=12, height=12),
        ]
        yy, xx = np.mgrid[:48, :64]
        expected = (yy < 40) & ((xx + 0.5 - 31.5) ** 2 + (yy + 0.5 - 16.5) ** 2 >= 36)
        actual = javascript(f"""
          import {{compileRoi}} from {json.dumps((APP / "ui/roi-geometry.js").as_uri())};
          console.log(JSON.stringify([...compileRoi({json.dumps(shapes)},64,48).mask]));
        """)
        np.testing.assert_array_equal(np.asarray(actual).reshape(48, 64), expected)

    def test_concave_parent_requires_the_whole_pixel_support(self):
        actual = javascript(f"""
          import {{compileRoi}} from {json.dumps((APP / "ui/roi-geometry.js").as_uri())};
          const shapes=[{{id:'l',type:'polygon',points:[[0,0],[40,0],[40,10],[10,10],[10,40],[0,40]]}},
            {{id:'r',type:'rectangle',x:5,y:5,width:20,height:20}}];
          console.log(JSON.stringify(compileRoi(shapes,48,48).shapes.map(s=>s.role)));
        """)
        self.assertEqual(actual, ["domain", "domain"])

    def test_multiple_domains_holes_islands_and_overlaps_are_order_independent(self):
        shapes = [
            dict(id="outer", type="rectangle", x=0, y=0, width=40, height=40),
            dict(id="second", type="rectangle", x=36, y=0, width=28, height=20),
            dict(id="hole", type="circle", x=12.5, y=12.5, width=16, height=16),
            dict(id="island", type="rectangle", x=18, y=18, width=5, height=5),
        ]
        yy, xx = np.mgrid[:48, :64]
        xx = xx + 0.5
        yy = yy + 0.5
        expected = (
            ((xx < 40) & (yy < 40) & ((xx - 20.5) ** 2 + (yy - 20.5) ** 2 >= 64))
            | ((xx >= 18) & (xx < 23) & (yy >= 18) & (yy < 23))
            | ((xx >= 36) & (xx < 64) & (yy < 20))
        )
        result = javascript(f"""
          import {{compileRoi}} from {json.dumps((APP / "ui/roi-geometry.js").as_uri())};
          const shapes={json.dumps(shapes)};
          console.log(JSON.stringify([shapes,[...shapes].reverse(),[...shapes,{{...shapes[0],id:'duplicate'}}]].map(s=>[...compileRoi(s,64,48).mask])));
        """)
        for mask in result:
            np.testing.assert_array_equal(np.array(mask).reshape(48, 64), expected)

    def test_ellipse_polygon_and_resize_use_pixel_centres(self):
        yy, xx = np.mgrid[:37, :59]
        xx = xx + 0.5
        yy = yy + 0.5
        expected = ((xx - 22.25) / 16) ** 2 + ((yy - 17.75) / 10) ** 2 < 1
        result = javascript(f"""
          import {{compileRoi,transformShape}} from {json.dumps((APP / "ui/roi-geometry.js").as_uri())};
          const ellipse={{id:'e',type:'ellipse',x:6.25,y:7.75,width:32,height:20}};
          const polygon={{id:'p',type:'polygon',points:[[0,0],[10,0],[10,10],[0,10]]}};
          const moved=transformShape(polygon,[12,4,20,8]);
          console.log(JSON.stringify({{ellipse:[...compileRoi([ellipse],59,37).mask],polygon:moved.points,original:polygon.points}}));
        """)
        np.testing.assert_array_equal(np.array(result["ellipse"]).reshape(37, 59), expected)
        self.assertEqual(result["polygon"], [[12, 4], [32, 4], [32, 12], [12, 12]])
        self.assertEqual(result["original"], [[0, 0], [10, 0], [10, 10], [0, 10]])

    def test_native_uhd_plate_preset_matches_the_original_mask_exactly(self):
        yy, xx = np.ogrid[:2160, :3840]
        expected = np.asarray((xx - 1920) ** 2 + (yy - 1080) ** 2 >= 112**2, dtype=np.uint8)
        actual = javascript(f"""
          import {{compileRoi,platePreset}} from {json.dumps((APP / "ui/roi-geometry.js").as_uri())};
          import {{createHash}} from 'node:crypto';
          const result=compileRoi(platePreset(),3840,2160);
          console.log(JSON.stringify({{hash:createHash('sha256').update(result.mask).digest('hex'),roles:result.shapes.map(s=>s.role)}}));
        """)
        self.assertEqual(actual["hash"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(actual["roles"], ["domain", "hole"])

    def test_analytic_plate_and_equivalent_strain_match_the_numpy_oracle(self):
        points = np.array([[350, 200], [1880, 920], [2100, 1100], [2500, 1850], [1920, 952]], float)
        actual = javascript(f"""
          import {{plateTruth}} from {json.dumps((APP / "ui/plate-truth.js").as_uri())};
          import {{equivalentStrain}} from {json.dumps((APP / "render/field.js").as_uri())};
          console.log(JSON.stringify({json.dumps(points.tolist())}.map(([x,y])=>{{const t=plateTruth(x,y,250,80);return {{...t,display:equivalentStrain(...t.strain)}};}})));
        """)
        np.testing.assert_allclose(
            [r["displacement"] for r in actual],
            displacement(points[:, 0], points[:, 1], 250, 80),
            atol=2e-13,
            rtol=1e-12,
        )
        expected = strain(points[:, 0], points[:, 1], 250, 80)
        np.testing.assert_allclose([r["strain"] for r in actual], expected, atol=2e-15, rtol=1e-12)
        np.testing.assert_allclose(
            [r["display"] for r in actual], von_mises_strain(expected), atol=2e-15, rtol=1e-12
        )
