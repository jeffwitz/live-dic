"""Exact mask image conversion and immutable vector/base composition."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "web-app/ui/mask-image.js"


def javascript(script):
    return json.loads(
        subprocess.check_output(
            ["node", "--input-type=module"],
            input=f"import * as m from {json.dumps(MODULE.as_uri())};" + script,
            text=True,
            encoding="utf-8",
        )
    )


@unittest.skipUnless(shutil.which("node"), "Node is needed for mask image contracts")
class MaskImageTests(unittest.TestCase):
    def test_exact_black_white_pixels_without_threshold_or_aliasing(self):
        result = javascript("""
          const rgba=new Uint8ClampedArray([0,0,0,255,255,255,255,255,0,0,0,255,255,255,255,255]);
          const mask=m.maskFromPixels(rgba,2,2);rgba.fill(0);
          console.log(JSON.stringify([...mask]));
        """)
        self.assertEqual(result, [0, 1, 0, 1])

    def test_gray_colour_and_transparency_require_explicit_conversion(self):
        result = javascript("""
          console.log(JSON.stringify([[127,127,127,255],[255,0,0,255],[255,255,255,0]].map(pixel=>{
            try{m.maskFromPixels(new Uint8Array(pixel),1,1);return false;}catch(e){return e.message;}
          })));
        """)
        self.assertTrue(all("not opaque black/white" in value for value in result))

    def test_explicit_luminance_and_alpha_thresholds_are_not_implicit(self):
        result = javascript("""
          const rgba=new Uint8Array([127,127,127,255,128,128,128,255,255,255,255,127,255,255,255,128]);
          console.log(JSON.stringify([...m.maskFromPixels(rgba,4,1,{threshold:128})]));
        """)
        self.assertEqual(result, [0, 1, 0, 1])

    def test_image_and_vector_intersection_preserves_both_sources(self):
        result = javascript("""
          const image=new Uint8Array([1,1,0,1]),shapes=new Uint8Array([1,0,1,1]);
          const merged=m.intersectMasks(image,shapes),expected=[...merged],full=m.intersectMasks(image,null),vector=m.intersectMasks(null,shapes);
          merged.fill(0);full.fill(0);vector.fill(0);
          console.log(JSON.stringify({expected,image:[...image],shapes:[...shapes],all:m.intersectMasks(null,null)}));
        """)
        self.assertEqual(result["expected"], [1, 0, 0, 1])
        self.assertEqual(result["image"], [1, 1, 0, 1])
        self.assertEqual(result["shapes"], [1, 0, 1, 1])
        self.assertIsNone(result["all"])
