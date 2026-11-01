"""Presentation geometry and cached review values, independently of GPU timing."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "web-app"


@unittest.skipUnless(shutil.which("node"), "Node is needed for presentation contracts")
class LiveVectorTests(unittest.TestCase):
    def test_norm_scale_and_preview_do_not_change_field_geometry(self):
        script = f"""
          import {{vectorAppearance,vectorColour}} from {json.dumps((APP / "render/live-vectors.js").as_uri())};
          const result={{width:3840,height:2160,grid:[240,135],subsetSize:16,subsetStride:16,maxAbs:[3,4,5]}};
          const automatic=vectorAppearance(result);
          const before=JSON.stringify(result),a=vectorAppearance(result,{{previewMaxWidth:1920,arrowStep:4}});
          const b=vectorAppearance(result,{{scale:10,arrowGain:3,arrowStep:8}});
          const invalid=[{{arrowStep:0}},{{scale:0}},{{arrowGain:-1}}].map(options=>{{try{{vectorAppearance(result,options);return false;}}catch{{return true;}}}});
          console.log(JSON.stringify({{a,b,automatic,invalid,unchanged:JSON.stringify(result)===before,
            colours:[0,.5,1].map(vectorColour)}}));
        """
        value = self.node(script)
        self.assertEqual([value["a"]["viewWidth"], value["a"]["viewHeight"]], [1920, 1080])
        self.assertEqual(value["a"]["instances"], 60 * 34)
        self.assertEqual(value["a"]["maximumPx"], 5)
        self.assertAlmostEqual(value["a"]["gain"], 5.12)
        self.assertEqual(value["b"]["maximumPx"], 10)
        self.assertEqual(value["b"]["gain"], 3)
        self.assertEqual(value["automatic"]["arrowStep"], 12)
        self.assertEqual(value["automatic"]["instances"], 240)
        self.assertEqual(value["colours"], [[21, 57, 99], [31, 132, 138], [241, 206, 95]])
        self.assertTrue(value["unchanged"])
        self.assertEqual(value["invalid"], [True] * 3)

    def test_review_cache_reuses_arrays_but_not_a_changed_analysis(self):
        script = f"""
          import {{createViewCache}} from {json.dumps((APP / "ui/view-cache.js").as_uri())};
          let derives=0,summaries=0;
          const cache=createViewCache((a,c)=>{{derives++;return {{values:new Float32Array([a.value]),valid:new Uint8Array([1]),unit:c}};}},
            values=>{{summaries++;return {{mean:values[0]}};}});
          const analysis={{value:2}},first=cache.get(analysis,'u');
          const same=cache.get(analysis,'u'),other=cache.get(analysis,'v'),changed=cache.get({{value:3}},'u');
          console.log(JSON.stringify({{same:first===same&&first.values===same.values,other:first!==other,
            changed:changed.statistics.mean,derives,summaries,report:cache.report()}}));
        """
        value = self.node(script)
        self.assertTrue(value["same"] and value["other"])
        self.assertEqual(value["changed"], 3)
        self.assertEqual(value["derives"], 3)
        self.assertEqual(value["summaries"], 3)
        self.assertEqual(value["report"], {"computations": 3, "hits": 1})

    @staticmethod
    def node(script):
        return json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
