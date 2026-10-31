"""Check that offered UI combinations satisfy the actual core option contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("node"), "Node is needed for solver-control contracts")
class SolverControlTests(unittest.TestCase):
    def test_preset_is_unchanged_and_custom_combinations_reach_the_core(self):
        ui = ROOT / "web-app/ui/workspace.js"
        core = ROOT / "python/src/lucas_kanade_icgn/web/core/engine.js"
        script = f"""
          import {{solverOptions}} from {json.dumps(ui.as_uri())};
          import {{DICEngine}} from {json.dumps(core.as_uri())};
          const values={{profile:{{value:'portable'}},'custom-solver':{{checked:false}},'subset-size':{{value:'16'}},'subset-stride':{{value:'16'}},'pyramid-levels':{{value:'2'}},'coarse-iterations':{{value:'8'}},'fine-iterations':{{value:'16'}},'temporal-start':{{checked:true}},'motion-model':{{value:'affine'}},interpolation:{{value:'bspline'}},'outlier-policy':{{value:'median'}}}};
          globalThis.document={{getElementById:id=>values[id]}};const preset=solverOptions();values['custom-solver'].checked=true;
          let count=0;const errors=[];
          for(const model of ['translation','affine'])for(const level of [1,2,3])for(const mode of ['bspline','linear','cubic'])for(const policy of ['median','replace','flag','disabled'])for(const temporal of [false,true]){{
            values['motion-model'].value=model;values['pyramid-levels'].value=String(level);values.interpolation.value=mode;values['outlier-policy'].value=policy;values['temporal-start'].checked=temporal;
            try{{const {{profile,...options}}=solverOptions();const engine=new DICEngine({{...options,width:3840,height:2160}});engine.destroy();count++;}}catch(e){{errors.push(e.message);}}
          }}console.log(JSON.stringify({{preset,count,errors}}));
        """
        result = json.loads(
            subprocess.check_output(
                ["node", "--input-type=module"], input=script, text=True, encoding="utf-8"
            )
        )
        self.assertEqual(result["preset"], {"profile": "portable"})
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["count"], 144)
