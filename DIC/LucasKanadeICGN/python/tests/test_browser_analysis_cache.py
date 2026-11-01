"""Bounded saved-analysis caching, independent of GPU and browser storage."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "web-app/ui/view-cache.js"


@unittest.skipUnless(shutil.which("node"), "Node is required for analysis cache contracts")
class BrowserAnalysisCacheTests(unittest.TestCase):
    def test_lru_limits_recipe_identity_and_raw_field_ownership(self):
        script = (
            f"import {{createAnalysisCache}} from {json.dumps(MODULE.as_uri())};"
            + """
          const result=()=>({field:{parameters:new Float64Array(10000)},
            fields:{u:new Float64Array(8),v:new Float64Array(8)},
            valid:new Uint8Array(8),strainValid:new Uint8Array(8),recipe:{window:5}});
          const cache=createAnalysisCache({maxBytes:300,maxEntries:2});
          const a=JSON.stringify([1,{window:5}]),b=JSON.stringify([2,{window:5}]),c=JSON.stringify([3,{window:5}]);
          cache.set(a,result());cache.set(b,result());const hit=cache.get(a);
          cache.set(c,result());const oldest=cache.get(b),recipeMiss=cache.get(JSON.stringify([1,{window:7}]));
          const bounded=cache.report();cache.clear();const cleared=cache.report();
          const tiny=createAnalysisCache({maxBytes:143,maxEntries:8});tiny.set(a,result());
          const one=createAnalysisCache({maxBytes:10000,maxEntries:1});one.set(a,result());one.set(b,result());
          console.log(JSON.stringify({rawRetained:'field' in hit,oldest,recipeMiss,bounded,cleared,
            oversized:tiny.report(),entryLimit:one.report(),oldEntry:one.get(a)}));
        """
        )
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertFalse(result["rawRetained"])
        self.assertIsNone(result["oldest"])
        self.assertIsNone(result["recipeMiss"])
        self.assertEqual(result["bounded"]["entries"], 2)
        self.assertEqual(result["bounded"]["bytes"], 288)
        self.assertEqual(result["bounded"]["hits"], 1)
        self.assertEqual(result["cleared"]["bytes"], 0)
        self.assertEqual(result["oversized"]["entries"], 0)
        self.assertEqual(result["entryLimit"]["entries"], 1)
        self.assertIsNone(result["oldEntry"])
