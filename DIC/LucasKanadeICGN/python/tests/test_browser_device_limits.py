"""Real geometry negotiates WebGPU limits without requesting every adapter maximum."""

import json
import shutil
import subprocess
import unittest
from importlib.resources import files

MODULE = files("lucas_kanade_icgn").joinpath("web/core/device.js")


@unittest.skipUnless(shutil.which("node"), "Node is required for WebGPU limit contracts")
class BrowserDeviceLimitTests(unittest.TestCase):
    def test_geometry_requests_only_limits_above_portable_defaults(self):
        script = (
            f"import {{acquireDevice,imageRequirements}} from {json.dumps(MODULE.as_uri())};"
            + """
          const MiB=1024**2,requests=[],limits={maxStorageBufferBindingSize:1024*MiB,maxBufferSize:1024*MiB,maxTextureDimension2D:16384};
          const adapter={limits,requestDevice:async request=>{requests.push(request.requiredLimits);return {limits};}};
          Object.defineProperty(globalThis,'navigator',{value:{gpu:{requestAdapter:async()=>adapter}},configurable:true});
          for(const options of [{width:3840,height:2160},{width:4112,height:3008},
            {width:6000,height:4000},{width:9000,height:64},{width:1024,height:1024,subsetStride:1}])await acquireDevice(options);
          const owned=await acquireDevice({device:{limits},width:4112,height:3008});
          const before=requests.length;adapter.limits={...limits,maxStorageBufferBindingSize:128*MiB};let refused;
          try{await acquireDevice({width:4112,height:3008});}catch(e){refused=e.message;}
          console.log(JSON.stringify({requests,refused,early:before===requests.length,owns:owned.ownsDevice,
            metadata:imageRequirements({width:1024,height:1024,subsetStride:1}).largestResource}));
        """
        )
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(result["requests"][0], {"maxStorageBuffersPerShaderStage": 8})
        self.assertEqual(
            result["requests"][1],
            {"maxStorageBuffersPerShaderStage": 8, "maxStorageBufferBindingSize": 4112 * 3008 * 16},
        )
        self.assertEqual(result["requests"][2]["maxBufferSize"], 6000 * 4000 * 16)
        self.assertEqual(result["requests"][3]["maxTextureDimension2D"], 9000)
        self.assertEqual(result["requests"][4]["maxStorageBufferBindingSize"], 1009**2 * 52 * 4)
        self.assertTrue(result["early"])
        self.assertFalse(result["owns"])
        self.assertIn("GPU adapter maxStorageBufferBindingSize", result["refused"])
        self.assertIn("No resizing", result["refused"])
        self.assertEqual(result["metadata"], "subset metadata and inverse Hessians")
