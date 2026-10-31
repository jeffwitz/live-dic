"""Local file admission and image ownership without a browser/GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "web-app/ui/local-media.js"


def javascript(script):
    return json.loads(
        subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
    )


@unittest.skipUnless(shutil.which("node"), "Node is needed for local-media contracts")
class LocalMediaTests(unittest.TestCase):
    def test_natural_relative_order_and_non_image_admission(self):
        result = javascript(f"""
          import {{imageFiles,fileInfo}} from {json.dumps(MODULE.as_uri())};
          const files=['frame_10.png','frame_2.png','README.md','frame_1.png'].map(name=>({{name,webkitRelativePath:'folder/'+name,size:42,lastModified:0,type:'image/png'}}));
          const input=files.map(f=>f.name),result=imageFiles(files);
          console.log(JSON.stringify({{names:result.images.map(f=>fileInfo(f).name),ignored:result.ignored,input:files.map(f=>f.name),frozen:Object.isFrozen(result.images)}}));
        """)
        self.assertEqual(
            result["names"], ["folder/frame_1.png", "folder/frame_2.png", "folder/frame_10.png"]
        )
        self.assertEqual(result["ignored"], ["folder/README.md"])
        self.assertEqual(
            result["input"], ["frame_10.png", "frame_2.png", "README.md", "frame_1.png"]
        )
        self.assertTrue(result["frozen"])

    def test_decode_cancellation_and_dimension_rejection_close_owned_images(self):
        result = javascript(f"""
          import {{decodeImage}} from {json.dumps(MODULE.as_uri())};
          let closed=0,resolve;const file={{name:'frame_4.png'}},abort=new AbortController();
          const pending=decodeImage(file,null,abort.signal,()=>new Promise(r=>resolve=r)).catch(e=>e.name);
          abort.abort();resolve({{width:64,height:64,close:()=>closed++}});const cancelled=await pending;
          const mismatch=await decodeImage(file,[128,64],null,async()=>({{width:64,height:64,close:()=>closed++}})).catch(e=>e.message);
          const good=await decodeImage(file,[64,64],null,async()=>({{width:64,height:64,close:()=>closed++}}));
          console.log(JSON.stringify({{cancelled,mismatch,closed,width:good.width}}));good.close();
        """)
        self.assertEqual(result["cancelled"], "AbortError")
        self.assertIn("frame_4.png is 64×64", result["mismatch"])
        self.assertIn("never resized", result["mismatch"])
        self.assertEqual(result["closed"], 2)
        self.assertEqual(result["width"], 64)

    def test_video_wait_cleans_up_on_abort_and_decoder_error(self):
        result = javascript(f"""
          import {{waitForMedia}} from {json.dumps(MODULE.as_uri())};
          const video=new EventTarget(),abort=new AbortController();
          const cancelled=waitForMedia(video,'loadeddata',abort.signal).catch(e=>e.name);abort.abort();
          const failed=waitForMedia(video,'seeked').catch(e=>e.message);video.error={{code:4}};video.dispatchEvent(new Event('error'));
          const ready=waitForMedia(video,'seeked');video.dispatchEvent(new Event('seeked'));await ready;
          console.log(JSON.stringify({{cancelled:await cancelled,failed:await failed}}));
        """)
        self.assertEqual(result["cancelled"], "AbortError")
        self.assertIn("media error 4", result["failed"])

    def test_empty_admission_and_decoder_failure_are_explicit(self):
        result = javascript(f"""
          import {{imageFiles,decodeImage}} from {json.dumps(MODULE.as_uri())};
          let empty;try{{imageFiles([{{name:'frame.tiff'}}]);}}catch(e){{empty=e.message;}}
          const failed=await decodeImage({{name:'broken.png'}},null,null,async()=>{{throw new Error('invalid pixels');}}).catch(e=>e.message);
          console.log(JSON.stringify({{empty,failed}}));
        """)
        self.assertIn("No supported image files", result["empty"])
        self.assertIn("Cannot decode broken.png", result["failed"])
