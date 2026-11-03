"""Image sequence orchestration stays separate from the application shell."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "web-app/ui/image-sequence.js"


@unittest.skipUnless(shutil.which("node"), "Node is needed for image-sequence contracts")
class BrowserImageSequenceTests(unittest.TestCase):
    def test_pair_processing_and_seek_keep_frame_ownership_and_position(self):
        script = f"""
          import {{createImageSequenceController}} from {json.dumps(MODULE.as_uri())};
          const elements = new Map();
          const $ = id => {{
            if (!elements.has(id)) elements.set(id, {{value: id === 'comparison' ? 'consecutive' : '0', textContent: '', hidden: false, disabled: false}});
            return elements.get(id);
          }};
          let paused = true, sequence, decoded = [], processed = null, previewCount = 0, closed = 0, detailCount = 0;
          const video = {{currentTime:0,duration:1}};
          const context = () => ({{source:'images',generation:4,signal:null,referenceInfo:{{width:4,height:3}},sourceSettings:{{}},paused,video}});
          sequence = createImageSequenceController({{
            $,context,setPaused:value=>{{paused=value;}},
            pauseSource:()=>{{paused=true;sequence.wakePlayback();}},drain:async()=>{{}},
            decodeImage:async file=>{{decoded.push(file.name);return {{width:4,height:3,close:()=>closed++}};}},
            fileInfo:file=>({{name:file.name}}),
            processFrame:async (target,index,options)=>{{processed={{index,options}};target.close();options.referenceFrame?.close();return true;}},
            captureReference:async frame=>frame.close(),
            showInputPreview:frame=>{{previewCount++;frame.close();}},
            frameFromVideo:async()=>({{width:4,height:3,close:()=>closed++}}),seekVideo:async()=>{{}},
            details:()=>{{detailCount++;}},message:()=>{{}},
          }});
          sequence.setImages([{{name:'frame_0.png'}},{{name:'frame_1.png'}}]);
          const registered = await sequence.runImage(1,true,true);
          await sequence.localSeek(0);
          console.log(JSON.stringify({{
            registered,decoded,processedIndex:processed.index,force:processed.options.force,
            reset:processed.options.reset,playback:processed.options.playback,index:sequence.index,
            previewCount,closed,position:$('local-position').textContent,seek:$('local-seek').value,
            details:detailCount,
          }}));
        """
        result = json.loads(
            subprocess.check_output(
                ["node", "--input-type=module"], input=script, text=True, encoding="utf-8"
            )
        )

        self.assertTrue(result["registered"])
        self.assertEqual(result["decoded"], ["frame_1.png", "frame_0.png", "frame_0.png"])
        self.assertEqual(result["processedIndex"], 1)
        self.assertTrue(result["force"])
        self.assertTrue(result["reset"])
        self.assertTrue(result["playback"])
        self.assertEqual(result["index"], 0)
        self.assertEqual(result["previewCount"], 1)
        self.assertEqual(result["closed"], 3)
        self.assertEqual(result["position"], "1 / 2 · frame_0.png")
        self.assertEqual(result["seek"], 0)
        self.assertEqual(result["details"], 1)


if __name__ == "__main__":
    unittest.main()
