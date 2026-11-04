"""Presentation geometry and cached review values, independently of GPU timing."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "web-app"


@unittest.skipUnless(shutil.which("node"), "Node is needed for presentation contracts")
class LiveVectorTests(unittest.TestCase):
    def test_saved_vectors_preserve_canvas_centres_and_physical_unit_conversion(self):
        script = f"""
          import {{renderFieldRaster}} from {json.dumps((APP / "ui/field-rendering.js").as_uri())};
          const paths=[];
          globalThis.Path2D=class{{constructor(){{this.points=[];paths.push(this);}}moveTo(x,y){{this.points.push(['M',x,y]);}}lineTo(x,y){{this.points.push(['L',x,y]);}}}};
          const context={{createImageData:(w,h)=>({{data:new Uint8ClampedArray(w*h*4)}}),
            putImageData(){{}},fillRect(){{}},drawImage(){{}},stroke(){{}}}};
          globalThis.OffscreenCanvas=class{{getContext(){{return context;}}}};
          const canvas={{width:64,height:64,getContext:()=>context}};
          const analysis={{field:{{width:64,height:64,grid:[1,1],subsetSize:16,subsetStride:16}},
            valid:new Uint8Array([1]),fields:{{u:new Float32Array([20]),v:new Float32Array([0])}},recipe:{{unitsPerPixel:2}}}};
          renderFieldRaster({{canvas,overlay:{{width:64,height:64}},analysis,values:[1],valid:[1],
            component:'u',low:0,high:2,palette:'sequential',showVectors:true,vectorStep:1,vectorGain:3}});
          console.log(JSON.stringify(paths[0].points.slice(0,2)));
        """
        # Canvas pixels occupy unit squares: a 16-pixel support is centred at 8.
        self.assertEqual(self.node(script), [["M", 8, 8], ["L", 38, 8]])

    def test_canvas_resize_reconfigures_webgpu_before_texture_acquisition(self):
        script = f"""
          import {{resizeAndConfigureCanvas}} from {json.dumps((APP / "render/canvas-context.js").as_uri())};
          let configured=false,configureCalls=0;
          const canvas={{_width:300,_height:150}};
          Object.defineProperties(canvas,{{
            width:{{get(){{return this._width;}},set(value){{this._width=value;configured=false;}}}},
            height:{{get(){{return this._height;}},set(value){{this._height=value;configured=false;}}}},
          }});
          const context={{configure(){{configured=true;configureCalls++;}},getCurrentTexture(){{
            if(!configured)throw new Error('Canvas not configured');return {{}};
          }}}};
          const device={{}},size={{width:canvas.width,height:canvas.height}};
          context.configure();
          resizeAndConfigureCanvas(canvas,context,device,'bgra8unorm',640,480,size);
          context.getCurrentTexture();
          const firstCalls=configureCalls;
          resizeAndConfigureCanvas(canvas,context,device,'bgra8unorm',640,480,size);
          const stableCalls=configureCalls;
          canvas.width=200;
          resizeAndConfigureCanvas(canvas,context,device,'bgra8unorm',640,480,size);
          context.getCurrentTexture();
          console.log(JSON.stringify({{firstCalls,stableCalls,finalCalls:configureCalls,configured}}));
        """
        value = self.node(script)
        self.assertEqual(
            value, {"firstCalls": 2, "stableCalls": 2, "finalCalls": 3, "configured": True}
        )

    def test_norm_scale_and_preview_do_not_change_field_geometry(self):
        script = f"""
          import {{vectorAppearance,vectorColour}} from {json.dumps((APP / "render/live-vectors.js").as_uri())};
          const result={{width:3840,height:2160,grid:[240,135],subsetSize:16,subsetStride:16,maxAbs:[3,4,5]}};
          const automatic=vectorAppearance(result);
          const before=JSON.stringify(result),a=vectorAppearance(result,{{previewMaxWidth:1920,arrowStep:4}});
          const b=vectorAppearance(result,{{scale:10,arrowGain:3,arrowStep:8}});
          const invalid=[{{arrowStep:0}},{{scale:0}},{{arrowGain:-1}},{{vectorColourMap:'unknown'}}].map(options=>{{try{{vectorAppearance(result,options);return false;}}catch{{return true;}}}});
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
        self.assertEqual(value["automatic"]["vectorColourMap"], "original")
        self.assertEqual(value["colours"], [[21, 57, 99], [31, 132, 138], [241, 206, 95]])
        self.assertTrue(value["unchanged"])
        self.assertEqual(value["invalid"], [True] * 4)

    def test_named_vector_colours_reuse_the_legend_tables_and_reject_invalid_names(self):
        script = f"""
          import {{vectorColour,vectorMapColour,vectorGradient,VECTOR_COLOUR_MAPS}} from {json.dumps((APP / "render/vector-colours.js").as_uri())};
          const legacy=[0,.5,1].map(vectorColour);
          const invalid=['__proto__','constructor','unknown',null,2].map(name=>{{try{{vectorMapColour(name,.5);return false;}}catch{{return true;}}}});
          const maps=Object.fromEntries(VECTOR_COLOUR_MAPS.map(name=>[name,[0,.5,1].map(value=>vectorMapColour(name,value))]));
          const gradient=vectorGradient('viridis');
          console.log(JSON.stringify({{legacy,invalid,maps,gradient,stable:gradient===vectorGradient('viridis')}}));
        """
        value = self.node(script)
        self.assertEqual(value["legacy"], [[21, 57, 99], [31, 132, 138], [241, 206, 95]])
        self.assertEqual(value["maps"]["original"], value["legacy"])
        self.assertEqual(value["maps"]["viridis"], [[68, 1, 84], [33, 145, 140], [253, 231, 37]])
        self.assertEqual(value["invalid"], [True] * 5)
        self.assertTrue(value["stable"])
        self.assertEqual(value["gradient"].count("rgb("), 256)

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
