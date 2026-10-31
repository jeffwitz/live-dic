"""Independent analytic oracles for the optional browser results workbench."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[2] / "web-app/ui/postprocess.js"
IMPORT = f"import * as p from {json.dumps(MODULE.as_uri())};"
FIXTURE = """
function fixture(width=3840,height=2160,size=16,stride=16) {
  const grid=[1+Math.floor((width-size)/stride),1+Math.floor((height-size)/stride)],n=grid[0]*grid[1];
  return {width,height,subsetSize:size,subsetStride:stride,grid,config:{motionModel:'affine'},
    parameters:new Float64Array(n*6),status:new Float32Array(n).fill(1),iterations:new Float32Array(n).fill(4)};
}
"""


def javascript(script):
    return json.loads(
        subprocess.check_output(
            ["node", "--input-type=module"],
            input=IMPORT + FIXTURE + script,
            text=True,
            encoding="utf-8",
        )
    )


@unittest.skipUnless(shutil.which("node"), "Node is needed for browser analysis contracts")
class BrowserPostprocessTests(unittest.TestCase):
    def test_uhd_plane_fit_and_direct_strain_match_analytic_affine_field(self):
        result = javascript("""
          const f=fixture(),c=(f.subsetSize-1)/2;
          for(let i=0;i<f.status.length;i++){const x=i%f.grid[0]*16+c,y=Math.floor(i/f.grid[0])*16+c;
            f.parameters.set([.3+.001*x+.002*y,.001,.002,-.2-.003*x+.004*y,-.003,.004],i*6);}
          const direct=p.analyzeField(f),vsg=p.analyzeField(f,{strainSource:'vsg',window:9});
          let error=0;for(const k of ['exx','eyy','exy','vonMises','principal1','principal2'])for(let i=0;i<f.status.length;i++)error=Math.max(error,Math.abs(direct.fields[k][i]-vsg.fields[k][i]));
          console.log(JSON.stringify({count:f.status.length,error,valid:vsg.strainValid.reduce((a,b)=>a+b),sample:Object.fromEntries(Object.entries(vsg.fields).map(([k,v])=>[k,v[12345]]))}));
        """)
        self.assertEqual(result["count"], 32400)
        self.assertEqual(result["valid"], 32400)
        self.assertLess(result["error"], 1e-13)
        e = np.array([[0.001, -0.0005], [-0.0005, 0.004]])
        eigen = np.linalg.eigvalsh(e)
        self.assertAlmostEqual(result["sample"]["principal1"], eigen[1], places=13)
        self.assertAlmostEqual(result["sample"]["principal2"], eigen[0], places=13)

    def test_green_strain_and_polar_rotation_of_exact_rigid_rotation(self):
        theta = 0.37
        result = javascript(f"""
          const f=fixture(128,128),t={theta},c=Math.cos(t),s=Math.sin(t);
          for(let i=0;i<f.status.length;i++)f.parameters.set([2,c-1,-s,-3,s,c-1],i*6);
          const a=p.analyzeField(f,{{strainMeasure:'green'}});
          console.log(JSON.stringify({{strain:['exx','eyy','exy'].map(k=>a.fields[k][0]),rotation:a.fields.rotation[0]}}));
        """)
        np.testing.assert_allclose(result["strain"], 0, atol=1e-15)
        self.assertAlmostEqual(result["rotation"], theta, places=14)

    def test_smoothing_respects_thin_mask_crack_and_preserves_solver_arrays(self):
        result = javascript("""
          const f=fixture(128,128),mask=new Uint8Array(128*128).fill(1);
          for(let y=0;y<128;y++)mask[y*128+64]=0;
          for(let i=0;i<f.status.length;i++)f.parameters[i*6]=i%8<4?1:9;
          const bytes=Array.from(f.parameters),a=p.analyzeField(f,{smoothing:'median',strainSource:'vsg',window:9},mask);
          console.log(JSON.stringify({same:bytes.every((v,i)=>v===f.parameters[i]),left:a.fields.u[3],right:a.fields.u[4],exx:Array.from(a.fields.exx),valid:Array.from(a.strainValid)}));
        """)
        self.assertTrue(result["same"])
        self.assertEqual(result["left"], 1)
        self.assertEqual(result["right"], 9)
        np.testing.assert_allclose(result["exx"], 0, atol=1e-15)
        self.assertTrue(all(result["valid"]))

    def test_rank_deficient_support_does_not_invent_strain_or_fill_failed_cells(self):
        result = javascript("""
          const f=fixture(80,80);f.status.fill(-2);for(let i=10;i<15;i++){f.status[i]=1;f.parameters[i*6]=i;}
          const a=p.analyzeField(f,{strainSource:'vsg',window:5});
          console.log(JSON.stringify({displacements:a.valid.reduce((a,b)=>a+b),strains:a.strainValid.reduce((a,b)=>a+b),empty:p.statistics(a.fields.exx,a.strainValid),finite:Object.values(a.fields).every(v=>v.every(Number.isFinite))}));
        """)
        self.assertEqual(result["displacements"], 5)
        self.assertEqual(result["strains"], 0)
        self.assertEqual(result["empty"]["mean"], None)
        self.assertTrue(result["finite"])

    def test_gaussian_smoothing_matches_independent_numpy_weights(self):
        result = javascript("""
          const f=fixture(112,112);for(let i=0;i<49;i++)f.parameters[i*6]=Math.sin(i*.73);
          const a=p.analyzeField(f,{smoothing:'gaussian',strainSource:'vsg',window:5});
          console.log(JSON.stringify({u:a.fields.u[24]}));
        """)
        y, x = np.mgrid[-2:3, -2:3]
        weights = np.exp(-(x**2 + y**2) / 2)
        indices = (y + 3) * 7 + x + 3
        expected = np.sum(weights * np.sin(indices * 0.73)) / np.sum(weights)
        self.assertAlmostEqual(result["u"], expected, places=14)

    def test_extensometer_uses_snapped_valid_endpoints_and_physical_units(self):
        result = javascript("""
          const f=fixture(128,128);for(let i=0;i<64;i++)f.parameters[i*6]=.01*(i%8*16+7.5);
          const a=p.analyzeField(f,{unit:'mm',unitsPerPixel:.05}),g=p.extensometer(a,[7.5,7.5],[103.5,7.5]);
          f.status[6]=-2;const bad=p.extensometer(p.analyzeField(f),[7.5,7.5],[103.5,7.5]);
          console.log(JSON.stringify({g,bad}));
        """)
        self.assertAlmostEqual(result["g"]["referenceLength"], 4.8)
        self.assertAlmostEqual(result["g"]["engineeringStrain"], 0.01)
        self.assertEqual(result["g"]["unit"], "mm")
        self.assertIsNone(result["bad"])

    def test_invalid_recipe_and_inconsistent_field_are_rejected(self):
        result = javascript("""
          const errors=[];for(const o of [{window:4},{smoothing:'median'},{unitsPerPixel:NaN},{unit:'px',unitsPerPixel:2},{unknown:1}]){try{p.validateRecipe(o);}catch(e){errors.push(e.message);}}
          const f=fixture(128,128);f.grid=[1,1];try{p.analyzeField(f);}catch(e){errors.push(e.message);}
          console.log(JSON.stringify(errors));
        """)
        self.assertEqual(len(result), 6)
        self.assertIn("geometry", result[-1])

    def test_numpy_archive_roundtrip_keeps_raw_and_processed_arrays_without_pickle(self):
        import base64
        import io

        archive_module = MODULE.with_name("archive.js")
        result = javascript(f"""
          import {{fieldArchive,importFieldArchive}} from {json.dumps(archive_module.as_uri())};
          const f=fixture(128,128),mask=new Uint8Array(128*128).fill(1);
          for(let i=0;i<64;i++)f.parameters[i*6]=.123456789012345+i/100;
          const a=p.analyzeField(f),bytes=fieldArchive(f,{{name:'Plaque · µε',comparison:{{mode:'fixed'}}}},a,mask);
          const imported=await importFieldArchive(bytes);
          console.log(JSON.stringify({{base64:Buffer.from(bytes).toString('base64'),equal:imported.field.parameters.every((v,i)=>v===f.parameters[i]),name:imported.metadata.name,recipe:imported.recipe}}));
        """)
        self.assertTrue(result["equal"])
        self.assertEqual(result["name"], "Plaque · µε")
        with np.load(io.BytesIO(base64.b64decode(result["base64"])), allow_pickle=False) as data:
            self.assertEqual(data["parameters"].shape, (8, 8, 6))
            self.assertEqual(data["roi_mask"].shape, (128, 128))
            self.assertEqual(json.loads(str(data["metadata_json"]))["name"], "Plaque · µε")
            self.assertEqual(data["width"], 128)
            np.testing.assert_allclose(data["processed_u"], data["parameters"][..., 0])

    def test_numpy_archive_rejects_crc_corruption_and_object_arrays(self):
        archive_module = MODULE.with_name("archive.js")
        result = javascript(f"""
          import {{fieldArchive,readZip,readNpy}} from {json.dumps(archive_module.as_uri())};
          const bytes=fieldArchive(fixture(128,128));bytes[100]^=1;
          let crc;try{{await readZip(bytes);}}catch(e){{crc=e.message;}}
          const prefix=Buffer.from([147,78,85,77,80,89,1,0]),text=Buffer.from("{{'descr':'|O8','fortran_order':False,'shape':(1,),}}"),length=Buffer.alloc(2);length.writeUInt16LE(text.length);
          let object;try{{readNpy(Buffer.concat([prefix,length,text,Buffer.alloc(8)]));}}catch(e){{object=e.message;}}
          console.log(JSON.stringify({{crc,object}}));
        """)
        self.assertIn("integrity", result["crc"])
        self.assertIn("pickle", result["object"])

    def test_quadratic_public_layout_retains_affine_first_six_components(self):
        result = javascript("""
          const f=fixture(128,128);f.config.motionModel='quadratic';f.parameters=new Float64Array(64*12);
          for(let i=0;i<64;i++)f.parameters.set([2,.001,.002,-3,.004,.005,11,12,13,14,15,16],i*12);
          const a=p.analyzeField(f);console.log(JSON.stringify({u:a.fields.u[0],v:a.fields.v[0],exx:a.fields.exx[0],eyy:a.fields.eyy[0],exy:a.fields.exy[0]}));
        """)
        self.assertEqual(result["u"], 2)
        self.assertEqual(result["v"], -3)
        self.assertAlmostEqual(result["exx"], 0.001)
        self.assertAlmostEqual(result["eyy"], 0.005)
        self.assertAlmostEqual(result["exy"], 0.003)
