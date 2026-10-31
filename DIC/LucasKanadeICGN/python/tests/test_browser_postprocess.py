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

    def test_compressed_native_numpy_archive_uses_exact_roi_geometry(self):
        import base64
        import io

        output = io.BytesIO()
        parameters = np.zeros((8, 8, 6), dtype=np.float32)
        parameters[..., 0] = 0.3
        parameters[..., 3] = -0.2
        np.savez_compressed(
            output,
            parameters=parameters,
            status=np.ones((8, 8)),
            iterations=np.full((8, 8), 3, dtype=np.int64),
            roi_mask=np.ones((128, 128), dtype=bool),
            config_json=np.asarray(
                json.dumps(dict(motionModel="affine", subsetSize=16, subsetStride=16))
            ),
        )
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        archive_module = MODULE.with_name("archive.js")
        result = javascript(f"""
          import {{importFieldArchive}} from {json.dumps(archive_module.as_uri())};
          const result=await importFieldArchive(Buffer.from({json.dumps(encoded)},'base64'));
          console.log(JSON.stringify({{width:result.field.width,height:result.field.height,iterations:result.field.iterations[0],v:result.field.parameters[3],geometry:result.metadata.importGeometry}}));
        """)
        self.assertEqual((result["width"], result["height"]), (128, 128))
        self.assertEqual(result["iterations"], 3)
        self.assertAlmostEqual(result["v"], -0.2, places=7)
        self.assertEqual(result["geometry"]["imageExtentSource"], "archived ROI shape")

    def test_native_sequence_requires_explicit_geometry_and_frame_selection(self):
        import base64
        import io

        output = io.BytesIO()
        parameters = np.zeros((2, 8, 8, 6), dtype=np.float32)
        parameters[1, ..., 0] = 2
        np.savez_compressed(
            output,
            parameters=parameters,
            status=np.ones((2, 8, 8)),
            iterations=np.ones((2, 8, 8)),
            config_json=np.asarray(
                json.dumps(dict(motionModel="translation", subsetSize=16, subsetStride=16))
            ),
        )
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        archive_module = MODULE.with_name("archive.js")
        result = javascript(f"""
          import {{importFieldArchive}} from {json.dumps(archive_module.as_uri())};
          const buffer=Buffer.from({json.dumps(encoded)},'base64');let missing;
          try{{await importFieldArchive(buffer);}}catch(e){{missing={{code:e.code,geometry:e.geometry}};}}
          const result=await importFieldArchive(buffer,{{width:128,height:128,frameIndex:1}});
          console.log(JSON.stringify({{missing,u:result.field.parameters[0],geometry:result.metadata.importGeometry}}));
        """)
        self.assertEqual(result["missing"]["code"], "FIELD_GEOMETRY_REQUIRED")
        self.assertEqual(result["missing"]["geometry"]["frameCount"], 2)
        self.assertEqual(result["u"], 2)
        self.assertEqual(result["geometry"]["frameIndex"], 1)
        self.assertEqual(result["geometry"]["imageExtentSource"], "explicit user input")

    def test_uhd_point_history_preserves_failure_gaps_and_order(self):
        history_module = MODULE.with_name("history.js")
        result = javascript(f"""
          import {{extractHistory}} from {json.dumps(history_module.as_uri())};
          const entries=[.1,.2,.3].map((value,i)=>{{const field=fixture();for(let j=0;j<field.status.length;j++)field.parameters[j*6]=value;if(i===1)field.status[62+56*240]=-2;return {{id:i+1,field,metadata:{{analysisEpoch:'same-reference',comparison:{{mode:'fixed',target:{{index:i}}}}}},mask:null}};}});
          const rows=await extractHistory(entries,{{recipe:p.DEFAULT_RECIPE,component:'u',points:[[999.5,903.5]],tool:'probe'}});
          console.log(JSON.stringify(rows));
        """)
        self.assertEqual([r["order"] for r in result], [1, 2, 3])
        self.assertEqual([r["value"] for r in result], [0.1, None, 0.3])
        self.assertEqual([r["status"] for r in result], [1, -2, 1])

    def test_gauge_history_is_not_cumulative_and_rejects_unrelated_references(self):
        history_module = MODULE.with_name("history.js")
        result = javascript(f"""
          import {{extractHistory,sameAnalysis}} from {json.dumps(history_module.as_uri())};
          const entries=[.001,.002,.003].map((strain,i)=>{{const field=fixture(128,128);for(let j=0;j<64;j++)field.parameters[j*6]=strain*(j%8*16+7.5);return {{id:i+1,field,metadata:{{analysisEpoch:'same-reference'}},mask:null}};}});
          const rows=await extractHistory(entries,{{recipe:p.DEFAULT_RECIPE,component:'u',points:[[7.5,7.5],[103.5,7.5]],tool:'gauge'}});
          entries[1].metadata.analysisEpoch='another-reference';let error;
          try{{await extractHistory(entries,{{recipe:p.DEFAULT_RECIPE,component:'u',points:[[7.5,7.5]],tool:'probe'}});}}catch(e){{error=e.message;}}
          console.log(JSON.stringify({{rows,error,same:sameAnalysis(entries[0],entries[1])}}));
        """)
        np.testing.assert_allclose(
            [r["value"] for r in result["rows"]], [0.001, 0.002, 0.003], atol=1e-15
        )
        self.assertFalse(result["same"])
        self.assertIn("one analysis epoch", result["error"])

    def test_history_cancellation_stops_before_the_next_field(self):
        history_module = MODULE.with_name("history.js")
        result = javascript(f"""
          import {{extractHistory}} from {json.dumps(history_module.as_uri())};
          const entries=Array.from({{length:5}},(_,i)=>({{id:i,field:fixture(128,128),metadata:{{analysisEpoch:'same'}},mask:null}})),abort=new AbortController();let processed=0,cancelled;
          try{{await extractHistory(entries,{{recipe:p.DEFAULT_RECIPE,component:'u',points:[[7.5,7.5]],tool:'probe',signal:abort.signal,progress:n=>{{processed=n;if(n===2)abort.abort();}}}});}}catch(e){{cancelled=e.name;}}
          console.log(JSON.stringify({{processed,cancelled}}));
        """)
        self.assertEqual(result["processed"], 2)
        self.assertEqual(result["cancelled"], "AbortError")

    def test_results_project_roundtrip_shares_masks_and_preserves_computed_fields(self):
        import base64
        import io
        import zipfile

        archive_module = MODULE.with_name("archive.js")
        result = javascript(f"""
          import {{projectArchive,importProjectArchive,readZip}} from {json.dumps(archive_module.as_uri())};
          const mask=new Uint8Array(128*128).fill(1),records=[1,2].map((n,i)=>{{const field=fixture(128,128);field.parameters[0]=n+.1234567890123;return {{id:n,field,mask,metadata:{{analysisEpoch:'one experiment',name:'µε'}},recipe:p.DEFAULT_RECIPE}};}});
          const bytes=projectArchive(records,{{selectedId:2}}),project=await importProjectArchive(bytes),zip=await readZip(bytes);
          console.log(JSON.stringify({{base64:Buffer.from(bytes).toString('base64'),maskEntries:Object.keys(zip).filter(k=>k.startsWith('masks/')),shared:project.records[0].mask===project.records[1].mask,values:project.records.map(r=>r.field.parameters[0]),selected:project.settings.selectedId}}));
        """)
        self.assertTrue(result["shared"])
        self.assertEqual(len(result["maskEntries"]), 1)
        self.assertEqual(result["selected"], 2)
        np.testing.assert_allclose(
            result["values"], [1.1234567890123, 2.1234567890123], atol=0, rtol=0
        )
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["base64"]))) as saved:
            manifest = json.loads(saved.read("manifest.json"))
            self.assertEqual(manifest["format"], "live-dic-results")
            self.assertEqual(len(manifest["fields"]), 2)
            values = np.load(io.BytesIO(saved.read("fields/0/parameters.npy")), allow_pickle=False)
            self.assertEqual(values.shape, (8, 8, 6))
            self.assertEqual(values[0, 0, 0], 1.1234567890123)

    def test_numpy_roi_hash_mismatch_is_rejected(self):
        archive_module = MODULE.with_name("archive.js")
        result = javascript(f"""
          import {{fieldArchive,importFieldArchive}} from {json.dumps(archive_module.as_uri())};
          const f=fixture(128,128);f.config.roi={{maskSha256:'0'.repeat(64)}};const mask=new Uint8Array(128*128).fill(1);
          let error;try{{await importFieldArchive(fieldArchive(f,{{}},null,mask));}}catch(e){{error=e.message;}}
          console.log(JSON.stringify(error));
        """)
        self.assertIn("hash does not match", result)

    def test_smoothing_then_vsg_reports_its_composed_spatial_support(self):
        result = javascript("""
          const f=fixture(512,512);f.parameters[(16*32+16)*6]=1;
          const a=p.analyzeField(f,{smoothing:'gaussian',strainSource:'vsg',window:5});
          console.log(JSON.stringify({uSpan:a.smoothingSpanPx,strainSpan:a.strainSpanPx,inside:a.fields.exx[16*32+19],outside:a.fields.exx[16*32+21]}));
        """)
        self.assertEqual(result["uSpan"], 80)
        self.assertEqual(result["strainSpan"], 144)
        self.assertGreater(abs(result["inside"]), 1e-8)
        self.assertEqual(result["outside"], 0)
