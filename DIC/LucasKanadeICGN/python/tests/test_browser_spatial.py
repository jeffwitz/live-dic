"""Shared-WGSL browser propagation: configuration, real re-solves and bitmaps."""

import json
import os
import subprocess
import unittest
from importlib.resources import files
from pathlib import Path

CORE = Path(str(files("lucas_kanade_icgn").joinpath("web/core")))
ROOT = Path(__file__).resolve().parents[2]


class BrowserSpatialContractTests(unittest.TestCase):
    def test_defaults_and_explicit_configuration_validation(self):
        script = f"""
          import {{DICEngine}} from {json.dumps((CORE / "engine.js").as_uri())};
          const defaults=new DICEngine();
          const unchanged=!('spatialPropagationPasses' in defaults.host.configuration);defaults.destroy();
          const good={{motionModel:'translation',spatialPropagationPasses:2}};
          const engine=new DICEngine(good),before=engine.host.configure();let preserved=false;
          try{{engine.host.configure({{interpolation:'linear'}});}}catch{{preserved=JSON.stringify(before)===JSON.stringify(engine.host.configure());}}
          const configured=engine.host.configure();engine.destroy();
          const bad=[{{spatialPropagationPasses:null}},{{spatialPropagationPasses:true}},{{spatialPropagationPasses:5}},
            {{...good,motionModel:'affine'}},{{...good,motionModel:'quadratic'}},
            {{...good,interpolation:'linear'}},{{...good,referencePreparation:'gpu'}},{{...good,pipelineDepth:2}}];
          const errors=bad.map(o=>{{try{{const e=new DICEngine(o);e.destroy();return null;}}catch(e){{return e.message;}}}});
          console.log(JSON.stringify({{unchanged,preserved,configured,errors}}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertTrue(result["unchanged"])
        self.assertTrue(result["preserved"])
        self.assertEqual(result["configured"]["spatialPropagationPasses"], 2)
        self.assertTrue(all(result["errors"]))

    def test_solve_setting_preserves_global_affine_initialization_and_is_cold(self):
        script = f"""
          import {{solverOptions,solverPreparationKey}} from {json.dumps((ROOT / "web-app/ui/workspace.js").as_uri())};
          const values={{'custom-solver':{{checked:true}},'global-affine-start':{{checked:true}},'global-affine-levels':{{value:'6'}},
            'fourier-start':{{checked:false}},'temporal-start':{{checked:false}},'subset-size':{{value:'32'}},
            'subset-stride':{{value:'16'}},'pyramid-levels':{{value:'2'}},'coarse-iterations':{{value:'8'}},
            'fine-iterations':{{value:'16'}},'motion-model':{{value:'translation'}},'interpolation':{{value:'bspline'}},
            'outlier-policy':{{value:'median'}},'median-filter-size':{{value:'3'}},'spatial-propagation':{{value:'0'}}}};
          globalThis.document={{body:{{dataset:{{workspaceMode:'analysis'}}}},getElementById:id=>values[id]??null}};
          const before=solverOptions();values['spatial-propagation'].value='2';const after=solverOptions();
          values['solver-workgroup-size']={{value:'32'}};const wide=solverOptions(),live=solverOptions('live');
          values['motion-model'].value='affine';let rejected=false;try{{solverOptions();}}catch{{rejected=true;}}
          console.log(JSON.stringify({{before,after,wide,live,rejected,cold:solverPreparationKey(before)!==solverPreparationKey(after)}}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertNotIn("spatialPropagationPasses", result["before"])
        self.assertEqual(result["after"]["initialization"], "global-affine")
        self.assertEqual(result["after"]["motionModel"], "translation")
        self.assertEqual(result["after"]["spatialPropagationPasses"], 2)
        self.assertTrue(result["cold"])
        self.assertTrue(result["rejected"])
        self.assertEqual(result["wide"]["solverWorkgroupSize"], 32)
        self.assertNotIn("solverWorkgroupSize", result["live"])


@unittest.skipUnless(os.environ.get("ICGN_RUN_BROWSER_TESTS") == "1", "Browser GPU tests disabled")
class BrowserSpatialExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lucas_kanade_icgn import ICGNWebGL

        flags = (
            (
                "--use-angle=swiftshader",
                "--enable-unsafe-webgpu",
                "--use-webgpu-adapter=swiftshader",
                "--use-vulkan=swiftshader",
            )
            if os.environ.get("ICGN_BROWSER_SOFTWARE") == "1"
            else ()
        )
        cls.browser = ICGNWebGL(
            backend="webgpu-wgsl", chromium_args=flags, startup_timeout=60, call_timeout=240
        )
        cls.browser.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()

    def evaluate(self, body):
        return self.browser._evaluate(
            """(async()=>{
          const {DICEngine}=await import('../core/engine.js');
          const check=(ok,message)=>{if(!ok)throw new Error(message);};
          let seed=31;const texture=(w,h)=>Uint8Array.from({length:w*h},()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed&255;});
          const read=async(device,buffer)=>{const out=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
            try{const enc=device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,out,0,buffer.size);device.queue.submit([enc.finish()]);
              await out.mapAsync(GPUMapMode.READ);return new Float32Array(out.getMappedRange()).slice();}finally{out.unmap();out.destroy();}};
        """
            + body
            + "})()"
        )

    def test_32_thread_pipelines_cover_models_preparation_and_compact_propagation(self):
        result = self.evaluate("""
          const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice();
          const entries=[];
          for(const name of ['createComputePipeline','createComputePipelineAsync']){
            const original=device[name].bind(device);
            device[name]=descriptor=>{entries.push(descriptor.compute);return original(descriptor);};
          }
          const w=128,h=128,f=texture(w,h),g=new Uint8Array(f.length),rows=[];
          for(let y=0;y<h;y++)for(let x=0;x<w;x++)g[y*w+x]=f[((y+1)%h)*w+(x-2+w)%w];
          try{
            for(const referencePreparation of ['cpu','gpu'])for(const motionModel of ['translation','affine','quadratic']){
              if(referencePreparation==='gpu'&&motionModel==='quadratic'){
                let refused=false;try{new DICEngine({solverWorkgroupSize:32,motionModel,referencePreparation});}catch{refused=true;}
                check(refused,'Unqualified GPU quadratic preparation was accepted');continue;
              }
              const e=await DICEngine.create({device,width:w,height:h,solverWorkgroupSize:32,motionModel,referencePreparation,
                spatialPropagationPasses:motionModel==='translation'&&referencePreparation==='cpu'?2:0,outlierPolicy:'disabled'});
              try{
                const info=await e.setReference(f),r=await e.register(g);
                check(info.config.solverWorkgroupSize===32&&r.config.solverWorkgroupSize===32,'Missing width provenance');
                let count=0,maximum=0;const n=motionModel==='quadratic'?12:6;
                // The rolled fixture has a seam at its boundary. Evaluate
                // the same one-window interior for every model and width.
                for(let i=0;i<r.status.length;i++){
                  const x=i%r.grid[0],y=Math.floor(i/r.grid[0]);
                  if(r.status[i]<=.5||x<1||y<1||x>=r.grid[0]-1||y>=r.grid[1]-1)continue;
                  count++;maximum=Math.max(maximum,Math.hypot(r.parameters[i*n]-2,r.parameters[i*n+3]+1));
                }
                check(count>20&&maximum<.03,'Width 32 integer translation differs');
                rows.push({referencePreparation,motionModel,count,maximum});
              }finally{e.destroy();}
            }
            const solvers=entries.filter(d=>['solve','solve_seeded','solve_compact_translation'].includes(d.entryPoint));
            check(solvers.some(d=>d.entryPoint==='solve_compact_translation'),'Compact solver not compiled');
            check(solvers.every(d=>d.constants.SOLVER_WORKGROUP_SIZE===32),'A solver ignored the requested width');
            return {rows,pipelines:solvers.length};
          }finally{device.destroy();}
        """)
        self.assertEqual(len(result["rows"]), 5)
        self.assertGreater(result["pipelines"], 6)

    def test_failed_point_really_uses_a_neighbour_and_preserves_every_other_state(self):
        actual = self.evaluate("""
          const w=128,h=128,f=texture(w,h),g=new Uint8Array(f.length),mask=new Uint8Array(f.length);
          for(let y=0;y<h;y++)for(let x=0;x<w;x++){g[y*w+x]=f[((y+1)%h)*w+(x-2+w)%w];mask[y*w+x]=+(x>=32&&x<96&&y>=32&&y<96);}
          const outputs=[];for(const solverWorkgroupSize of [16,32]){
          const e=await DICEngine.create({width:w,height:h,solverWorkgroupSize,motionModel:'translation',spatialPropagationPasses:2,outlierPolicy:'disabled'});
          try{
            await e.setReference(f,{roiMask:mask});await e.register(g);
            const slot=e.host.memoryResources().slots[0],buffer=slot.levels[0].buffers.state,states=new Float32Array(64*8);
            for(let i=0;i<64;i++){states[i*8]=2;states[i*8+3]=-1;states[i*8+6]=1;states[i*8+7]=4;}
            states[27*8]=0;states[27*8+3]=0;states[27*8+6]=-1;e.device.queue.writeBuffer(buffer,0,states);
            const enc=e.device.createCommandEncoder();slot.spatial.afterLevel(enc,0,false);e.device.queue.submit([enc.finish()]);
            const output=await read(e.device,buffer);
            for(let i=0;i<output.length;i++)if(Math.floor(i/8)!==27)check(output[i]===states[i],'Unqueued point changed');
            check(output[27*8+6]>.5,'Failed point did not converge');
            check(Math.abs(output[27*8]-2)<.03&&Math.abs(output[27*8+3]+1)<.03,'Wrong refined displacement');
            outputs.push({solverWorkgroupSize,status:output[27*8+6],u:output[27*8],v:output[27*8+3]});
          }finally{e.destroy();}
          }return outputs;
        """)
        self.assertEqual([row["solverWorkgroupSize"] for row in actual], [16, 32])
        for row in actual:
            self.assertGreater(row["status"], 0.5)

    def test_external_bitmap_median_filter_and_warm_reference_resources(self):
        actual = self.evaluate("""
          const w=128,h=128,f=texture(w,h),rgba=new Uint8ClampedArray(w*h*4);
          for(let i=0;i<f.length;i++){rgba[4*i]=rgba[4*i+1]=rgba[4*i+2]=f[i];rgba[4*i+3]=255;}
          const bitmap=await createImageBitmap(new ImageData(rgba,w,h));
          const e=await DICEngine.create({motionModel:'translation',spatialPropagationPasses:2,outlierPolicy:'median',medianFilterSize:5});
          try{
            const info=await e.setReference(bitmap);const first=await e.register(bitmap);
            const resources=e.memoryUsage();let allocation=false;const saved=[];
            for(const name of ['createBuffer','createTexture','createComputePipeline','createComputePipelineAsync']){
              saved.push([name,e.device[name]]);const original=e.device[name].bind(e.device);e.device[name]=(...args)=>{allocation=true;return original(...args);};
            }
            let second;try{second=await e.register(bitmap);}finally{for(const [name,value]of saved)e.device[name]=value;}
            check(!allocation,'Lazy spatial allocation/compilation');check(second.accepted===first.accepted,'Warm acceptance changed');
            check(second.config.spatialPropagationPasses===2&&info.config.spatialPropagationPasses===2,'Missing provenance');
            check(e.memoryUsage().knownResourceBytes===resources.knownResourceBytes,'Warm resources grew');
            return {accepted:second.accepted,configuration:second.config};
          }finally{bitmap.close();e.destroy();}
        """)
        self.assertGreater(actual["accepted"], 0)
        self.assertEqual(actual["configuration"]["outlierPolicy"], "median")
