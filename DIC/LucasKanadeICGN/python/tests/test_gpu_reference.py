"""GPU reference preparation versus the existing double-precision CPU oracle.

These are browser numerical/ownership tests, never FPS gates. The new path is
explicitly opt-in and does not replace native or fixed-reference defaults.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from importlib.resources import files
from pathlib import Path

CORE = Path(str(files("lucas_kanade_icgn").joinpath("web/core")))


@unittest.skipUnless(shutil.which("node"), "Node is needed to execute ES modules")
class GpuReferenceContractTests(unittest.TestCase):
    def test_opt_in_and_unsupported_combinations(self):
        script = f"""
          import {{DICEngine}} from {json.dumps((CORE / "engine.js").as_uri())};
          const cases=[{{referencePreparation:null}},{{referencePreparation:'other'}},
            {{referencePreparation:'gpu',motionModel:'quadratic'}},
            {{referencePreparation:'gpu',pipelineDepth:2}}];
          const rejected=cases.map(options=>{{try{{new DICEngine(options);return false;}}catch{{return true;}}}});
          const engine=new DICEngine();let pairRejected=false;
          try{{engine.registerPair(new Uint8Array(16),new Uint8Array(16));}}catch{{pairRejected=true;}}
          const unchanged=!('referencePreparation' in engine.host.configuration);engine.destroy();
          console.log(JSON.stringify({{rejected,pairRejected,unchanged}}));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertTrue(all(actual["rejected"]))
        self.assertTrue(actual["pairRejected"])
        self.assertTrue(actual["unchanged"])


@unittest.skipUnless(
    os.environ.get("ICGN_RUN_BROWSER_TESTS") == "1", "set ICGN_RUN_BROWSER_TESTS=1"
)
class GpuReferenceExecutionTests(unittest.TestCase):
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
          const {prepareReferenceGray,reduceHalfGray}=await import('../core/reference.js');
          const check=(value,message)=>{if(!value)throw new Error(message);};
          const texture=(width,height)=>Uint8Array.from({length:width*height},(_,i)=>(i*7+i*i+Math.floor(i/width)*17)%251);
        """
            + body
            + "})()"
        )

    def test_reference_quantities_match_cpu_for_models_interpolators_and_grids(self):
        actual = self.evaluate("""
          let cases=0,maxReferenceError=0,maxInverseError=0;
          for(const motionModel of ['translation','affine'])for(const interpolation of ['bspline','linear','cubic']) {
            for(const [subsetSize,subsetStride,pyramidLevels] of [[16,16,2],[17,8,1],[32,16,1],[33,33,1]]) {
              const width=99,height=97,pixels=texture(width,height), options={width,height,motionModel,interpolation,subsetSize,subsetStride,pyramidLevels};
              const engine=await DICEngine.create({...options,referencePreparation:'gpu'});
              try {
                await engine.setReference(pixels);let level={values:pixels,width,height};
                for(let i=0;i<pyramidLevels;i++) {
                  const expected=prepareReferenceGray(level.values,{...options,width:level.width,height:level.height});
                  const actual=await engine.exportReference(i);
                  check(actual.reference.length===expected.reference.length,'reference shape');
                  for(let k=0;k<actual.reference.length;k++) {
                    const error=Math.abs(actual.reference[k]-expected.reference[k]);maxReferenceError=Math.max(maxReferenceError,error);
                    check(error<=3e-4+3e-6*Math.abs(expected.reference[k]),`reference ${motionModel}/${interpolation}/${subsetSize}/${i}/${k}: ${error}`);
                  }
                  for(let k=0;k<actual.metadata.length;k++) {
                    const component=k%16,a=actual.metadata[k],b=expected.metadata[k];
                    const tolerance=component>=2&&component<8?0.1+1e-4*Math.abs(b):component>=8&&component<14?.003+1e-4*Math.abs(b):.001+1e-5*Math.abs(b);
                    check(Math.abs(a-b)<=tolerance,`metadata ${component}: ${a} versus ${b}`);
                  }
                  for(let k=0;k<actual.inverseHessian.length;k++) {
                    const a=actual.inverseHessian[k],b=expected.inverseHessian[k],error=Math.abs(a-b);
                    maxInverseError=Math.max(maxInverseError,error);check(error<=5e-9+5e-4*Math.abs(b),`inverse ${error}`);
                  }
                  check(actual.referenceValid.every((v,k)=>v===+(expected.metadata[k*16+1]>0)),'validity differs');
                  level=reduceHalfGray(level.values,level.width,level.height);
                }
                cases++;
              } finally {engine.destroy();}
            }
          }
          return {cases,maxReferenceError,maxInverseError};
        """)
        self.assertEqual(actual["cases"], 24)
        self.assertLess(actual["maxReferenceError"], 0.001)

    def test_flat_references_are_invalid_and_finite_including_odd_subsets(self):
        actual = self.evaluate("""
          for(const subsetSize of [16,17])for(const motionModel of ['translation','affine']) {
            const engine=await DICEngine.create({width:68,height:68,subsetSize,referencePreparation:'gpu',motionModel});
            try {
              await engine.setReference(new Uint8Array(68*68));
              for(const value of [0,137,255]) {
                const image=new Uint8Array(68*68).fill(value),result=await engine.registerPair(image,image);
                check(result.referenceAdmissible===0,'flat reference admitted');
                check(result.accepted===0&&result.parameters.every(Number.isFinite),'flat field');
                const reference=await engine.exportReference();
                check(reference.referenceValid.every(v=>v===0),'flat numeric validity');
                check(reference.inverseHessian.every(Number.isFinite),'flat inverse finite');
              }
            }finally{engine.destroy();}
          }return true;
        """)
        self.assertTrue(actual)

    def test_pairs_reuse_resources_submit_once_and_keep_inputs_independent(self):
        actual = self.evaluate("""
          const width=128,height=128,options={width,height,outlierPolicy:'disabled',pyramidLevels:2,readback:'full'};
          const gpu=await DICEngine.create({...options,referencePreparation:'gpu'}),cpu=await DICEngine.create(options);
          const reference=texture(width,height),target=new Uint8Array(reference.length);
          for(let y=0;y<height;y++)for(let x=0;x<width;x++)target[y*width+x]=reference[y*width+(x+width-1)%width];
          try {
            await gpu.setReference(reference);await gpu.registerPair(reference,target);
            await cpu.setReference(reference);const expected=await cpu.register(target);
            const before=gpu.memoryUsage(),device=gpu.device;
            const originals={},counts={createBuffer:0,createTexture:0,createComputePipelineAsync:0,submit:0};
            for(const key of ['createBuffer','createTexture','createComputePipelineAsync']) {
              originals[key]=device[key].bind(device);device[key]=(...args)=>{counts[key]++;return originals[key](...args);};
            }
            const submit=device.queue.submit.bind(device.queue);device.queue.submit=(...args)=>{counts.submit++;return submit(...args);};
            let result;
            try{result=await gpu.registerPair(reference,target);}finally{
              for(const key of Object.keys(originals))device[key]=originals[key];device.queue.submit=submit;
            }
            check(counts.submit===1,`submissions ${counts.submit}`);
            check(!counts.createBuffer&&!counts.createTexture&&!counts.createComputePipelineAsync,'hot allocation/compilation');
            check(gpu.memoryUsage().knownResourceBytes===before.knownResourceBytes,'memory grew');
            check(gpu.reference.referenceValid===null&&gpu.reference.referenceValidity==='gpu-resident','stale validity');
            let maxError=0,count=0;
            for(let i=0;i<result.status.length;i++)if(result.status[i]>0&&expected.status[i]>0) {
              check(Math.abs(result.parameters[6*i]-1)<.05,'translation accuracy');
              for(let k=0;k<6;k++)maxError=Math.max(maxError,Math.abs(result.parameters[6*i+k]-expected.parameters[6*i+k]));count++;
            }
            check(count>30&&maxError<.005,`pair parity ${count}/${maxError}`);
            const reverse=await gpu.registerPair(target,reference);
            check(reverse.parameters.some((v,k)=>k%6===0&&v<-.95),'new reference not used');
            let rejected=false;try{gpu.registerPair(reference,new Uint8Array(16));}catch{rejected=true;}
            check(rejected&&gpu.reference!==null,'invalid dimensions damaged prepared state');
            return {counts,count,maxError};
          }finally{gpu.destroy();cpu.destroy();}
        """)
        self.assertEqual(actual["counts"]["submit"], 1)
        self.assertGreater(actual["count"], 30)

    def test_fractional_translation_and_affine_against_analytic_truth(self):
        actual = self.evaluate("""
          const width=160,height=160;
          const signal=(x,y)=>128+24*Math.sin(.43*x+.17*y)+23*Math.cos(.19*x-.39*y)+20*Math.sin(.31*x+.37*y)+14*Math.cos(.61*x-.29*y);
          const reference=Uint8Array.from({length:width*height},(_,i)=>Math.round(signal(i%width,Math.floor(i/width))));
          const results=[];
          for(const motionModel of ['translation','affine']) {
            const a=motionModel==='affine'?.0015:0,b=motionModel==='affine'?.001:0,c=motionModel==='affine'?-.0008:0,d=motionModel==='affine'?-.001:0;
            const tx=.35,ty=-.2,det=(1+a)*(1+d)-b*c;
            const target=Uint8Array.from({length:width*height},(_,i)=>{
              const x=i%width-tx,y=Math.floor(i/width)-ty;
              return Math.round(signal(((1+d)*x-b*y)/det,(-c*x+(1+a)*y)/det));
            });
            const engine=await DICEngine.create({width,height,motionModel,referencePreparation:'gpu',outlierPolicy:'disabled'});
            try {
              await engine.setReference(reference);const result=await engine.registerPair(reference,target);
              let displacement=0,gradient=0,count=0;
              for(let y=2;y<8;y++)for(let x=2;x<8;x++) {
                const index=y*10+x;if(result.status[index]<=0)continue;
                const px=x*16+7.5,py=y*16+7.5,truth=[tx+a*px+b*py,a,b,ty+c*px+d*py,c,d];
                for(let k=0;k<6;k++) {const e=result.parameters[index*6+k]-truth[k];if(k===0||k===3)displacement+=e*e;else gradient+=e*e;}
                count++;
              }
              const displacementRmse=Math.sqrt(displacement/(2*count)),gradientRmse=Math.sqrt(gradient/(4*count));
              check(count>=32&&displacementRmse<.05&&gradientRmse<.002,`analytic ${motionModel}: ${count}/${displacementRmse}/${gradientRmse}`);
              results.push({motionModel,count,displacementRmse,gradientRmse});
            }finally{engine.destroy();}
          }return results;
        """)
        self.assertEqual(len(actual), 2)

    def test_roi_empty_and_holes_remain_static_across_new_pairs(self):
        actual = self.evaluate("""
          const width=128,height=128,image=texture(width,height),mask=new Uint8Array(width*height).fill(1);
          mask.fill(0,32*width,48*width);const engine=await DICEngine.create({width,height,referencePreparation:'gpu',outlierPolicy:'median'});
          try {
            await engine.setReference(image,{roiMask:mask});const selected=engine.reference.roiSelected.slice();mask.fill(0);
            for(let i=0;i<3;i++) {
              const result=await engine.registerPair(image,image);
              check(result.status.every((v,k)=>selected[k]||v===-3),'excluded subset reactivated');
              check(result.accepted>0&&result.referenceAdmissible===selected.reduce((a,b)=>a+b,0),'selection/validity');
            }
            await engine.setReference(image,{roiMask:mask});
            const submit=engine.device.queue.submit.bind(engine.device.queue);let submissions=0;
            engine.device.queue.submit=(...args)=>{submissions++;return submit(...args);};
            let result;try{result=await engine.registerPair(image,image);}finally{engine.device.queue.submit=submit;}
            check(submissions===0,'empty ROI submitted work');
            check(result.referenceAdmissible===0&&result.status.every(v=>v===-3),'empty ROI result');return true;
          }finally{engine.destroy();}
        """)
        self.assertTrue(actual)
