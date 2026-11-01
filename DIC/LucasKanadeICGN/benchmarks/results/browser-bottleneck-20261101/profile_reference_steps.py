import json, sys, shutil, traceback
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'tools'))
from qualify_local_sources import wait
from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer
site=Path('/tmp/live-dic-reference-step-site')
shutil.copytree('/tmp/live-dic-cause-site',site,dirs_exist_ok=True)
p=site/'web/core/reference-gpu.js'
s=p.read_text()
needle='    return { copy, encode, readValidity,'
addition='''    async function profile(frame, repeats) {
      const steps=[];
      const recorder={
        clearBuffer(...args){steps.push({label:'clear reference counts',run:encoder=>encoder.clearBuffer(...args)});},
        beginComputePass(options){
          const calls=[];let label=options?.label??'reference input conversion';
          return {setPipeline(pipeline){calls.push(['setPipeline',[pipeline]]);},
            setBindGroup(...args){calls.push(['setBindGroup',args]);},
            dispatchWorkgroups(...args){calls.push(['dispatchWorkgroups',args]);},
            end(){steps.push({label,run(encoder){const pass=encoder.beginComputePass(options);for(const [name,args] of calls)pass[name](...args);pass.end();}});}};
        }
      };
      copy(frame);encode(recorder);
      const samples=[];
      for(let repeat=0;repeat<repeats;repeat++){
        let started=performance.now();copy(frame);const whole=device.createCommandEncoder();encode(whole);device.queue.submit([whole.finish()]);await device.queue.onSubmittedWorkDone();
        const wholeMs=performance.now()-started;
        started=performance.now();const empty=device.createCommandEncoder();device.queue.submit([empty.finish()]);await device.queue.onSubmittedWorkDone();const emptyFenceMs=performance.now()-started;
        started=performance.now();copy(frame);const copyApiMs=performance.now()-started;await device.queue.onSubmittedWorkDone();const copyCompletionMs=performance.now()-started;
        const measured=[];
        for(const step of steps){
          const start=performance.now(),encoder=device.createCommandEncoder();step.run(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
          measured.push({label:step.label,wallMs:performance.now()-start});
        }
        samples.push({wholeMs,emptyFenceMs,copyApiMs,copyCompletionMs,steps:measured});
      }
      return {samples,boundary:'Whole: copy + unchanged full preparation + completion. Steps: each original pass submitted/fenced separately; extra fences perturb execution and these are wall times, not GPU timestamps.'};
    }
'''
assert s.count(needle)==1
p.write_text(s.replace(needle,addition+needle.replace('copy, encode, readValidity,','copy, encode, readValidity, profile,')))
p=site/'web/core/pyramid.js';s=p.read_text();needle='    exportField() { return readResult('
assert s.count(needle)==1
p.write_text(s.replace(needle,'    profileReference(frame,repeats) {return gpuReference.profile(frame,repeats);},\n'+needle))
server=_StaticServer(site.parent);server.start()
report={'base':json.loads((site/'build.json').read_text()),'kind':'Intrusive per-pass completion diagnosis; no shader change; not FPS qualification'}
out=Path('/tmp/live-dic-reference-steps.json')
try:
 with ICGNWebGL(backend='webgpu-wgsl',startup_timeout=60,call_timeout=300) as browser:
  url=f'http://127.0.0.1:{server.port}/{site.name}/index.html?smoke=1&case=plate'
  browser._evaluate(f'location.assign({json.dumps(url)});true')
  wait(browser,'globalThis.liveDicStatus?.completed>=1',timeout=180)
  if not browser._evaluate('liveDicSourceState().paused'):browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
  report['browser']=browser._evaluate("""(async()=>{
    const {DICEngine}=await import('./web/core/engine.js'),{LATENCY_OPTIONS}=await import('./web/core/device.js');
    const pixels=new Uint8Array(await fetch('./reference.gray').then(r=>r.arrayBuffer()));
    const rgba=new Uint8ClampedArray(pixels.length*4);for(let i=0;i<pixels.length;i++){rgba[4*i]=rgba[4*i+1]=rgba[4*i+2]=pixels[i];rgba[4*i+3]=255;}
    const bitmap=await createImageBitmap(new ImageData(rgba,3840,2160));
    const engine=await DICEngine.create({...LATENCY_OPTIONS,width:3840,height:2160,referencePreparation:'gpu',iterationsPerLevel:[8,16],tolerancePerLevel:[.005,.005]});
    try{
      await engine.setReference(pixels);for(let i=0;i<5;i++)await engine.registerPair(pixels,pixels);
      const bytes=await engine.host.profileReference(pixels,12),external=await engine.host.profileReference(bitmap,12);
      await engine.registerPair(pixels,pixels);const field=await engine.exportField();
      let count=0,sum=0;for(let i=0;i<field.status.length;i++)if(field.status[i]>=.5){count++;sum+=field.parameters[6*i]**2+field.parameters[6*i+3]**2;}
      return {bytes,external,quality:{width:field.width,height:field.height,accepted:count,finite:field.parameters.every(Number.isFinite),rmse:Math.sqrt(sum/count)}};
    }finally{bitmap.close();engine.destroy();}
  })()""")
except BaseException as error:
 report['failure']=str(error);report['traceback']=traceback.format_exc();raise
finally:
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');server.close()
print(out)
