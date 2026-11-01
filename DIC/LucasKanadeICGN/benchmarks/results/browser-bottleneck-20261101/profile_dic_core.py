import json, hashlib, time, shutil, sys, traceback
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path.cwd() / 'tools'))
from qualify_local_sources import wait
from lucas_kanade_icgn import ICGNWebGL, ICGNWGPUFullPyramid
from lucas_kanade_icgn.engine import _StaticServer

root = Path.cwd()
site = Path('/tmp/live-dic-cause-site')
shutil.copytree('/tmp/live-dic-cadence-site', site, dirs_exist_ok=True)
shutil.copyfile('/tmp/profile_dic_gpu.js', site / 'profile-gpu.js')
source = root / 'DIC/LucasKanadeICGN/assets/uhd_tensile/reference.png'
pixels = np.array(Image.open(source).convert('L'))
(site / 'reference.gray').write_bytes(pixels.tobytes())
report = {'base': json.loads((site / 'build.json').read_text()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'kind': 'Stage diagnosis on unchanged native-UHD static identity pair; not camera FPS', 'native': []}
out = Path('/tmp/live-dic-core-cause.json')
for temporal in [False, True]:
    opts = dict(device_type='gpu', motion_model='affine', interpolation='bspline', outlier_policy='median', iterations_per_level=(8,16), tolerance_per_level=(.005,.005))
    if temporal:
        opts.update(initialization='temporal', temporal_prediction='velocity', temporal_fine_correction=True, divergence_patience=3, divergence_min_iterations=4)
    with ICGNWGPUFullPyramid(**opts) as engine:
        started = time.perf_counter()
        engine.set_reference(pixels)
        prep = (time.perf_counter()-started)*1000
        for _ in range(5): engine.register(pixels)
        samples = []
        for _ in range(30):
            started = time.perf_counter()
            result = engine.register(pixels)
            samples.append((time.perf_counter()-started)*1000)
        report['native'].append({'temporal': temporal, 'adapter': engine.adapter_info, 'options': opts, 'preparationMs': prep, 'wallMs': samples, 'accepted': int(result.accepted_mask.sum()), 'rmse': result.rmse((0,0))})
        print('native', temporal, np.percentile(samples,[50,95]).tolist(), flush=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
server = _StaticServer(site.parent)
server.start()
try:
    with ICGNWebGL(backend='webgpu-wgsl', startup_timeout=60, call_timeout=300) as browser:
        url = f'http://127.0.0.1:{server.port}/{site.name}/index.html?smoke=1&case=plate'
        browser._evaluate(f'location.assign({json.dumps(url)});true')
        wait(browser, 'globalThis.liveDicStatus?.completed>=1', timeout=180)
        if not browser._evaluate('liveDicSourceState().paused'):
            browser._evaluate("document.getElementById('pause').onclick().then(()=>true)")
        report['browser'] = browser._evaluate("""(async()=>{
          const {DICEngine}=await import('./web/core/engine.js'),{LATENCY_OPTIONS}=await import('./web/core/device.js'),{instrument}=await import('./profile-gpu.js');
          const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
          if(!adapter.features.has('timestamp-query'))throw new Error('Hardware GPU timestamp queries required for this diagnostic');
          const device=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:{maxStorageBuffersPerShaderStage:8}});
          const trace=instrument(device),pixels=new Uint8Array(await fetch('./reference.gray').then(r=>r.arrayBuffer()));
          const rgba=new Uint8ClampedArray(pixels.length*4);for(let i=0;i<pixels.length;i++){rgba[4*i]=rgba[4*i+1]=rgba[4*i+2]=pixels[i];rgba[4*i+3]=255;}
          const bitmap=await createImageBitmap(new ImageData(rgba,3840,2160));
          const cases=[];
          const variants=[['fixed-full',false,'full',false],['fixed-summary-export',false,'none',false],['pair-full',true,'full',false],['pair-summary-export',true,'none',false],['pair-external-summary-export',true,'none',true]];
          for(const [name,pair,readback,external] of variants){
            const options={...LATENCY_OPTIONS,device,width:3840,height:2160,readback,referencePreparation:'gpu',iterationsPerLevel:[8,16],tolerancePerLevel:[.005,.005]};
            const engine=await DICEngine.create(options),source=external?bitmap:pixels;
            const prepStart=performance.now();await engine.setReference(source);const preparationMs=performance.now()-prepStart;
            let last;const samples=[];
            async function call(profile,index){
              const id=name+':'+index;if(profile)trace.begin(id);
              const started=performance.now();last=pair?await engine.registerPair(source,source):await engine.register(source);
              const observed=performance.now(),field=readback==='none'?await engine.exportField():last;
              const finished=performance.now();if(profile)trace.end();
              let accepted=0,sum=0;for(let i=0;i<field.status.length;i++)if(field.status[i]>=.5){accepted++;sum+=field.parameters[6*i]**2+field.parameters[6*i+3]**2;}
              return {id,profile,callMs:observed-started,exportMs:finished-observed,totalMs:finished-started,accepted,rmse:Math.sqrt(sum/accepted),finite:field.parameters.every(Number.isFinite),config:field.config};
            }
            for(let i=0;i<5;i++)await call(false,-1);
            for(let i=0;i<15;i++)samples.push(await call(false,i));
            for(let i=15;i<25;i++)samples.push(await call(true,i));
            for(let i=25;i<40;i++)samples.push(await call(false,i));
            cases.push({name,preparationMs,samples});engine.destroy();
          }
          const gpu=await trace.read();bitmap.close();device.destroy();return {adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},features:[...adapter.features],cases,gpu};
        })()""")
        out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
except BaseException as error:
    report['failure'] = str(error)
    report['traceback'] = traceback.format_exc()
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    raise
finally:
    server.close()
print('saved', out, flush=True)
