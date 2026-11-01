import base64, csv, hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path.cwd()/'tools'))
from qualify_local_sources import wait
from lucas_kanade_icgn import ICGNWGPUFullPyramid, ICGNWebGL, RegistrationResult
from lucas_kanade_icgn.engine import _StaticServer

root=Path.cwd();assets=root/'DIC/LucasKanadeICGN/assets/uhd_tensile'
images={name:np.array(Image.open(assets/name).convert('L')) for name in ['reference.png','frame_040.png']}
sequence=['reference.png','frame_040.png','reference.png','frame_040.png']
limits=json.loads((assets/'manifest.json').read_text())['regression_limits']
with (assets/'machine_truth.csv').open() as file:truth=next(row for row in csv.DictReader(file) if int(row['frame'])==40)
ex,ey=float(truth['stretch_x'])-1,float(truth['stretch_y'])-1
report={'kind':'Fixed first-reference UHD qualification, not an FPS campaign','inputs':{name:hashlib.sha256((assets/name).read_bytes()).hexdigest() for name in images},'sequence':sequence,'native':[],'browser':[]}
def check(result,name,output):
 assert result.parameters.shape==(135,240,6) and np.isfinite(result.parameters).all()
 y,x=np.indices(result.status.shape);x=x*16+7.5;y=y*16+7.5
 interior=(x>=32)&(x<3840-32)&(y>=32)&(y<2160-32);valid=interior&result.accepted_mask
 expected=np.stack((ex*(x-1920),ey*(y-1080)),axis=-1) if name=='frame_040.png' else np.zeros((135,240,2))
 errors=np.linalg.norm(result.displacement[valid]-expected[valid],axis=-1)
 rmse=float(np.sqrt(np.mean(errors**2)));coverage=float(valid.sum()/interior.sum())
 assert coverage>=limits['affine_frame_040_minimum_accepted_fraction']
 assert rmse<(limits['affine_frame_040_displacement_rmse_px'] if name=='frame_040.png' else limits['identity_displacement_rmse_px'])
 p95=float(np.percentile(errors,95));assert p95<limits['affine_frame_040_displacement_p95_px']
 gradient=result.parameters[..., [1,2,4,5]][valid]-([ex,0,0,ey] if name=='frame_040.png' else [0,0,0,0])
 gradient_rmse=float(np.sqrt(np.mean(gradient**2)));assert gradient_rmse<limits['affine_frame_040_median_gradient_rmse_per_component']
 result.save_npz(output)
 with np.load(output,allow_pickle=False) as stored:
  assert np.array_equal(stored['parameters'],result.parameters) and np.array_equal(stored['status'],result.status) and np.array_equal(stored['iterations'],result.iterations)
  json.loads(str(stored['config_json']))
 return {'image':name,'coverage':coverage,'rmsePx':rmse,'p95Px':p95,'gradientRmse':gradient_rmse,'npzExact':True}

with tempfile.TemporaryDirectory(prefix='fixed-first-fields-') as temporary:
 with ICGNWGPUFullPyramid(device_type='gpu',outlier_policy='median') as engine:
  engine.set_reference(images['reference.png'])
  for index,name in enumerate(sequence):report['native'].append(check(engine.register(images[name]),name,Path(temporary)/f'native-{index}.npz'))
 site=Path('/tmp/live-dic-fixed-first-site');site.mkdir(exist_ok=True)
 import shutil
 shutil.copytree('/tmp/live-dic-cadence-site',site,dirs_exist_ok=True)
 for name,pixels in images.items():(site/(name+'.gray')).write_bytes(pixels.tobytes())
 server=_StaticServer(site.parent);server.start()
 try:
  with ICGNWebGL(backend='webgpu-wgsl',startup_timeout=60,call_timeout=240) as browser:
   url=f'http://127.0.0.1:{server.port}/{site.name}/?smoke=1&case=plate';browser._evaluate(f'location.assign({json.dumps(url)});true');wait(browser,'globalThis.liveDicStatus?.completed>=1',timeout=180)
   records=browser._evaluate("""(async()=>{
    const {DICEngine}=await import('./web/core/engine.js');
    const reference=new Uint8Array(await fetch('./reference.png.gray').then(r=>r.arrayBuffer())),target=new Uint8Array(await fetch('./frame_040.png.gray').then(r=>r.arrayBuffer()));
    function encode(values){const view=new DataView(new ArrayBuffer(values.length*4));for(let i=0;i<values.length;i++)view.setFloat32(i*4,values[i],true);const bytes=new Uint8Array(view.buffer),chunks=[];for(let i=0;i<bytes.length;i+=32768)chunks.push(String.fromCharCode(...bytes.subarray(i,i+32768)));return btoa(chunks.join(''));}
    const output=[];
    for(const referencePreparation of ['cpu','gpu']){
      const engine=await DICEngine.create({width:3840,height:2160,motionModel:'affine',interpolation:'bspline',outlierPolicy:'median',referencePreparation,readback:'full'});
      let prepares=0;const prepare=engine.host.setReference.bind(engine.host);engine.host.setReference=(...args)=>{prepares++;return prepare(...args);};
      try{
        await engine.setReference(reference);const metadata=engine.host.memoryResources().levels.map(level=>level.buffers.metadata);
        async function snapshot(){const hashes=[];for(const buffer of metadata){const copy=engine.device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const encoder=engine.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,copy,0,buffer.size);engine.device.queue.submit([encoder.finish()]);await copy.mapAsync(GPUMapMode.READ);hashes.push(Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',copy.getMappedRange()))).map(v=>v.toString(16).padStart(2,'0')).join(''));copy.unmap();copy.destroy();}return hashes;}
        const before=await snapshot(),fields=[];
        for(const input of [reference,target,reference,target]){const field=await engine.register(input);fields.push({...field,parameters:encode(field.parameters),status:encode(field.status),iterations:encode(field.iterations)});}
        const after=await snapshot();if(prepares!==1||JSON.stringify(before)!==JSON.stringify(after))throw new Error('Reference cache changed');
        output.push({referencePreparation,prepares,before,after,fields});
      }finally{engine.destroy();}
    }return output;
   })()""")
   for case in records:
    scores=[]
    for index,(name,field) in enumerate(zip(sequence,case.pop('fields'),strict=True)):
     shape=(field['grid'][1],field['grid'][0]);arrays={key:np.frombuffer(base64.b64decode(field[key]),dtype='<f4').reshape(shape+((6,) if key=='parameters' else ())) for key in ['parameters','status','iterations']}
     result=RegistrationResult(**arrays,accepted=field['accepted'],replaced=field['replaced'],mean_translation=tuple(field['meanTranslation']),mean_iterations=field['meanIterations'],timing=field['timing'],config=field['config'],width=3840,height=2160)
     scores.append(check(result,name,Path(temporary)/f'browser-{case["referencePreparation"]}-{index}.npz'))
    report['browser'].append({**case,'fields':scores})
 finally:server.close()
Path('/tmp/live-dic-fixed-first-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
