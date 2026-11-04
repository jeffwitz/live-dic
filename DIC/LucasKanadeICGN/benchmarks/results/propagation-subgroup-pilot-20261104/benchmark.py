import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import wgpu.backends.wgpu_native._api as api
sys.path.insert(0,str(Path("DIC/LucasKanadeICGN/benchmarks").resolve()))
from gpu_spatial_seed import GpuSpatialPyramid
real_request=api.GPUAdapter.request_device_sync
real_shader=api.GPUDevice.create_shader_module
source=Path("/tmp/live-dic-night-spatial-subgroup.wgsl").read_text()
def request(self,**kwargs):
 features=set(kwargs.pop("required_features",[]));features.add("subgroup")
 return real_request(self,required_features=list(features),**kwargs)
def shader(self,**kwargs):
 if "Experimental translation seed selection" in kwargs.get("code",""):kwargs["code"]=source
 return real_shader(self,**kwargs)

import json,time,hashlib,csv
from PIL import Image
root=Path("DIC/LucasKanadeICGN/assets/uhd_tensile")
f,g=[np.array(Image.open(root/name)) for name in ("reference.png","frame_040.png")]
engines=[];rows=[];fields=[]
try:
 with patch.object(api.GPUAdapter,"request_device_sync",request):
  for subgroup in (False,True):
   e=GpuSpatialPyramid(strategy="spatial",failed_only=True,passes=2,subset_size=32,subset_stride=16,device_type="gpu",tolerance_per_level=(.005,.005));engines.append(e)
   if subgroup:
    with patch.object(api.GPUDevice,"create_shader_module",shader):e.prepare_research(f)
   else:e.prepare_research(f)
   for _ in range(8):r=e.register(g)
   fields.append(r);rows.append(dict(subgroup=subgroup,accepted=r.accepted,samplesByRoundMs=[]))
   print(subgroup,r.accepted,flush=True)
 np.testing.assert_array_equal(fields[0].status,fields[1].status)
 delta=float(np.max(np.abs(fields[0].parameters-fields[1].parameters)))
 for block in range(6):
  for i in (range(2) if block%2==0 else reversed(range(2))):
   for _ in range(8):engines[i].register(g)
   samples=[]
   for _ in range(60):
    t=time.perf_counter();r=engines[i].register(g);samples.append(1000*(time.perf_counter()-t))
   rows[i]["samplesByRoundMs"].append(samples);print(block,i,np.median(samples),flush=True)
 for row in rows:
  samples=np.array(row["samplesByRoundMs"]).ravel();row.update(medianMs=float(np.median(samples)),p95Ms=float(np.percentile(samples,95)),fps=float(1000/np.mean(samples)))
 Path("/tmp/live-dic-night-subgroup-pilot.json").write_text(json.dumps(dict(adapter=engines[0].adapter_info,rows=rows,parameterMaxAbs=delta,shaderSha256=hashlib.sha256(source.encode()).hexdigest()),indent=2))
finally:
 for e in engines:e.close()
