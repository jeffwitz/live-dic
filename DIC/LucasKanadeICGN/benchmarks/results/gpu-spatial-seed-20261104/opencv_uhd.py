import json,time,hashlib,platform
from pathlib import Path
import cv2,numpy as np
from PIL import Image
root=Path("DIC/LucasKanadeICGN/assets/uhd_tensile");out=Path("/tmp/live-dic-gpu-seed-20261104")
f=np.array(Image.open(root/"reference.png"));g=np.array(Image.open(root/"frame_040.png"))
import csv
with (root/"machine_truth.csv").open() as z:truth=next(r for r in csv.DictReader(z) if int(r["frame"])==40)
y,x=np.mgrid[0:134,0:239];x=x*16+15;y=y*16+15;pop=(x+.5>=32)&(x+.5<3840-32)&(y+.5>=32)&(y+.5<2160-32)
expected=np.stack(((float(truth["stretch_x"])-1)*(x+.5-1920),(float(truth["stretch_y"])-1)*(y+.5-1080)),axis=-1)
cv2.setNumThreads(8);cv2.ocl.setUseOpenCL(False)
rows=[]
for name in ("ultrafast","fast","medium","user-4-2","matched-32-16-no-var","matched-32-16-var10"):
 preset={"ultrafast":cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,"fast":cv2.DISOPTICAL_FLOW_PRESET_FAST,"medium":cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}.get(name)
 d=cv2.DISOpticalFlow_create(preset) if preset is not None else cv2.DISOpticalFlow_create()
 if preset is None:
  d.setFinestScale(0);d.setPatchSize(4 if name=="user-4-2" else 32);d.setPatchStride(2 if name=="user-4-2" else 16)
  d.setVariationalRefinementAlpha(5);d.setVariationalRefinementGamma(1);d.setVariationalRefinementDelta(0);d.setVariationalRefinementEpsilon(.3)
  d.setVariationalRefinementIterations(0 if name.endswith("no-var") else 10)
  if name!="user-4-2":d.setCoarsestScale(1)
 options={key:getattr(d,"get"+key)() for key in ("FinestScale","CoarsestScale","PatchSize","PatchStride","GradientDescentIterations","VariationalRefinementIterations","UseSpatialPropagation","UseMeanNormalization")}
 print("START",name,options,flush=True)
 d.calc(f,g,None)
 samples=[]
 for _ in range(5):
  t=time.perf_counter();flow=d.calc(f,g,None);samples.append((time.perf_counter()-t)*1000)
 values=(flow[y,x]+flow[y,x+1]+flow[y+1,x]+flow[y+1,x+1])*.25
 error=np.linalg.norm(values[pop]-expected[pop],axis=-1)
 row=dict(name=name,options=options,timesMs=samples,medianMs=float(np.median(samples)),p95Ms=float(np.percentile(samples,95)),fps=float(5000/sum(samples)),interiorPopulation=int(pop.sum()),vectorRmse=float(np.sqrt(np.mean(error**2))),p95Error=float(np.percentile(error,95)))
 rows.append(row);print(json.dumps(row),flush=True);d.collectGarbage();del d,flow
 (out/"opencv-uhd.json").write_text(json.dumps(dict(opencv=cv2.__version__,threads=cv2.getNumThreads(),opencl=cv2.ocl.useOpenCL(),platform=platform.platform(),scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2,allow_nan=False))
