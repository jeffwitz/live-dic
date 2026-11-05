"""Qualify opt-in browser GPU reference updates on original analytical UHD data.

Measures CPU-array input to a materialized JS typed field, with a new reference
on each call. This excludes Python transport, camera capture and recording.
No fixture is regenerated, resized or downloaded. Small crop is optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from qualify_local_sources import wait

from live_dic import ICGNBrowser
from lucas_kanade_icgn.engine import _StaticServer

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "DIC/LucasKanadeICGN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--software", action="store_true")
    parser.add_argument("--motion-model", choices=["affine", "translation"], default="affine")
    parser.add_argument("--resolution", choices=["3840x2160", "1920x1080"], default="3840x2160")
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 3:
        parser.error("nonnegative warmup and at least three repetitions required")
    fixture = PROJECT / "assets/uhd_tensile"
    manifest = json.loads((fixture / "manifest.json").read_text())
    for record in manifest["images"]:
        if hashlib.sha256((fixture / record["file"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Original fixture hash mismatch: {record['file']}")
    flags = (
        (
            "--use-angle=swiftshader",
            "--enable-unsafe-webgpu",
            "--use-webgpu-adapter=swiftshader",
            "--use-vulkan=swiftshader",
        )
        if args.software
        else ()
    )
    server = _StaticServer(ROOT)
    server.start()
    try:
        with ICGNBrowser(
            backend="webgpu-wgsl", startup_timeout=60, call_timeout=600, chromium_args=flags
        ) as browser:
            url = f"http://127.0.0.1:{server.port}/"
            browser._evaluate(f"location.assign({json.dumps(url)});true")
            wait(browser, f"location.href==={json.dumps(url)}")
            options = dict(
                warmup=args.warmup,
                repetitions=args.repetitions,
                resolution=args.resolution,
                motionModel=args.motion_model,
            )
            script = r"""(async()=>{
              const request=__OPTIONS__;
              if(!['translation','affine'].includes(request.motionModel))throw new Error('Explicit qualification motion model required');
              const {DICEngine}=await import('/DIC/LucasKanadeICGN/python/src/lucas_kanade_icgn/web/core/engine.js');
              const width=request.resolution==='3840x2160'?3840:1920,height=width===3840?2160:1080;
              const load=async name=>{
                const blob=await(await fetch('/DIC/LucasKanadeICGN/assets/uhd_tensile/'+name)).blob();
                const image=await createImageBitmap(blob),canvas=new OffscreenCanvas(width,height),context=canvas.getContext('2d',{willReadFrequently:true});
                // Native pixels; never scale the fixture for 1080p.
                context.drawImage(image,width===3840?0:-900,width===3840?0:-500);image.close();
                const rgba=context.getImageData(0,0,width,height).data;
                return Uint8Array.from({length:width*height},(_,i)=>rgba[i*4]);
              };
              const reference=await load('reference.png');
              let target=await load('frame_040.png');
              if(request.motionModel==='translation') {
                target=Uint8Array.from({length:reference.length},(_,i)=>reference[((Math.floor(i/width)+1)%height)*width+(i%width+width-2)%width]);
              }
              const metadata=await(await fetch('/DIC/LucasKanadeICGN/assets/uhd_tensile/manifest.json')).json();
              const truthText=await(await fetch('/DIC/LucasKanadeICGN/assets/uhd_tensile/machine_truth.csv')).text();
              const lines=truthText.trim().split(/\r?\n/),keys=lines[0].split(','),truthRow=Object.fromEntries(keys.map((key,i)=>[key,lines[41].split(',')[i]]));
              const ex=Number(truthRow.stretch_x)-1,ey=Number(truthRow.stretch_y)-1;
              const engineOptions={width,height,pyramidLevels:2,motionModel:request.motionModel,subsetSize:16,subsetStride:16,interpolation:'bspline',outlierPolicy:'median',readback:'full'};
              const engines={cpu:await DICEngine.create(engineOptions),gpu:await DICEngine.create({...engineOptions,referencePreparation:'gpu'})};
              const costs={},fields={},cold={};
              const run=async mode=>{
                const t=performance.now();
                if(mode==='cpu'){await engines.cpu.setReference(reference);fields.cpu=await engines.cpu.register(target);}
                else fields.gpu=await engines.gpu.registerPair(reference,target);
                return performance.now()-t;
              };
              try {
                for(const mode of ['cpu','gpu']){const t=performance.now();await engines[mode].setReference(reference);cold[mode]=performance.now()-t;costs[mode]=[];}
                for(let i=0;i<request.warmup;i++)for(const mode of ['cpu','gpu'])await run(mode);
                // Alternate order in every pair to limit systematic drift.
                for(let i=0;i<request.repetitions;i++)for(const mode of (i%2?['gpu','cpu']:['cpu','gpu']))costs[mode].push(await run(mode));
                const quantile=(values,q)=>{const a=[...values].sort((a,b)=>a-b),i=(a.length-1)*q,lo=Math.floor(i),hi=Math.ceil(i);return a[lo]+(a[hi]-a[lo])*(i-lo);};
                const timings=Object.fromEntries(Object.entries(costs).map(([key,values])=>[key,{samplesMs:values,medianMs:quantile(values,.5),p95Ms:quantile(values,.95),p99Ms:quantile(values,.99),throughputFps:1000*values.length/values.reduce((a,b)=>a+b,0)}]));
                const a=fields.cpu,b=fields.gpu;let common=0,statusMismatch=0,sum=0,max=0,grad=0,finite=true;
                for(let i=0;i<a.status.length;i++){
                  statusMismatch+=+(a.status[i]!==b.status[i]);
                  for(let k=0;k<6;k++)finite&&=Number.isFinite(b.parameters[6*i+k]);
                  if(a.status[i]>0&&b.status[i]>0){common++;for(let k=0;k<6;k++){
                    const e=b.parameters[6*i+k]-a.parameters[6*i+k];max=Math.max(max,Math.abs(e));
                    if(k===0||k===3)sum+=e*e;else grad+=e*e;
                  }}
                }
                if(!finite||common<.9*Math.min(a.accepted,b.accepted)||Math.sqrt(sum/(2*common))>=.005||Math.sqrt(grad/(4*common))>=.0005)throw new Error('GPU/CPU field parity failed');
                const analytical={};
                for(const [mode,result] of Object.entries(fields)) {
                  const errors=[],gradientErrors=[];let population=0,accepted=0;
                  for(let i=0;i<result.status.length;i++) {
                    const x=(i%result.grid[0])*16+7.5,y=Math.floor(i/result.grid[0])*16+7.5;
                    if(x<32||x>=width-32||y<32||y>=height-32)continue;population++;
                    if(result.status[i]<=0)continue;accepted++;
                    const gx=x+(width===3840?0:900),gy=y+(width===3840?0:500),p=result.parameters.subarray(6*i,6*i+6);
                    const truth=request.motionModel==='translation'?[2,0,0,-1,0,0]:[ex*(gx-1920),ex,0,ey*(gy-1080),0,ey];
                    errors.push(Math.hypot(p[0]-truth[0],p[3]-truth[3]));
                    gradientErrors.push((p[1]-truth[1])**2+p[2]**2+p[4]**2+(p[5]-truth[5])**2);
                  }
                  const rmse=Math.sqrt(errors.reduce((a,b)=>a+b*b,0)/errors.length),p95=quantile(errors,.95),gradientRmse=Math.sqrt(gradientErrors.reduce((a,b)=>a+b,0)/(4*gradientErrors.length));
                  if(accepted/population<.9||rmse>=.05||p95>=.1||gradientRmse>=.002)throw new Error(`Analytical truth failed ${mode}: ${accepted}/${population}/${rmse}/${p95}/${gradientRmse}`);
                  analytical[mode]={population,accepted,coverage:accepted/population,displacementRmse:rmse,displacementP95:p95,gradientRmse};
                }
                return {boundary:'Resident Uint8Array pair -> materialized JS field; CPU includes setReference per pair; GPU reuses resources. No Python transport/capture/recording.',
                  browser:navigator.userAgent,adapter:await engines.gpu.host.info(),engineOptions,request,coldPreparationMs:cold,timings,
                  quality:{finite,cpuAccepted:a.accepted,gpuAccepted:b.accepted,commonAccepted:common,statusMismatch,displacementDifferenceRmse:Math.sqrt(sum/(2*common)),gradientDifferenceRmse:Math.sqrt(grad/(4*common)),maxParameterDifference:max},
                  analytical,gpuMemory:engines.gpu.memoryUsage(),fixtureMetadata:metadata,targetRecipe:request.motionModel==='translation'?'Original reference rolled (+2,-1) pixels; interior truth only':'Original frame_040.png'};
              }finally{engines.cpu.destroy();engines.gpu.destroy();}
            })()""".replace("__OPTIONS__", json.dumps(options))
            report = browser._evaluate(script)
    finally:
        server.close()
    report["source"] = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [
                PROJECT / "python/src/lucas_kanade_icgn/web/core",
                PROJECT / "python/src/lucas_kanade_icgn/shaders",
            ]
            for p in sorted(folder.glob("*"))
            if p.is_file()
        },
    )
    report["command"] = [sys.executable, *sys.argv]
    report["scriptSha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["softwareRequested"] = args.software
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"timings": report["timings"], "quality": report["quality"]}, indent=2))


if __name__ == "__main__":
    main()
