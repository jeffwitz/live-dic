"""Check actual live glyph pixels and UHD current-image presentation."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib.parse import quote

from qualify_local_sources import wait

from lucas_kanade_icgn import ICGNWebGL
from lucas_kanade_icgn.engine import _StaticServer


def qualify_renderer(browser):
    """Use independent pixel bounds, colours and readback, not copied shader math."""
    result = browser._evaluate("""(async()=>{
      const {LiveVectorRenderer}=await import('./render/live-vectors.js');
      const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice();
      const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
      const canvas=new OffscreenCanvas(128,128),renderer=new LiveVectorRenderer(device,canvas);
      const values=new Float32Array(4*8);for(let i=0;i<4;i++)values.set([8,0,0,0,0,0,-3,3],i*8);values[6]=1;
      const buffer=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
      device.queue.writeBuffer(buffer,0,values);
      const texture=device.createTexture({size:[128,128],format:'rgba8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      const pixels=new Uint8Array(128*128*4);for(let i=0;i<128*128;i++)pixels.set([73,73,73,255],i*4);
      pixels.set([30,40,50,255],(100*128+100)*4);device.queue.writeTexture({texture},pixels,{bytesPerRow:512},[128,128]);
      const reference=device.createTexture({size:[128,128],format:'rgba8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      const referencePixels=new Uint8Array(128*128*4);for(let i=0;i<128*128;i++)referencePixels.set([111,111,111,255],i*4);
      device.queue.writeTexture({texture:reference},referencePixels,{bytesPerRow:512},[128,128]);
      const engine={fieldBuffer:buffer,imageTexture:texture};
      const field={width:128,height:128,grid:[2,2],subsetSize:64,subsetStride:64,config:{motionModel:'affine'},maxAbs:[8,0,8]};
      async function draw(options={}){
        renderer.draw(engine,field,{arrowStep:1,arrowGain:1,scale:16,...options});await device.queue.onSubmittedWorkDone();
        const bitmap=canvas.transferToImageBitmap(),read=new OffscreenCanvas(canvas.width,canvas.height),ctx=read.getContext('2d');
        ctx.drawImage(bitmap,0,0);bitmap.close();return ctx.getImageData(0,0,canvas.width,canvas.height).data;
      }
      const background=await draw({mode:8}),normal=await draw(),amplified=await draw({arrowGain:3});
      const outlined=await draw({vectorOutline:true});
      function bounds(data){let x0=128,y0=128,x1=-1,y1=-1,count=0;const colours=new Set();
        for(let y=0;y<128;y++)for(let x=0;x<128;x++){const i=(y*128+x)*4;
          if(data[i]!==background[i]||data[i+1]!==background[i+1]||data[i+2]!==background[i+2]){
            x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);count++;colours.add([...data.slice(i,i+3)].join(','));}}
        return {x0,y0,x1,y1,count,colours:[...colours]};}
      const a=bounds(normal),b=bounds(amplified),outlinedBounds=bounds(outlined),sentinel=[...amplified.slice((100*128+100)*4,(100*128+100)*4+4)];
      const read=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,read,0,values.byteLength);device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);const unchanged=new Float32Array(read.getMappedRange()).every((v,i)=>v===values[i]);read.unmap();
      // Step two samples the centre of its block (bottom-right subset here),
      // not the unavailable upper-left subset. Check actual raster coordinates.
      values[6]=-3;values[30]=1;device.queue.writeBuffer(buffer,0,values);
      const blockCentre=bounds(await draw({arrowStep:2}));values[30]=-3;
      values[6]=-3;device.queue.writeBuffer(buffer,0,values);const excluded=await draw();
      values[6]=0;device.queue.writeBuffer(buffer,0,values);const unavailable=bounds(await draw());
      values[6]=1;values[0]=0;device.queue.writeBuffer(buffer,0,values);const zero=bounds(await draw());
      const referenceView=await draw({imageTexture:reference}),referenceOnly=renderer.legend.imageOnly;
      await draw({mode:8,previewMaxWidth:64});const preview=[canvas.width,canvas.height];
      const report={adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,
          device:adapter.info.device,description:adapter.info.description},
        errors,normal:a,amplified:b,outlined:outlinedBounds,blockCentre,sentinel,unchanged,
        backgroundExact:background.every((v,i)=>v===pixels[i]),roiExcludedExact:excluded.every((v,i)=>v===background[i]),
        unavailable,zero,referenceOnly,referenceExact:referenceView.every((v,i)=>v===referencePixels[i]),preview,grid:field.grid};
      renderer.destroy();renderer.destroy();let closedRefused=false;
      try{renderer.draw(engine,field);}catch(error){closedRefused=error.message.includes('closed');}
      report.closedRefused=closedRefused;
      read.destroy();buffer.destroy();texture.destroy();reference.destroy();device.destroy();return report;
    })()""")
    assert not result["errors"], result
    assert result["backgroundExact"] and result["sentinel"] == [30, 40, 50, 255], result
    a, b = result["normal"], result["amplified"]
    assert a["count"] > 10 and b["count"] > a["count"], result
    assert a["x0"] - b["x0"] == 16 and a["x1"] == b["x1"], result
    assert a["x1"] in [38, 39] and a["y0"] >= 29 and a["y1"] <= 33, result
    assert a["colours"] == b["colours"] == ["31,132,138"], result
    outlined = result["outlined"]
    assert outlined["count"] > a["count"], result
    assert {"31,132,138", "6,6,6", "250,250,250"}.issubset(outlined["colours"]), result
    assert outlined["x1"] == a["x1"], result  # Same measured tip/length, added lateral contrast.
    assert result["unchanged"] and result["roiExcludedExact"], result
    assert result["blockCentre"]["x0"] == 95 and result["blockCentre"]["x1"] == 102, result
    assert 93 <= result["blockCentre"]["y0"] <= result["blockCentre"]["y1"] <= 98, result
    assert result["unavailable"]["count"] > 0, result
    assert result["zero"]["colours"] == ["21,57,99"], result
    assert result["referenceOnly"] and result["referenceExact"], result
    assert result["preview"] == [64, 64] and result["grid"] == [2, 2], result
    assert result["closedRefused"], result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    site = args.site.resolve()
    server = _StaticServer(site.parent)
    server.start()
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
    try:
        with ICGNWebGL(backend="webgpu-wgsl", chromium_args=flags, call_timeout=240) as browser:
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=plate')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.completed>=1")
            report = dict(
                software=args.software,
                renderer=qualify_renderer(browser),
                build=json.loads((site / "build.json").read_text()),
            )
            report["uhd"] = browser._evaluate("""({completed:liveDicStatus.completed,
              score:liveDicStatus.results.at(-1).plateScore,canvas:[document.getElementById('field').width,document.getElementById('field').height],
              colourbar:document.getElementById('high').textContent,lengthKey:document.getElementById('vector-key-label').textContent,
              keyHidden:document.getElementById('live-vector-key').hidden,workbench:liveDicWorkbench.report(),errors:liveDicStatus.errors})""")
            assert report["uhd"]["canvas"] == [1920, 1080], report["uhd"]
            reference_before = browser._evaluate(
                "JSON.parse(document.getElementById('details').textContent).reference"
            )
            completed_before = browser._evaluate("liveDicStatus.completed")
            previews = []
            for outlined in (False, True):
                data_url = browser._evaluate(f"""(async()=>{{
                  const control=document.getElementById('vector-outline');
                  control.checked={str(outlined).lower()};await control.onchange();
                  const original=document.getElementById('field'),crop=document.createElement('canvas');
                  crop.width=640;crop.height=360;crop.getContext('2d').drawImage(original,480,300,640,360,0,0,640,360);
                  return crop.toDataURL('image/png');
                }})()""")
                path = args.report.with_name(
                    args.report.stem + ("-outline.png" if outlined else "-plain.png")
                )
                path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
                previews.append(path.name)
            assert browser._evaluate("liveDicStatus.completed") == completed_before
            assert (
                browser._evaluate(
                    "JSON.parse(document.getElementById('details').textContent).reference"
                )
                == reference_before
            )
            report["previews"] = previews
            assert (
                not report["uhd"]["keyHidden"] and "Tips mark current" in report["uhd"]["lengthKey"]
            )
            assert report["uhd"]["workbench"]["analysisRuns"] == 0
            assert report["uhd"]["score"]["displacementRmsePx"] < 0.05
            assert not report["uhd"]["errors"], report["uhd"]
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(report, indent=2, allow_nan=False))
    finally:
        server.close()


if __name__ == "__main__":
    main()
