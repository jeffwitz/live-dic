import { presentationSize } from '../ui/supervision.js';

export const VECTOR_COLOURS = [[21, 57, 99], [31, 132, 138], [241, 206, 95]];

/** Same sequential palette for the GPU glyphs and the HTML colour bar. */
export function vectorColour(fraction) {
  const position = Math.max(0, Math.min(1, fraction)) * 2;
  const index = Math.min(1, Math.floor(position)), weight = position - index;
  return VECTOR_COLOURS[index].map((value, channel) => Math.round(value + weight * (VECTOR_COLOURS[index + 1][channel] - value)));
}

function roundedMaximum(value) {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value)), fraction = value / power;
  return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power;
}

/** Presentation values only: never normalize or modify the measured field. */
export function vectorAppearance(result, { scale = null, arrowStep = null, arrowGain = null, previewMaxWidth = 0 } = {}) {
  // A monitoring overview needs a few hundred glyphs, not thousands of tiny
  // arrows. Keep small fields dense and let users select a fixed step instead.
  arrowStep ??= Math.max(1, Math.ceil(result.grid[0] / 20), Math.ceil(result.grid[1] / 12));
  if (!Number.isInteger(arrowStep) || arrowStep < 1) throw new Error('Vector spacing must be a positive subset count');
  if (scale !== null && (!Number.isFinite(scale) || scale <= 0)) throw new Error('Vector colour maximum must be positive');
  if (arrowGain !== null && (!Number.isFinite(arrowGain) || arrowGain <= 0)) throw new Error('Vector magnification must be positive or automatic');
  const [viewWidth, viewHeight] = presentationSize(result.width, result.height, previewMaxWidth);
  const measuredMaximum = result.maxAbs?.[2] ?? 1;
  const maximumPx = scale ?? roundedMaximum(measuredMaximum);
  // Fit ordinary vectors within part of a sampled cell. The one-pixel floor
  // prevents floating-point noise in an identity field becoming long arrows.
  const gain = arrowGain ?? .4 * result.subsetStride * arrowStep / Math.max(1, measuredMaximum);
  return { maximumPx, gain, arrowStep, viewWidth, viewHeight,
    imageWidth: result.width, imageHeight: result.height,
    instances: Math.ceil(result.grid[0] / arrowStep) * Math.ceil(result.grid[1] / arrowStep),
    tipAnchor: 'Current subset centre: reference centre plus measured displacement',
    unit: 'image px', colours: VECTOR_COLOURS, accepted: result.accepted ?? null };
}

/** Lightweight live view: image copy plus sparse instanced vector geometry.
 * The background fragment never reads a field value. No scalar raster,
 * displacement interpolation, strain calculation or CPU field readback runs here.
 */
export class LiveVectorRenderer {
  constructor(device, canvas) {
    this.device = device; this.canvas = canvas;
    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('This canvas cannot present WebGPU');
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format, alphaMode: 'opaque' });
    this.config = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.backgroundGroups = new WeakMap(); this.vectorGroups = new WeakMap(); this.legend = null;
    const image = device.createShaderModule({ label: 'Live image without field sampling', code: `
      struct Config { width:u32,height:u32,gx:u32,gy:u32,stride:u32,parameters:u32,step:u32,gray:u32,
        subset:f32,spacing:f32,maximum:f32,gain:f32,view_width:f32,view_height:f32,opacity:f32,padding:f32 }
      @group(0) @binding(1) var image:texture_2d<f32>;
      @group(0) @binding(2) var<uniform> config:Config;
      @vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
        let x=f32((i<<1u)&2u);let y=f32(i&2u);return vec4<f32>(x*2.0-1.0,y*2.0-1.0,0,1);
      }
      @fragment fn background(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32> {
        let xy=vec2<i32>(p.xy*vec2<f32>(f32(config.width)/config.view_width,f32(config.height)/config.view_height));
        let value=textureLoad(image,clamp(xy,vec2<i32>(0),vec2<i32>(i32(config.width)-1,i32(config.height)-1)),0);
        return vec4<f32>(select(value.rgb,vec3<f32>(value.r),config.gray!=0u),1);
      }
    ` });
    this.background = device.createRenderPipeline({ label: 'Live current image', layout: 'auto',
      vertex: { module: image, entryPoint: 'vertex' }, fragment: { module: image, entryPoint: 'background', targets: [{ format }] },
      primitive: { topology: 'triangle-list' } });
    const vectors = device.createShaderModule({ label: 'Sparse norm-coloured live vectors', code: `
      struct Field { values:array<f32> }
      struct Config { width:u32,height:u32,gx:u32,gy:u32,stride:u32,parameters:u32,step:u32,gray:u32,
        subset:f32,spacing:f32,maximum:f32,gain:f32,view_width:f32,view_height:f32,opacity:f32,padding:f32 }
      @group(0) @binding(0) var<storage,read> field:Field;
      @group(0) @binding(2) var<uniform> config:Config;
      struct Vertex { @builtin(position) position:vec4<f32>, @location(0) colour:vec4<f32> }
      fn colour(norm:f32)->vec3<f32> {
        let t=clamp(norm/max(config.maximum,1e-8),0.0,1.0)*2.0;
        let low=vec3<f32>(21,57,99)/255.0;let middle=vec3<f32>(31,132,138)/255.0;let high=vec3<f32>(241,206,95)/255.0;
        return select(mix(low,middle,t),mix(middle,high,t-1.0),t>=1.0);
      }
      @vertex fn glyph(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->Vertex {
        let columns=(config.gx+config.step-1u)/config.step;
        let ix=min(config.gx-1u,(instance%columns)*config.step+config.step/2u);
        let iy=min(config.gy-1u,(instance/columns)*config.step+config.step/2u);
        let base=(iy*config.gx+ix)*config.stride;
        let status=field.values[base+config.parameters];
        var output:Vertex;output.position=vec4<f32>(-2,-2,0,1);output.colour=vec4<f32>(0);
        if status == -3.0 {return output;}
        let factor=vec2<f32>(config.view_width/f32(config.width),config.view_height/f32(config.height));
        let centre=(vec2<f32>(f32(ix),f32(iy))*config.spacing+vec2<f32>((config.subset-1.0)/2.0));
        let corners=array<vec2<f32>,6>(vec2<f32>(0,-1),vec2<f32>(1,-1),vec2<f32>(1,1),vec2<f32>(0,-1),vec2<f32>(1,1),vec2<f32>(0,1));
        var point=centre*factor;
        if status < 0.5 {
          // An unavailable requested subset has no reliable deformed centre.
          // Mark its reference centre, never manufacture a displacement.
          let axis=select(vec2<f32>(.7071,.7071),vec2<f32>(.7071,-.7071),vertex>=6u);
          let normal=vec2<f32>(-axis.y,axis.x);let corner=corners[vertex%6u];
          point+=(corner.x*2.0-1.0)*axis*3.0+normal*corner.y*.65;
          output.colour=vec4<f32>(.63,.67,.70,config.opacity);
        } else {
          let displacement=vec2<f32>(field.values[base],field.values[base+3u]);
          let end=(centre+displacement)*factor;
          let delta=displacement*config.gain*factor;
          let distance=length(delta);output.colour=vec4<f32>(colour(length(displacement)),config.opacity);
          if distance < 1.0 {
            // A short measured vector is a dot; its colour still reports norm.
            point=end;
            if vertex<6u {let c=corners[vertex];point+=vec2<f32>(c.x*2.0-1.0,c.y)*1.25;}
          } else {
            let direction=delta/distance;let normal=vec2<f32>(-direction.y,direction.x);
            let head=min(distance*.4,12.0);let start=end-delta;let neck=end-direction*head;
            if vertex<6u {let c=corners[vertex];point=mix(start,neck,c.x)+normal*c.y*2.0;}
            else if vertex<9u {
              let points=array<vec2<f32>,3>(end,neck+normal*head*.5,neck-normal*head*.5);point=points[vertex-6u];
            } else {point=start;}
          }
        }
        output.position=vec4<f32>(point.x/config.view_width*2.0-1.0,1.0-point.y/config.view_height*2.0,0,1);
        return output;
      }
      @fragment fn ink(input:Vertex)->@location(0) vec4<f32> {return input.colour;}
    ` });
    this.vectors = device.createRenderPipeline({ label: 'Live sparse displacement glyphs', layout: 'auto',
      vertex: { module: vectors, entryPoint: 'glyph' }, fragment: { module: vectors, entryPoint: 'ink',
        targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] }, primitive: { topology: 'triangle-list' } });
  }
  draw(engine, result, { mode = 2, scale = null, opacity = 1, imageTexture = null, arrowStep = null, arrowGain = null, previewMaxWidth = 0 } = {}) {
    if (![2,8].includes(mode)) throw new Error('Live view supports norm-coloured vectors or the current image; use Results for scalar fields');
    if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1) throw new Error('Vector opacity must be 0..1');
    const appearance = vectorAppearance(result, { scale, arrowStep, arrowGain, previewMaxWidth });
    const { viewWidth, viewHeight, maximumPx, gain } = appearance;
    if (this.canvas.width !== viewWidth) this.canvas.width = viewWidth;
    if (this.canvas.height !== viewHeight) this.canvas.height = viewHeight;
    const texture = imageTexture ?? engine.imageTexture;
    const bytes = new ArrayBuffer(64), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    u.set([result.width,result.height,...result.grid,result.config.motionModel==='quadratic'?16:8,
      result.config.motionModel==='quadratic'?12:6,appearance.arrowStep,+(texture.format==='r8unorm')]);
    f.set([result.subsetSize,result.subsetStride,maximumPx,gain,viewWidth,viewHeight,opacity,0],8);
    this.device.queue.writeBuffer(this.config,0,bytes);
    let background = this.backgroundGroups.get(texture);
    if (!background) { background = this.device.createBindGroup({ layout: this.background.getBindGroupLayout(0), entries: [
      { binding:1,resource:texture.createView() },{ binding:2,resource:{buffer:this.config} }] });this.backgroundGroups.set(texture,background); }
    const encoder=this.device.createCommandEncoder({label:'Live image and sparse vectors'}),pass=encoder.beginRenderPass({colorAttachments:[
      {view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:'clear',storeOp:'store'}]});
    pass.setPipeline(this.background);pass.setBindGroup(0,background);pass.draw(3);
    const visible=mode===2&&!imageTexture;
    if (visible) {
      let group=this.vectorGroups.get(engine.fieldBuffer);
      if(!group){group=this.device.createBindGroup({layout:this.vectors.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:engine.fieldBuffer}},{binding:2,resource:{buffer:this.config}}]});this.vectorGroups.set(engine.fieldBuffer,group);}
      pass.setPipeline(this.vectors);pass.setBindGroup(0,group);pass.draw(12,appearance.instances);
    }
    pass.end();this.device.queue.submit([encoder.finish()]);
    this.legend={...appearance,imageOnly:!visible};
    return maximumPx;
  }
  destroy(){this.config.destroy();this.context.unconfigure();}
}
