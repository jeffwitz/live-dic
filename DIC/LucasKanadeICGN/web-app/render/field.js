/** Plane-strain equivalent strain, matching the archived plate convention. */
export const equivalentStrain = (exx, eyy, exy) => 2 / 3 * Math.sqrt(Math.max(0, exx * exx - exx * eyy + eyy * eyy + 3 * exy * exy));

/** Direct GPU field rendering. No field values are read into JavaScript here. */
export class FieldRenderer {
  constructor(device, canvas) {
    this.device = device; this.canvas = canvas; this.groups = new WeakMap();
    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('This canvas cannot present WebGPU');
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format, alphaMode: 'opaque' });
    this.config = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: `
      struct Field { values: array<f32> }
      struct Config { width:u32,height:u32,gx:u32,gy:u32,stride:u32,parameters:u32,mode:u32,gray:u32,
        size:f32,step:f32,scale:f32,opacity:f32,
        arrow_step:u32,arrow_gain:f32,arrow_width:f32,padding:f32 }
      @group(0) @binding(0) var<storage,read> field:Field;
      @group(0) @binding(1) var camera:texture_2d<f32>;
      @group(0) @binding(2) var<uniform> config:Config;
      @vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
        let x=f32((i<<1u)&2u);let y=f32(i&2u);return vec4<f32>(x*2.0-1.0,y*2.0-1.0,0.0,1.0);
      }
      @fragment fn fragment(@builtin(position) pixel:vec4<f32>)->@location(0) vec4<f32> {
        let xy=vec2<i32>(pixel.xy);
        let image=textureLoad(camera,clamp(xy,vec2<i32>(0),vec2<i32>(i32(config.width)-1,i32(config.height)-1)),0);
        var background=image.rgb;if config.gray!=0u { background=vec3<f32>(image.r); }
        let grid=vec2<i32>(floor((pixel.xy-vec2<f32>((config.size-1.0)/2.0))/config.step+0.5));
        if any(grid<vec2<i32>(0)) || grid.x>=i32(config.gx) || grid.y>=i32(config.gy) {return vec4<f32>(background,1.0);}
        let base=(u32(grid.y)*config.gx+u32(grid.x))*config.stride;
        let status=field.values[base+config.parameters];
        var color=vec3<f32>(0.48);
        if status>=0.5 {
          let u=field.values[base];let v=field.values[base+3u];
          var value=u;
          switch config.mode {
            case 1u:{value=v;} case 2u:{value=length(vec2<f32>(u,v));}
            case 3u:{value=field.values[base+1u];} case 4u:{value=field.values[base+5u];}
            case 5u:{value=0.5*(field.values[base+2u]+field.values[base+4u]);}
            case 7u:{let exx=field.values[base+1u];let eyy=field.values[base+5u];
              let exy=0.5*(field.values[base+2u]+field.values[base+4u]);
              value=(2.0/3.0)*sqrt(max(0.0,exx*exx-exx*eyy+eyy*eyy+3.0*exy*exy));}
            default:{}
          }
          let t=clamp(value/max(config.scale,1e-8),-1.0,1.0);
          color=mix(vec3<f32>(0.12,0.32,0.8),vec3<f32>(0.98),clamp(t+1.0,0.0,1.0));
          if t>=0.0 { color=mix(vec3<f32>(0.98),vec3<f32>(0.84,0.18,0.12),t); }
          if config.mode==6u { color=select(vec3<f32>(0.1,0.65,0.4),vec3<f32>(1.0,0.65,0.1),status>1.5); }
        }
        if config.mode==6u {
          color=vec3<f32>(0.48);
          if status == -1.0 {color=vec3<f32>(0.23);}
          if status == -2.0 {color=vec3<f32>(0.8,0.12,0.16);}
          if status == 0.0 {color=vec3<f32>(0.98,0.8,0.15);}
          if status >= 0.5 {color=select(vec3<f32>(0.1,0.65,0.4),vec3<f32>(1.0,0.5,0.1),status>1.5);}
        }
        return vec4<f32>(mix(background,color,config.opacity),1.0);
      }
    ` });
    this.pipeline = device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vertex' },
      fragment: { module, entryPoint: 'fragment', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    const arrows = device.createShaderModule({ code: `
      struct Field { values:array<f32> }
      struct Config { width:u32,height:u32,gx:u32,gy:u32,stride:u32,parameters:u32,mode:u32,gray:u32,
        size:f32,step:f32,scale:f32,opacity:f32,arrow_step:u32,arrow_gain:f32,arrow_width:f32,padding:f32 }
      @group(0) @binding(0) var<storage,read> field:Field;
      @group(0) @binding(2) var<uniform> config:Config;
      struct Vertex { @builtin(position) position:vec4<f32>, @location(0) edge:f32 }
      @vertex fn arrow(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->Vertex {
        let columns=(config.gx+config.arrow_step-1u)/config.arrow_step;
        let ix=(instance%columns)*config.arrow_step;let iy=(instance/columns)*config.arrow_step;
        let base=(iy*config.gx+ix)*config.stride;
        let delta=vec2<f32>(field.values[base],field.values[base+3u])*config.arrow_gain;
        let distance=length(delta);var result:Vertex;
        if field.values[base+config.parameters]<0.5 || distance<0.25 {
          result.position=vec4<f32>(-2.0,-2.0,0.0,1.0);result.edge=0.0;return result;
        }
        let start=vec2<f32>(f32(ix),f32(iy))*config.step+vec2<f32>((config.size-1.0)/2.0);
        let end=start+delta;let direction=delta/distance;let normal=vec2<f32>(-direction.y,direction.x);
        let head=min(distance*0.35,config.arrow_width*3.0);
        let segment=vertex/6u;var a=start;var b=end;
        if segment>0u {a=end;b=end-direction*head+normal*head*select(-0.5,0.5,segment==1u);}
        let along=normalize(b-a);let across=vec2<f32>(-along.y,along.x);
        let corners=array<vec2<f32>,6>(vec2<f32>(0,-1),vec2<f32>(1,-1),vec2<f32>(1,1),vec2<f32>(0,-1),vec2<f32>(1,1),vec2<f32>(0,1));
        let corner=corners[vertex%6u];let p=mix(a,b,corner.x)+across*corner.y*config.arrow_width*0.5;
        result.position=vec4<f32>(p.x/f32(config.width)*2.0-1.0,1.0-p.y/f32(config.height)*2.0,0.0,1.0);
        result.edge=corner.y;return result;
      }
      @fragment fn ink(input:Vertex)->@location(0) vec4<f32> {
        return vec4<f32>(select(vec3<f32>(0.04,0.16,0.23),vec3<f32>(0.2,1.0,0.95),abs(input.edge)<0.55),1.0);
      }
    ` });
    this.arrowPipeline = device.createRenderPipeline({ layout:'auto',vertex:{module:arrows,entryPoint:'arrow'},
      fragment:{module:arrows,entryPoint:'ink',targets:[{format}]},primitive:{topology:'triangle-list'} });
    this.arrowGroups = new WeakMap();

  }
  draw(engine, result, { mode = 2, scale = null, opacity = .7, imageTexture = null, arrows = false, arrowStep = 4, arrowGain = 5 } = {}) {
    if (!Number.isInteger(arrowStep) || arrowStep < 1 || !Number.isFinite(arrowGain) || arrowGain < 0) throw new Error('Invalid arrow spacing or magnification');
    const width = result.width, height = result.height;
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    const max = scale ?? (mode === 7 ? (result.maxAbs?.[3] ?? 1) * Math.sqrt(8 / 3) : result.maxAbs?.[mode < 3 ? mode : 3]) ?? 1;
    const bytes = new ArrayBuffer(64), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    const texture = imageTexture ?? engine.imageTexture;
    u.set([width, height, ...result.grid, result.config.motionModel === 'quadratic' ? 16 : 8,
      result.config.motionModel === 'quadratic' ? 12 : 6, mode, +(texture.format === 'r8unorm')]);
    f.set([result.subsetSize, result.subsetStride, Math.max(max, 1e-8), opacity], 8);
    u[12] = arrowStep; f[13] = arrowGain; f[14] = Math.max(2, width / 500);
    this.device.queue.writeBuffer(this.config, 0, bytes);
    let byTexture = this.groups.get(engine.fieldBuffer);
    if (!byTexture) { byTexture = new WeakMap(); this.groups.set(engine.fieldBuffer, byTexture); }
    let bindGroup = byTexture.get(texture);
    if (!bindGroup) {
    bindGroup = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: engine.fieldBuffer } },
      { binding: 1, resource: texture.createView() }, { binding: 2, resource: { buffer: this.config } },
    ] });
    byTexture.set(texture, bindGroup);
    }
    const encoder = this.device.createCommandEncoder(), pass = encoder.beginRenderPass({ colorAttachments: [
      { view: this.context.getCurrentTexture().createView(), clearValue: { r: .1, g: .1, b: .1, a: 1 }, loadOp: 'clear', storeOp: 'store' },
    ] });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, bindGroup); pass.draw(3);
    if (arrows && !imageTexture) {
      let group = this.arrowGroups.get(engine.fieldBuffer);
      if (!group) { group = this.device.createBindGroup({layout:this.arrowPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:engine.fieldBuffer}},{binding:2,resource:{buffer:this.config}}]});this.arrowGroups.set(engine.fieldBuffer,group); }
      pass.setPipeline(this.arrowPipeline);pass.setBindGroup(0,group);
      pass.draw(18,Math.ceil(result.grid[0]/arrowStep)*Math.ceil(result.grid[1]/arrowStep));
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return max;
  }
  destroy() { this.config.destroy(); this.context.unconfigure(); }
}
