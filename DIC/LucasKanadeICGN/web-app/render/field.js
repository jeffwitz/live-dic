/** Direct GPU field rendering. No field values are read into JavaScript here. */
export class FieldRenderer {
  constructor(device, canvas) {
    this.device = device; this.canvas = canvas; this.groups = new WeakMap();
    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('This canvas cannot present WebGPU');
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format, alphaMode: 'opaque' });
    this.config = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: `
      struct Field { values: array<f32> }
      struct Config { width:u32,height:u32,gx:u32,gy:u32,stride:u32,parameters:u32,mode:u32,gray:u32,
        size:f32,step:f32,scale:f32,opacity:f32 }
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
            default:{}
          }
          let t=clamp(value/max(config.scale,1e-8),-1.0,1.0);
          color=mix(vec3<f32>(0.12,0.32,0.8),vec3<f32>(0.98),clamp(t+1.0,0.0,1.0));
          if t>=0.0 { color=mix(vec3<f32>(0.98),vec3<f32>(0.84,0.18,0.12),t); }
          if config.mode==6u { color=select(vec3<f32>(0.1,0.65,0.4),vec3<f32>(1.0,0.65,0.1),status>1.5); }
        }
        return vec4<f32>(mix(background,color,config.opacity),1.0);
      }
    ` });
    this.pipeline = device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vertex' },
      fragment: { module, entryPoint: 'fragment', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
  }
  draw(engine, result, { mode = 2, scale = null, opacity = .7 } = {}) {
    const width = result.width, height = result.height;
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    const max = scale ?? result.maxAbs?.[mode < 3 ? mode : 3] ?? 1;
    const bytes = new ArrayBuffer(48), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    const texture = engine.imageTexture;
    u.set([width, height, ...result.grid, result.config.motionModel === 'quadratic' ? 16 : 8,
      result.config.motionModel === 'quadratic' ? 12 : 6, mode, +(texture.format === 'r8unorm')]);
    f.set([result.subsetSize, result.subsetStride, Math.max(max, 1e-8), opacity], 8);
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
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, bindGroup); pass.draw(3); pass.end();
    this.device.queue.submit([encoder.finish()]);
    return max;
  }
  destroy() { this.config.destroy(); this.context.unconfigure(); }
}
