/** GPU API resource sizes, not physical VRAM occupancy or available device memory. */
export function gpuMemoryUsage(resources) {
  const seen = new Set(), bytesPerTexel = {r8unorm:1,r32float:4,rg32float:8,rgba8unorm:4,'rgba8unorm-srgb':4,bgra8unorm:4,'bgra8unorm-srgb':4,rgba16float:8,rgba32float:16};
  const report = {bufferCount:0,bufferBytes:0,mappedBufferBytes:0,textureCount:0,texturePayloadBytes:0,unknownTextureFormats:[],physicalVramBytes:null,freeVramBytes:null,memoryArchitecture:'not exposed by WebGPU'};
  function visit(value) {
    if (!value || typeof value !== 'object' || seen.has(value) || ArrayBuffer.isView(value)) return;
    seen.add(value); const tag=Object.prototype.toString.call(value);
    if (tag==='[object GPUBuffer]') {
      report.bufferCount++;report.bufferBytes+=value.size;
      if(value.usage&3)report.mappedBufferBytes+=value.size;
    } else if(tag==='[object GPUTexture]') {
      report.textureCount++;
      const stride=bytesPerTexel[value.format];
      if(!stride){report.unknownTextureFormats.push(value.format);return;}
      for(let level=0;level<value.mipLevelCount;level++)report.texturePayloadBytes+=Math.max(1,Math.floor(value.width/2**level))*Math.max(1,Math.floor(value.height/2**level))*value.depthOrArrayLayers*value.sampleCount*stride;
    } else if(Array.isArray(value)||value instanceof Set)for(const item of value)visit(item);
    else if(value instanceof Map)for(const item of value.values())visit(item);
    else if(Object.getPrototypeOf(value)===Object.prototype||Object.getPrototypeOf(value)===null)for(const item of Object.values(value))visit(item);
  }
  visit(resources);
  report.knownResourceBytes=report.bufferBytes+report.texturePayloadBytes;
  report.scope='Live engine/input resources. Buffer sizes are requested GPU API bytes; texture payload excludes driver padding, pipelines and canvas/swapchain resources. Mapped buffers may use host memory. Unified GPUs share host RAM; these numbers must not be interpreted as independent free RAM/VRAM.';
  return report;
}
