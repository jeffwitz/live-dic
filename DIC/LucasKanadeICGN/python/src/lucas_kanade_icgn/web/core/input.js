import { loadShader } from './shaders.js';
export function frameSize(source) {
  return [source.displayWidth ?? source.videoWidth ?? source.width,
    source.displayHeight ?? source.videoHeight ?? source.height];
}
/** Owns GPU resources, never the caller's VideoFrame/ImageBitmap. */
export async function createInputConverter(device, width, height) {
  const source = device.createTexture({ size: [width, height], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
  const texture = device.createTexture({ size: [width, height], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
    compute: { module: await loadShader(device, 'input.wgsl'), entryPoint: 'luminance' } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: source.createView() }, { binding: 1, resource: texture.createView() },
  ] });
  function copy(frame) {
    const [w, h] = frameSize(frame);
    if (w !== width || h !== height) throw new Error('Frame dimensions must match the reference');
    device.queue.copyExternalImageToTexture({ source: frame }, { texture: source }, [width, height]);
  }
  function encode(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16)); pass.end();
  }
  async function referencePixels(frame, region = null) {
    const x=region?.x??0,y=region?.y??0,w=region?.size??width,h=region?.size??height;
    if(x<0||y<0||x+w>width||y+h>height)throw new Error('Readback region exceeds the image');
    copy(frame);
    const encoder = device.createCommandEncoder(); encode(encoder);
    const bytesPerRow = Math.ceil(w * 4 / 256) * 256;
    const readback = device.createBuffer({ size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyTextureToBuffer({ texture,origin:[x,y] }, { buffer: readback, bytesPerRow, rowsPerImage: h }, [w, h]);
    device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(readback.getMappedRange()), pixels = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pixels[y * w + x] = mapped[y * bytesPerRow + 4 * x];
      return pixels;
    } finally {if(readback.mapState==='mapped')readback.unmap();readback.destroy();}
  }
  return { texture, source, pipeline, bindGroup, copy, encode, referencePixels,
    destroy() { source.destroy(); texture.destroy(); } };
}
