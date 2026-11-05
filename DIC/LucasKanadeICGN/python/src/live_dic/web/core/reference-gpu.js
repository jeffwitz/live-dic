import { loadShader } from './shaders.js';
import { createInputConverter, frameSize } from './input.js';

/** Persistent opt-in preparation graph; the IC-GN solver/layout are unchanged.
 *
 * Cold geometry, ROI links and resources are shared between pairs. Each hot
 * call updates intensities, gradients, inverse Hessians and numeric validity
 * on GPU, before solving in the same command encoder. No input is owned here.
 */
export async function createGpuReferenceGraph(device, levels, reducePipeline, reduceLayout, options) {
  const owned = [], fine = levels[0], size = options.subsetSize, stride = options.subsetStride;
  const allocate = (bytes, usage, label) => {
    const buffer = device.createBuffer({ size: Math.max(4, bytes), usage, label });
    owned.push(buffer);
    return buffer;
  };
  let converter, byteTexture;
  try {
    converter = await createInputConverter(device, fine.width, fine.height);
    // Reference and target bytes must never alias: both are uploaded before the
    // single submission. External inputs likewise have a separate converter.
    byteTexture = device.createTexture({ label: 'Independent pair reference bytes',
      size: [fine.width, fine.height], format: 'r8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
    const shader = await loadShader(device, 'reference.wgsl');
    const interpolationMode = { bspline: 0, linear: 1, cubic: 2 }[options.interpolation];
    // Five storage buffers, one uniform and one texture: below baseline limits.
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[2, 3, 4, 6].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } })),
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = {};
    for (const entry of ['gaussian_x', 'gaussian_y', 'direct_intensity', 'spline_rows',
      'spline_columns', 'gradients', 'prepare_subsets']) {
      // Entry points only accept overrides that they actually use.
      let constants;
      if (entry === 'gradients') constants = { SUBSET_SIZE: size, SUBSET_STRIDE: stride, INTERPOLATION_MODE: interpolationMode };
      if (entry === 'prepare_subsets') constants = { SUBSET_SIZE: size, SUBSET_STRIDE: stride,
        MOTION_MODEL: options.motionModel === 'translation' ? 1 : 0, ROI_ENABLED: !!fine.roi };
      pipelines[entry] = await device.createComputePipelineAsync({ label: `Reference ${entry}`,
        layout: pipelineLayout, compute: { module: shader, entryPoint: entry, constants } });
    }
    // Headers count admissible references; following words store dense numeric
    // validity independently of ROI selection. Only a count joins hot readback.
    let words = levels.length;
    const offsets = levels.map(level => { const start = words; words += level.gridCount; return start; });
    const validity = allocate(words * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      'Reference validity and counts');
    const validityReadback = allocate(words * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      'Cold reference validity readback');
    const jobs = levels.map((level, i) => {
      // Three float planes; 3840x2160 fits a 128 MiB storage binding.
      const scratch = allocate(level.pixelCount * 12, GPUBufferUsage.STORAGE, `Reference scratch level ${i}`);
      const config = allocate(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, `Reference config level ${i}`);
      device.queue.writeBuffer(config, 0, new Uint32Array([level.width, level.height, level.gridWidth,
        level.gridCount, level.inverseOffset * 4, offsets[i], i, 0]));
      const selection = allocate(level.roi ? level.gridCount * 4 : 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, `Reference ROI level ${i}`);
      device.queue.writeBuffer(selection, 0, level.roi ? Uint32Array.from(level.roi.selected) : new Uint32Array([1]));
      const group = texture => device.createBindGroup({ layout, entries: [
        { binding: 0, resource: texture.createView() },
        ...[[1, config], [2, scratch], [3, level.buffers.reference], [4, level.buffers.metadata],
          [5, selection], [6, validity]].map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      ] });
      return { level, external: group(i ? level.sourceTexture : converter.texture), bytes: i ? null : group(byteTexture) };
    });
    const reductions = levels.slice(1).map((level, i) => {
      const group = texture => device.createBindGroup({ layout: reduceLayout, entries: [
        { binding: 0, resource: texture.createView() }, { binding: 1, resource: level.sourceTexture.createView() },
      ] });
      return { level, external: group(i ? levels[i].sourceTexture : converter.texture), bytes: i ? null : group(byteTexture) };
    });
    let external = true;
    function copy(frame) {
      if (frame instanceof Uint8Array) {
        if (frame.length !== fine.pixelCount) throw new Error('Reference bytes must match prepared dimensions');
        device.queue.writeTexture({ texture: byteTexture }, frame,
          { bytesPerRow: fine.width, rowsPerImage: fine.height }, [fine.width, fine.height]);
        external = false;
      } else {
        const [width, height] = frameSize(frame);
        if (width !== fine.width || height !== fine.height) throw new Error('Pair reference dimensions must match the prepared reference');
        converter.copy(frame);
        external = true;
      }
    }
    function encode(encoder) {
      if (external) converter.encode(encoder);
      encoder.clearBuffer(validity, 0, levels.length * 4);
      for (const edge of reductions) {
        const pass = encoder.beginComputePass({ label: 'reference pyramid reduction' });
        pass.setPipeline(reducePipeline);
        pass.setBindGroup(1, !external && edge.bytes ? edge.bytes : edge.external);
        pass.dispatchWorkgroups(Math.ceil(edge.level.width / 16), Math.ceil(edge.level.height / 16));
        pass.end();
      }
      for (const job of jobs) {
        const { level } = job;
        const dispatch = (name, x, y = 1) => {
          const pass = encoder.beginComputePass({ label: `reference level ${level.index} ${name}` });
          pass.setPipeline(pipelines[name]);
          pass.setBindGroup(0, !external && job.bytes ? job.bytes : job.external);
          pass.dispatchWorkgroups(x, y); pass.end();
        };
        if (options.interpolation === 'bspline') {
          dispatch('gaussian_x', Math.ceil(level.width / 16), Math.ceil(level.height / 16));
          dispatch('gaussian_y', Math.ceil(level.width / 16), Math.ceil(level.height / 16));
          dispatch('spline_rows', Math.ceil(level.height / 64));
          dispatch('spline_columns', Math.ceil(level.width / 64));
        } else dispatch('direct_intensity', Math.ceil(level.width / 16), Math.ceil(level.height / 16));
        dispatch('gradients', Math.ceil(level.width / 16), Math.ceil(level.height / 16));
        dispatch('prepare_subsets', level.gridWidth, level.gridHeight);
      }
    }
    // Explicit cold inspection only. Hot pairs never read intermediate images,
    // Hessians or validity masks; they read four count bytes with the final field.
    async function readValidity(encoder) {
      encoder.copyBufferToBuffer(validity, 0, validityReadback, 0, words * 4);
      device.queue.submit([encoder.finish()]);
      await validityReadback.mapAsync(GPUMapMode.READ);
      try {
        const data = new Uint32Array(validityReadback.getMappedRange());
        return { counts: Array.from(data.subarray(0, levels.length)),
          masks: levels.map((level, i) => Uint8Array.from(data.subarray(offsets[i], offsets[i] + level.gridCount))) };
      } finally { validityReadback.unmap(); }
    }
    return { copy, encode, readValidity,
      async prepare(frame) {
        copy(frame);
        const encoder = device.createCommandEncoder(); encode(encoder);
        return readValidity(encoder);
      },
      copyCount(encoder, destination, offset) { encoder.copyBufferToBuffer(validity, 0, destination, offset, 4); },
      memoryResources: () => ({ owned, converter, byteTexture }),
      destroy() { owned.forEach(buffer => buffer.destroy()); converter.destroy(); byteTexture.destroy(); },
    };
  } catch (error) {
    owned.forEach(buffer => buffer.destroy()); converter?.destroy(); byteTexture?.destroy();
    throw error;
  }
}
