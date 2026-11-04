/** Optional failed-point seed search. Canonical WGSL owns all numerical work. */
import { loadShader } from './shaders.js';

export function validateSpatialPropagation(options = {}) {
  const passes = options.spatialPropagationPasses === undefined ? 0 : options.spatialPropagationPasses;
  if (!Number.isInteger(passes) || passes < 0 || passes > 4) {
    throw new Error('spatialPropagationPasses must be an integer from 0 to 4');
  }
  if (!passes) return 0;
  if ((options.motionModel ?? 'affine') !== 'translation') {
    throw new Error('GPU spatial propagation currently requires the translation local model');
  }
  if ((options.interpolation ?? 'bspline') !== 'bspline'
      || (options.spatialWeighting ?? 'uniform') !== 'uniform') {
    throw new Error('GPU spatial propagation requires B-spline and uniform integration');
  }
  if ((options.referencePreparation ?? 'cpu') !== 'cpu') {
    throw new Error('GPU spatial propagation currently requires CPU reference preparation');
  }
  if ((options.pipelineDepth ?? 1) !== 1) {
    throw new Error('GPU spatial propagation currently supports one processing slot');
  }
  return passes;
}

/** Prepare every texture, permission table and pipeline before registration. */
export async function createSpatialGraph(device, solverModule, levels, seededLayout,
  options, references, externalTexture = null) {
  const passes = validateSpatialPropagation(options);
  if (!passes) return null;
  const resources = [];
  const data = (label, bytes, usage) => {
    const buffer = device.createBuffer({ label, size: bytes.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    resources.push(buffer); device.queue.writeBuffer(buffer, 0, bytes); return buffer;
  };
  const storage = (label, size, extra = 0) => {
    const buffer = device.createBuffer({ label, size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra });
    resources.push(buffer); return buffer;
  };
  const entry = (binding, buffer) => ({ binding, resource: { buffer } });
  try {
    const module = await loadShader(device, 'spatial_seed.wgsl');
    const choose = await device.createComputePipelineAsync({ layout: 'auto',
      compute: { module, entryPoint: 'choose', constants: { FAILED_ONLY: true, CANDIDATE_MODE: 0 } } });
    const finalize = await device.createComputePipelineAsync({ layout: 'auto',
      compute: { module, entryPoint: 'finalize_chain' } });
    const stages = [];
    for (const level of levels) {
      const { width, height, gridWidth, gridHeight, gridCount, buffers } = level;
      const links = new Uint32Array(gridCount);
      for (let index = 0; index < gridCount; index++) {
        if (!level.referenceValid[index] || (level.roi && !level.roi.selected[index])) continue;
        if (level.roi) links[index] = level.roi.neighbours[index];
        else {
          const x = index % gridWidth, y = Math.floor(index / gridWidth);
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            if (x + dx >= 0 && y + dy >= 0 && x + dx < gridWidth && y + dy < gridHeight) {
              links[index] |= 1 << ((dy + 1) * 3 + dx + 1);
            }
          }
        }
      }
      const permissions = data('Spatial ROI/reference permissions', links, GPUBufferUsage.STORAGE);
      const reference = device.createTexture({ label: 'Spatial raw reference', size: [width, height],
        format: 'r8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
      resources.push(reference);
      device.queue.writeTexture({ texture: reference }, Uint8Array.from(references[level.index].values),
        { bytesPerRow: width, rowsPerImage: height }, [width, height]);
      const configBytes = new ArrayBuffer(48), words = new Uint32Array(configBytes), floats = new Float32Array(configBytes);
      words.set([width, height, gridWidth, gridCount, options.subsetSize, options.subsetStride, 1, 0]);
      floats[8] = 1e-4; floats[9] = 1e-3;
      const config = data('Spatial seed configuration', new Uint8Array(configBytes), GPUBufferUsage.UNIFORM);
      const work = storage('Spatial indirect work list', (gridCount + 4) * 4, GPUBufferUsage.INDIRECT);
      const continuation = storage('Spatial continuation arguments', 12, GPUBufferUsage.INDIRECT);
      const prior = storage('Spatial initial level seeds', gridCount * 32);
      const scores = storage('Spatial candidate scores', gridCount * 16);
      const chooseGroup = texture => device.createBindGroup({ layout: choose.getBindGroupLayout(0), entries: [
        { binding: 0, resource: reference.createView() }, { binding: 1, resource: texture.createView() },
        entry(2, config), entry(3, buffers.state), entry(4, permissions), entry(5, buffers.seedState),
        entry(6, work), entry(7, scores), entry(8, prior),
      ] });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [seededLayout] });
      const compact = await device.createComputePipelineAsync({ layout, compute: {
        module: solverModule, entryPoint: 'solve_compact_translation', constants: {
          MOTION_MODEL: 1, SUBSET_SIZE: options.subsetSize, SUBSET_STRIDE: options.subsetStride,
          SOLVER_WORKGROUP_SIZE: options.solverWorkgroupSize ?? 16, ROI_ENABLED: +(!!level.roi),
          ...(level.index === 0 && options.divergencePatience ? {
            DIVERGENCE_PATIENCE: options.divergencePatience,
            DIVERGENCE_MIN_ITERATIONS: options.divergenceMinIterations ?? 4,
          } : {}),
        },
      } });
      const compactGroup = texture => device.createBindGroup({ layout: seededLayout, entries: [
        { binding: 0, resource: texture.createView() },
        ...[[1, 'config'], [2, 'temporary'], [4, 'reference'], [6, 'metadata'], [7, 'state'],
          [8, 'fir'], [9, 'filteredState'], [10, 'seedState']].map(([binding, name]) => ({ binding,
          resource: { buffer: buffers[name], ...(binding === 2 ? { size: level.pixelCount * 4 } : {}) } })),
        entry(11, work),
      ] });
      stages.push({ level, work, continuation, prior, choose: chooseGroup(level.sourceTexture), compact: compactGroup(level.sourceTexture),
        externalChoose: level.index === 0 && externalTexture ? chooseGroup(externalTexture) : null,
        externalCompact: level.index === 0 && externalTexture ? compactGroup(externalTexture) : null,
        compactPipeline: compact,
        finalize: device.createBindGroup({ layout: finalize.getBindGroupLayout(0),
          entries: [entry(2, config), entry(6, work), entry(9, continuation)] }),
      });
    }
    return {
      resources,
      beforeLevel(encoder, levelIndex, seeded) {
        const stage = stages[levelIndex];
        if (seeded) encoder.copyBufferToBuffer(stage.level.buffers.seedState, 0, stage.prior, 0, stage.prior.size);
        else encoder.clearBuffer(stage.prior);
      },
      afterLevel(encoder, levelIndex, external) {
        const stage = stages[levelIndex];
        for (let step = 0; step < passes; step++) {
          encoder.clearBuffer(stage.work);
          const compute = encoder.beginComputePass();
          compute.setPipeline(choose); compute.setBindGroup(0, external && levelIndex === 0 ? stage.externalChoose : stage.choose);
          if (step === 0) compute.dispatchWorkgroups(stage.level.gridWidth, stage.level.gridHeight);
          else compute.dispatchWorkgroupsIndirect(stage.continuation, 0);
          compute.setPipeline(finalize); compute.setBindGroup(0, stage.finalize); compute.dispatchWorkgroups(1);
          compute.setPipeline(stage.compactPipeline);
          compute.setBindGroup(0, external && levelIndex === 0 ? stage.externalCompact : stage.compact);
          compute.dispatchWorkgroupsIndirect(stage.work, 4); compute.end();
        }
      },
      destroy() { resources.forEach(resource => resource.destroy()); resources.length = 0; },
    };
  } catch (error) { resources.forEach(resource => resource.destroy()); throw error; }
}
