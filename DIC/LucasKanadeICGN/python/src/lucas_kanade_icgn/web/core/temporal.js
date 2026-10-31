/** Host-only temporal graph: canonical full_pyramid.wgsl owns all equations. */
import { seedPermissions } from './roi.js';

export function validateTemporal(options) {
  const initialization = options.initialization ?? 'zero';
  const prediction = options.temporalPrediction ?? 'none';
  if (!['zero', 'temporal'].includes(initialization)) throw new Error("initialization must be 'zero' or 'temporal'");
  if (!['none', 'velocity'].includes(prediction)) throw new Error("temporalPrediction must be 'none' or 'velocity'");
  if (prediction !== 'none' && initialization !== 'temporal') throw new Error('temporalPrediction requires temporal initialization');
  if (options.temporalFineCorrection && (initialization !== 'temporal' || (options.pyramidLevels ?? 2) < 2)) {
    throw new Error('temporalFineCorrection requires temporal initialization and at least two levels');
  }
  if (options.temporalMaxAge != null && (!Number.isInteger(options.temporalMaxAge) || options.temporalMaxAge < 0)) {
    throw new Error('temporalMaxAge must be a nonnegative integer or null');
  }
  if (options.pyramidPolicy && options.pyramidPolicy !== 'fixed') throw new Error('Adaptive pyramids are experimental and not supported in this browser engine');
}

export async function createTemporalGraph(device, shader, levels, seedEdges, options, roiMask, size, stride) {
  if ((options.initialization ?? 'zero') === 'zero') return null;
  const owned = [], fine = levels[0], coarse = levels.at(-1);
  const velocity = options.temporalPrediction === 'velocity', correction = !!options.temporalFineCorrection;
  const model = options.motionModel === 'quadratic' ? 2 : options.motionModel === 'translation' ? 1 : 0;
  const stateStride = model === 2 ? 16 : 8, parameterCount = model === 2 ? 12 : 6;
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const alloc = (size, usage = storage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
  const data = (array, usage) => { const b = alloc(array.byteLength, usage | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(b, 0, array); return b; };
  const history = (level, coarseFlag = 0) => ({
    state: alloc(level.gridCount * stateStride * 4), ages: alloc(level.gridCount * 4),
    config: data(new Uint32Array([level.gridCount, options.temporalMaxAge ?? 0xfffffffe,
      +(options.outlierPolicy === 'replace' && !velocity), coarseFlag]), GPUBufferUsage.UNIFORM),
  });
  const current = history(fine), coarseHistory = correction ? history(coarse, 1) : null;
  const previous = velocity ? history(fine) : null;
  const prediction = velocity ? alloc(fine.gridCount * stateStride * 4) : null;
  const counts = alloc(16, storage | GPUBufferUsage.COPY_SRC);
  const permissions = roiMask ? seedPermissions(fine.roi, coarse.roi, 1, 2 ** (levels.length - 1),
    size, stride, roiMask, fine.width, fine.height) : new Uint32Array(coarse.gridCount).fill(31);
  const links = data(permissions, GPUBufferUsage.STORAGE);
  const seedBytes = new ArrayBuffer(32), u = new Uint32Array(seedBytes), f = new Float32Array(seedBytes);
  u.set([fine.gridWidth, fine.gridHeight, coarse.gridWidth, coarse.gridCount]); f.set([1, 2 ** (levels.length - 1)], 4);
  const seedConfig = data(new Uint8Array(seedBytes), GPUBufferUsage.UNIFORM);
  const empty = device.createBindGroupLayout({ entries: [] });
  async function operation(entryPoint, group, specifications, constants, count) {
    const layout = device.createBindGroupLayout({ entries: specifications.map(([binding, type]) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
    })) });
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [...Array(group).fill(empty), layout] }),
      compute: { module: shader, entryPoint, constants: { MOTION_MODEL: model, ...constants } },
    });
    const bindGroup = device.createBindGroup({ layout, entries: specifications.map(([binding, , buffer]) => ({ binding, resource: { buffer } })) });
    return { pipeline, bindGroup, group, workgroups: [Math.ceil(count / 256), 1, 1] };
  }
  const out = (level) => options.outlierPolicy === 'disabled' ? level.buffers.state : level.buffers.filteredState;
  const projection = await operation('project_temporal_seed', 2, [
    [0, 'read-only-storage', prediction ?? current.state], [1, 'storage', coarse.buffers.seedState],
    [2, 'storage', coarse.buffers.activeMask], [3, 'uniform', seedConfig],
    [4, 'read-only-storage', links], [5, 'storage', counts],
  ], { TRACK_SEED_AVAILABILITY: correction, SUBSET_SIZE: size, SUBSET_STRIDE: stride }, coarse.gridCount);
  const merge = async (level, h) => operation(velocity ? 'update_temporal_state_grouped' : 'update_temporal_state', 3, [
    [0, 'storage', h.state], [1, 'storage', h.ages], [2, 'read-only-storage', level.buffers.state],
    [3, 'read-only-storage', out(level)], [4, 'uniform', h.config], [5, 'storage', counts],
  ], { REPORT_HISTORY_EXPIRY: false, AGGREGATE_HISTORY_COUNTS: velocity }, level.gridCount);
  const mergeFine = await merge(fine, current), mergeCoarse = correction ? await merge(coarse, coarseHistory) : null;
  let predict = null, save = null, refine = null;
  if (velocity) {
    predict = await operation('predict_temporal_state', 2, [
      [0, 'read-only-storage', current.state], [1, 'storage', prediction], [5, 'storage', counts],
      [6, 'read-only-storage', previous.state], [7, 'read-only-storage', current.ages], [9, 'read-only-storage', previous.ages],
    ], {}, fine.gridCount);
    save = await operation('save_previous_temporal_state', 3, [
      [0, 'storage', current.state], [1, 'storage', current.ages], [2, 'read-only-storage', fine.buffers.state],
      [3, 'read-only-storage', out(fine)], [4, 'uniform', current.config],
      [6, 'storage', previous.state], [7, 'storage', previous.ages],
    ], {}, fine.gridCount);
  }
  if (correction) {
    const edge = seedEdges.get(0);
    const refinementLinks = edge.links ?? data(new Uint32Array(fine.gridCount).fill(31), GPUBufferUsage.STORAGE);
    refine = await operation('refine_temporal_seed', 2, [
      [0, 'read-only-storage', coarseHistory.state], [1, 'storage', fine.buffers.seedState],
      [2, 'storage', fine.buffers.activeMask], [3, 'uniform', edge.config], [4, 'read-only-storage', refinementLinks],
      [6, 'read-only-storage', current.state], [7, 'read-only-storage', current.ages],
      [8, 'uniform', current.config], [9, 'read-only-storage', coarseHistory.ages],
    ], { SUBSET_SIZE: size, SUBSET_STRIDE: stride }, fine.gridCount);
  }
  function reset() {
    for (const [h, count] of [[current, fine.gridCount], [coarseHistory, coarse.gridCount], [previous, fine.gridCount]]) {
      if (!h) continue;
      const states = new Float32Array(count * stateStride);
      for (let i = 0; i < count; i++) states[i * stateStride + parameterCount] = -1;
      device.queue.writeBuffer(h.state, 0, states);
      device.queue.writeBuffer(h.ages, 0, new Uint32Array(count).fill(0xffffffff));
    }
  }
  reset();
  return { counts, before: [predict, projection].filter(Boolean), refine,
    after: [save, mergeFine, mergeCoarse].filter(Boolean), reset,
    destroy() { owned.forEach((b) => b.destroy()); } };
}
