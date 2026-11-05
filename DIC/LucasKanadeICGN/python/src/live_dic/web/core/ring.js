/** Bounded admission and submission order; no unbounded promise queue. */
export class SlotRing {
  constructor(depth = 1) {
    if (!Number.isInteger(depth) || depth < 1) throw new Error('pipelineDepth must be at least 1');
    this.depth = depth;
    this.free = Array.from({ length: depth }, (_, i) => i);
    this.active = 0;
    this.tail = Promise.resolve();
  }
  run(operation) {
    if (!this.free.length) return Promise.reject(new Error('Frame backpressure: all slots are busy'));
    const index = this.free.shift();
    this.active++;
    // Begin submission now. Only delivery is chained, so GPU slots overlap.
    const pending = Promise.resolve().then(() => operation(index));
    pending.catch(() => {});
    const ordered = this.tail.then(() => pending);
    this.tail = ordered.catch(() => {});
    return ordered.finally(() => { this.active--; this.free.push(index); });
  }
  get idle() { return this.active === 0; }
}

/** Clone only mutable target resources. Reference/configuration/ROI are shared. */
export function createTargetSlot(device, primary, layouts, configuration) {
  const map = new Map();
  const levels = primary.levels.map((original) => {
    const buffers = { ...original.buffers };
    for (const name of ['temporary', 'state', 'filteredState', 'seedState', 'activeMask']) {
      const source = original.buffers[name];
      buffers[name] = device.createBuffer({ size: source.size, usage: source.usage });
    }
    const sourceTexture = device.createTexture({ size: [original.width, original.height],
      format: original.index ? 'rgba8unorm' : 'r8unorm', usage: original.sourceTexture.usage });
    function group(layout, wide, seeded = false) {
      const entries = [{ binding: 0, resource: sourceTexture.createView() },
        ...[[1, 'config'], [2, 'temporary'], [4, 'reference'], [6, 'metadata'],
          [7, 'state'], [8, 'fir'], [9, 'filteredState'], ...(seeded ? [[10, 'seedState'], [11, 'activeMask']] : [])]
          .map(([binding, name]) => ({ binding, resource: { buffer: buffers[name],
            ...(binding === 2 && !wide ? { size: original.pixelCount * 4 } : {}) } }))];
      return device.createBindGroup({ layout, entries });
    }
    const level = { ...original, buffers, sourceTexture,
      bindGroup: group(layouts.base, false), prefilterBindGroup: group(layouts.base, true),
      seededBindGroup: group(layouts.seeded, false, true) };
    map.set(original, level);
    return level;
  });
  const seedEdges = new Map([...primary.seedEdges].map(([index, edge]) => {
    const source = levels[index + 1], destination = levels[index];
    return [index, { ...edge, bindGroup: device.createBindGroup({ layout: layouts.seedTransfer, entries: [
      { binding: 0, resource: { buffer: configuration.outlierPolicy === 'disabled' ? source.buffers.state : source.buffers.filteredState } },
      { binding: 1, resource: { buffer: destination.buffers.seedState } },
      { binding: 2, resource: { buffer: destination.buffers.activeMask } },
      { binding: 3, resource: { buffer: edge.config } },
      ...(edge.links ? [{ binding: 4, resource: { buffer: edge.links } }] : []),
    ] }) }];
  }));
  const reductionEdges = primary.reductionEdges.map((edge) => {
    const source = map.get(edge.source), destination = map.get(edge.destination);
    return { source, destination, intermediate: null, verticalBindGroup: device.createBindGroup({ layout: layouts.reduce,
      entries: [{ binding: 0, resource: source.sourceTexture.createView() },
        { binding: 1, resource: destination.sourceTexture.createView() }] }) };
  });
  return { levels, seedEdges, reductionEdges, temporal: null };
}

export function attachExternalInput(device, slot, layouts, converter) {
  const level = slot.levels[0];
  function group(layout, wide, seeded = false) {
    return device.createBindGroup({ layout, entries: [
      { binding: 0, resource: converter.texture.createView() },
      ...[[1, 'config'], [2, 'temporary'], [4, 'reference'], [6, 'metadata'], [7, 'state'], [8, 'fir'], [9, 'filteredState'],
        ...(seeded ? [[10, 'seedState'], [11, 'activeMask']] : [])].map(([binding, name]) => ({ binding,
        resource: { buffer: level.buffers[name], ...(binding === 2 && !wide ? { size: level.pixelCount * 4 } : {}) } })),
    ] });
  }
  slot.external = { converter, base: group(layouts.base, false), wide: group(layouts.base, true), seeded: group(layouts.seeded, false, true),
    reduce: slot.reductionEdges.length ? device.createBindGroup({ layout: layouts.reduce, entries: [
      { binding: 0, resource: converter.texture.createView() },
      { binding: 1, resource: slot.levels[1].sourceTexture.createView() },
    ] }) : null };
}
