/** Dense packed field decoding; no DOM or GPU dependencies. */
export function decodePacked(packed, { width: FRAME_WIDTH, height: FRAME_HEIGHT,
  gridWidth: GRID_WIDTH, gridHeight: GRID_HEIGHT, subsetSize: SUBSET_SIZE,
  subsetStride: SUBSET_STRIDE, configuration, timing }) {
  const GRID_COUNT = GRID_WIDTH * GRID_HEIGHT;
  const parameterCount = configuration.motionModel === 'quadratic' ? 12 : 6;
  const stateStride = configuration.motionModel === 'quadratic' ? 16 : 8;
  const parameters = new Float32Array(GRID_COUNT * parameterCount);
  const status = new Float32Array(GRID_COUNT);
  const iterations = new Float32Array(GRID_COUNT);
  let accepted = 0;
  let replaced = 0;
  let meanX = 0;
  let meanY = 0;
  let iterationTotal = 0;
  for (let index = 0; index < GRID_COUNT; ++index) {
    const source = stateStride * index;
    const target = parameterCount * index;
    for (let component = 0; component < parameterCount; ++component) {
      parameters[target + component] = packed[source + component];
    }
    status[index] = packed[source + parameterCount];
    iterations[index] = packed[source + parameterCount + 1];
    if (status[index] > 1.5) replaced += 1;
    if (status[index] >= 0.5) {
      accepted += 1;
      meanX += parameters[target];
      meanY += parameters[target + 3];
      iterationTotal += iterations[index];
    }
  }
  return {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    subsetSize: SUBSET_SIZE,
    subsetStride: SUBSET_STRIDE,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    stateStride,
    parameters,
    status,
    iterations,
    accepted,
    replaced,
    meanTranslation: accepted ? [meanX / accepted, meanY / accepted] : [null, null],
    meanIterations: accepted ? iterationTotal / accepted : null,
    timing,
    config: { ...configuration },
  };
}

/** Small per-workgroup summaries keep GPU display independent of field copies. */
export async function createResultReader(device, module, level, options, field) {
  const count = level.gridCount, stride = options.motionModel === 'quadratic' ? 16 : 8;
  const parameters = options.motionModel === 'quadratic' ? 12 : 6;
  const groups = Math.ceil(count / 256), step = Math.max(1, Math.ceil(count / 1024));
  const sampleCount = Math.ceil(count / step), summaryBytes = groups * 48;
  const summary = device.createBuffer({ size: summaryBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const samples = device.createBuffer({ size: sampleCount * stride * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const config = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bytes = new ArrayBuffer(64), ints = new Uint32Array(bytes), floats = new Float32Array(bytes);
  ints.set([count, stride, parameters, level.gridWidth]);
  floats.set([(options.subsetSize - 1) / 2, options.subsetStride, level.width / 2, level.height / 2], 4);
  ints[12] = step;
  device.queue.writeBuffer(config, 0, bytes);
  const statistics = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'statistics' } });
  const decimate = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'decimate' } });
  const resource = (binding, buffer) => ({ binding, resource: { buffer } });
  const statsGroup = device.createBindGroup({ layout: statistics.getBindGroupLayout(0), entries: [resource(0, field), resource(1, summary), resource(2, config)] });
  const samplesGroup = device.createBindGroup({ layout: decimate.getBindGroupLayout(0), entries: [resource(0, field), resource(2, config), resource(3, samples)] });
  function setTruth(truth) {
    floats[8] = truth?.[0] ?? 0; floats[9] = truth?.[1] ?? 0; floats[10] = truth ? 1 : 0;
    device.queue.writeBuffer(config, 0, bytes);
  }
  return { summary, samples, summaryBytes, sampleBytes: samples.size, step, sampleCount,
    truthEnabled: () => floats[10] !== 0, setTruth,
    encode(encoder, mode) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(statistics); pass.setBindGroup(0, statsGroup); pass.dispatchWorkgroups(groups);
      if (mode === 'decimated') { pass.setPipeline(decimate); pass.setBindGroup(0, samplesGroup); pass.dispatchWorkgroups(Math.ceil(sampleCount / 256)); }
      pass.end();
    },
    decode(mapped) {
      const values = new Float32Array(mapped, 0, summaryBytes / 4);
      let accepted = 0, iterationSum = 0, selected = 0, replaced = 0, squared = 0;
      const maxAbs = [0, 0, 0, 0];
      for (let i = 0; i < values.length; i += 12) {
        accepted += values[i]; iterationSum += values[i + 1]; selected += values[i + 2]; replaced += values[i + 3]; squared += values[i + 8];
        for (let k = 0; k < 4; k++) maxAbs[k] = Math.max(maxAbs[k], values[i + 4 + k]);
      }
      return { accepted, replaced, selected, excluded: count - selected, meanIterations: accepted ? iterationSum / accepted : null,
        acceptance: selected ? accepted / selected : null, maxAbs,
        truthRmse: floats[10] && accepted ? Math.sqrt(squared / accepted) : null };
    },
    destroy() { summary.destroy(); samples.destroy(); config.destroy(); },
  };
}
