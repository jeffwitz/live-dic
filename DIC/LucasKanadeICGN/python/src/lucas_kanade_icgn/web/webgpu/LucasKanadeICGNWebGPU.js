/*
 * Raw browser WebGPU adapter for the same portable WGSL used by wgpu-py.
 *
 * This adapter deliberately contains host orchestration only. The FIR,
 * affine IC-GN solve, convergence rules, and median replacement are loaded
 * from the canonical shaders/icgn.wgsl package resource.
 */

import {
  imageToGray,
  makeFIRWeights,
  prepareReferenceData,
} from '../common/ReferencePreparation.js';

const SUBSET_SIZE = 16;
const DEFAULT_MAX_ITERATIONS = 30;
const DEFAULT_TOLERANCE = 1e-3;
const DEFAULT_ZNCC_MIN = 0.8;
const FIR_RADIUS = 15;
const SHADER_URL = new URL('../../shaders/icgn.wgsl', import.meta.url);

const configuration = {
  iterations: DEFAULT_MAX_ITERATIONS,
  tolerance: DEFAULT_TOLERANCE,
  znccMin: DEFAULT_ZNCC_MIN,
};

let adapter = null;
let device = null;
let shaderModule = null;
let pipelines = null;
let bindGroupLayout = null;
let bindGroup = null;
let sourceTexture = null;
let buffers = null;
let querySet = null;
let queryResolveBuffer = null;
let timestampQueries = false;
let referenceReady = false;
let FRAME_WIDTH = 0;
let FRAME_HEIGHT = 0;
let PIXEL_COUNT = 0;
let GRID_WIDTH = 0;
let GRID_HEIGHT = 0;
let GRID_COUNT = 0;

const loadImage = async (dataUrl) => {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return createImageBitmap(blob);
};

const createBufferWithData = (label, data, usage) => {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, data.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
};

const destroyRuntimeResources = () => {
  if (sourceTexture) sourceTexture.destroy();
  if (buffers) Object.values(buffers).forEach((buffer) => buffer.destroy());
  if (querySet) querySet.destroy();
  if (queryResolveBuffer) queryResolveBuffer.destroy();
  sourceTexture = null;
  buffers = null;
  querySet = null;
  queryResolveBuffer = null;
  bindGroup = null;
  bindGroupLayout = null;
  pipelines = null;
  referenceReady = false;
};

const ensureDevice = async () => {
  if (device) return device;
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser');
  adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter is available');
  timestampQueries = adapter.features.has('timestamp-query');
  device = await adapter.requestDevice({
    requiredFeatures: timestampQueries ? ['timestamp-query'] : [],
  });
  const shaderResponse = await fetch(SHADER_URL);
  if (!shaderResponse.ok) {
    throw new Error(`Cannot load the shared WGSL shader: HTTP ${shaderResponse.status}`);
  }
  shaderModule = device.createShaderModule({
    label: 'DIC-RT shared portable WGSL',
    code: await shaderResponse.text(),
  });
  const compilation = await shaderModule.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === 'error');
  if (errors.length) {
    throw new Error(errors.map((message) => message.message).join('\n'));
  }
  return device;
};

const createRuntime = async (width, height) => {
  if (!Number.isInteger(width) || !Number.isInteger(height)
      || width < SUBSET_SIZE || height < SUBSET_SIZE) {
    throw new Error(`Image dimensions must be integers of at least ${SUBSET_SIZE} pixels`);
  }
  if (buffers && width === FRAME_WIDTH && height === FRAME_HEIGHT) return false;
  await ensureDevice();
  destroyRuntimeResources();
  FRAME_WIDTH = width;
  FRAME_HEIGHT = height;
  PIXEL_COUNT = width * height;
  GRID_WIDTH = Math.floor(width / SUBSET_SIZE);
  GRID_HEIGHT = Math.floor(height / SUBSET_SIZE);
  GRID_COUNT = GRID_WIDTH * GRID_HEIGHT;

  const largestBufferBytes = PIXEL_COUNT * 4 * Float32Array.BYTES_PER_ELEMENT;
  if (largestBufferBytes > device.limits.maxStorageBufferBindingSize
      || largestBufferBytes > device.limits.maxBufferSize) {
    throw new Error(`The ${width}x${height} reference exceeds this adapter's buffer limits`);
  }

  sourceTexture = device.createTexture({
    label: 'DIC-RT target luma',
    size: [width, height, 1],
    format: 'r8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  const storage = GPUBufferUsage.STORAGE;
  buffers = {
    config: device.createBuffer({
      label: 'DIC-RT configuration',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    temporary: device.createBuffer({
      label: 'DIC-RT FIR temporary', size: PIXEL_COUNT * 4, usage: storage,
    }),
    coefficients: device.createBuffer({
      label: 'DIC-RT target coefficients', size: PIXEL_COUNT * 4, usage: storage,
    }),
    reference: device.createBuffer({
      label: 'DIC-RT reference', size: PIXEL_COUNT * 16,
      usage: storage | GPUBufferUsage.COPY_DST,
    }),
    inverse: device.createBuffer({
      label: 'DIC-RT inverse Hessians', size: GRID_COUNT * 36 * 4,
      usage: storage | GPUBufferUsage.COPY_DST,
    }),
    metadata: device.createBuffer({
      label: 'DIC-RT metadata', size: GRID_COUNT * 16 * 4,
      usage: storage | GPUBufferUsage.COPY_DST,
    }),
    state: device.createBuffer({
      label: 'DIC-RT state', size: GRID_COUNT * 8 * 4,
      usage: storage | GPUBufferUsage.COPY_DST,
    }),
    fir: createBufferWithData(
      'DIC-RT FIR weights',
      makeFIRWeights(FIR_RADIUS),
      storage,
    ),
    filteredState: device.createBuffer({
      label: 'DIC-RT filtered state', size: GRID_COUNT * 8 * 4,
      usage: storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }),
  };

  bindGroupLayout = device.createBindGroupLayout({
    label: 'DIC-RT shared WGSL bindings',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = async (label, entryPoint) => {
    try {
      return await device.createComputePipelineAsync({
        label,
        layout,
        compute: { module: shaderModule, entryPoint },
      });
    } catch (error) {
      const compilation = await shaderModule.getCompilationInfo();
      const diagnostics = compilation.messages.map((message) => {
        const location = message.lineNum
          ? `${message.lineNum}:${message.linePos}`
          : 'unknown location';
        return `${message.type} at ${location}: ${message.message}`;
      });
      const details = diagnostics.length ? `\n${diagnostics.join('\n')}` : '';
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`${message}${details}`);
    }
  };
  pipelines = {
    prefilterX: await pipeline('DIC-RT shared prefilter X', 'prefilter_x'),
    prefilterY: await pipeline('DIC-RT shared prefilter Y', 'prefilter_y'),
    solver: await pipeline('DIC-RT shared affine solver', 'solve'),
    median: await pipeline('DIC-RT shared median filter', 'median_filter'),
  };
  const entries = [
    { binding: 0, resource: sourceTexture.createView() },
    ...['config', 'temporary', 'coefficients', 'reference', 'inverse', 'metadata', 'state', 'fir', 'filteredState']
      .map((name, index) => ({ binding: index + 1, resource: { buffer: buffers[name] } })),
  ];
  bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });
  if (timestampQueries) {
    querySet = device.createQuerySet({ type: 'timestamp', count: 8 });
    queryResolveBuffer = device.createBuffer({
      size: 8 * BigUint64Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
  }
  updateConfig();
  referenceReady = false;
  return true;
};

const updateConfig = () => {
  if (!buffers) return;
  const bytes = new ArrayBuffer(32);
  const view = new DataView(bytes);
  const integers = [
    FRAME_WIDTH,
    FRAME_HEIGHT,
    GRID_WIDTH,
    GRID_COUNT,
    configuration.iterations,
    PIXEL_COUNT,
  ];
  integers.forEach((value, index) => view.setUint32(index * 4, value, true));
  view.setFloat32(24, configuration.tolerance, true);
  view.setFloat32(28, configuration.znccMin, true);
  device.queue.writeBuffer(buffers.config, 0, bytes);
};

const configure = (options = {}) => {
  const next = {
    iterations: options.iterations === undefined
      ? configuration.iterations : Number(options.iterations),
    tolerance: options.tolerance === undefined
      ? configuration.tolerance : Number(options.tolerance),
    znccMin: options.znccMin === undefined
      ? configuration.znccMin : Number(options.znccMin),
  };
  if (!Number.isInteger(next.iterations) || next.iterations < 1 || next.iterations > 30) {
    throw new Error('iterations must be an integer between 1 and 30');
  }
  if (!Number.isFinite(next.tolerance) || next.tolerance <= 0) {
    throw new Error('tolerance must be greater than zero');
  }
  if (!Number.isFinite(next.znccMin) || next.znccMin < 0 || next.znccMin > 1) {
    throw new Error('znccMin must be between 0 and 1');
  }
  Object.assign(configuration, next);
  updateConfig();
  return { ...configuration };
};

const setTarget = (image) => {
  const gray = imageToGray(image, FRAME_WIDTH, FRAME_HEIGHT, 1);
  const bytes = Uint8Array.from(gray);
  device.queue.writeTexture(
    { texture: sourceTexture },
    bytes,
    { bytesPerRow: FRAME_WIDTH, rowsPerImage: FRAME_HEIGHT },
    [FRAME_WIDTH, FRAME_HEIGHT, 1],
  );
};

const executePipeline = async ({ measureGpu = false } = {}) => {
  const encoder = device.createCommandEncoder({ label: 'DIC-RT shared WGSL registration' });
  encoder.clearBuffer(buffers.state);
  encoder.clearBuffer(buffers.filteredState);
  const jobs = [
    [pipelines.prefilterX, Math.ceil(PIXEL_COUNT / 256)],
    [pipelines.prefilterY, Math.ceil(PIXEL_COUNT / 256)],
    [pipelines.solver, GRID_COUNT],
    [pipelines.median, Math.ceil(GRID_COUNT / 256)],
  ];
  jobs.forEach(([pipeline, workgroups], index) => {
    const timestampWrites = measureGpu && timestampQueries ? {
      querySet,
      beginningOfPassWriteIndex: 2 * index,
      endOfPassWriteIndex: 2 * index + 1,
    } : undefined;
    const pass = encoder.beginComputePass({ timestampWrites });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  });
  let timestampReadback = null;
  if (measureGpu && timestampQueries) {
    timestampReadback = device.createBuffer({
      size: 8 * BigUint64Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.resolveQuerySet(querySet, 0, 8, queryResolveBuffer, 0);
    encoder.copyBufferToBuffer(queryResolveBuffer, 0, timestampReadback, 0, 64);
  }
  device.queue.submit([encoder.finish()]);
  if (!timestampReadback) {
    await device.queue.onSubmittedWorkDone();
    return null;
  }
  await timestampReadback.mapAsync(GPUMapMode.READ);
  const timestamps = new BigUint64Array(timestampReadback.getMappedRange());
  let nanoseconds = 0n;
  for (let index = 0; index < 4; ++index) {
    nanoseconds += timestamps[2 * index + 1] - timestamps[2 * index];
  }
  const gpuMs = Number(nanoseconds) / 1e6;
  timestampReadback.unmap();
  timestampReadback.destroy();
  return gpuMs;
};

const readResult = async (timing) => {
  const size = GRID_COUNT * 8 * 4;
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: 'DIC-RT shared WGSL readback' });
  encoder.copyBufferToBuffer(buffers.filteredState, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const packed = new Float32Array(readback.getMappedRange());
  const parameters = new Float32Array(GRID_COUNT * 6);
  const status = new Float32Array(GRID_COUNT);
  const iterations = new Float32Array(GRID_COUNT);
  let accepted = 0;
  let replaced = 0;
  let meanX = 0;
  let meanY = 0;
  let iterationTotal = 0;
  for (let index = 0; index < GRID_COUNT; ++index) {
    const source = 8 * index;
    const target = 6 * index;
    for (let component = 0; component < 6; ++component) {
      parameters[target + component] = packed[source + component];
    }
    status[index] = packed[source + 6];
    iterations[index] = packed[source + 7];
    if (status[index] > 1.5) replaced += 1;
    if (status[index] >= 0.5) {
      accepted += 1;
      meanX += parameters[target];
      meanY += parameters[target + 3];
      iterationTotal += iterations[index];
    }
  }
  const result = {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    subsetSize: SUBSET_SIZE,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    parameters: Array.from(parameters),
    status: Array.from(status),
    iterations: Array.from(iterations),
    accepted,
    replaced,
    meanTranslation: accepted ? [meanX / accepted, meanY / accepted] : [null, null],
    meanIterations: accepted ? iterationTotal / accepted : null,
    timing,
    config: { ...configuration },
  };
  readback.unmap();
  readback.destroy();
  return result;
};

const setReference = async (dataUrl, options = {}) => {
  await ensureDevice();
  configure(options);
  const image = await loadImage(dataUrl);
  const started = performance.now();
  const runtimeStarted = performance.now();
  await createRuntime(image.width, image.height);
  const runtimeBuildWallMs = performance.now() - runtimeStarted;
  const referenceStarted = performance.now();
  try {
    const prepared = prepareReferenceData(image, {
      width: FRAME_WIDTH,
      height: FRAME_HEIGHT,
      subsetSize: SUBSET_SIZE,
      intensityScale: 1,
    });
    device.queue.writeBuffer(buffers.reference, 0, prepared.reference);
    device.queue.writeBuffer(buffers.inverse, 0, prepared.inverseHessian);
    device.queue.writeBuffer(buffers.metadata, 0, prepared.metadata);
  } finally {
    image.close();
  }
  await device.queue.onSubmittedWorkDone();
  const referenceBuildWallMs = performance.now() - referenceStarted;
  referenceReady = true;
  return {
    preparationWallMs: performance.now() - started,
    runtimeBuildWallMs,
    referenceBuildWallMs,
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    subsetSize: SUBSET_SIZE,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    config: { ...configuration },
    device: (await info()).device,
  };
};

const register = async (dataUrl, options = {}) => {
  await ensureDevice();
  if (!referenceReady) throw new Error('A reference image must be configured first');
  configure(options);
  const image = await loadImage(dataUrl);
  try {
    setTarget(image);
  } finally {
    image.close();
  }
  const started = performance.now();
  const gpuMs = await executePipeline({ measureGpu: true });
  const wallMs = performance.now() - started;
  return readResult({ gpuMs, wallMs });
};

const percentile = (values, probability) => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(probability * sorted.length) - 1)];
};

const benchmark = async (dataUrl, repeats = 20, warmups = 3, options = {}) => {
  await ensureDevice();
  if (!referenceReady) throw new Error('A reference image must be configured first');
  configure(options);
  const image = await loadImage(dataUrl);
  try {
    setTarget(image);
  } finally {
    image.close();
  }
  for (let index = 0; index < warmups; ++index) await executePipeline();
  const gpuTimes = [];
  const wallTimes = [];
  for (let index = 0; index < repeats; ++index) {
    const started = performance.now();
    const gpuMs = await executePipeline({ measureGpu: true });
    wallTimes.push(performance.now() - started);
    if (gpuMs !== null) gpuTimes.push(gpuMs);
  }
  const registration = await readResult({
    gpuMs: gpuTimes.length ? gpuTimes[gpuTimes.length - 1] : null,
    wallMs: wallTimes[wallTimes.length - 1],
  });
  return {
    repeats,
    warmups,
    gpuTimes,
    wallTimes,
    gpuMedianMs: gpuTimes.length ? percentile(gpuTimes, 0.5) : null,
    gpuP95Ms: gpuTimes.length ? percentile(gpuTimes, 0.95) : null,
    wallMedianMs: percentile(wallTimes, 0.5),
    wallP95Ms: percentile(wallTimes, 0.95),
    device: (await info()).device,
    registration,
  };
};

const info = async () => {
  await ensureDevice();
  const adapterInfo = adapter.info || {};
  const identity = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device]
    .filter(Boolean).join(' ');
  return {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    subsetSize: SUBSET_SIZE,
    device: `Raw WebGPU shared WGSL${identity ? ` (${identity})` : ''}`,
    timerQuery: timestampQueries,
    backend: 'raw-webgpu-wgsl',
    threeRevision: null,
    timestampQueries,
    vendor: adapterInfo.vendor || null,
    architecture: adapterInfo.architecture || null,
    adapterDevice: adapterInfo.device || null,
    description: adapterInfo.description || null,
    subgroups: false,
    subgroupWidth: 0,
    solver: 'portable-workgroup',
  };
};

window.ICGNHeadless = { info, configure, setReference, register, benchmark };
window.ICGNHeadlessReady = ensureDevice();
