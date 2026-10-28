/*
 * Raw browser WebGPU adapter for the same portable WGSL used by wgpu-py.
 *
 * This adapter deliberately contains host orchestration only. Pyramid
 * reduction and seed transfer come from full_pyramid.wgsl; FIR, affine IC-GN,
 * convergence, and median replacement come from the canonical icgn.wgsl.
 */

import {
  imageToGray,
  makeFIRWeights,
  prepareReferenceGray,
  reduceHalfGray,
} from '../common/ReferencePreparation.js';

const SUBSET_SIZE = 16;
const DEFAULT_MAX_ITERATIONS = 30;
const DEFAULT_TOLERANCE = 1e-3;
const DEFAULT_ZNCC_MIN = 0.8;
const FIR_RADIUS = 15;
const SHADER_URL = new URL('../../shaders/icgn.wgsl', import.meta.url);
const PYRAMID_SHADER_URL = new URL('../../shaders/full_pyramid.wgsl', import.meta.url);
const DEFAULT_PYRAMID_LEVELS = 2;
const ITERATION_SCHEDULES = {
  1: [30],
  2: [12, 30],
  3: [6, 8, 20],
};
const TOLERANCE_SCHEDULES = {
  1: [1e-3],
  2: [5e-3, 1e-3],
  3: [1e-2, 5e-3, 1e-3],
};

const configuration = {
  iterations: DEFAULT_MAX_ITERATIONS,
  tolerance: DEFAULT_TOLERANCE,
  znccMin: DEFAULT_ZNCC_MIN,
  pyramidLevels: DEFAULT_PYRAMID_LEVELS,
  iterationsPerLevel: [...ITERATION_SCHEDULES[DEFAULT_PYRAMID_LEVELS]],
  tolerancePerLevel: [...TOLERANCE_SCHEDULES[DEFAULT_PYRAMID_LEVELS]],
};

let adapter = null;
let device = null;
let shaderModule = null;
let pyramidShaderModule = null;
let pipelines = null;
let bindGroupLayout = null;
let seededBindGroupLayout = null;
let reduceXBindGroupLayout = null;
let reduceYBindGroupLayout = null;
let seedTransferBindGroupLayout = null;
let bindGroup = null;
let sourceTexture = null;
let buffers = null;
let querySet = null;
let queryResolveBuffer = null;
let timestampQueries = false;
let referenceReady = false;
let levels = [];
let reductionEdges = [];
let seedEdges = new Map();
let runtimePyramidLevels = 0;
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
  levels.forEach((level) => {
    level.sourceTexture.destroy();
    Object.values(level.buffers).forEach((buffer) => buffer.destroy());
  });
  reductionEdges.forEach((edge) => edge.intermediate.destroy());
  seedEdges.forEach((edge) => edge.config.destroy());
  if (querySet) querySet.destroy();
  if (queryResolveBuffer) queryResolveBuffer.destroy();
  levels = [];
  reductionEdges = [];
  seedEdges = new Map();
  runtimePyramidLevels = 0;
  sourceTexture = null;
  buffers = null;
  querySet = null;
  queryResolveBuffer = null;
  bindGroup = null;
  bindGroupLayout = null;
  seededBindGroupLayout = null;
  reduceXBindGroupLayout = null;
  reduceYBindGroupLayout = null;
  seedTransferBindGroupLayout = null;
  pipelines = null;
  referenceReady = false;
};

const ensureDevice = async () => {
  if (device) return device;
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser');
  adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter is available');
  const requestedStorageBuffers = Math.min(
    10,
    adapter.limits.maxStorageBuffersPerShaderStage,
  );
  timestampQueries = adapter.features.has('timestamp-query');
  device = await adapter.requestDevice({
    requiredFeatures: timestampQueries ? ['timestamp-query'] : [],
    requiredLimits: {
      maxStorageBuffersPerShaderStage: requestedStorageBuffers,
    },
  });
  const [shaderResponse, pyramidShaderResponse] = await Promise.all([
    fetch(SHADER_URL),
    fetch(PYRAMID_SHADER_URL),
  ]);
  if (!shaderResponse.ok || !pyramidShaderResponse.ok) {
    throw new Error(
      `Cannot load the shared WGSL shaders: HTTP ${shaderResponse.status}`
      + ` / ${pyramidShaderResponse.status}`,
    );
  }
  shaderModule = device.createShaderModule({
    label: 'DIC-RT shared portable WGSL',
    code: await shaderResponse.text(),
  });
  pyramidShaderModule = device.createShaderModule({
    label: 'DIC-RT shared full-pyramid WGSL',
    code: await pyramidShaderResponse.text(),
  });
  const compilation = await Promise.all([
    shaderModule.getCompilationInfo(),
    pyramidShaderModule.getCompilationInfo(),
  ]);
  const errors = compilation.flatMap((result) => result.messages)
    .filter((message) => message.type === 'error');
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
  const levelCount = configuration.pyramidLevels;
  const coarsestScale = 2 ** (levelCount - 1);
  if (Math.floor(width / coarsestScale) < SUBSET_SIZE
      || Math.floor(height / coarsestScale) < SUBSET_SIZE) {
    throw new Error(`The coarsest pyramid level must be at least ${SUBSET_SIZE} pixels`);
  }
  if (buffers && width === FRAME_WIDTH && height === FRAME_HEIGHT
      && runtimePyramidLevels === levelCount) return false;
  await ensureDevice();
  if (levelCount > 1 && device.limits.maxStorageBuffersPerShaderStage < 10) {
    throw new Error(
      'A multi-level pyramid requires at least 10 storage buffers per shader stage; '
      + `this adapter exposes ${device.limits.maxStorageBuffersPerShaderStage}. `
      + 'Use pyramidLevels=1 on this adapter.',
    );
  }
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
  const baseLayoutEntries = [
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
  ];
  bindGroupLayout = device.createBindGroupLayout({
    label: 'DIC-RT shared WGSL bindings',
    entries: baseLayoutEntries,
  });
  seededBindGroupLayout = levelCount > 1 ? device.createBindGroupLayout({
    label: 'DIC-RT shared seeded WGSL bindings',
    entries: [
      ...baseLayoutEntries,
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  }) : null;
  reduceXBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float' },
      },
    ],
  });
  reduceYBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' },
      },
    ],
  });
  seedTransferBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const seededLayout = seededBindGroupLayout
    ? device.createPipelineLayout({ bindGroupLayouts: [seededBindGroupLayout] })
    : null;
  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  const reduceXLayout = device.createPipelineLayout({
    bindGroupLayouts: [reduceXBindGroupLayout],
  });
  const reduceYLayout = device.createPipelineLayout({
    bindGroupLayouts: [emptyLayout, reduceYBindGroupLayout],
  });
  const seedLayout = device.createPipelineLayout({
    bindGroupLayouts: [emptyLayout, emptyLayout, seedTransferBindGroupLayout],
  });
  const pipeline = async (label, entryPoint, module = shaderModule, pipelineLayout = layout) => {
    try {
      return await device.createComputePipelineAsync({
        label,
        layout: pipelineLayout,
        compute: { module, entryPoint },
      });
    } catch (error) {
      const compilation = await module.getCompilationInfo();
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
    seededSolver: seededLayout ? await pipeline(
      'DIC-RT shared seeded affine solver',
      'solve_seeded',
      shaderModule,
      seededLayout,
    ) : null,
    median: await pipeline('DIC-RT shared median filter', 'median_filter'),
    reduceX: await pipeline(
      'DIC-RT pyramid reduce X', 'reduce_x', pyramidShaderModule, reduceXLayout,
    ),
    reduceY: await pipeline(
      'DIC-RT pyramid reduce Y', 'reduce_y', pyramidShaderModule, reduceYLayout,
    ),
    transferSeed: await pipeline(
      'DIC-RT pyramid seed transfer', 'transfer_seed', pyramidShaderModule, seedLayout,
    ),
  };

  const storage = GPUBufferUsage.STORAGE;
  for (let levelIndex = 0; levelIndex < levelCount; ++levelIndex) {
    const scale = 2 ** levelIndex;
    const levelWidth = Math.floor(width / scale);
    const levelHeight = Math.floor(height / scale);
    const pixelCount = levelWidth * levelHeight;
    const gridWidth = Math.floor(levelWidth / SUBSET_SIZE);
    const gridHeight = Math.floor(levelHeight / SUBSET_SIZE);
    const gridCount = gridWidth * gridHeight;
    const gpuWritable = levelIndex > 0;
    const levelSourceTexture = device.createTexture({
      label: `DIC-RT target luma level ${levelIndex}`,
      size: [levelWidth, levelHeight, 1],
      format: gpuWritable ? 'rgba8unorm' : 'r8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
        | (gpuWritable ? GPUTextureUsage.STORAGE_BINDING : 0),
    });
    const levelBuffers = {
      config: device.createBuffer({
        label: `DIC-RT configuration level ${levelIndex}`,
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      temporary: device.createBuffer({
        label: `DIC-RT FIR temporary level ${levelIndex}`,
        size: pixelCount * 4,
        usage: storage,
      }),
      coefficients: device.createBuffer({
        label: `DIC-RT target coefficients level ${levelIndex}`,
        size: pixelCount * 4,
        usage: storage,
      }),
      reference: device.createBuffer({
        label: `DIC-RT reference level ${levelIndex}`,
        size: pixelCount * 16,
        usage: storage | GPUBufferUsage.COPY_DST,
      }),
      inverse: device.createBuffer({
        label: `DIC-RT inverse Hessians level ${levelIndex}`,
        size: gridCount * 36 * 4,
        usage: storage | GPUBufferUsage.COPY_DST,
      }),
      metadata: device.createBuffer({
        label: `DIC-RT metadata level ${levelIndex}`,
        size: gridCount * 16 * 4,
        usage: storage | GPUBufferUsage.COPY_DST,
      }),
      state: device.createBuffer({
        label: `DIC-RT state level ${levelIndex}`,
        size: gridCount * 8 * 4,
        usage: storage | GPUBufferUsage.COPY_DST,
      }),
      fir: createBufferWithData(
        `DIC-RT FIR weights level ${levelIndex}`,
        makeFIRWeights(FIR_RADIUS),
        storage,
      ),
      filteredState: device.createBuffer({
        label: `DIC-RT filtered state level ${levelIndex}`,
        size: gridCount * 8 * 4,
        usage: storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    };
    if (levelIndex < levelCount - 1) {
      levelBuffers.seedState = device.createBuffer({
        label: `DIC-RT seed state level ${levelIndex}`,
        size: gridCount * 8 * 4,
        usage: storage,
      });
      levelBuffers.activeMask = device.createBuffer({
        label: `DIC-RT active mask level ${levelIndex}`,
        size: gridCount * 4,
        usage: storage,
      });
    }
    const baseEntries = [
      { binding: 0, resource: levelSourceTexture.createView() },
      ...['config', 'temporary', 'coefficients', 'reference', 'inverse', 'metadata', 'state', 'fir', 'filteredState']
        .map((name, index) => ({
          binding: index + 1,
          resource: { buffer: levelBuffers[name] },
        })),
    ];
    const levelBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: baseEntries,
    });
    const seededBindGroup = levelBuffers.seedState ? device.createBindGroup({
      layout: seededBindGroupLayout,
      entries: [
        ...baseEntries,
        { binding: 10, resource: { buffer: levelBuffers.seedState } },
        { binding: 11, resource: { buffer: levelBuffers.activeMask } },
      ],
    }) : null;
    levels.push({
      index: levelIndex,
      width: levelWidth,
      height: levelHeight,
      pixelCount,
      gridWidth,
      gridHeight,
      gridCount,
      sourceTexture: levelSourceTexture,
      buffers: levelBuffers,
      bindGroup: levelBindGroup,
      seededBindGroup,
    });
  }

  for (let levelIndex = 1; levelIndex < levelCount; ++levelIndex) {
    const source = levels[levelIndex - 1];
    const destination = levels[levelIndex];
    const intermediate = device.createTexture({
      label: `DIC-RT horizontal pyramid level ${levelIndex}`,
      size: [destination.width, source.height, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    reductionEdges.push({
      source,
      destination,
      intermediate,
      horizontalBindGroup: device.createBindGroup({
        layout: reduceXBindGroupLayout,
        entries: [
          { binding: 0, resource: source.sourceTexture.createView() },
          { binding: 1, resource: intermediate.createView() },
        ],
      }),
      verticalBindGroup: device.createBindGroup({
        layout: reduceYBindGroupLayout,
        entries: [
          { binding: 0, resource: intermediate.createView() },
          { binding: 1, resource: destination.sourceTexture.createView() },
        ],
      }),
    });
  }

  for (let destinationIndex = levelCount - 2; destinationIndex >= 0; --destinationIndex) {
    const source = levels[destinationIndex + 1];
    const destination = levels[destinationIndex];
    const bytes = new ArrayBuffer(32);
    const view = new DataView(bytes);
    [source.gridWidth, source.gridHeight, destination.gridWidth, destination.gridCount]
      .forEach((value, index) => view.setUint32(index * 4, value, true));
    view.setFloat32(16, 2 ** (destinationIndex + 1), true);
    view.setFloat32(20, 2 ** destinationIndex, true);
    const config = createBufferWithData(
      `DIC-RT seed transfer level ${destinationIndex}`,
      new Uint8Array(bytes),
      GPUBufferUsage.UNIFORM,
    );
    seedEdges.set(destinationIndex, {
      config,
      bindGroup: device.createBindGroup({
        layout: seedTransferBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: source.buffers.filteredState } },
          { binding: 1, resource: { buffer: destination.buffers.seedState } },
          { binding: 2, resource: { buffer: destination.buffers.activeMask } },
          { binding: 3, resource: { buffer: config } },
        ],
      }),
    });
  }

  sourceTexture = levels[0].sourceTexture;
  buffers = levels[0].buffers;
  bindGroup = levels[0].bindGroup;
  runtimePyramidLevels = levelCount;
  if (timestampQueries) {
    const passCount = 7 * levelCount - 3;
    querySet = device.createQuerySet({ type: 'timestamp', count: 2 * passCount });
    queryResolveBuffer = device.createBuffer({
      size: 2 * passCount * BigUint64Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
  }
  updateConfig();
  referenceReady = false;
  return true;
};

const updateConfig = () => {
  if (!levels.length) return;
  levels.forEach((level) => {
    const scheduleIndex = configuration.pyramidLevels - 1 - level.index;
    const bytes = new ArrayBuffer(32);
    const view = new DataView(bytes);
    const integers = [
      level.width,
      level.height,
      level.gridWidth,
      level.gridCount,
      configuration.iterationsPerLevel[scheduleIndex],
      level.pixelCount,
    ];
    integers.forEach((value, index) => view.setUint32(index * 4, value, true));
    view.setFloat32(24, configuration.tolerancePerLevel[scheduleIndex], true);
    view.setFloat32(28, configuration.znccMin, true);
    device.queue.writeBuffer(level.buffers.config, 0, bytes);
  });
};

const configure = (options = {}, allowPyramidRebuild = false) => {
  const pyramidLevels = options.pyramidLevels === undefined
    ? configuration.pyramidLevels : Number(options.pyramidLevels);
  if (!Number.isInteger(pyramidLevels) || pyramidLevels < 1 || pyramidLevels > 3) {
    throw new Error('pyramidLevels must be an integer between 1 and 3');
  }
  if (!allowPyramidRebuild && levels.length && pyramidLevels !== runtimePyramidLevels) {
    throw new Error('Changing pyramidLevels requires setReference() to rebuild the runtime');
  }
  const changedLevels = pyramidLevels !== configuration.pyramidLevels;
  const iterationsPerLevel = options.iterationsPerLevel === undefined
    ? (changedLevels
      ? [...ITERATION_SCHEDULES[pyramidLevels]]
      : [...configuration.iterationsPerLevel])
    : Array.from(options.iterationsPerLevel, Number);
  const tolerancePerLevel = options.tolerancePerLevel === undefined
    ? (changedLevels
      ? [...TOLERANCE_SCHEDULES[pyramidLevels]]
      : [...configuration.tolerancePerLevel])
    : Array.from(options.tolerancePerLevel, Number);
  if (iterationsPerLevel.length !== pyramidLevels
      || tolerancePerLevel.length !== pyramidLevels) {
    throw new Error('Per-level schedules must contain one value per pyramid level');
  }
  const next = {
    iterations: options.iterations === undefined
      ? iterationsPerLevel[pyramidLevels - 1] : Number(options.iterations),
    tolerance: options.tolerance === undefined
      ? tolerancePerLevel[pyramidLevels - 1] : Number(options.tolerance),
    znccMin: options.znccMin === undefined
      ? configuration.znccMin : Number(options.znccMin),
    pyramidLevels,
    iterationsPerLevel,
    tolerancePerLevel,
  };
  next.iterationsPerLevel[pyramidLevels - 1] = next.iterations;
  next.tolerancePerLevel[pyramidLevels - 1] = next.tolerance;
  if (!Number.isInteger(next.iterations) || next.iterations < 1 || next.iterations > 30) {
    throw new Error('iterations must be an integer between 1 and 30');
  }
  if (!Number.isFinite(next.tolerance) || next.tolerance <= 0) {
    throw new Error('tolerance must be greater than zero');
  }
  if (!Number.isFinite(next.znccMin) || next.znccMin < 0 || next.znccMin > 1) {
    throw new Error('znccMin must be between 0 and 1');
  }
  next.iterationsPerLevel.forEach((value) => {
    if (!Number.isInteger(value) || value < 1 || value > 30) {
      throw new Error('Each per-level iteration count must be between 1 and 30');
    }
  });
  next.tolerancePerLevel.forEach((value) => {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error('Each per-level tolerance must be greater than zero');
    }
  });
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
  const encoder = device.createCommandEncoder({ label: 'DIC-RT full-pyramid registration' });
  levels.forEach((level) => {
    encoder.clearBuffer(level.buffers.state);
    encoder.clearBuffer(level.buffers.filteredState);
  });
  const jobs = [];
  reductionEdges.forEach((edge) => {
    jobs.push({
      pipeline: pipelines.reduceX,
      bindGroup: edge.horizontalBindGroup,
      group: 0,
      workgroups: [Math.ceil(edge.destination.width / 16), Math.ceil(edge.source.height / 16), 1],
    });
    jobs.push({
      pipeline: pipelines.reduceY,
      bindGroup: edge.verticalBindGroup,
      group: 1,
      workgroups: [
        Math.ceil(edge.destination.width / 16),
        Math.ceil(edge.destination.height / 16),
        1,
      ],
    });
  });
  const appendLevel = (level, seeded) => {
    jobs.push(
      {
        pipeline: pipelines.prefilterX,
        bindGroup: level.bindGroup,
        group: 0,
        workgroups: [Math.ceil(level.pixelCount / 256), 1, 1],
      },
      {
        pipeline: pipelines.prefilterY,
        bindGroup: level.bindGroup,
        group: 0,
        workgroups: [Math.ceil(level.pixelCount / 256), 1, 1],
      },
      {
        pipeline: seeded ? pipelines.seededSolver : pipelines.solver,
        bindGroup: seeded ? level.seededBindGroup : level.bindGroup,
        group: 0,
        workgroups: [level.gridCount, 1, 1],
      },
      {
        pipeline: pipelines.median,
        bindGroup: level.bindGroup,
        group: 0,
        workgroups: [Math.ceil(level.gridCount / 256), 1, 1],
      },
    );
  };
  const coarsestIndex = levels.length - 1;
  appendLevel(levels[coarsestIndex], false);
  for (let destinationIndex = coarsestIndex - 1; destinationIndex >= 0; --destinationIndex) {
    const edge = seedEdges.get(destinationIndex);
    const destination = levels[destinationIndex];
    jobs.push({
      pipeline: pipelines.transferSeed,
      bindGroup: edge.bindGroup,
      group: 2,
      workgroups: [Math.ceil(destination.gridCount / 256), 1, 1],
    });
    appendLevel(destination, true);
  }
  jobs.forEach((job, index) => {
    const timestampWrites = measureGpu && timestampQueries ? {
      querySet,
      beginningOfPassWriteIndex: 2 * index,
      endOfPassWriteIndex: 2 * index + 1,
    } : undefined;
    const pass = encoder.beginComputePass({ timestampWrites });
    pass.setPipeline(job.pipeline);
    pass.setBindGroup(job.group, job.bindGroup);
    pass.dispatchWorkgroups(...job.workgroups);
    pass.end();
  });
  let timestampReadback = null;
  if (measureGpu && timestampQueries) {
    const timestampBytes = 2 * jobs.length * BigUint64Array.BYTES_PER_ELEMENT;
    timestampReadback = device.createBuffer({
      size: timestampBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.resolveQuerySet(querySet, 0, 2 * jobs.length, queryResolveBuffer, 0);
    encoder.copyBufferToBuffer(
      queryResolveBuffer,
      0,
      timestampReadback,
      0,
      timestampBytes,
    );
  }
  device.queue.submit([encoder.finish()]);
  if (!timestampReadback) {
    await device.queue.onSubmittedWorkDone();
    return null;
  }
  await timestampReadback.mapAsync(GPUMapMode.READ);
  const timestamps = new BigUint64Array(timestampReadback.getMappedRange());
  let nanoseconds = 0n;
  for (let index = 0; index < jobs.length; ++index) {
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
  configure(options, true);
  const image = await loadImage(dataUrl);
  const started = performance.now();
  const runtimeStarted = performance.now();
  await createRuntime(image.width, image.height);
  const runtimeBuildWallMs = performance.now() - runtimeStarted;
  const referenceStarted = performance.now();
  try {
    const referenceLevels = [{
      values: imageToGray(image, FRAME_WIDTH, FRAME_HEIGHT, 1),
      width: FRAME_WIDTH,
      height: FRAME_HEIGHT,
    }];
    for (let levelIndex = 1; levelIndex < configuration.pyramidLevels; ++levelIndex) {
      const previous = referenceLevels[levelIndex - 1];
      referenceLevels.push(reduceHalfGray(previous.values, previous.width, previous.height));
    }
    levels.forEach((level, levelIndex) => {
      const source = referenceLevels[levelIndex];
      const prepared = prepareReferenceGray(source.values, {
        width: source.width,
        height: source.height,
        subsetSize: SUBSET_SIZE,
      });
      device.queue.writeBuffer(level.buffers.reference, 0, prepared.reference);
      device.queue.writeBuffer(level.buffers.inverse, 0, prepared.inverseHessian);
      device.queue.writeBuffer(level.buffers.metadata, 0, prepared.metadata);
    });
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
    pyramidLevels: configuration.pyramidLevels,
    iterationsPerLevel: [...configuration.iterationsPerLevel],
    tolerancePerLevel: [...configuration.tolerancePerLevel],
    solver: configuration.pyramidLevels > 1
      ? 'portable-workgroup-full-pyramid'
      : 'portable-workgroup',
  };
};

window.ICGNHeadless = { info, configure, setReference, register, benchmark };
window.ICGNHeadlessReady = ensureDevice();
