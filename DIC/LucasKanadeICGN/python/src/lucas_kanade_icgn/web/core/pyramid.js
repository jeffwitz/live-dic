import { createTemporalGraph, validateTemporal } from './temporal.js';
import { validateMask, planRoi, reduceMask, seedPermissions } from './roi.js';
import { decodePacked } from './result.js';
/*
 * Raw browser WebGPU adapter for the same portable WGSL used by wgpu-py.
 *
 * This adapter deliberately contains host orchestration only. Pyramid
 * reduction and seed transfer come from full_pyramid.wgsl; FIR, affine IC-GN,
 * convergence, and displacement UOD come from the canonical icgn.wgsl.
 */

import {
  imageToGray,
  makeFIRWeights,
  prepareReferenceGray,
  reduceHalfGray,
} from './reference.js';

export function createPyramidHost(options = {}) {
validateTemporal(options);
const SUBSET_SIZE = Number(options.subsetSize ?? 16);
if (![8, 15, 16, 17, 31, 32, 33].includes(SUBSET_SIZE)) {
  throw new Error('Unsupported subsetSize');
}
const SUBSET_STRIDE = Number(options.subsetStride ?? SUBSET_SIZE);
if (!Number.isInteger(SUBSET_STRIDE) || SUBSET_STRIDE < 1 || SUBSET_STRIDE > SUBSET_SIZE) {
  throw new Error('subsetStride must be an integer between 1 and subsetSize');
}
const DEFAULT_MAX_ITERATIONS = 30;
const DEFAULT_TOLERANCE = 1e-3;
const DEFAULT_ZNCC_MIN = 0.8;
const FIR_RADIUS = 15;
const SHADER_URL = new URL('../../shaders/icgn.wgsl', import.meta.url);
const PYRAMID_SHADER_URL = new URL('../../shaders/full_pyramid.wgsl', import.meta.url);
const DEFAULT_PYRAMID_LEVELS = 2;
const ITERATION_SCHEDULES = {
  1: [30],
  2: [8, 16],
  3: [6, 8, 16],
};
const TOLERANCE_SCHEDULES = {
  1: [1e-3],
  2: [5e-3, 5e-3],
  3: [1e-2, 5e-3, 5e-3],
};
const INTERPOLATION_CONSTANTS = {
  bspline: 0,
  linear: 1,
  cubic: 2,
};
const MOTION_MODEL_CONSTANTS = { affine: 0, translation: 1, quadratic: 2 };
const OUTLIER_POLICY_CONSTANTS = { flag: 0, replace: 1, disabled: 2, median: 3 };

const configuration = {
  subsetStride: SUBSET_STRIDE,
  iterations: DEFAULT_MAX_ITERATIONS,
  tolerance: DEFAULT_TOLERANCE,
  znccMin: DEFAULT_ZNCC_MIN,
  pyramidLevels: DEFAULT_PYRAMID_LEVELS,
  iterationsPerLevel: [...ITERATION_SCHEDULES[DEFAULT_PYRAMID_LEVELS]],
  tolerancePerLevel: [...TOLERANCE_SCHEDULES[DEFAULT_PYRAMID_LEVELS]],
  interpolation: 'bspline',
  motionModel: 'affine',
  outlierPolicy: 'replace',
};

let roiMask = null;
let roiPlans = [];
let temporal = null;
let adapter = null;
let device = null;
let ownsDevice = false;
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
let runtimeMotionModel = null;
let runtimeOutlierPolicy = null;
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
  temporal?.destroy(); temporal = null;
  levels.forEach((level) => {
    level.sourceTexture.destroy();
    Object.values(level.buffers).forEach((buffer) => buffer.destroy());
  });
  reductionEdges.forEach((edge) => edge.intermediate.destroy());
  seedEdges.forEach((edge) => { edge.config.destroy(); edge.links?.destroy(); });
  if (querySet) querySet.destroy();
  if (queryResolveBuffer) queryResolveBuffer.destroy();
  levels = [];
  reductionEdges = [];
  seedEdges = new Map();
  runtimePyramidLevels = 0;
  runtimeMotionModel = null;
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
  if (options.device) {
    device = options.device;
  }
  if (!device && !globalThis.navigator?.gpu) throw new Error('WebGPU is unavailable in this browser');
  if (!device) {
    adapter = await globalThis.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter is available');
    timestampQueries = false; // Wall time is authoritative until query sanity is qualified.
    device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 8 } });
    ownsDevice = true;
  }
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
      && runtimePyramidLevels === levelCount
      && runtimeMotionModel === configuration.motionModel
      && runtimeOutlierPolicy === configuration.outlierPolicy) return false;
  await ensureDevice();
  destroyRuntimeResources();
  FRAME_WIDTH = width;
  FRAME_HEIGHT = height;
  PIXEL_COUNT = width * height;
  GRID_WIDTH = 1 + Math.floor((width - SUBSET_SIZE) / SUBSET_STRIDE);
  GRID_HEIGHT = 1 + Math.floor((height - SUBSET_SIZE) / SUBSET_STRIDE);
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
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ];
  bindGroupLayout = device.createBindGroupLayout({
    label: 'DIC-RT shared WGSL bindings',
    entries: baseLayoutEntries,
  });
  seededBindGroupLayout = device.createBindGroupLayout({
    label: 'DIC-RT shared seeded WGSL bindings',
    entries: [
      ...baseLayoutEntries,
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  reduceXBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float' },
      },
    ],
  });
  reduceYBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' },
      },
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
      ...(roiMask ? [{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }] : []),
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
  const pipeline = async (
    label,
    entryPoint,
    module = shaderModule,
    pipelineLayout = layout,
    constants = undefined,
  ) => {
    try {
      return await device.createComputePipelineAsync({
        label,
        layout: pipelineLayout,
        compute: { module, entryPoint, constants },
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
  const solvers = {};
  const seededSolvers = {};
  const fineSolvers = {};
  const fineSeededSolvers = {};
  for (const [mode, constant] of Object.entries(INTERPOLATION_CONSTANTS)) {
    const constants = {
      INTERPOLATION_MODE: constant,
      MOTION_MODEL: MOTION_MODEL_CONSTANTS[configuration.motionModel],
      SUBSET_SIZE,
      SUBSET_STRIDE,
      ROI_ENABLED: +!!roiMask,
    };
    solvers[mode] = await pipeline(
      `DIC-RT shared ${mode} affine solver`, 'solve', shaderModule, layout, constants,
    );
    seededSolvers[mode] = seededLayout ? await pipeline(
      `DIC-RT shared seeded ${mode} affine solver`,
      'solve_seeded',
      shaderModule,
      seededLayout,
      constants,
    ) : null;
  }
  const prefilterXFast = await pipeline(
    'DIC-RT shared prefilter X', 'prefilter_x', shaderModule, layout,
    { PREFILTER_CLAMP_HALO: 0 },
  );
  const prefilterYFast = await pipeline(
    'DIC-RT shared prefilter Y', 'prefilter_y', shaderModule, layout,
    { PREFILTER_CLAMP_HALO: 0 },
  );
  for (const [mode, constant] of Object.entries(INTERPOLATION_CONSTANTS)) {
    const constants = { INTERPOLATION_MODE: constant, MOTION_MODEL: MOTION_MODEL_CONSTANTS[configuration.motionModel],
      SUBSET_SIZE, SUBSET_STRIDE, ROI_ENABLED: +!!roiMask,
      DIVERGENCE_PATIENCE: options.divergencePatience ?? 0,
      DIVERGENCE_MIN_ITERATIONS: options.divergenceMinIterations ?? 4 };
    fineSolvers[mode] = options.divergencePatience ? await pipeline('Fine stagnation solver', 'solve', shaderModule, layout, constants) : solvers[mode];
    fineSeededSolvers[mode] = options.divergencePatience ? await pipeline('Fine seeded stagnation solver', 'solve_seeded', shaderModule, seededLayout, constants) : seededSolvers[mode];
  }
  const smallestScale = 2 ** (configuration.pyramidLevels - 1);
  const needsSafeX = Math.floor(FRAME_WIDTH / smallestScale) < 44;
  const needsSafeY = Math.floor(FRAME_HEIGHT / smallestScale) < 44;
  const prefilterXSafe = needsSafeX ? await pipeline(
    'DIC-RT shared safe-halo prefilter X', 'prefilter_x', shaderModule, layout,
    { PREFILTER_CLAMP_HALO: 1 },
  ) : prefilterXFast;
  const prefilterYSafe = needsSafeY ? await pipeline(
    'DIC-RT shared safe-halo prefilter Y', 'prefilter_y', shaderModule, layout,
    { PREFILTER_CLAMP_HALO: 1 },
  ) : prefilterYFast;
  pipelines = {
    prefilterXFast,
    prefilterYFast,
    prefilterXSafe,
    prefilterYSafe,
    solvers,
    seededSolvers,
    fineSolvers,
    fineSeededSolvers,
    median: await pipeline(
      'DIC-RT shared displacement UOD', 'median_filter', shaderModule, layout,
      {
        MOTION_MODEL: MOTION_MODEL_CONSTANTS[configuration.motionModel],
        OUTLIER_POLICY: OUTLIER_POLICY_CONSTANTS[configuration.outlierPolicy],
        ROI_ENABLED: +!!roiMask,
        SUBSET_SIZE,
        SUBSET_STRIDE,
      },
    ),
    reduceX: await pipeline(
      'DIC-RT pyramid reduce X', 'reduce_x', pyramidShaderModule, reduceXLayout,
    ),
    reduceY: await pipeline(
      'DIC-RT fused pyramid reduction', 'reduce_fused', pyramidShaderModule, reduceYLayout,
    ),
    transferSeed: await pipeline(
      'DIC-RT pyramid seed transfer', roiMask ? 'transfer_seed_roi' : 'transfer_seed', pyramidShaderModule, seedLayout,
      {
        MOTION_MODEL: MOTION_MODEL_CONSTANTS[configuration.motionModel],
        SUBSET_SIZE,
        SUBSET_STRIDE,
      },
    ),
  };

  const storage = GPUBufferUsage.STORAGE;
  const parameterCount = configuration.motionModel === 'quadratic' ? 12 : 6;
  const stateStride = configuration.motionModel === 'quadratic' ? 16 : 8;
  const metadataStride = configuration.motionModel === 'quadratic' ? 28 : 16;
  for (let levelIndex = 0; levelIndex < levelCount; ++levelIndex) {
    const scale = 2 ** levelIndex;
    const levelWidth = Math.floor(width / scale);
    const levelHeight = Math.floor(height / scale);
    const pixelCount = levelWidth * levelHeight;
    const gridWidth = 1 + Math.floor((levelWidth - SUBSET_SIZE) / SUBSET_STRIDE);
    const gridHeight = 1 + Math.floor((levelHeight - SUBSET_SIZE) / SUBSET_STRIDE);
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
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      temporary: device.createBuffer({
        label: `DIC-RT packed FIR planes level ${levelIndex}`,
        size: pixelCount * 8,
        usage: storage,
      }),
      reference: device.createBuffer({
        label: `DIC-RT reference level ${levelIndex}`,
        size: pixelCount * 16,
        usage: storage | GPUBufferUsage.COPY_DST,
      }),
      metadata: device.createBuffer({
        label: `DIC-RT metadata and inverse Hessians level ${levelIndex}`,
        size: gridCount * (metadataStride + (roiMask ? 4 : 0) + parameterCount * parameterCount) * 4,
        usage: storage | GPUBufferUsage.COPY_DST,
      }),
      state: device.createBuffer({
        label: `DIC-RT state level ${levelIndex}`,
        size: gridCount * stateStride * 4,
        usage: storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      fir: createBufferWithData(
        `DIC-RT FIR weights level ${levelIndex}`,
        makeFIRWeights(FIR_RADIUS),
        storage,
      ),
      filteredState: device.createBuffer({
        label: `DIC-RT filtered state level ${levelIndex}`,
        size: gridCount * stateStride * 4,
        usage: storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    };
    { // Coarsest temporal initialization also needs a seed buffer.
      levelBuffers.seedState = device.createBuffer({
        label: `DIC-RT seed state level ${levelIndex}`,
        size: gridCount * stateStride * 4,
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
      ...[[1, 'config'], [2, 'temporary'], [4, 'reference'], [6, 'metadata'],
        [7, 'state'], [8, 'fir'], [9, 'filteredState']].map(([binding, name]) => ({
          binding, resource: { buffer: levelBuffers[name] },
        })),
    ];
    const prefilterBindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: baseEntries });
    baseEntries.find((e) => e.binding === 2).resource.size = pixelCount * 4;
    const levelBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: baseEntries,
    });
    const seededBindGroup = device.createBindGroup({
      layout: seededBindGroupLayout,
      entries: [
        ...baseEntries,
        { binding: 10, resource: { buffer: levelBuffers.seedState } },
        { binding: 11, resource: { buffer: levelBuffers.activeMask } },
      ],
    });
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
      roi: roiPlans[levelIndex] ?? null,
      inverseOffset: gridCount * (metadataStride + (roiMask ? 4 : 0)) / 4,
      seededBindGroup,
      prefilterBindGroup,
    });
  }

  for (let levelIndex = 1; levelIndex < levelCount; ++levelIndex) {
    const source = levels[levelIndex - 1];
    const destination = levels[levelIndex];
    const intermediate = device.createTexture({
      label: `DIC-RT horizontal pyramid level ${levelIndex}`,
      size: [destination.width, source.height, 1],
      format: 'r32float',
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
          { binding: 0, resource: source.sourceTexture.createView() },
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
    const links = roiMask ? createBufferWithData('ROI seed links', seedPermissions(
      source.roi, destination.roi, 2 ** source.index, 2 ** destination.index,
      SUBSET_SIZE, SUBSET_STRIDE, roiMask, FRAME_WIDTH, FRAME_HEIGHT), GPUBufferUsage.STORAGE) : null;
    seedEdges.set(destinationIndex, {
      config, links,
      bindGroup: device.createBindGroup({
        layout: seedTransferBindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: {
              buffer: configuration.outlierPolicy === 'disabled'
                ? source.buffers.state : source.buffers.filteredState,
            },
          },
          { binding: 1, resource: { buffer: destination.buffers.seedState } },
          { binding: 2, resource: { buffer: destination.buffers.activeMask } },
          { binding: 3, resource: { buffer: config } },
          ...(links ? [{ binding: 4, resource: { buffer: links } }] : []),
        ],
      }),
    });
  }

  sourceTexture = levels[0].sourceTexture;
  buffers = levels[0].buffers;
  bindGroup = levels[0].bindGroup;
  runtimePyramidLevels = levelCount;
  runtimeMotionModel = configuration.motionModel;
  runtimeOutlierPolicy = configuration.outlierPolicy;
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
    const bytes = new ArrayBuffer(48);
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
    view.setUint32(32, level.inverseOffset, true);
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
  const interpolation = options.interpolation === undefined
    ? configuration.interpolation : String(options.interpolation).toLowerCase();
  if (!(interpolation in INTERPOLATION_CONSTANTS)) {
    throw new Error("interpolation must be 'bspline', 'linear', or 'cubic'");
  }
  if (!allowPyramidRebuild && referenceReady
      && interpolation !== configuration.interpolation) {
    throw new Error('Changing interpolation requires setReference()');
  }
  const motionModel = options.motionModel === undefined
    ? configuration.motionModel : String(options.motionModel).toLowerCase();
  if (!(motionModel in MOTION_MODEL_CONSTANTS)) {
    throw new Error("motionModel must be 'affine', 'translation', or 'quadratic'");
  }
  if (!allowPyramidRebuild && referenceReady && motionModel !== configuration.motionModel) {
    throw new Error('Changing motionModel requires setReference()');
  }
  const outlierPolicy = options.outlierPolicy === undefined
    ? configuration.outlierPolicy : String(options.outlierPolicy).toLowerCase();
  if (!(outlierPolicy in OUTLIER_POLICY_CONSTANTS)) {
    throw new Error("outlierPolicy must be 'replace', 'flag', 'disabled', or 'median'");
  }
  if (!allowPyramidRebuild && referenceReady
      && outlierPolicy !== configuration.outlierPolicy) {
    throw new Error('Changing outlierPolicy requires setReference()');
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
    interpolation,
    motionModel,
    outlierPolicy,
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

/** Upload an already-grayscale frame without decoding an intermediate image. */
const setTargetBytes = (pixels, width, height) => {
  if (width !== FRAME_WIDTH || height !== FRAME_HEIGHT) {
    throw new Error(
      `Target dimensions ${width}x${height} do not match the reference ${FRAME_WIDTH}x${FRAME_HEIGHT}`,
    );
  }
  if (!(pixels instanceof Uint8Array) || pixels.byteLength !== PIXEL_COUNT) {
    throw new Error(`Expected ${PIXEL_COUNT} grayscale bytes`);
  }
  device.queue.writeTexture(
    { texture: sourceTexture },
    pixels,
    { bytesPerRow: FRAME_WIDTH, rowsPerImage: FRAME_HEIGHT },
    [FRAME_WIDTH, FRAME_HEIGHT, 1],
  );
};

const executePipeline = async ({ measureGpu = false } = {}) => {
  if (roiPlans[0]?.count === 0) return null;
  const encoder = device.createCommandEncoder({ label: 'DIC-RT full-pyramid registration' });
  levels.forEach((level) => {
    encoder.clearBuffer(level.buffers.state);
    if (configuration.outlierPolicy !== 'disabled') {
      encoder.clearBuffer(level.buffers.filteredState);
    }
  });
  if (temporal) encoder.clearBuffer(temporal.counts);
  const jobs = [];
  reductionEdges.forEach((edge) => {
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
    if (configuration.interpolation === 'bspline') {
      jobs.push({
        pipeline: level.width < 44 ? pipelines.prefilterXSafe : pipelines.prefilterXFast,
        bindGroup: level.prefilterBindGroup,
        group: 0,
        workgroups: [Math.ceil(level.width / 32), Math.ceil(level.height / 8), 1],
      }, {
        pipeline: level.height < 44 ? pipelines.prefilterYSafe : pipelines.prefilterYFast,
        bindGroup: level.prefilterBindGroup,
        group: 0,
        workgroups: [Math.ceil(level.width / 8), Math.ceil(level.height / 32), 1],
      });
    }
    jobs.push({
      pipeline: seeded
        ? (level.index === 0 ? pipelines.fineSeededSolvers : pipelines.seededSolvers)[configuration.interpolation]
        : (level.index === 0 ? pipelines.fineSolvers : pipelines.solvers)[configuration.interpolation],
      bindGroup: seeded ? level.seededBindGroup : level.bindGroup,
      group: 0,
      workgroups: [level.gridWidth, level.gridHeight, 1],
    });
    if (configuration.outlierPolicy !== 'disabled') {
      jobs.push({
        pipeline: pipelines.median,
        bindGroup: level.bindGroup,
        group: 0,
        workgroups: [Math.ceil(level.gridCount / 256), 1, 1],
      });
    }
  };
  const coarsestIndex = levels.length - 1;
  if (temporal) jobs.push(...temporal.before);
  appendLevel(levels[coarsestIndex], !!temporal);
  for (let destinationIndex = coarsestIndex - 1; destinationIndex >= 0; --destinationIndex) {
    const edge = seedEdges.get(destinationIndex);
    const destination = levels[destinationIndex];
    jobs.push({
      pipeline: pipelines.transferSeed,
      bindGroup: edge.bindGroup,
      group: 2,
      workgroups: [Math.ceil(destination.gridCount / 256), 1, 1],
    });
    if (destinationIndex === 0 && temporal?.refine) jobs.push(temporal.refine);
    appendLevel(destination, true);
  }
  if (temporal) jobs.push(...temporal.after);
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
  const parameterCount = configuration.motionModel === 'quadratic' ? 12 : 6;
  const stateStride = configuration.motionModel === 'quadratic' ? 16 : 8;
  const size = GRID_COUNT * stateStride * 4;
  const readback = device.createBuffer({
    size: size + (temporal ? 16 : 0),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: 'DIC-RT shared WGSL readback' });
  const resultBuffer = configuration.outlierPolicy === 'disabled'
    ? buffers.state : buffers.filteredState;
  encoder.copyBufferToBuffer(resultBuffer, 0, readback, 0, size);
  if (temporal) encoder.copyBufferToBuffer(temporal.counts, 0, readback, size, 16);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = readback.getMappedRange();
  const packed = new Float32Array(mapped, 0, size / 4);
  const result = decodePacked(packed, { width: FRAME_WIDTH, height: FRAME_HEIGHT,
    gridWidth: GRID_WIDTH, gridHeight: GRID_HEIGHT, subsetSize: SUBSET_SIZE,
    subsetStride: SUBSET_STRIDE, configuration, timing });
  if (temporal) {
    const counters = new Uint32Array(mapped, size, 4);
    result.counters = { temporallySeededCoarseSubsets: counters[0], temporalAcceptedUpdates: counters[1], velocityPredictedSubsets: counters[2], temporalExpiredSubsets: counters[3] };
  }
  if (roiPlans[0]) result.roiSelected = roiPlans[0].selected;
  readback.unmap();
  readback.destroy();
  return result;
};

const setReference = async (dataUrl, options = {}) => {
  await ensureDevice();
  configure(options, true);
  const image = typeof dataUrl === 'string' ? await loadImage(dataUrl) : dataUrl;
  const started = performance.now();
  const ownedMask = validateMask(options.roiMask, image.width, image.height, options.roiPolicy ?? 'full', configuration.motionModel);
  roiMask = ownedMask && !ownedMask.every(Boolean) ? ownedMask : null;
  roiPlans = [];
  if (roiMask) {
    let reduced = { mask: roiMask, width: image.width, height: image.height };
    for (let i = 0; i < configuration.pyramidLevels; i++) {
      roiPlans.push(planRoi(reduced.mask, reduced.width, reduced.height, SUBSET_SIZE, SUBSET_STRIDE));
      if (i + 1 < configuration.pyramidLevels) reduced = reduceMask(reduced.mask, reduced.width, reduced.height);
    }
  }
  const runtimeStarted = performance.now();
  destroyRuntimeResources();
  await createRuntime(image.width, image.height);
  const runtimeBuildWallMs = performance.now() - runtimeStarted;
  const referenceStarted = performance.now();
  try {
    const referenceLevels = [{
      values: image.pixels ?? imageToGray(image, FRAME_WIDTH, FRAME_HEIGHT, 1),
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
        subsetStride: SUBSET_STRIDE,
        interpolation: configuration.interpolation,
        motionModel: configuration.motionModel,
      });
      device.queue.writeBuffer(level.buffers.reference, 0, prepared.reference);
      let metadata = prepared.metadata;
      if (level.roi) {
        metadata = new Float32Array(prepared.metadata.length + level.gridCount * 4);
        metadata.set(prepared.metadata);
        const metaStride = configuration.motionModel === 'quadratic' ? 28 : 16;
        for (let i = 0; i < level.gridCount; i++) {
          if (!level.roi.selected[i]) metadata[i * metaStride + 1] = -3;
          metadata[prepared.metadata.length + i * 4] = level.roi.neighbours[i];
        }
        if (level.roi.count === 0) {
          const stateStride = configuration.motionModel === 'quadratic' ? 16 : 8;
          const state = new Float32Array(level.gridCount * stateStride);
          for (let i = 0; i < level.gridCount; i++) state[i * stateStride + stateStride - 2] = -3;
          device.queue.writeBuffer(level.buffers.state, 0, state);
          device.queue.writeBuffer(level.buffers.filteredState, 0, state);
        }
      }
      device.queue.writeBuffer(level.buffers.metadata, metadata.byteLength, prepared.inverseHessian);
      device.queue.writeBuffer(level.buffers.metadata, 0, metadata);
    });
  } finally {
    image.close?.();
  }
  temporal = await createTemporalGraph(device, pyramidShaderModule, levels, seedEdges, { ...options, ...configuration }, roiMask, SUBSET_SIZE, SUBSET_STRIDE);
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
    subsetStride: SUBSET_STRIDE,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    config: { ...configuration },
    roiSelected: roiPlans[0]?.selected ?? null,
    device: (await info()).device,
  };
};

const register = async (dataUrl, options = {}) => {
  await ensureDevice();
  if (!referenceReady) throw new Error('A reference image must be configured first');
  configure(options);
  const image = typeof dataUrl === 'string' ? await loadImage(dataUrl) : dataUrl;
  try {
    setTarget(image);
  } finally {
    image.close?.();
  }
  const started = performance.now();
  const gpuMs = await executePipeline({ measureGpu: true });
  const wallMs = performance.now() - started;
  return readResult({ gpuMs, wallMs });
};

/** Register a raw contiguous uint8 grayscale frame received from Python. */
const registerBytes = async (pixels, width, height, options = {}) => {
  await ensureDevice();
  if (!referenceReady) throw new Error('A reference image must be configured first');
  configure(options);
  setTargetBytes(pixels, width, height);
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
  const image = typeof dataUrl === 'string' ? await loadImage(dataUrl) : dataUrl;
  try {
    setTarget(image);
  } finally {
    image.close?.();
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
  const adapterInfo = adapter?.info || {};
  const identity = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device]
    .filter(Boolean).join(' ');
  return {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    subsetSize: SUBSET_SIZE,
    subsetStride: SUBSET_STRIDE,
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
    interpolation: configuration.interpolation,
    motionModel: configuration.motionModel,
    outlierPolicy: configuration.outlierPolicy,
    iterationsPerLevel: [...configuration.iterationsPerLevel],
    tolerancePerLevel: [...configuration.tolerancePerLevel],
    solver: configuration.pyramidLevels > 1
      ? 'portable-workgroup-full-pyramid'
      : 'portable-workgroup',
  };
};

configure(options, true);
return {
  info, configure, setReference, register, registerBytes, benchmark, ensureDevice,
  resetTemporalState() { temporal?.reset(); },
  get device() { return device; },
  get fieldBuffer() { return configuration.outlierPolicy === 'disabled' ? buffers?.state : buffers?.filteredState; },
  get configuration() { return { ...configuration }; },
  destroy() {
    destroyRuntimeResources();
    if (ownsDevice) device?.destroy();
    device = null;
  },
};
}
