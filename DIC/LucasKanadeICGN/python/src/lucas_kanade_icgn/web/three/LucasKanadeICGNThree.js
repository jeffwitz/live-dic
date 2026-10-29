/*
 * Copyright (c) 2026 DIC-RT contributors
 *
 * Independent Three.js/TSL implementation of the six-parameter affine
 * inverse-compositional Gauss-Newton algorithm used by DIC-RT.
 *
 * This module deliberately contains no camera, video-codec, or graphical UI
 * code. It exposes the same headless contract as the GPU-IO implementation so
 * both backends can be driven from Python with identical inputs.
 */

import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  Return,
  array,
  bitAnd,
  float,
  int,
  ivec2,
  localId,
  select,
  shiftRight,
  storage,
  subgroupAdd,
  subgroupSize,
  uint,
  uniform,
  vec2,
  vec4,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
  instanceIndex,
} from 'three/tsl';
import {
  imageToGray,
  makeFIRWeights,
  prepareReferenceData,
} from '../common/ReferencePreparation.js';

const DEFAULT_FRAME_WIDTH = 512;
const DEFAULT_FRAME_HEIGHT = 512;
const URL_PARAMETERS = new URLSearchParams(window.location.search);
const SUBSET_SIZE = Number(URL_PARAMETERS.get('subsetSize') ?? 16);
if (![8, 16, 32].includes(SUBSET_SIZE)) {
  throw new Error('subsetSize must be 8, 16, or 32');
}
const SUBSET_HALF_EXTENT = (SUBSET_SIZE - 1) / 2;
const SUBSET_PIXELS = SUBSET_SIZE * SUBSET_SIZE;
const SOLVER_LANES = 16;
const DEFAULT_MAX_ITERATIONS = 30;
const DEFAULT_TOLERANCE = 1e-3;
const DEFAULT_ZNCC_MIN = 0.8;
const FIR_RADIUS = 15;
const PREFILTER_TILE_SIZE = 16;
const PREFILTER_TILE_STRIDE = PREFILTER_TILE_SIZE + 2 * FIR_RADIUS;
const PREFILTER_TILE_PIXELS = PREFILTER_TILE_SIZE * PREFILTER_TILE_STRIDE;
const FORCE_WEBGL = URL_PARAMETERS.get('forceWebGL') === '1';

const configuration = {
  iterations: DEFAULT_MAX_ITERATIONS,
  tolerance: DEFAULT_TOLERANCE,
  znccMin: DEFAULT_ZNCC_MIN,
  interpolation: 'bspline',
  motionModel: 'affine',
  outlierPolicy: 'replace',
};
const toleranceNode = uniform(DEFAULT_TOLERANCE);
const znccMinNode = uniform(DEFAULT_ZNCC_MIN);
const iterationCountNode = uniform(DEFAULT_MAX_ITERATIONS, 'int');

let renderer = null;
let rendererPromise = null;
let referenceReady = false;
let kernels = null;
let FRAME_WIDTH = DEFAULT_FRAME_WIDTH;
let FRAME_HEIGHT = DEFAULT_FRAME_HEIGHT;
let PIXEL_COUNT = FRAME_WIDTH * FRAME_HEIGHT;
let GRID_WIDTH = Math.floor(FRAME_WIDTH / SUBSET_SIZE);
let GRID_HEIGHT = Math.floor(FRAME_HEIGHT / SUBSET_SIZE);
let GRID_COUNT = GRID_WIDTH * GRID_HEIGHT;
let attributes = null;
let sourcePackedNode = null;
let sourceNode = null;
let temporaryNode = null;
let coefficientsNode = null;
let referenceNode = null;
let hessianInverseNode = null;
let metadataNode = null;
let translationNode = null;
let linearNode = null;
let statusNode = null;
let filteredTranslationNode = null;
let filteredLinearNode = null;
let filteredStatusNode = null;
let subgroupDiagnosticNode = null;
let subgroupProbeWidth = 0;
let subgroupSolverWidth = 0;
let subgroupValidation = 'unavailable';
let useSubgroupSolver = false;
let solverKind = 'uninitialized';
let runtimeInterpolation = null;
let runtimeMotionModel = null;
let runtimeOutlierPolicy = null;
let targetUsesPackedBytes = false;

/** Allocate all resolution-dependent storage and rebuild the TSL compute graph. */
const initializeRuntime = (width, height) => {
  if (!Number.isInteger(width) || !Number.isInteger(height)
      || width < SUBSET_SIZE || height < SUBSET_SIZE) {
    throw new Error(`Image dimensions must be integers of at least ${SUBSET_SIZE} pixels`);
  }
  if (attributes && width === FRAME_WIDTH && height === FRAME_HEIGHT
      && runtimeInterpolation === configuration.interpolation
      && runtimeMotionModel === configuration.motionModel
      && runtimeOutlierPolicy === configuration.outlierPolicy) return false;

  FRAME_WIDTH = width;
  FRAME_HEIGHT = height;
  PIXEL_COUNT = FRAME_WIDTH * FRAME_HEIGHT;
  GRID_WIDTH = Math.floor(FRAME_WIDTH / SUBSET_SIZE);
  GRID_HEIGHT = Math.floor(FRAME_HEIGHT / SUBSET_SIZE);
  GRID_COUNT = GRID_WIDTH * GRID_HEIGHT;

  if (renderer.backend.isWebGPUBackend === true) {
    const largestBufferBytes = PIXEL_COUNT * 4 * Float32Array.BYTES_PER_ELEMENT;
    const { limits } = renderer.backend.device;
    if (largestBufferBytes > limits.maxStorageBufferBindingSize) {
      throw new Error(
        `The ${FRAME_WIDTH}x${FRAME_HEIGHT} reference buffer needs ${largestBufferBytes} bytes, `
        + `but this WebGPU adapter allows ${limits.maxStorageBufferBindingSize}`,
      );
    }
    if (largestBufferBytes > limits.maxBufferSize) {
      throw new Error(
        `The ${FRAME_WIDTH}x${FRAME_HEIGHT} reference buffer exceeds the adapter's maximum buffer size`,
      );
    }
  }

  const StorageAttribute = renderer.backend.isWebGPUBackend === true
    ? THREE.StorageBufferAttribute
    : THREE.StorageInstancedBufferAttribute;
  const storageAttribute = (length, itemSize, ArrayType = Float32Array) => new StorageAttribute(
    new ArrayType(length),
    itemSize,
  );
  attributes = {
    sourcePacked: renderer.backend.isWebGPUBackend === true
      ? storageAttribute(Math.ceil(PIXEL_COUNT / 4), 1, Uint32Array)
      : null,
    source: storageAttribute(PIXEL_COUNT, 1),
    temporary: storageAttribute(PIXEL_COUNT, 1),
    coefficients: storageAttribute(PIXEL_COUNT, 1),
    reference: storageAttribute(PIXEL_COUNT * 4, 4),
    hessianInverse: storageAttribute(GRID_COUNT * 9 * 4, 4),
    metadata: storageAttribute(GRID_COUNT * 4 * 4, 4),
    translation: storageAttribute(GRID_COUNT * 2, 2),
    linear: storageAttribute(GRID_COUNT * 4, 4),
    status: storageAttribute(GRID_COUNT * 2, 2),
    filteredTranslation: storageAttribute(GRID_COUNT * 2, 2),
    filteredLinear: storageAttribute(GRID_COUNT * 4, 4),
    filteredStatus: storageAttribute(GRID_COUNT * 2, 2),
    subgroupDiagnostic: storageAttribute(1, 1),
  };

  sourcePackedNode = attributes.sourcePacked
    ? storage(attributes.sourcePacked, 'uint', Math.ceil(PIXEL_COUNT / 4)).toReadOnly()
    : null;
  sourceNode = storage(attributes.source, 'float', PIXEL_COUNT);
  temporaryNode = storage(attributes.temporary, 'float', PIXEL_COUNT);
  coefficientsNode = storage(attributes.coefficients, 'float', PIXEL_COUNT);
  referenceNode = storage(attributes.reference, 'vec4', PIXEL_COUNT).toReadOnly();
  hessianInverseNode = storage(attributes.hessianInverse, 'vec4', GRID_COUNT * 9).toReadOnly();
  metadataNode = storage(attributes.metadata, 'vec4', GRID_COUNT * 4).toReadOnly();
  translationNode = storage(attributes.translation, 'vec2', GRID_COUNT);
  linearNode = storage(attributes.linear, 'vec4', GRID_COUNT);
  statusNode = storage(attributes.status, 'vec2', GRID_COUNT);
  filteredTranslationNode = storage(attributes.filteredTranslation, 'vec2', GRID_COUNT);
  filteredLinearNode = storage(attributes.filteredLinear, 'vec4', GRID_COUNT);
  filteredStatusNode = storage(attributes.filteredStatus, 'vec2', GRID_COUNT);
  subgroupDiagnosticNode = storage(attributes.subgroupDiagnostic, 'float', 1);
  if (renderer.backend.isWebGPUBackend !== true) {
    [
      sourceNode,
      temporaryNode,
      coefficientsNode,
      referenceNode,
      hessianInverseNode,
      metadataNode,
      translationNode,
      linearNode,
      statusNode,
    ].forEach((node) => node.setPBO(true));
  }
  useSubgroupSolver = renderer.backend.isWebGPUBackend === true
    && subgroupProbeWidth >= SOLVER_LANES;
  subgroupSolverWidth = 0;
  subgroupValidation = useSubgroupSolver ? 'pending' : 'unavailable';
  kernels = buildKernels();
  runtimeInterpolation = configuration.interpolation;
  runtimeMotionModel = configuration.motionModel;
  runtimeOutlierPolicy = configuration.outlierPolicy;
  referenceReady = false;
  targetUsesPackedBytes = false;
  return true;
};

/** Prepare all reference-dependent IC-GN quantities once on the CPU. */
const prepareReference = (image) => {
  const prepared = prepareReferenceData(image, {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    subsetSize: SUBSET_SIZE,
    interpolation: configuration.interpolation,
    motionModel: configuration.motionModel,
  });
  attributes.reference.array.set(prepared.reference);
  attributes.hessianInverse.array.set(prepared.inverseHessian);
  attributes.metadata.array.set(prepared.metadata);
  attributes.reference.needsUpdate = true;
  attributes.hessianInverse.needsUpdate = true;
  attributes.metadata.needsUpdate = true;
  referenceReady = true;
};

const FIR_WEIGHTS = makeFIRWeights();

const mirrorNode = (coordinate, size) => {
  const reflected = select(
    coordinate.lessThan(0),
    coordinate.negate(),
    select(coordinate.greaterThanEqual(size), int(2 * size - 2).sub(coordinate), coordinate),
  );
  // Extra halo cells in a partial tile may remain outside the image after one
  // reflection. They are not used by valid outputs, but their loads must still
  // be in bounds because every invocation participates in the following barrier.
  // The tile geometry needs the guard only below 31 pixels; keep larger kernels
  // free of the additional GPU clamp because `size` is a host-side constant.
  return size < 31 ? reflected.clamp(0, size - 1) : reflected;
};

const makePrefilterKernel = (input, output, horizontal) => {
  if (renderer.backend.isWebGPUBackend !== true) {
    return Fn(() => {
      const index = instanceIndex;
      const x = int(index.mod(uint(FRAME_WIDTH)));
      const y = int(index.div(uint(FRAME_WIDTH)));
      const result = float(0).toVar();
      for (let tap = -FIR_RADIUS; tap <= FIR_RADIUS; ++tap) {
        const sx = horizontal ? mirrorNode(x.add(tap), FRAME_WIDTH) : x;
        const sy = horizontal ? y : mirrorNode(y.add(tap), FRAME_HEIGHT);
        const sampleIndex = uint(sy.mul(FRAME_WIDTH).add(sx));
        result.addAssign(input.element(sampleIndex).mul(FIR_WEIGHTS[tap + FIR_RADIUS]));
      }
      output.element(index).assign(result);
    })().compute(PIXEL_COUNT, [256])
      .setName(horizontal ? 'DIC-RT WebGL2 prefilter X' : 'DIC-RT WebGL2 prefilter Y');
  }
  const tileColumns = Math.ceil(FRAME_WIDTH / PREFILTER_TILE_SIZE);
  const tileRows = Math.ceil(FRAME_HEIGHT / PREFILTER_TILE_SIZE);
  return Fn(() => {
    const lane = int(localId.x);
    const tileColumn = int(workgroupId.x.mod(uint(tileColumns)));
    const tileRow = int(workgroupId.x.div(uint(tileColumns)));
    const baseX = tileColumn.mul(PREFILTER_TILE_SIZE);
    const baseY = tileRow.mul(PREFILTER_TILE_SIZE);
    const tile = workgroupArray('float', PREFILTER_TILE_PIXELS);

    // A 256-lane workgroup cooperatively loads the 16x46 halo. Each lane
    // performs at most three global reads instead of every output doing 31.
    for (let batch = 0; batch < 3; ++batch) {
      const tileIndex = lane.add(batch * 256);
      If(tileIndex.lessThan(PREFILTER_TILE_PIXELS), () => {
        const tileX = horizontal
          ? tileIndex.mod(PREFILTER_TILE_STRIDE)
          : tileIndex.mod(PREFILTER_TILE_SIZE);
        const tileY = horizontal
          ? tileIndex.div(PREFILTER_TILE_STRIDE)
          : tileIndex.div(PREFILTER_TILE_SIZE);
        const candidateX = horizontal ? baseX.add(tileX).sub(FIR_RADIUS) : baseX.add(tileX);
        const candidateY = horizontal ? baseY.add(tileY) : baseY.add(tileY).sub(FIR_RADIUS);
        const sampleX = horizontal
          ? mirrorNode(candidateX, FRAME_WIDTH)
          : select(candidateX.lessThan(FRAME_WIDTH), candidateX, FRAME_WIDTH - 1);
        const sampleY = horizontal
          ? select(candidateY.lessThan(FRAME_HEIGHT), candidateY, FRAME_HEIGHT - 1)
          : mirrorNode(candidateY, FRAME_HEIGHT);
        const sampleIndex = uint(sampleY.mul(FRAME_WIDTH).add(sampleX));
        tile.element(tileIndex).assign(input.element(sampleIndex));
      });
    }
    workgroupBarrier();

    const localX = lane.mod(PREFILTER_TILE_SIZE);
    const localY = lane.div(PREFILTER_TILE_SIZE);
    const x = baseX.add(localX);
    const y = baseY.add(localY);
    If(x.lessThan(FRAME_WIDTH).and(y.lessThan(FRAME_HEIGHT)), () => {
      const result = float(0).toVar();
      for (let tap = -FIR_RADIUS; tap <= FIR_RADIUS; ++tap) {
        const tileIndex = horizontal
          ? localY.mul(PREFILTER_TILE_STRIDE).add(localX).add(FIR_RADIUS + tap)
          : localY.add(FIR_RADIUS + tap).mul(PREFILTER_TILE_SIZE).add(localX);
        result.addAssign(tile.element(tileIndex).mul(FIR_WEIGHTS[tap + FIR_RADIUS]));
      }
      output.element(uint(y.mul(FRAME_WIDTH).add(x))).assign(result);
    });
  })().compute(tileColumns * tileRows * 256, [256])
    .setName(horizontal ? 'DIC-RT tiled prefilter X' : 'DIC-RT tiled prefilter Y');
};

/** Expand four packed uint8 pixels per word into the float source buffer. */
const makeUnpackKernel = () => Fn(() => {
  const index = instanceIndex;
  const packed = sourcePackedNode.element(index.div(uint(4)));
  const shift = index.mod(uint(4)).mul(uint(8));
  const value = bitAnd(shiftRight(packed, shift), uint(255));
  sourceNode.element(index).assign(float(value).div(255));
})().compute(PIXEL_COUNT, [256]).setName('DIC-RT unpack uint8 target');

const packedComponent = (buffer, itemIndex, componentIndex) => {
  const packed = buffer.element(itemIndex);
  return [packed.x, packed.y, packed.z, packed.w][componentIndex];
};

const metadataValue = (subsetIndex, componentIndex) => packedComponent(
  metadataNode,
  subsetIndex.mul(4).add(uint(Math.floor(componentIndex / 4))),
  componentIndex % 4,
);

const hessianValue = (subsetIndex, componentIndex) => packedComponent(
  hessianInverseNode,
  subsetIndex.mul(9).add(uint(Math.floor(componentIndex / 4))),
  componentIndex % 4,
);

const cubicWeights = (fraction) => {
  const squared = fraction.mul(fraction);
  const cubed = squared.mul(fraction);
  const complement = float(1).sub(fraction);
  return [
    complement.mul(complement).mul(complement).div(6),
    cubed.mul(3).sub(squared.mul(6)).add(4).div(6),
    cubed.mul(-3).add(squared.mul(3)).add(fraction.mul(3)).add(1).div(6),
    cubed.div(6),
  ];
};

const sampleCubic = (position) => {
  const base = ivec2(position.floor());
  const fraction = position.sub(vec2(base));
  const wx = cubicWeights(fraction.x);
  const wy = cubicWeights(fraction.y);
  const value = float(0).toVar();
  for (let yy = 0; yy < 4; ++yy) {
    const rowValue = float(0).toVar();
    for (let xx = 0; xx < 4; ++xx) {
      const sx = base.x.add(xx - 1);
      const sy = base.y.add(yy - 1);
      rowValue.addAssign(coefficientsNode.element(uint(sy.mul(FRAME_WIDTH).add(sx))).mul(wx[xx]));
    }
    value.addAssign(rowValue.mul(wy[yy]));
  }
  return value;
};

const sampleLinear = (position) => {
  const base = ivec2(position.floor());
  const fraction = position.sub(vec2(base));
  const topLeft = sourceNode.element(uint(base.y.mul(FRAME_WIDTH).add(base.x)));
  const topRight = sourceNode.element(uint(base.y.mul(FRAME_WIDTH).add(base.x.add(1))));
  const bottomLeft = sourceNode.element(uint(base.y.add(1).mul(FRAME_WIDTH).add(base.x)));
  const bottomRight = sourceNode.element(uint(base.y.add(1).mul(FRAME_WIDTH).add(base.x.add(1))));
  const top = topLeft.mul(float(1).sub(fraction.x)).add(topRight.mul(fraction.x));
  const bottom = bottomLeft.mul(float(1).sub(fraction.x)).add(bottomRight.mul(fraction.x));
  return top.mul(float(1).sub(fraction.y)).add(bottom.mul(fraction.y));
};

const keysCubicWeights = (fraction) => {
  const squared = fraction.mul(fraction);
  const cubed = squared.mul(fraction);
  return [
    fraction.mul(-0.5).add(squared).sub(cubed.mul(0.5)),
    float(1).sub(squared.mul(2.5)).add(cubed.mul(1.5)),
    fraction.mul(0.5).add(squared.mul(2)).sub(cubed.mul(1.5)),
    squared.mul(-0.5).add(cubed.mul(0.5)),
  ];
};

const sampleKeysCubic = (position) => {
  const base = ivec2(position.floor());
  const fraction = position.sub(vec2(base));
  const wx = keysCubicWeights(fraction.x);
  const wy = keysCubicWeights(fraction.y);
  const value = float(0).toVar();
  for (let yy = 0; yy < 4; ++yy) {
    const rowValue = float(0).toVar();
    for (let xx = 0; xx < 4; ++xx) {
      const sx = base.x.add(xx - 1);
      const sy = base.y.add(yy - 1);
      rowValue.addAssign(sourceNode.element(uint(sy.mul(FRAME_WIDTH).add(sx))).mul(wx[xx]));
    }
    value.addAssign(rowValue.mul(wy[yy]));
  }
  return value;
};

const sampleTarget = (position) => {
  if (configuration.interpolation === 'linear') return sampleLinear(position);
  if (configuration.interpolation === 'cubic') return sampleKeysCubic(position);
  return sampleCubic(position);
};

const activeMotionComponents = () => (configuration.motionModel === 'translation'
  ? [0, 3] : [0, 1, 2, 3, 4, 5]);

const displacementAt = (translation, linear, localX, localY) => (
  configuration.motionModel === 'translation'
    ? translation
    : translation.add(vec2(
      linear.x.mul(localX).add(linear.y.mul(localY)),
      linear.z.mul(localX).add(linear.w.mul(localY)),
    ))
);

const solveIncrement = (subsetIndex, reducedJ, currentMean, ratio) => {
  if (configuration.motionModel === 'translation') {
    const residualX = metadataValue(subsetIndex, 2).sub(ratio.mul(
      reducedJ[0].sub(currentMean.mul(metadataValue(subsetIndex, 8))),
    ));
    const residualY = metadataValue(subsetIndex, 5).sub(ratio.mul(
      reducedJ[3].sub(currentMean.mul(metadataValue(subsetIndex, 11))),
    ));
    const deltaX = hessianValue(subsetIndex, 0).mul(residualX)
      .add(hessianValue(subsetIndex, 3).mul(residualY)).negate();
    const deltaY = hessianValue(subsetIndex, 18).mul(residualX)
      .add(hessianValue(subsetIndex, 21).mul(residualY)).negate();
    return [deltaX, float(0), float(0), deltaY, float(0), float(0)];
  }
  const residual = Array.from({ length: 6 }, (_, component) => metadataValue(
    subsetIndex,
    2 + component,
  ).sub(ratio.mul(
    reducedJ[component].sub(currentMean.mul(metadataValue(subsetIndex, 8 + component))),
  )));
  return Array.from({ length: 6 }, (_, row) => {
    const value = float(0).toVar();
    for (let column = 0; column < 6; ++column) {
      value.addAssign(hessianValue(subsetIndex, 6 * row + column).mul(residual[column]));
    }
    return value.negate();
  });
};

const incrementDeterminant = (delta) => (configuration.motionModel === 'translation'
  ? float(1)
  : delta[1].add(1).mul(delta[5].add(1)).sub(delta[2].mul(delta[4])));

const cornerChange = (delta) => {
  if (configuration.motionModel === 'translation') return vec2(delta[0], delta[3]).length();
  const maximum = float(0).toVar();
  for (let corner = 0; corner < 4; ++corner) {
    const x = corner === 1 || corner === 3 ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
    const y = corner >= 2 ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
    const change = vec2(
      delta[0].add(delta[1].mul(x)).add(delta[2].mul(y)),
      delta[3].add(delta[4].mul(x)).add(delta[5].mul(y)),
    ).length();
    maximum.assign(maximum.max(change));
  }
  return maximum;
};

const composeInverseUpdate = (translation, linear, delta, determinant) => {
  if (configuration.motionModel === 'translation') {
    return {
      translation: translation.sub(vec2(delta[0], delta[3])),
      linear: vec4(0),
      cornerChange: cornerChange(delta),
    };
  }
  const inverse11 = delta[5].add(1).div(determinant);
  const inverse12 = delta[2].negate().div(determinant);
  const inverse21 = delta[4].negate().div(determinant);
  const inverse22 = delta[1].add(1).div(determinant);
  const warp11 = linear.x.add(1);
  const warp12 = linear.y;
  const warp21 = linear.z;
  const warp22 = linear.w.add(1);
  const inverseTranslationX = inverse11.mul(delta[0]).add(inverse12.mul(delta[3])).negate();
  const inverseTranslationY = inverse21.mul(delta[0]).add(inverse22.mul(delta[3])).negate();
  const updatedTranslation = translation.add(vec2(
    warp11.mul(inverseTranslationX).add(warp12.mul(inverseTranslationY)),
    warp21.mul(inverseTranslationX).add(warp22.mul(inverseTranslationY)),
  ));
  const updated11 = warp11.mul(inverse11).add(warp12.mul(inverse21));
  const updated12 = warp11.mul(inverse12).add(warp12.mul(inverse22));
  const updated21 = warp21.mul(inverse11).add(warp22.mul(inverse21));
  const updated22 = warp21.mul(inverse12).add(warp22.mul(inverse22));
  return {
    translation: updatedTranslation,
    linear: vec4(updated11.sub(1), updated12, updated21, updated22.sub(1)),
    cornerChange: cornerChange(delta),
  };
};

const convergenceStatus = (change, zncc) => select(
  change.lessThan(toleranceNode),
  select(zncc.greaterThanEqual(znccMinNode), 1, -2),
  0,
);

/**
 * Build the WebGL2 transform-feedback solver.
 *
 * WebGL2 has no workgroups, shared memory, or subgroup reductions. One
 * invocation therefore owns one complete subset and accumulates its 256
 * pixels serially. Numerical helpers and the packed result layout remain
 * common with the cooperative WebGPU solvers.
 */
const makeSerialSolverKernel = () => Fn(() => {
  const subsetIndex = uint(instanceIndex);
  const currentTranslation = translationNode.element(subsetIndex).toVar();
  const currentLinear = linearNode.element(subsetIndex).toVar();
  const initialStatus = statusNode.element(subsetIndex).toVar();
  const currentStatus = float(select(
    metadataValue(subsetIndex, 1).greaterThanEqual(0.5),
    initialStatus.x,
    -1,
  )).toVar();
  const currentIterations = float(initialStatus.y).toVar();
  const gridX = int(subsetIndex.mod(uint(GRID_WIDTH)));
  const gridY = int(subsetIndex.div(uint(GRID_WIDTH)));
  const center = vec2(float(gridX).add(0.5), float(gridY).add(0.5)).mul(SUBSET_SIZE);

  Loop({
    start: int(0),
    end: iterationCountNode,
    type: 'int',
    condition: '<',
    name: 'iteration',
  }, ({ iteration }) => {
    If(currentStatus.abs().greaterThan(0.5), () => Break());

    const inside = float(1).toVar();
    for (let corner = 0; corner < 4; ++corner) {
      const localX = corner === 1 || corner === 3
        ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
      const localY = corner >= 2 ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
      const position = center.sub(0.5).add(vec2(localX, localY))
        .add(displacementAt(currentTranslation, currentLinear, localX, localY));
      const valid = position.x.greaterThanEqual(2).and(position.y.greaterThanEqual(2))
        .and(position.x.lessThanEqual(FRAME_WIDTH - 3))
        .and(position.y.lessThanEqual(FRAME_HEIGHT - 3));
      inside.mulAssign(select(valid, 1, 0));
    }
    If(inside.lessThan(0.5), () => {
      currentStatus.assign(-1);
      Break();
    });

    const sumCurrent = float(0).toVar();
    const sumCurrent2 = float(0).toVar();
    const sumReferenceCurrent = float(0).toVar();
    const sumJCurrent = Array.from({ length: 6 }, () => float(0).toVar());
    Loop({
      start: int(0),
      end: int(SUBSET_SIZE),
      type: 'int',
      condition: '<',
      name: 'row',
    }, ({ row }) => {
      Loop({
        start: int(0),
        end: int(SUBSET_SIZE),
        type: 'int',
        condition: '<',
        name: 'column',
      }, ({ column }) => {
        const localX = float(column).sub(SUBSET_HALF_EXTENT);
        const localY = float(row).sub(SUBSET_HALF_EXTENT);
        const displacement = displacementAt(currentTranslation, currentLinear, localX, localY);
        const current = sampleTarget(
          center.sub(0.5).add(vec2(localX, localY)).add(displacement),
        );
        const templateIndex = uint(gridY.mul(SUBSET_SIZE).add(row).mul(FRAME_WIDTH)
          .add(gridX.mul(SUBSET_SIZE)).add(column));
        const reference = referenceNode.element(templateIndex);
        const jacobian = [
          reference.y,
          reference.y.mul(localX),
          reference.y.mul(localY),
          reference.z,
          reference.z.mul(localX),
          reference.z.mul(localY),
        ];
        sumCurrent.addAssign(current);
        sumCurrent2.addAssign(current.mul(current));
        sumReferenceCurrent.addAssign(reference.x.mul(current));
        for (const component of activeMotionComponents()) {
          sumJCurrent[component].addAssign(jacobian[component].mul(current));
        }
      });
    });

    const currentMean = sumCurrent.div(SUBSET_PIXELS);
    const currentNorm = sumCurrent2.sub(sumCurrent.mul(currentMean)).max(0).sqrt();
    const referenceNorm = metadataValue(subsetIndex, 0);
    currentIterations.assign(float(iteration).add(1));
    If(currentNorm.lessThan(1e-9).or(referenceNorm.lessThan(1e-9)), () => {
      currentStatus.assign(-2);
    }).Else(() => {
      const ratio = referenceNorm.div(currentNorm);
      const delta = solveIncrement(subsetIndex, sumJCurrent, currentMean, ratio);
      const determinant = incrementDeterminant(delta);
      If(determinant.abs().lessThan(1e-8), () => {
        currentStatus.assign(-1);
      }).Else(() => {
        const update = composeInverseUpdate(
          currentTranslation,
          currentLinear,
          delta,
          determinant,
        );
        currentTranslation.assign(update.translation);
        currentLinear.assign(update.linear);
        const zncc = sumReferenceCurrent.div(referenceNorm.mul(currentNorm));
        currentStatus.assign(convergenceStatus(update.cornerChange, zncc));
      });
    });
  });

  translationNode.element(subsetIndex).assign(currentTranslation);
  linearNode.element(subsetIndex).assign(currentLinear);
  statusNode.element(subsetIndex).assign(vec2(currentStatus, currentIterations));
})().compute(GRID_COUNT, [64]).setName('DIC-RT serial WebGL2 affine IC-GN solver');


/**
 * Build the complete affine IC-GN solve as one cooperative dispatch.
 *
 * Keeping the warp in workgroup memory avoids global state round-trips between
 * iterations. Converged subsets skip interpolation and updates while all lanes
 * continue through the barriers required by portable WGSL.
 */
const makeFusedCooperativeSolverKernel = () => Fn(() => {
  const subsetIndex = uint(workgroupId.y.mul(GRID_WIDTH).add(workgroupId.x));
  const lane = int(localId.x);
  const initialTranslation = translationNode.element(subsetIndex).toVar();
  const initialLinear = linearNode.element(subsetIndex).toVar();
  const initialStatus = statusNode.element(subsetIndex).toVar();
  const gridX = int(subsetIndex.mod(uint(GRID_WIDTH)));
  const gridY = int(subsetIndex.div(uint(GRID_WIDTH)));
  const center = vec2(float(gridX).add(0.5), float(gridY).add(0.5)).mul(SUBSET_SIZE);

  const sharedState = workgroupArray('vec4', 2);
  const sharedIterations = workgroupArray('float', 1);
  const sharedA = workgroupArray('vec4', SOLVER_LANES);
  const sharedB = workgroupArray('vec4', SOLVER_LANES);
  const sharedC = workgroupArray('vec4', SOLVER_LANES);
  If(lane.equal(0), () => {
    const state = select(metadataValue(subsetIndex, 1).greaterThanEqual(0.5), initialStatus.x, -1);
    sharedState.element(int(0)).assign(vec4(
      initialTranslation.x,
      initialTranslation.y,
      initialLinear.x,
      initialLinear.y,
    ));
    sharedState.element(int(1)).assign(vec4(
      initialLinear.z,
      initialLinear.w,
      state,
      initialStatus.y,
    ));
    sharedIterations.element(int(0)).assign(initialStatus.y);
  });
  workgroupBarrier();

  Loop({
    start: int(0),
    end: iterationCountNode,
    type: 'int',
    condition: '<',
    name: 'iteration',
  }, ({ iteration }) => {
    const state0 = sharedState.element(int(0)).toVar();
    const state1 = sharedState.element(int(1)).toVar();
    const oldTranslation = vec2(state0.x, state0.y);
    const oldLinear = vec4(state0.z, state0.w, state1.x, state1.y);
    const inside = float(1).toVar();
    for (let corner = 0; corner < 4; ++corner) {
      const localX = corner === 1 || corner === 3 ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
      const localY = corner >= 2 ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
      const position = center.sub(0.5).add(vec2(localX, localY))
        .add(displacementAt(oldTranslation, oldLinear, localX, localY));
      const valid = position.x.greaterThanEqual(2).and(position.y.greaterThanEqual(2))
        .and(position.x.lessThanEqual(FRAME_WIDTH - 3)).and(position.y.lessThanEqual(FRAME_HEIGHT - 3));
      inside.mulAssign(select(valid, 1, 0));
    }
    If(lane.equal(0).and(inside.lessThan(0.5)), () => {
      sharedState.element(int(1)).assign(vec4(state1.x, state1.y, -1, state1.w));
    });
    workgroupBarrier();

    const sumCurrent = float(0).toVar();
    const sumCurrent2 = float(0).toVar();
    const sumReferenceCurrent = float(0).toVar();
    const sumJCurrent = Array.from({ length: 6 }, () => float(0).toVar());
    const active = state1.z.abs().lessThanEqual(0.5).and(inside.greaterThanEqual(0.5));
    If(active, () => {
      Loop({ start: int(0), end: int(SUBSET_SIZE), type: 'int', condition: '<', name: 'row' }, ({ row }) => {
        const accumulateColumn = (column) => {
          const localX = float(column).sub(SUBSET_HALF_EXTENT);
          const localY = float(row).sub(SUBSET_HALF_EXTENT);
          const displacement = displacementAt(oldTranslation, oldLinear, localX, localY);
          const current = sampleTarget(
            center.sub(0.5).add(vec2(localX, localY)).add(displacement),
          );
          const templateIndex = uint(gridY.mul(SUBSET_SIZE).add(row).mul(FRAME_WIDTH)
            .add(gridX.mul(SUBSET_SIZE)).add(column));
          const reference = referenceNode.element(templateIndex);
          const jacobian = [
            reference.y,
            reference.y.mul(localX),
            reference.y.mul(localY),
            reference.z,
            reference.z.mul(localX),
            reference.z.mul(localY),
          ];
          sumCurrent.addAssign(current);
          sumCurrent2.addAssign(current.mul(current));
          sumReferenceCurrent.addAssign(reference.x.mul(current));
          for (const component of activeMotionComponents()) {
            sumJCurrent[component].addAssign(jacobian[component].mul(current));
          }
        };

        if (SUBSET_SIZE === SOLVER_LANES) {
          // Preserve the original generated graph for the default 16x16 path.
          accumulateColumn(lane);
        } else {
          for (let offset = 0; offset < SUBSET_SIZE; offset += SOLVER_LANES) {
            const column = lane.add(offset);
            If(column.lessThan(SUBSET_SIZE), () => accumulateColumn(column));
          }
        }
      });
    });

    sharedA.element(lane).assign(vec4(sumCurrent, sumCurrent2, sumReferenceCurrent, sumJCurrent[0]));
    sharedB.element(lane).assign(vec4(sumJCurrent[1], sumJCurrent[2], sumJCurrent[3], sumJCurrent[4]));
    sharedC.element(lane).assign(vec4(sumJCurrent[5], 0, 0, 0));
    workgroupBarrier();
    for (const offset of [8, 4, 2, 1]) {
      If(lane.lessThan(offset), () => {
        sharedA.element(lane).addAssign(sharedA.element(lane.add(offset)));
        sharedB.element(lane).addAssign(sharedB.element(lane.add(offset)));
        sharedC.element(lane).addAssign(sharedC.element(lane.add(offset)));
      });
      workgroupBarrier();
    }

    If(lane.equal(0).and(active), () => {
      const totalsA = sharedA.element(int(0));
      const totalsB = sharedB.element(int(0));
      const totalsC = sharedC.element(int(0));
      const reducedJ = [totalsA.w, totalsB.x, totalsB.y, totalsB.z, totalsB.w, totalsC.x];
      const currentMean = totalsA.x.div(SUBSET_PIXELS);
      const currentNorm = totalsA.y.sub(totalsA.x.mul(currentMean)).max(0).sqrt();
      const referenceNorm = metadataValue(subsetIndex, 0);
      const nextIteration = float(iteration).add(1);
      sharedIterations.element(int(0)).assign(nextIteration);

      If(currentNorm.lessThan(1e-9).or(referenceNorm.lessThan(1e-9)), () => {
        sharedState.element(int(1)).assign(vec4(state1.x, state1.y, -2, nextIteration));
      }).Else(() => {
        const ratio = referenceNorm.div(currentNorm);
        const delta = solveIncrement(subsetIndex, reducedJ, currentMean, ratio);
        const determinant = incrementDeterminant(delta);
        If(determinant.abs().lessThan(1e-8), () => {
          sharedState.element(int(1)).assign(vec4(state1.x, state1.y, -1, nextIteration));
        }).Else(() => {
          const update = composeInverseUpdate(oldTranslation, oldLinear, delta, determinant);
          const zncc = totalsA.z.div(referenceNorm.mul(currentNorm));
          const nextState = convergenceStatus(update.cornerChange, zncc);
          sharedState.element(int(0)).assign(vec4(
            update.translation.x,
            update.translation.y,
            update.linear.x,
            update.linear.y,
          ));
          sharedState.element(int(1)).assign(vec4(
            update.linear.z,
            update.linear.w,
            nextState,
            nextIteration,
          ));
        });
      });
    });
    workgroupBarrier();
  });

  If(lane.equal(0), () => {
    const finalState0 = sharedState.element(int(0)).toVar();
    const finalState1 = sharedState.element(int(1)).toVar();
    translationNode.element(subsetIndex).assign(vec2(finalState0.x, finalState0.y));
    linearNode.element(subsetIndex).assign(vec4(finalState0.z, finalState0.w, finalState1.x, finalState1.y));
    statusNode.element(subsetIndex).assign(vec2(finalState1.z, sharedIterations.element(int(0))));
  });
})().compute([GRID_WIDTH, GRID_HEIGHT, 1], [SOLVER_LANES, 1, 1])
  .setName('DIC-RT fused cooperative affine IC-GN solver');

/**
 * Build the fast subgroup solver used when WebGPU exposes subgroup operations.
 *
 * All 16 lanes retain the same warp state in registers. Hardware collectives
 * reduce the per-column sums without workgroup barriers or global state
 * round-trips; converged subsets skip the expensive interpolation and update.
 */
const makeSubgroupSolverKernel = () => {
  const solverLanes = Math.min(subgroupProbeWidth, SUBSET_PIXELS);
  const pixelsPerLane = Math.ceil(SUBSET_PIXELS / solverLanes);
  return Fn(() => {
  const subsetIndex = uint(workgroupId.y.mul(GRID_WIDTH).add(workgroupId.x));
  const lane = int(localId.x);
  If(subsetIndex.equal(0).and(lane.equal(0)), () => {
    // This value comes from the actual solver pipeline, not the lightweight
    // discovery shader. The host validates it before accepting this kernel.
    subgroupDiagnosticNode.element(int(0)).assign(float(subgroupSize));
  });
  const currentTranslation = translationNode.element(subsetIndex).toVar();
  const currentLinear = linearNode.element(subsetIndex).toVar();
  const initialStatus = statusNode.element(subsetIndex).toVar();
  const currentStatus = float(select(
    metadataValue(subsetIndex, 1).greaterThanEqual(0.5),
    initialStatus.x,
    -1,
  )).toVar();
  const currentIterations = float(initialStatus.y).toVar();
  const gridX = int(subsetIndex.mod(uint(GRID_WIDTH)));
  const gridY = int(subsetIndex.div(uint(GRID_WIDTH)));
  const center = vec2(float(gridX).add(0.5), float(gridY).add(0.5)).mul(SUBSET_SIZE);

  Loop({
    start: int(0),
    end: iterationCountNode,
    type: 'int',
    condition: '<',
    name: 'iteration',
  }, ({ iteration }) => {
    const inside = float(1).toVar();
    for (let corner = 0; corner < 4; ++corner) {
      const localX = corner === 1 || corner === 3 ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
      const localY = corner >= 2 ? SUBSET_HALF_EXTENT : -SUBSET_HALF_EXTENT;
      const position = center.sub(0.5).add(vec2(localX, localY))
        .add(displacementAt(currentTranslation, currentLinear, localX, localY));
      const valid = position.x.greaterThanEqual(2).and(position.y.greaterThanEqual(2))
        .and(position.x.lessThanEqual(FRAME_WIDTH - 3)).and(position.y.lessThanEqual(FRAME_HEIGHT - 3));
      inside.mulAssign(select(valid, 1, 0));
    }
    If(inside.lessThan(0.5), () => currentStatus.assign(-1));

    const active = currentStatus.abs().lessThanEqual(0.5).and(inside.greaterThanEqual(0.5));
    const sumCurrent = float(0).toVar();
    const sumCurrent2 = float(0).toVar();
    const sumReferenceCurrent = float(0).toVar();
    const sumJCurrent = Array.from({ length: 6 }, () => float(0).toVar());
    If(active, () => {
      for (let batch = 0; batch < pixelsPerLane; ++batch) {
        const pixel = lane.add(batch * solverLanes);
        If(pixel.lessThan(SUBSET_PIXELS), () => {
          const pixelX = pixel.mod(SUBSET_SIZE);
          const pixelY = pixel.div(SUBSET_SIZE);
          const localX = float(pixelX).sub(SUBSET_HALF_EXTENT);
          const localY = float(pixelY).sub(SUBSET_HALF_EXTENT);
          const displacement = displacementAt(currentTranslation, currentLinear, localX, localY);
          const current = sampleTarget(center.sub(0.5).add(vec2(localX, localY)).add(displacement));
          const templateIndex = uint(gridY.mul(SUBSET_SIZE).add(pixelY).mul(FRAME_WIDTH)
            .add(gridX.mul(SUBSET_SIZE)).add(pixelX));
          const reference = referenceNode.element(templateIndex);
          const jacobian = [
            reference.y,
            reference.y.mul(localX),
            reference.y.mul(localY),
            reference.z,
            reference.z.mul(localX),
            reference.z.mul(localY),
          ];
          sumCurrent.addAssign(current);
          sumCurrent2.addAssign(current.mul(current));
          sumReferenceCurrent.addAssign(reference.x.mul(current));
          for (const component of activeMotionComponents()) {
            sumJCurrent[component].addAssign(jacobian[component].mul(current));
          }
        });
      }
    });

    const totalsA = subgroupAdd(
      vec4(sumCurrent, sumCurrent2, sumReferenceCurrent, sumJCurrent[0]),
    ).toVar();
    const totalsB = subgroupAdd(
      vec4(sumJCurrent[1], sumJCurrent[2], sumJCurrent[3], sumJCurrent[4]),
    ).toVar();
    const totalsC = subgroupAdd(vec4(sumJCurrent[5], 0, 0, 0)).toVar();

    If(active, () => {
      const reducedJ = [totalsA.w, totalsB.x, totalsB.y, totalsB.z, totalsB.w, totalsC.x];
      const currentMean = totalsA.x.div(SUBSET_PIXELS);
      const currentNorm = totalsA.y.sub(totalsA.x.mul(currentMean)).max(0).sqrt();
      const referenceNorm = metadataValue(subsetIndex, 0);
      currentIterations.assign(float(iteration).add(1));

      If(currentNorm.lessThan(1e-9).or(referenceNorm.lessThan(1e-9)), () => {
        currentStatus.assign(-2);
      }).Else(() => {
        const ratio = referenceNorm.div(currentNorm);
        const delta = solveIncrement(subsetIndex, reducedJ, currentMean, ratio);
        const determinant = incrementDeterminant(delta);
        If(determinant.abs().lessThan(1e-8), () => {
          currentStatus.assign(-1);
        }).Else(() => {
          const update = composeInverseUpdate(
            currentTranslation,
            currentLinear,
            delta,
            determinant,
          );
          currentTranslation.assign(update.translation);
          currentLinear.assign(update.linear);
          const zncc = totalsA.z.div(referenceNorm.mul(currentNorm));
          currentStatus.assign(convergenceStatus(update.cornerChange, zncc));
        });
      });
    });
  });

  If(lane.equal(0), () => {
    translationNode.element(subsetIndex).assign(currentTranslation);
    linearNode.element(subsetIndex).assign(currentLinear);
    statusNode.element(subsetIndex).assign(vec2(currentStatus, currentIterations));
  });
  })().compute([GRID_WIDTH, GRID_HEIGHT, 1], [solverLanes, 1, 1])
    .setName(`DIC-RT ${solverLanes}-lane subgroup affine IC-GN solver`);
};

const median9Node = (values, count) => {
  Loop(9, () => {
    Loop(8, ({ i }) => {
      If(i.add(1).lessThan(count).and(values.element(i).greaterThan(values.element(i.add(1)))), () => {
        const temporary = float(values.element(i)).toVar();
        values.element(i).assign(values.element(i.add(1)));
        values.element(i.add(1)).assign(temporary);
      });
    });
  });
  const middle = count.div(2);
  return select(
    count.mod(2).equal(1),
    values.element(middle),
    values.element(middle.sub(1)).add(values.element(middle)).mul(0.5),
  );
};

/** Displacement UOD or an explicit centred 3x3 component median. */
const makeMedianKernel = () => Fn(() => {
  const subsetIndex = instanceIndex;
  const translation = translationNode.element(subsetIndex);
  const linear = linearNode.element(subsetIndex);
  const status = statusNode.element(subsetIndex);
  filteredTranslationNode.element(subsetIndex).assign(translation);
  filteredLinearNode.element(subsetIndex).assign(linear);
  filteredStatusNode.element(subsetIndex).assign(status);
  If(status.x.lessThan(0.5), () => Return());

  const predictedX = array('float', 9).toVar();
  const predictedY = array('float', 9).toVar();
  const p1 = array('float', 9).toVar();
  const p2 = array('float', 9).toVar();
  const p4 = array('float', 9).toVar();
  const p5 = array('float', 9).toVar();
  const pureMedian = configuration.outlierPolicy === 'median';
  const count = int(pureMedian ? 1 : 0).toVar();
  if (pureMedian) {
    predictedX.element(0).assign(translation.x);
    predictedY.element(0).assign(translation.y);
    p1.element(0).assign(linear.x);
    p2.element(0).assign(linear.y);
    p4.element(0).assign(linear.z);
    p5.element(0).assign(linear.w);
  }
  const pointX = int(subsetIndex.mod(uint(GRID_WIDTH)));
  const pointY = int(subsetIndex.div(uint(GRID_WIDTH)));

  for (let dy = -1; dy <= 1; ++dy) {
    for (let dx = -1; dx <= 1; ++dx) {
      if (dx === 0 && dy === 0) continue;
      const neighbourX = pointX.add(dx);
      const neighbourY = pointY.add(dy);
      const inside = neighbourX.greaterThanEqual(0).and(neighbourY.greaterThanEqual(0))
        .and(neighbourX.lessThan(GRID_WIDTH)).and(neighbourY.lessThan(GRID_HEIGHT));
      If(inside, () => {
        const neighbour = uint(neighbourY.mul(GRID_WIDTH).add(neighbourX));
        If(statusNode.element(neighbour).x.greaterThanEqual(0.5), () => {
          const neighbourTranslation = translationNode.element(neighbour);
          const neighbourLinear = linearNode.element(neighbour);
          const deltaX = -SUBSET_SIZE * dx;
          const deltaY = -SUBSET_SIZE * dy;
          predictedX.element(count).assign(neighbourTranslation.x
            .add(neighbourLinear.x.mul(deltaX)).add(neighbourLinear.y.mul(deltaY)));
          predictedY.element(count).assign(neighbourTranslation.y
            .add(neighbourLinear.z.mul(deltaX)).add(neighbourLinear.w.mul(deltaY)));
          p1.element(count).assign(neighbourLinear.x);
          p2.element(count).assign(neighbourLinear.y);
          p4.element(count).assign(neighbourLinear.z);
          p5.element(count).assign(neighbourLinear.w);
          count.addAssign(1);
        });
      });
    }
  }
  if (!pureMedian) If(count.lessThan(4), () => Return());

  const filteredTranslation = translation.toVar();
  const filteredLinear = linear.toVar();
  const filteredStatus = status.toVar();
  const componentMedian = (samples) => {
    const values = array('float', 9).toVar();
    Loop(9, ({ i }) => {
      values.element(i).assign(samples.element(i));
    });
    return float(median9Node(values, count)).toVar();
  };
  const displacementOutlier = (samples, ownValue, median) => {
    const values = array('float', 9).toVar();
    Loop(9, ({ i }) => {
      values.element(i).assign(samples.element(i).sub(median).abs());
    });
    const spread = median9Node(values, count);
    return ownValue.sub(median).abs().div(spread.add(0.1)).greaterThan(2);
  };

  const medianX = componentMedian(predictedX);
  const medianY = componentMedian(predictedY);
  let medianP1 = linear.x;
  let medianP2 = linear.y;
  let medianP4 = linear.z;
  let medianP5 = linear.w;
  if (['replace', 'median'].includes(configuration.outlierPolicy)
      && configuration.motionModel === 'affine') {
    medianP1 = componentMedian(p1);
    medianP2 = componentMedian(p2);
    medianP4 = componentMedian(p4);
    medianP5 = componentMedian(p5);
  }
  if (pureMedian) {
    filteredTranslation.assign(vec2(medianX, medianY));
    if (configuration.motionModel === 'affine') {
      filteredLinear.assign(vec4(medianP1, medianP2, medianP4, medianP5));
    }
  } else {
    const flagged = displacementOutlier(predictedX, translation.x, medianX)
      .or(displacementOutlier(predictedY, translation.y, medianY));
    If(flagged, () => {
      if (configuration.outlierPolicy === 'replace') {
        filteredTranslation.assign(vec2(medianX, medianY));
        if (configuration.motionModel === 'affine') {
          filteredLinear.assign(vec4(medianP1, medianP2, medianP4, medianP5));
        }
      }
      filteredStatus.x.assign(2);
    });
  }

  filteredTranslationNode.element(subsetIndex).assign(filteredTranslation);
  filteredLinearNode.element(subsetIndex).assign(filteredLinear);
  filteredStatusNode.element(subsetIndex).assign(filteredStatus);
})().compute(GRID_COUNT, [64]).setName(
  configuration.outlierPolicy === 'median'
    ? 'DIC-RT centred 3x3 field median' : 'DIC-RT displacement UOD',
);

const buildKernels = () => {
  const webgpu = renderer.backend.isWebGPUBackend === true;
  solverKind = webgpu
    ? (useSubgroupSolver ? 'subgroup-pending-validation' : 'portable-workgroup')
    : 'serial-webgl2';
  return {
    unpack: webgpu ? makeUnpackKernel() : null,
    prefilterX: makePrefilterKernel(sourceNode, temporaryNode, true),
    prefilterY: makePrefilterKernel(temporaryNode, coefficientsNode, false),
    solver: webgpu
      ? (useSubgroupSolver
        ? makeSubgroupSolverKernel()
        : makeFusedCooperativeSolverKernel())
      : makeSerialSolverKernel(),
    median: makeMedianKernel(),
  };
};

/** Return the subgroup width selected for a 16-invocation compute workgroup. */
const measureSubgroupWidth = async () => {
  if (renderer.backend.isWebGPUBackend !== true
      || !renderer.backend.device.features.has('subgroups')) return 0;
  const attribute = new THREE.StorageBufferAttribute(new Float32Array(1), 1);
  const output = storage(attribute, 'float', 1);
  const probe = Fn(() => {
    If(localId.x.equal(0), () => output.element(int(0)).assign(float(subgroupSize)));
  })().compute([1, 1, 1], [SOLVER_LANES]).setName('DIC-RT subgroup width probe');
  renderer.compute(probe);
  const data = await renderer.getArrayBufferAsync(attribute);
  return new Float32Array(data)[0];
};

const ensureRenderer = async () => {
  if (rendererPromise) return rendererPromise;
  rendererPromise = (async () => {
    renderer = new THREE.WebGPURenderer({
      antialias: false,
      powerPreference: 'high-performance',
      trackTimestamp: true,
      forceWebGL: FORCE_WEBGL,
    });
    await renderer.init();
    subgroupProbeWidth = await measureSubgroupWidth();
    useSubgroupSolver = subgroupProbeWidth >= SOLVER_LANES;
    solverKind = renderer.backend.isWebGPUBackend === true
      ? (useSubgroupSolver ? 'subgroup-pending-validation' : 'portable-workgroup')
      : 'serial-webgl2';
    return renderer;
  })();
  return rendererPromise;
};

const loadImage = async (dataUrl) => {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return createImageBitmap(blob);
};

const setTarget = (image) => {
  if (!attributes) throw new Error('A reference image must be configured first');
  const gray = imageToGray(image, FRAME_WIDTH, FRAME_HEIGHT);
  for (let index = 0; index < PIXEL_COUNT; ++index) attributes.source.array[index] = gray[index];
  attributes.source.needsUpdate = true;
  targetUsesPackedBytes = false;
};

/** Upload an already-grayscale frame without decoding an intermediate image. */
const setTargetBytes = (pixels, width, height) => {
  if (!attributes) throw new Error('A reference image must be configured first');
  if (width !== FRAME_WIDTH || height !== FRAME_HEIGHT) {
    throw new Error(
      `Target dimensions ${width}x${height} do not match the reference ${FRAME_WIDTH}x${FRAME_HEIGHT}`,
    );
  }
  if (!(pixels instanceof Uint8Array) || pixels.byteLength !== PIXEL_COUNT) {
    throw new Error(`Expected ${PIXEL_COUNT} grayscale bytes`);
  }
  if (renderer.backend.isWebGPUBackend === true) {
    new Uint8Array(attributes.sourcePacked.array.buffer).set(pixels);
    attributes.sourcePacked.needsUpdate = true;
    targetUsesPackedBytes = true;
    return;
  }
  for (let index = 0; index < PIXEL_COUNT; ++index) attributes.source.array[index] = pixels[index] / 255;
  attributes.source.needsUpdate = true;
  targetUsesPackedBytes = false;
};

const resetSolver = () => {
  attributes.translation.array.fill(0);
  attributes.linear.array.fill(0);
  attributes.status.array.fill(0);
  attributes.translation.needsUpdate = true;
  attributes.linear.needsUpdate = true;
  attributes.status.needsUpdate = true;
};

const synchronize = async () => {
  if (renderer.backend.isWebGPUBackend === true) {
    await renderer.backend.device.queue.onSubmittedWorkDone();
  } else {
    renderer.backend.gl.finish();
  }
};

/** Validate the subgroup size selected for the production solver pipeline. */
const validateSubgroupSolver = async () => {
  if (!useSubgroupSolver || renderer.backend.isWebGPUBackend !== true) return;
  attributes.subgroupDiagnostic.array[0] = 0;
  attributes.subgroupDiagnostic.needsUpdate = true;
  resetSolver();
  renderer.compute(kernels.solver);
  await synchronize();
  const data = await renderer.getArrayBufferAsync(attributes.subgroupDiagnostic);
  subgroupSolverWidth = Math.trunc(new Float32Array(data)[0]);
  const expectedWidth = Math.min(subgroupProbeWidth, SUBSET_PIXELS);
  if (subgroupSolverWidth !== expectedWidth) {
    useSubgroupSolver = false;
    subgroupValidation = 'mismatch';
    kernels.solver = makeFusedCooperativeSolverKernel();
    solverKind = 'portable-workgroup-subgroup-mismatch';
    return;
  }
  subgroupValidation = 'validated';
  solverKind = 'subgroup';
};

const resolveGpuTimestamp = async () => {
  if (!renderer.backend.trackTimestamp) return null;
  return renderer.resolveTimestampsAsync('compute');
};

const executePipeline = async ({ measureGpu = false } = {}) => {
  resetSolver();
  const inputKernels = targetUsesPackedBytes ? [kernels.unpack] : [];
  const outlierKernels = configuration.outlierPolicy === 'disabled' ? [] : [kernels.median];
  const computeKernels = configuration.interpolation === 'bspline'
    ? [...inputKernels, kernels.prefilterX, kernels.prefilterY, kernels.solver, ...outlierKernels]
    : [...inputKernels, kernels.solver, ...outlierKernels];
  if (renderer.backend.isWebGPUBackend !== true) {
    await renderer.computeAsync(computeKernels);
    if (measureGpu) return resolveGpuTimestamp();
    await synchronize();
    return null;
  }
  computeKernels.forEach((kernel) => renderer.compute(kernel));
  if (measureGpu) {
    const gpuMs = await resolveGpuTimestamp();
    if (gpuMs !== null) return gpuMs;
  }
  await synchronize();
  return null;
};

const readResult = async (timing) => {
  const resultAttributes = configuration.outlierPolicy === 'disabled'
    ? [attributes.translation, attributes.linear, attributes.status]
    : [attributes.filteredTranslation, attributes.filteredLinear, attributes.filteredStatus];
  const [translationBuffer, linearBuffer, statusBuffer] = await Promise.all([
    renderer.getArrayBufferAsync(resultAttributes[0]),
    renderer.getArrayBufferAsync(resultAttributes[1]),
    renderer.getArrayBufferAsync(resultAttributes[2]),
  ]);
  const translation = new Float32Array(translationBuffer);
  const linear = new Float32Array(linearBuffer);
  const statusPairs = new Float32Array(statusBuffer);
  let replaced = 0;
  const parameters = new Float32Array(GRID_COUNT * 6);
  const status = new Float32Array(GRID_COUNT);
  const iterations = new Float32Array(GRID_COUNT);
  let accepted = 0;
  let meanX = 0;
  let meanY = 0;
  let iterationTotal = 0;
  for (let index = 0; index < GRID_COUNT; ++index) {
    parameters[6 * index] = translation[2 * index];
    parameters[6 * index + 1] = linear[4 * index];
    parameters[6 * index + 2] = linear[4 * index + 1];
    parameters[6 * index + 3] = translation[2 * index + 1];
    parameters[6 * index + 4] = linear[4 * index + 2];
    parameters[6 * index + 5] = linear[4 * index + 3];
    status[index] = statusPairs[2 * index];
    if (status[index] > 1.5) replaced += 1;
    iterations[index] = statusPairs[2 * index + 1];
    if (status[index] >= 0.5) {
      accepted += 1;
      meanX += parameters[6 * index];
      meanY += parameters[6 * index + 3];
      iterationTotal += iterations[index];
    }
  }
  return {
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
};

const percentile = (values, probability) => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(probability * sorted.length) - 1)];
};

const register = async (dataUrl, options = {}) => {
  await ensureRenderer();
  if (!referenceReady) throw new Error('A reference image must be configured first');
  configure(options);
  const image = await loadImage(dataUrl);
  try {
    setTarget(image);
  } finally {
    image.close();
  }
  const wallStart = performance.now();
  const gpuMs = await executePipeline({ measureGpu: true });
  const wallMs = performance.now() - wallStart;
  return readResult({ gpuMs, wallMs });
};

/** Register a raw contiguous uint8 grayscale frame received from Python. */
const registerBytes = async (pixels, width, height, options = {}) => {
  await ensureRenderer();
  if (!referenceReady) throw new Error('A reference image must be configured first');
  configure(options);
  setTargetBytes(pixels, width, height);
  const wallStart = performance.now();
  const gpuMs = await executePipeline({ measureGpu: true });
  const wallMs = performance.now() - wallStart;
  return readResult({ gpuMs, wallMs });
};

const benchmark = async (dataUrl, repeats = 20, warmups = 3, options = {}) => {
  await ensureRenderer();
  if (!referenceReady) throw new Error('A reference image must be configured first');
  configure(options);
  const image = await loadImage(dataUrl);
  try {
    setTarget(image);
  } finally {
    image.close();
  }
  for (let index = 0; index < warmups; ++index) await executePipeline();
  // Discard warm-up timestamp queries so the first sample contains one
  // pipeline execution, not the accumulated warm-up command buffers.
  await resolveGpuTimestamp();
  const gpuTimes = [];
  const wallTimes = [];
  for (let index = 0; index < repeats; ++index) {
    const start = performance.now();
    const gpuMs = await executePipeline({ measureGpu: true });
    wallTimes.push(performance.now() - start);
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

function configure(options = {}, allowInterpolationRebuild = false) {
  const interpolation = options.interpolation === undefined
    ? configuration.interpolation : String(options.interpolation).toLowerCase();
  if (!['bspline', 'linear', 'cubic'].includes(interpolation)) {
    throw new Error("interpolation must be 'bspline', 'linear', or 'cubic'");
  }
  if (!allowInterpolationRebuild && referenceReady
      && interpolation !== configuration.interpolation) {
    throw new Error('Changing interpolation requires setReference()');
  }
  const motionModel = options.motionModel === undefined
    ? configuration.motionModel : String(options.motionModel).toLowerCase();
  if (!['affine', 'translation'].includes(motionModel)) {
    throw new Error("motionModel must be 'affine' or 'translation'");
  }
  if (!allowInterpolationRebuild && referenceReady && motionModel !== configuration.motionModel) {
    throw new Error('Changing motionModel requires setReference()');
  }
  const outlierPolicy = options.outlierPolicy === undefined
    ? configuration.outlierPolicy : String(options.outlierPolicy).toLowerCase();
  if (!['replace', 'flag', 'disabled', 'median'].includes(outlierPolicy)) {
    throw new Error("outlierPolicy must be 'replace', 'flag', 'disabled', or 'median'");
  }
  if (!allowInterpolationRebuild && referenceReady
      && outlierPolicy !== configuration.outlierPolicy) {
    throw new Error('Changing outlierPolicy requires setReference()');
  }
  const next = {
    iterations: options.iterations === undefined
      ? configuration.iterations : Number(options.iterations),
    tolerance: options.tolerance === undefined
      ? configuration.tolerance : Number(options.tolerance),
    znccMin: options.znccMin === undefined ? configuration.znccMin : Number(options.znccMin),
    interpolation,
    motionModel,
    outlierPolicy,
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
  toleranceNode.value = next.tolerance;
  znccMinNode.value = next.znccMin;
  iterationCountNode.value = next.iterations;
  return { ...configuration };
}

const setReference = async (dataUrl, options = {}) => {
  await ensureRenderer();
  configure(options, true);
  const image = await loadImage(dataUrl);
  const start = performance.now();
  const runtimeStart = performance.now();
  initializeRuntime(image.width, image.height);
  let runtimeBuildWallMs = performance.now() - runtimeStart;
  const referenceStart = performance.now();
  try {
    prepareReference(image);
  } finally {
    image.close();
  }
  const referenceBuildWallMs = performance.now() - referenceStart;
  const subgroupValidationStart = performance.now();
  await validateSubgroupSolver();
  runtimeBuildWallMs += performance.now() - subgroupValidationStart;
  const preparationWallMs = performance.now() - start;
  const details = await info();
  return {
    preparationWallMs,
    runtimeBuildWallMs,
    referenceBuildWallMs,
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    subsetSize: SUBSET_SIZE,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    config: { ...configuration },
    device: details.device,
  };
};

const info = async () => {
  await ensureRenderer();
  const adapterInfo = renderer.backend.device?.adapterInfo || {};
  const isWebGPU = renderer.backend.isWebGPUBackend === true;
  const timerQuery = isWebGPU
    ? Boolean(renderer.backend.trackTimestamp)
    : Boolean(renderer.backend.hasTimestamp);
  const identity = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device]
    .filter(Boolean).join(' ');
  return {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    subsetSize: SUBSET_SIZE,
    device: `Three.js r${THREE.REVISION} ${renderer.backend.constructor.name}${identity ? ` (${identity})` : ''}`,
    timerQuery,
    backend: renderer.backend.constructor.name,
    threeRevision: THREE.REVISION,
    timestampQueries: timerQuery,
    vendor: adapterInfo.vendor || null,
    architecture: adapterInfo.architecture || null,
    adapterDevice: adapterInfo.device || null,
    description: adapterInfo.description || null,
    subgroups: isWebGPU && renderer.backend.device.features.has('subgroups'),
    subgroupWidth: subgroupSolverWidth || subgroupProbeWidth,
    subgroupProbeWidth,
    subgroupSolverWidth,
    subgroupValidation,
    motionModel: configuration.motionModel,
    outlierPolicy: configuration.outlierPolicy,
    solver: solverKind,
  };
};

window.ICGNHeadless = {
  info,
  configure,
  setReference,
  register,
  registerBytes,
  benchmark,
};
window.ICGNHeadlessReady = ensureRenderer();

