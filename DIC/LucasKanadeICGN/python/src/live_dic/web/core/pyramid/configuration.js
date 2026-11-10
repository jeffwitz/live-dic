import { validateTemporal } from '../temporal.js';
import { validateSpatialPropagation } from '../spatial.js';

const DEFAULT_MAX_ITERATIONS = 30;
const DEFAULT_TOLERANCE = 1e-3;
const DEFAULT_ZNCC_MIN = 0.8;
const DEFAULT_PYRAMID_LEVELS = 2;

export const INTERPOLATION_CONSTANTS = { bspline: 0, linear: 1, cubic: 2 };
export const MOTION_MODEL_CONSTANTS = { affine: 0, translation: 1, quadratic: 2 };
export const OUTLIER_POLICY_CONSTANTS = { flag: 0, replace: 1, disabled: 2, median: 3 };
export const MEDIAN_FILTER_SCOPE_CONSTANTS = { rejected: 0, all: 1 };
// The shared solver uses signed 32-bit arithmetic for the flattened S × S loop.
export const MAX_SUBSET_SIZE = 46_340;

const ITERATION_SCHEDULES = {
  1: [30],
  2: [8, 16],
  3: [6, 8, 16],
  4: [12, 8, 8, 16],
  5: [12, 8, 8, 8, 16],
  6: [12, 8, 8, 8, 8, 16],
};

const TOLERANCE_SCHEDULES = {
  1: [1e-3],
  2: [5e-3, 5e-3],
  3: [1e-2, 5e-3, 5e-3],
  4: [1e-2, 1e-2, 5e-3, 5e-3],
  5: [1e-2, 1e-2, 5e-3, 5e-3, 5e-3],
  6: [1e-2, 1e-2, 1e-2, 5e-3, 5e-3, 5e-3],
};

/** Validate and build the unchanged initial solver configuration. */
export function createInitialConfiguration(options = {}) {
  validateTemporal(options);
  if (options.solverWorkgroupSize !== undefined && ![16, 32].includes(options.solverWorkgroupSize)) {
    throw new Error('solverWorkgroupSize must be 16 or 32');
  }
  const subsetSize = Number(options.subsetSize ?? 16);
  if (!Number.isInteger(subsetSize) || subsetSize < 4 || subsetSize > MAX_SUBSET_SIZE) {
    throw new Error(`subsetSize must be an integer from 4 to ${MAX_SUBSET_SIZE}`);
  }
  const subsetStride = Number(options.subsetStride ?? subsetSize);
  if (!Number.isInteger(subsetStride) || subsetStride < 1 || subsetStride > subsetSize) {
    throw new Error('subsetStride must be an integer between 1 and subsetSize');
  }

  return {
    ...(options.referencePreparation === 'gpu' ? { referencePreparation: 'gpu' } : {}),
    subsetSize,
    subsetStride,
    initialization: options.initialization ?? 'zero',
    temporalPrediction: options.temporalPrediction ?? 'none',
    temporalFineCorrection: options.temporalFineCorrection ?? false,
    temporalMaxAge: options.temporalMaxAge ?? null,
    divergencePatience: options.divergencePatience ?? null,
    divergenceMinIterations: options.divergenceMinIterations ?? 4,
    pyramidPolicy: 'fixed',
    spatialWeighting: 'uniform',
    solverWorkgroupSize: options.solverWorkgroupSize ?? 16,
    pipelineDepth: options.pipelineDepth ?? 1,
    iterations: DEFAULT_MAX_ITERATIONS,
    tolerance: DEFAULT_TOLERANCE,
    znccMin: DEFAULT_ZNCC_MIN,
    pyramidLevels: DEFAULT_PYRAMID_LEVELS,
    iterationsPerLevel: [...ITERATION_SCHEDULES[DEFAULT_PYRAMID_LEVELS]],
    tolerancePerLevel: [...TOLERANCE_SCHEDULES[DEFAULT_PYRAMID_LEVELS]],
    interpolation: 'bspline',
    motionModel: 'affine',
    outlierPolicy: 'replace',
    medianFilterSize: 3,
    medianFilterScope: 'rejected',
    ...(options.initialization === 'global-affine'
      ? { globalAffineLevels: Number(options.globalAffineLevels ?? 5) }
      : {}),
  };
}

/** Resolve a requested change without mutating the active configuration. */
export function resolveConfiguration(
  current,
  options = {},
  {
    hostOptions = {},
    allowPyramidRebuild = false,
    referenceReady = false,
    runtimePyramidLevels = 0,
  } = {},
) {
  if (Object.keys(options).length === 0) return { next: { ...current }, changed: false };
  if (options.solverWorkgroupSize !== undefined
      && options.solverWorkgroupSize !== current.solverWorkgroupSize) {
    throw new Error('Create a new DICEngine to change solverWorkgroupSize');
  }

  const initialization = options.initialization ?? current.initialization;
  validateTemporal({ ...hostOptions, ...current, ...options, initialization });
  if (!allowPyramidRebuild && referenceReady && initialization !== current.initialization) {
    throw new Error('Changing initialization requires setReference()');
  }
  if (initialization === 'fourier' && (options.motionModel ?? current.motionModel) === 'quadratic') {
    throw new Error('Fourier initialization currently supports translation and affine models');
  }

  const globalAffineLevels = options.globalAffineLevels === undefined
    ? Number(current.globalAffineLevels ?? 5)
    : Number(options.globalAffineLevels);
  if (!Number.isInteger(globalAffineLevels) || globalAffineLevels < 5 || globalAffineLevels > 8) {
    throw new Error('globalAffineLevels must be an integer between 5 and 8');
  }
  if (initialization !== 'global-affine' && options.globalAffineLevels !== undefined
      && globalAffineLevels !== 5) {
    throw new Error('globalAffineLevels requires initialization="global-affine"');
  }
  if (!allowPyramidRebuild && referenceReady && initialization === 'global-affine'
      && globalAffineLevels !== current.globalAffineLevels) {
    throw new Error('Changing globalAffineLevels requires setReference()');
  }

  const pyramidLevels = options.pyramidLevels === undefined
    ? current.pyramidLevels
    : Number(options.pyramidLevels);
  if (!Number.isInteger(pyramidLevels) || pyramidLevels < 1 || pyramidLevels > 6) {
    throw new Error('pyramidLevels must be an integer between 1 and 6');
  }
  if (!allowPyramidRebuild && runtimePyramidLevels
      && pyramidLevels !== runtimePyramidLevels) {
    throw new Error('Changing pyramidLevels requires setReference() to rebuild the runtime');
  }

  const interpolation = options.interpolation === undefined
    ? current.interpolation
    : String(options.interpolation).toLowerCase();
  if (!(interpolation in INTERPOLATION_CONSTANTS)) {
    throw new Error("interpolation must be 'bspline', 'linear', or 'cubic'");
  }
  if (!allowPyramidRebuild && referenceReady && interpolation !== current.interpolation) {
    throw new Error('Changing interpolation requires setReference()');
  }

  const motionModel = options.motionModel === undefined
    ? current.motionModel
    : String(options.motionModel).toLowerCase();
  if (!(motionModel in MOTION_MODEL_CONSTANTS)) {
    throw new Error("motionModel must be 'affine', 'translation', or 'quadratic'");
  }
  if (!allowPyramidRebuild && referenceReady && motionModel !== current.motionModel) {
    throw new Error('Changing motionModel requires setReference()');
  }

  const outlierPolicy = options.outlierPolicy === undefined
    ? current.outlierPolicy
    : String(options.outlierPolicy).toLowerCase();
  if (!(outlierPolicy in OUTLIER_POLICY_CONSTANTS)) {
    throw new Error("outlierPolicy must be 'replace', 'flag', 'disabled', or 'median'");
  }
  if (!allowPyramidRebuild && referenceReady && outlierPolicy !== current.outlierPolicy) {
    throw new Error('Changing outlierPolicy requires setReference()');
  }

  const medianFilterScope = options.medianFilterScope === undefined
    ? current.medianFilterScope
    : String(options.medianFilterScope).toLowerCase();
  if (!(medianFilterScope in MEDIAN_FILTER_SCOPE_CONSTANTS)) {
    throw new Error("medianFilterScope must be 'rejected' or 'all'");
  }
  if (medianFilterScope === 'all' && outlierPolicy !== 'median') {
    throw new Error("medianFilterScope='all' requires outlierPolicy='median'");
  }
  if (!allowPyramidRebuild && referenceReady && medianFilterScope !== current.medianFilterScope) {
    throw new Error('Changing medianFilterScope requires setReference()');
  }

  const medianFilterSize = options.medianFilterSize === undefined
    ? current.medianFilterSize
    : Number(options.medianFilterSize);
  if (![3, 5, 7, 9].includes(medianFilterSize)) {
    throw new Error('medianFilterSize must be 3, 5, 7, or 9 subsets');
  }
  if (medianFilterSize > 3 && outlierPolicy !== 'median') {
    throw new Error("medianFilterSize greater than 3 requires outlierPolicy='median'");
  }
  if (!allowPyramidRebuild && referenceReady && medianFilterSize !== current.medianFilterSize) {
    throw new Error('Changing medianFilterSize requires setReference()');
  }

  const changedLevels = pyramidLevels !== current.pyramidLevels;
  const iterationsPerLevel = options.iterationsPerLevel === undefined
    ? (changedLevels ? [...ITERATION_SCHEDULES[pyramidLevels]] : [...current.iterationsPerLevel])
    : Array.from(options.iterationsPerLevel, Number);
  const tolerancePerLevel = options.tolerancePerLevel === undefined
    ? (changedLevels ? [...TOLERANCE_SCHEDULES[pyramidLevels]] : [...current.tolerancePerLevel])
    : Array.from(options.tolerancePerLevel, Number);
  if (iterationsPerLevel.length !== pyramidLevels || tolerancePerLevel.length !== pyramidLevels) {
    throw new Error('Per-level schedules must contain one value per pyramid level');
  }

  const next = {
    initialization,
    iterations: options.iterations === undefined
      ? iterationsPerLevel[pyramidLevels - 1]
      : Number(options.iterations),
    tolerance: options.tolerance === undefined
      ? tolerancePerLevel[pyramidLevels - 1]
      : Number(options.tolerance),
    znccMin: options.znccMin === undefined ? current.znccMin : Number(options.znccMin),
    pyramidLevels,
    iterationsPerLevel,
    tolerancePerLevel,
    interpolation,
    motionModel,
    outlierPolicy,
    medianFilterSize,
    medianFilterScope,
    ...(initialization === 'global-affine' ? { globalAffineLevels } : {}),
    ...(options.spatialPropagationPasses !== undefined || current.spatialPropagationPasses !== undefined
      ? { spatialPropagationPasses: options.spatialPropagationPasses === undefined
        ? current.spatialPropagationPasses : options.spatialPropagationPasses } : {}),
  };
  validateSpatialPropagation({ ...hostOptions, ...current, ...next });
  if (!allowPyramidRebuild && referenceReady
      && (next.spatialPropagationPasses ?? 0) !== (current.spatialPropagationPasses ?? 0)) {
    throw new Error('Changing spatialPropagationPasses requires setReference()');
  }
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

  const changed = Object.entries(next).some(([key, value]) => Array.isArray(value)
    ? value.some((entry, index) => entry !== current[key]?.[index])
      || value.length !== current[key]?.length
    : value !== current[key]);
  return { next, changed };
}
