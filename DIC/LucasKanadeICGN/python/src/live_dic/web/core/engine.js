import { validateMask } from './roi.js';
import { createInputConverter, frameSize } from './input.js';
import { createPyramidHost } from './pyramid.js';
import { SlotRing } from './ring.js';
import { gpuMemoryUsage } from './gpu-memory.js';
import { validateSpatialPropagation } from './spatial.js';
export { probeCapabilities } from './device.js';

/** Dependency-free controller. Every instance owns its reference and queue. */
export class DICEngine {
  static async create(options = {}) {
    const engine = new DICEngine(options);
    try { await engine.host.ensureDevice(); return engine; }
    catch (error) { engine.destroy(); throw error; }
  }
  constructor(options = {}) {
    const supported = new Set(['device', 'width', 'height', 'subsetSize', 'subsetStride', 'motionModel',
      'interpolation', 'pyramidLevels', 'iterationsPerLevel', 'tolerancePerLevel', 'iterations', 'tolerance',
      'znccMin', 'outlierPolicy', 'medianFilterSize', 'medianFilterScope', 'initialization', 'temporalPrediction', 'temporalFineCorrection',
      'temporalMaxAge', 'divergencePatience', 'divergenceMinIterations', 'pipelineDepth', 'readback',
      'spatialWeighting', 'roiPolicy', 'roiTruncation', 'pyramidPolicy', 'solverWorkgroupSize',
      'referencePreparation', 'globalAffineLevels', 'spatialPropagationPasses']);
    for (const key of Object.keys(options)) if (!supported.has(key)) throw new Error(`Unsupported engine option: ${key}`);
    validateSpatialPropagation(options);
    if (options.referencePreparation !== undefined && !['cpu', 'gpu'].includes(options.referencePreparation)) {
      throw new Error("referencePreparation must be 'cpu' or 'gpu'");
    }
    if (options.referencePreparation === 'gpu'
        && ((options.motionModel ?? 'affine') === 'quadratic' || (options.pipelineDepth ?? 1) !== 1)) {
      throw new Error('GPU reference preparation supports translation/affine and one slot');
    }
    if (options.referencePreparation === 'gpu' && options.initialization === 'wgpu-matching') {
      throw new Error('WGPU-Matching initialization requires CPU reference preparation');
    }
    if (options.solverWorkgroupSize !== undefined && ![16, 32].includes(options.solverWorkgroupSize)) throw new Error('solverWorkgroupSize must be 16 or 32');
    if (options.roiTruncation !== undefined && options.roiTruncation !== 'none') throw new Error('ROI truncation is not supported in the browser engine');
    if (options.temporalFineCorrection !== undefined && typeof options.temporalFineCorrection !== 'boolean') throw new Error('temporalFineCorrection must be boolean');
    for (const key of ['width', 'height']) if (options[key] !== undefined && (!Number.isInteger(options[key]) || options[key] < 1)) throw new Error(`${key} must be a positive integer`);

    for (const key of ['subsetSize', 'subsetStride', 'pyramidLevels', 'iterations', 'tolerance', 'znccMin', 'medianFilterSize']) {
      if (options[key] !== undefined && typeof options[key] !== 'number') throw new Error(`${key} must be a number`);
    }
    if (options.medianFilterSize !== undefined
        && ![3, 5, 7, 9].includes(options.medianFilterSize)) {
      throw new Error('medianFilterSize must be 3, 5, 7, or 9 subsets');
    }
    if (options.medianFilterSize > 3 && options.outlierPolicy !== 'median') {
      throw new Error("medianFilterSize greater than 3 requires outlierPolicy='median'");
    }
    if (options.medianFilterScope !== undefined
        && !['rejected', 'all'].includes(options.medianFilterScope)) {
      throw new Error("medianFilterScope must be 'rejected' or 'all'");
    }
    if (options.medianFilterScope === 'all' && options.outlierPolicy !== 'median') {
      throw new Error("medianFilterScope='all' requires outlierPolicy='median'");
    }
    if (options.globalAffineLevels !== undefined
        && (!Number.isInteger(options.globalAffineLevels)
          || options.globalAffineLevels < 5 || options.globalAffineLevels > 8)) {
      throw new Error('globalAffineLevels must be an integer between 5 and 8');
    }
    if (options.globalAffineLevels !== undefined && options.globalAffineLevels !== 5
        && options.initialization !== 'global-affine') {
      throw new Error('globalAffineLevels requires initialization="global-affine"');
    }
    for (const key of ['iterationsPerLevel', 'tolerancePerLevel']) {
      if (options[key] !== undefined && (typeof options[key] === 'string' || options[key] === null
          || !Array.from(options[key]).every((value) => typeof value === 'number'))) throw new Error(`${key} must contain numbers`);
    }
    if (options.spatialWeighting !== undefined && options.spatialWeighting !== 'uniform') {
      throw new Error('Gaussian weighting is not supported in the browser engine');
    }
    if (options.roiPolicy !== undefined && !['full', 'partial'].includes(options.roiPolicy)) {
      throw new Error("roiPolicy must be 'full' or 'partial'");
    }
    if (!['full', 'decimated', 'none'].includes(options.readback === undefined ? 'full' : options.readback)) throw new Error('Unknown readback mode');
    this.options = Object.freeze({ readback: 'full', ...options });
    if (options.pipelineDepth !== undefined && (!Number.isInteger(options.pipelineDepth) || options.pipelineDepth < 1)) throw new Error('pipelineDepth must be a positive integer');
    this.ring = new SlotRing(options.pipelineDepth ?? 1);
    this.inputs = [];
    this.host = createPyramidHost({ ...options, inputConverterForSlot: (i) => this.inputs[i] });
    this.closed = false;
  }
  assertOpen() { if (this.closed) throw new Error('The engine has been destroyed'); }
  async setReference(source, options = {}) {
    this.assertOpen();
    for (const key of Object.keys(options)) if (!['roiMask', 'roiPolicy', 'roiTruncation'].includes(key)) throw new Error(`Unsupported reference option: ${key}`);
    if (!this.ring.idle) throw new Error('Finish the current sequence before replacing the reference');
    let pixels = source;
    const [width, height] = source instanceof Uint8Array ? [this.options.width, this.options.height] : frameSize(source);
    const size = this.options.subsetSize ?? 16, scale = 2 ** ((this.options.pyramidLevels ?? 2) - 1);
    if (!Number.isInteger(width) || !Number.isInteger(height) || Math.floor(width / scale) < size || Math.floor(height / scale) < size) throw new Error('Source dimensions must contain the coarsest subset');
    if (source instanceof Uint8Array && source.length !== width * height) throw new Error('Expected width * height grayscale bytes');
    if (options.roiTruncation !== undefined && options.roiTruncation !== 'none') throw new Error('ROI truncation is not supported in the browser engine');
    const ownedMask = validateMask(options.roiMask, width, height, options.roiPolicy === undefined ? 'full' : options.roiPolicy, this.host.configuration.motionModel);
    // Negotiate the real source shape before allocating conversion textures or
    // reading pixels. This also covers engines created without width/height.
    await this.host.ensureDevice({width,height,roiMask:ownedMask});
    this.reference = null;
    this.inputs.forEach((input) => input.destroy()); this.inputs = [];
    for (let i = 0; i < this.ring.depth; i++) this.inputs.push(await createInputConverter(this.device, width, height));
    if (this.options.referencePreparation === 'gpu') {
      if(['fourier','global-affine','wgpu-matching'].includes(this.options.initialization)&&!(source instanceof Uint8Array))pixels=await this.inputs[0].referencePixels(source);
      source = { width, height, gpuSource: source,...(['fourier','global-affine','wgpu-matching'].includes(this.options.initialization)?{pixels}:{}) };
    } else {
      if (!(source instanceof Uint8Array)) pixels = await this.inputs[0].referencePixels(source);
      if (pixels instanceof Uint8Array) {
        if (pixels.length !== width * height) throw new Error('Expected width * height grayscale bytes');
        source = { pixels, width, height };
      }
    }
    this.reference = await this.host.setReference(source, { ...this.options, ...options, roiMask: ownedMask });
    return this.reference;
  }
  register(source, { sourceKind = null, initialAffineEstimate = null } = {}) {
    this.assertOpen();
    if (!this.reference) throw new Error('A reference image must be configured first');
    if (initialAffineEstimate !== null && this.options.initialization !== 'wgpu-matching') {
      throw new Error('initialAffineEstimate requires initialization="wgpu-matching"');
    }
    return this.ring.run(async (slotIndex) => {
      const started = performance.now();
      const result = source instanceof Uint8Array
        ? await this.host.registerBytes(source, this.reference.width, this.reference.height, {}, slotIndex, initialAffineEstimate)
        : await this.host.registerExternal(source, slotIndex, { sourceKind, initialAffineEstimate });
      result.timing.totalWallMs = performance.now() - started;
      return result;
    });
  }
  registerPair(referenceFrame, currentFrame) {
    this.assertOpen();
    if (this.options.referencePreparation !== 'gpu') throw new Error('registerPair requires referencePreparation="gpu"');
    if (!this.reference) throw new Error('Prepare the static geometry/ROI with setReference first');
    for (const frame of [referenceFrame, currentFrame]) {
      const [width, height] = frame instanceof Uint8Array
        ? [this.reference.width, this.reference.height] : frameSize(frame);
      if (width !== this.reference.width || height !== this.reference.height
          || (frame instanceof Uint8Array && frame.length !== width * height)) {
        throw new Error('Pair images must match the prepared reference dimensions');
      }
    }
    return this.ring.run(async () => {
      const started = performance.now();
      try {
        const result=await this.host.registerPair(referenceFrame, currentFrame);
        result.timing.totalWallMs = performance.now() - started;
        // The current numeric validity mask stays on GPU. Do not expose a
        // stale cold-reference mask as though it described this new pair.
        this.reference = { ...this.reference, referenceValid: null, referenceValidity: 'gpu-resident',
          referenceAdmissible: result.referenceAdmissible };
        return result;
      } catch (error) { this.reference = null; throw error; }
    });
  }
  resetTemporalState() {
    this.assertOpen();
    if (!this.ring.idle) throw new Error('Finish the current sequence before resetting history');
    this.host.resetTemporalState?.();
  }
  setAnalyticTruth(truth) {
    this.assertOpen();
    if (truth != null && (!Array.isArray(truth) || truth.length !== 2 || !truth.every(Number.isFinite))) throw new Error('Truth must contain two finite affine diagonal gradients');
    this.host.setAnalyticTruth(truth);
  }
  exportField() {
    this.assertOpen();
    if (!this.ring.idle) throw new Error('Pause and finish current frames before exporting');
    return this.host.exportField();
  }
  exportReference(level = 0) {
    this.assertOpen();
    if (!this.ring.idle) throw new Error('Pause and finish current frames before inspecting a reference');
    return this.host.exportReference(level);
  }
  get device() { return this.host?.device; }
  get imageTexture() { this.assertOpen(); return this.host.imageTexture; }
  get fieldBuffer() { this.assertOpen(); return this.host.fieldBuffer; }
  memoryUsage() { const usage=gpuMemoryUsage(this.closed?null:{host:this.host.memoryResources(),inputs:this.inputs});
    if(!this.closed&&this.options.initialization==='fourier')usage.cpuInitializationBytes=this.host.cpuMemoryBytes();return usage; }
  destroy() { if (this.closed) return; this.closed = true; this.host.destroy(); this.inputs.forEach((input) => input.destroy()); this.inputs = []; this.host = null; this.options = null; }
}
