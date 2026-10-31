import { validateMask } from './roi.js';
import { createInputConverter, frameSize } from './input.js';
import { createPyramidHost } from './pyramid.js';
import { SlotRing } from './ring.js';
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
      'znccMin', 'outlierPolicy', 'initialization', 'temporalPrediction', 'temporalFineCorrection',
      'temporalMaxAge', 'divergencePatience', 'divergenceMinIterations', 'pipelineDepth', 'readback',
      'spatialWeighting', 'roiPolicy', 'roiTruncation', 'pyramidPolicy', 'solverWorkgroupSize']);
    for (const key of Object.keys(options)) if (!supported.has(key)) throw new Error(`Unsupported engine option: ${key}`);
    if (options.solverWorkgroupSize !== undefined && options.solverWorkgroupSize !== 16) throw new Error('The browser engine supports only solverWorkgroupSize=16');
    if (options.roiTruncation !== undefined && options.roiTruncation !== 'none') throw new Error('ROI truncation is not supported in the browser engine');
    if (options.temporalFineCorrection !== undefined && typeof options.temporalFineCorrection !== 'boolean') throw new Error('temporalFineCorrection must be boolean');
    for (const key of ['width', 'height']) if (options[key] !== undefined && (!Number.isInteger(options[key]) || options[key] < 1)) throw new Error(`${key} must be a positive integer`);

    for (const key of ['subsetSize', 'subsetStride', 'pyramidLevels', 'iterations', 'tolerance', 'znccMin']) {
      if (options[key] !== undefined && typeof options[key] !== 'number') throw new Error(`${key} must be a number`);
    }
    for (const key of ['iterationsPerLevel', 'tolerancePerLevel']) {
      if (options[key] !== undefined && (typeof options[key] === 'string' || options[key] === null
          || !Array.from(options[key]).every((value) => typeof value === 'number'))) throw new Error(`${key} must contain numbers`);
    }
    if (options.spatialWeighting !== undefined && options.spatialWeighting !== 'uniform') {
      throw new Error('Gaussian weighting is not supported in the browser engine');
    }
    if (options.roiPolicy !== undefined && options.roiPolicy !== 'full') {
      throw new Error('partial ROI is not supported in the browser engine');
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
    this.reference = null;
    this.inputs.forEach((input) => input.destroy()); this.inputs = [];
    for (let i = 0; i < this.ring.depth; i++) this.inputs.push(await createInputConverter(this.device, width, height));
    if (!(source instanceof Uint8Array)) pixels = await this.inputs[0].referencePixels(source);
    if (pixels instanceof Uint8Array) {
      if (pixels.length !== width * height) throw new Error('Expected width * height grayscale bytes');
      source = { pixels, width, height };
    }
    this.reference = await this.host.setReference(source, { ...this.options, ...options, roiMask: ownedMask });
    return this.reference;
  }
  register(source) {
    this.assertOpen();
    if (!this.reference) throw new Error('A reference image must be configured first');
    return this.ring.run(async (slotIndex) => {
      const started = performance.now();
      const result = source instanceof Uint8Array
        ? await this.host.registerBytes(source, this.reference.width, this.reference.height, {}, slotIndex)
        : await this.host.registerExternal(source, slotIndex);
      result.timing.totalWallMs = performance.now() - started;
      return result;
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
  get device() { return this.host?.device; }
  get imageTexture() { this.assertOpen(); return this.host.imageTexture; }
  get fieldBuffer() { this.assertOpen(); return this.host.fieldBuffer; }
  destroy() { if (this.closed) return; this.closed = true; this.host.destroy(); this.inputs.forEach((input) => input.destroy()); this.inputs = []; this.host = null; this.options = null; }
}
