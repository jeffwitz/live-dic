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
  constructor(options) {
    if (options.spatialWeighting && options.spatialWeighting !== 'uniform') {
      throw new Error('Gaussian weighting is not supported in the browser engine');
    }
    if (options.roiPolicy && options.roiPolicy !== 'full') {
      throw new Error('partial ROI is not supported in the browser engine');
    }
    if (!['full', 'decimated', 'none'].includes(options.readback ?? 'full')) throw new Error('Unknown readback mode');
    this.options = { readback: 'full', ...options };
    this.ring = new SlotRing(options.pipelineDepth ?? 1);
    this.inputs = [];
    this.host = createPyramidHost({ ...options, inputConverterForSlot: (i) => this.inputs[i] });
    this.closed = false;
  }
  assertOpen() { if (this.closed) throw new Error('The engine has been destroyed'); }
  async setReference(source, options = {}) {
    this.assertOpen();
    if (!this.ring.idle) throw new Error('Finish the current sequence before replacing the reference');
    let pixels = source;
    const [width, height] = source instanceof Uint8Array ? [this.options.width, this.options.height] : frameSize(source);
    if (!width || !height) throw new Error('Source dimensions are required');
    this.inputs.forEach((input) => input.destroy()); this.inputs = [];
    for (let i = 0; i < this.ring.depth; i++) this.inputs.push(await createInputConverter(this.device, width, height));
    if (!(source instanceof Uint8Array)) pixels = await this.inputs[0].referencePixels(source);
    if (pixels instanceof Uint8Array) {
      if (pixels.length !== width * height) throw new Error('Expected width * height grayscale bytes');
      source = { pixels, width, height };
    }
    this.reference = await this.host.setReference(source, { ...this.options, ...options });
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
  get device() { return this.host.device; }
  get imageTexture() { this.assertOpen(); return this.host.imageTexture; }
  get fieldBuffer() { this.assertOpen(); return this.host.fieldBuffer; }
  destroy() { if (this.closed) return; this.closed = true; this.host.destroy(); this.inputs.forEach((input) => input.destroy()); this.inputs = []; }
}
