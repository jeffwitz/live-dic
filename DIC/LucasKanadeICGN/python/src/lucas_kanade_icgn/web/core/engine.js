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
    if (options.initialization && options.initialization !== 'zero') throw new Error('Temporal initialization is not yet supported in this extraction stage');
    if (options.readback && options.readback !== 'full') throw new Error('Only full readback is supported in this extraction stage');
    this.options = { readback: 'full', ...options };
    this.ring = new SlotRing(options.pipelineDepth ?? 1);
    this.host = createPyramidHost(options);
    this.closed = false;
  }
  assertOpen() { if (this.closed) throw new Error('The engine has been destroyed'); }
  async setReference(source, options = {}) {
    this.assertOpen();
    if (!this.ring.idle) throw new Error('Finish the current sequence before replacing the reference');
    if (options.roiMask) throw new Error('ROI is not supported in this extraction stage');
    if (source instanceof Uint8Array) {
      if (!this.options.width || !this.options.height) throw new Error('Uint8Array input requires width and height');
      source = { pixels: source, width: this.options.width, height: this.options.height };
    }
    this.reference = await this.host.setReference(source, this.options);
    return this.reference;
  }
  register(source) {
    this.assertOpen();
    if (!this.reference) throw new Error('A reference image must be configured first');
    return this.ring.run(async () => {
      const started = performance.now();
      const result = source instanceof Uint8Array
        ? await this.host.registerBytes(source, this.reference.width, this.reference.height)
        : await this.host.register(source);
      result.timing.totalWallMs = performance.now() - started;
      return result;
    });
  }
  resetTemporalState() { this.assertOpen(); this.host.resetTemporalState?.(); }
  get device() { return this.host.device; }
  get fieldBuffer() { this.assertOpen(); return this.host.fieldBuffer; }
  destroy() { if (this.closed) return; this.closed = true; this.host.destroy(); }
}
