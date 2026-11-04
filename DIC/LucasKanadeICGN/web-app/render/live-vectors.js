import { LIVE_BACKGROUND_SHADER, LIVE_GLYPH_SHADER } from './live-vector-shaders.js';
import { presentationSize } from '../ui/supervision.js';
import { resizeAndConfigureCanvas } from './canvas-context.js';

import { VECTOR_COLOURS, validateVectorColourMap, vectorMapColour } from './vector-colours.js';
export { VECTOR_COLOURS, vectorColour } from './vector-colours.js';

function roundedMaximum(value) {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value)), fraction = value / power;
  return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power;
}

/** Presentation values only: never normalize or modify the measured field. */
export function vectorAppearance(result, { scale = null, arrowStep = null, arrowGain = null, vectorColourMap = 'original', previewMaxWidth = 0 } = {}) {
  // A monitoring overview needs a few hundred glyphs, not thousands of tiny
  // arrows. Keep small fields dense and let users select a fixed step instead.
  arrowStep ??= Math.max(1, Math.ceil(result.grid[0] / 20), Math.ceil(result.grid[1] / 12));
  if (!Number.isInteger(arrowStep) || arrowStep < 1) throw new Error('Vector spacing must be a positive subset count');
  if (scale !== null && (!Number.isFinite(scale) || scale <= 0)) throw new Error('Vector colour maximum must be positive');
  if (arrowGain !== null && (!Number.isFinite(arrowGain) || arrowGain <= 0)) throw new Error('Vector magnification must be positive or automatic');
  validateVectorColourMap(vectorColourMap);
  const [viewWidth, viewHeight] = presentationSize(result.width, result.height, previewMaxWidth);
  const measuredMaximum = result.maxAbs?.[2] ?? 1;
  const maximumPx = scale ?? roundedMaximum(measuredMaximum);
  // Fit ordinary vectors within part of a sampled cell. The one-pixel floor
  // prevents floating-point noise in an identity field becoming long arrows.
  const gain = arrowGain ?? .4 * result.subsetStride * arrowStep / Math.max(1, measuredMaximum);
  return { maximumPx, gain, arrowStep, viewWidth, viewHeight,
    imageWidth: result.width, imageHeight: result.height,
    instances: Math.ceil(result.grid[0] / arrowStep) * Math.ceil(result.grid[1] / arrowStep),
    tipAnchor: 'Current subset centre: reference centre plus measured displacement',
    unit: 'image px', colours: vectorColourMap === 'original' ? VECTOR_COLOURS : null, vectorColourMap, accepted: result.accepted ?? null };
}

/** Lightweight live view: image copy plus sparse instanced vector geometry.
 * The background fragment never reads a field value. No scalar raster,
 * displacement interpolation, strain calculation or CPU field readback runs here.
 */
export class LiveVectorRenderer {
  constructor(device, canvas) {
    this.closed = false;
    this.device = device; this.canvas = canvas;
    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('This canvas cannot present WebGPU');
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.format = format;
    this.context.configure({ device, format, alphaMode: 'opaque' });
    this.configuredCanvasSize = { width: canvas.width, height: canvas.height };
    this.config = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paletteTexture = device.createTexture({ label: 'Live vector colour lookup', size: [256, 1],
      format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.paletteView = this.paletteTexture.createView(); this.paletteName = 'original';
    this.backgroundGroups = new WeakMap(); this.vectorGroups = new WeakMap(); this.legend = null;
    const image = device.createShaderModule({ label: 'Live image without field sampling', code: LIVE_BACKGROUND_SHADER });
    this.background = device.createRenderPipeline({ label: 'Live current image', layout: 'auto',
      vertex: { module: image, entryPoint: 'vertex' }, fragment: { module: image, entryPoint: 'background', targets: [{ format }] },
      primitive: { topology: 'triangle-list' } });
    const vectors = device.createShaderModule({ label: 'Sparse norm-coloured live vectors', code: LIVE_GLYPH_SHADER });
    const vectorBindings = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', minBindingSize: 64 } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float' } },
    ] });
    const vectorLayout = device.createPipelineLayout({ bindGroupLayouts: [vectorBindings] });
    this.vectors = device.createRenderPipeline({ label: 'Live sparse displacement glyphs', layout: vectorLayout,
      vertex: { module: vectors, entryPoint: 'glyph' }, fragment: { module: vectors, entryPoint: 'ink',
        targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] }, primitive: { topology: 'triangle-list' } });
    this.outlines = [2.2, 1.1].map(width => device.createRenderPipeline({
      label: `Live vector contrast outline ${width}`, layout: vectorLayout,
      vertex: { module: vectors, entryPoint: 'glyph', constants: { OUTLINE_WIDTH: width } },
      fragment: { module: vectors, entryPoint: 'ink', targets: [{ format,
        blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
      primitive: { topology: 'triangle-list' },
    }));
  }
  draw(engine, result, {
    mode = 2, scale = null, opacity = 1, imageTexture = null,
    arrowStep = null, arrowGain = null, vectorOutline = false,
    vectorColourMap = 'original', previewMaxWidth = 0,
  } = {}) {
    if (this.closed) throw new Error('Live vector renderer is closed');
    if (![2,8].includes(mode)) throw new Error('Live view supports norm-coloured vectors or the current image; use Results for scalar fields');
    if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1) throw new Error('Vector opacity must be 0..1');
    if (typeof vectorOutline !== 'boolean') throw new Error('Vector outline must be boolean');
    const appearance = vectorAppearance(result, { scale, arrowStep, arrowGain, vectorColourMap, previewMaxWidth });
    const { viewWidth, viewHeight, maximumPx, gain } = appearance;
    this.updatePalette(vectorColourMap);
    resizeAndConfigureCanvas(
      this.canvas,
      this.context,
      this.device,
      this.format,
      viewWidth,
      viewHeight,
      this.configuredCanvasSize,
    );
    const texture = imageTexture ?? engine.imageTexture;
    const bytes = new ArrayBuffer(64), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    u.set([result.width,result.height,...result.grid,result.config.motionModel==='quadratic'?16:8,
      result.config.motionModel==='quadratic'?12:6,appearance.arrowStep,+(texture.format==='r8unorm')]);
    f.set([result.subsetSize,result.subsetStride,maximumPx,gain,viewWidth,viewHeight,opacity,+(vectorColourMap !== 'original')],8);
    this.device.queue.writeBuffer(this.config,0,bytes);
    let background = this.backgroundGroups.get(texture);
    if (!background) {
      background = this.device.createBindGroup({
        layout: this.background.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: texture.createView() },
          { binding: 2, resource: { buffer: this.config } },
        ],
      });
      this.backgroundGroups.set(texture, background);
    }
    const encoder = this.device.createCommandEncoder({ label: 'Live image and sparse vectors' });
    const pass = encoder.beginRenderPass({ colorAttachments: [{
      view: this.context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store',
    }] });
    pass.setPipeline(this.background);
    pass.setBindGroup(0, background);
    pass.draw(3);
    const visible = mode === 2 && !imageTexture;
    if (visible) {
      let group = this.vectorGroups.get(engine.fieldBuffer);
      if (!group) {
        group = this.device.createBindGroup({
          layout: this.vectors.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: engine.fieldBuffer } },
            { binding: 2, resource: { buffer: this.config } },
            { binding: 3, resource: this.paletteView },
          ],
        });
        this.vectorGroups.set(engine.fieldBuffer, group);
      }
      if (vectorOutline) {
        for (const outline of this.outlines) {
          pass.setPipeline(outline);
          pass.setBindGroup(0, group);
          pass.draw(12, appearance.instances);
        }
      }
      pass.setPipeline(this.vectors);
      pass.setBindGroup(0, group);
      pass.draw(12, appearance.instances);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.legend = { ...appearance, imageOnly: !visible, vectorOutline };
    return maximumPx;
  }
  updatePalette(name) {
    if (name === this.paletteName) return;
    if (name !== 'original') {
      const pixels = new Uint8Array(256 * 4);
      for (let index = 0; index < 256; index++) {
        pixels.set([...vectorMapColour(name, index / 255), 255], index * 4);
      }
      this.device.queue.writeTexture({ texture: this.paletteTexture }, pixels,
        { bytesPerRow: 1024, rowsPerImage: 1 }, [256, 1]);
    }
    this.paletteName = name;
  }
  destroy() {
    if (this.closed) return;
    this.closed = true;
    this.config.destroy();
    this.paletteTexture.destroy();
    this.paletteTexture = this.paletteView = null;
    this.context.unconfigure();
    this.backgroundGroups = new WeakMap();
    this.vectorGroups = new WeakMap();
    this.outlines = [];
    this.background = this.vectors = this.config = this.legend = null;
  }
}
