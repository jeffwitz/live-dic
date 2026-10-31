/** Capability checks use limits, never vendor-name performance guesses. */
export const RESOLUTIONS = [[1280, 720], [1920, 1080], [3840, 2160]];
export function fitsResolution(limits, width, height) {
  return width <= limits.maxTextureDimension2D && height <= limits.maxTextureDimension2D
    && width * height * 16 <= limits.maxStorageBufferBindingSize
    && width * height * 16 <= limits.maxBufferSize;
}
export async function probeCapabilities({ gpu = globalThis.navigator?.gpu, benchmark = true } = {}) {
  if (!gpu) return { supported: false, reasons: ['WebGPU requires a supported browser and secure context'] };
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { supported: false, reasons: ['No WebGPU adapter is available'] };
  const keys = ['maxStorageBuffersPerShaderStage', 'maxBindGroups', 'maxStorageBufferBindingSize',
    'maxBufferSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup',
    'maxTextureDimension2D'];
  const limits = Object.fromEntries(keys.map((key) => [key, adapter.limits[key]]));
  const supported = limits.maxStorageBuffersPerShaderStage >= 8 && limits.maxBindGroups >= 4
    && limits.maxComputeInvocationsPerWorkgroup >= 256 && limits.maxComputeWorkgroupStorageSize >= 16384;
  const capabilities = { supported, adapter: { ...adapter.info, vendor: adapter.info?.vendor,
    architecture: adapter.info?.architecture, device: adapter.info?.device,
    description: adapter.info?.description }, limits, features: [...adapter.features],
    resolutions: RESOLUTIONS.filter(([w, h]) => fitsResolution(limits, w, h)),
    recommendedProfile: 'portable', reasons: ['Start at 720p; measure the real graph before raising resolution'],
    profiles: { portable: true, 'qualified-latency': true, minimal: false },
  };
  return supported && benchmark ? measureProfiles(capabilities) : capabilities;
}

export const LATENCY_OPTIONS = Object.freeze({ pyramidLevels: 2, interpolation: 'bspline',
  motionModel: 'affine', outlierPolicy: 'median', initialization: 'temporal',
  temporalPrediction: 'velocity', temporalFineCorrection: true, divergencePatience: 3,
  divergenceMinIterations: 4, pipelineDepth: 1, readback: 'none' });

/** Advisory zero-motion graph benchmark, not a camera/trajectory qualification. */
export async function measureProfiles(capabilities, { resolutions = capabilities.resolutions } = {}) {
  const { DICEngine } = await import('./engine.js');
  const samples = [];
  for (const [width, height] of resolutions) {
    let engine;
    try {
      const pixels = Uint8Array.from({ length: width * height }, (_, i) =>
        Math.round(128 + 38 * Math.sin(i % width * .41) * Math.cos(Math.floor(i / width) * .29)
          + 21 * Math.sin(i % width * .19 + Math.floor(i / width) * .23)));
      engine = await DICEngine.create({ ...LATENCY_OPTIONS, width, height });
      const preparation = await engine.setReference(pixels), times = [];
      let result;
      for (let i = 0; i < 7; i++) {
        const start = performance.now(); result = await engine.register(pixels);
        if (i >= 2) times.push(performance.now() - start);
      }
      const sorted = [...times].sort((a, b) => a - b);
      samples.push({ width, height, times, medianMs: sorted[2], p95Ms: sorted.at(-1),
        acceptance: result.acceptance, preparationWallMs: preparation.preparationWallMs });
    } catch (error) { samples.push({ width, height, error: String(error) }); }
    finally { engine?.destroy(); }
  }
  const feasible = samples.filter((s) => !s.error && s.p95Ms < 1000 / 60 && s.acceptance > .9);
  const best = feasible.at(-1);
  return { ...capabilities, measurements: samples,
    recommendedProfile: best && best.width >= 1920 ? 'qualified-latency' : 'portable',
    recommendedResolution: best ? [best.width, best.height] : capabilities.resolutions[0],
    reasons: [best ? `Zero-motion graph p95 ${best.p95Ms.toFixed(1)} ms at ${best.width}×${best.height}`
      : 'No measured candidate sustained the 60 Hz advisory bound; use portable at the lowest resolution',
    'Short microbenchmark only; camera motion, decoding, display and thermal throttling require a real run'],
  };
}

/** Request only the explicit limit needed by the common graph. */
export async function acquireDevice({ device: injected } = {}) {
  if (injected) return { device: injected, adapter: null, ownsDevice: false };
  if (!globalThis.navigator?.gpu) throw new Error('WebGPU is unavailable in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter is available');
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 8 } });
  return { device, adapter, ownsDevice: true };
}
