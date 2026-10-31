/** Capability checks use limits, never vendor-name performance guesses. */
export const RESOLUTIONS = [[1280, 720], [1920, 1080], [3840, 2160]];
export function fitsResolution(limits, width, height) {
  return width <= limits.maxTextureDimension2D && height <= limits.maxTextureDimension2D
    && width * height * 16 <= limits.maxStorageBufferBindingSize
    && width * height * 16 <= limits.maxBufferSize;
}
export async function probeCapabilities({ gpu = globalThis.navigator?.gpu } = {}) {
  if (!gpu) return { supported: false, reasons: ['WebGPU requires a supported browser and secure context'] };
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { supported: false, reasons: ['No WebGPU adapter is available'] };
  const keys = ['maxStorageBuffersPerShaderStage', 'maxBindGroups', 'maxStorageBufferBindingSize',
    'maxBufferSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup',
    'maxTextureDimension2D'];
  const limits = Object.fromEntries(keys.map((key) => [key, adapter.limits[key]]));
  const supported = limits.maxStorageBuffersPerShaderStage >= 8 && limits.maxBindGroups >= 4
    && limits.maxComputeInvocationsPerWorkgroup >= 256 && limits.maxComputeWorkgroupStorageSize >= 16384;
  return { supported, adapter: { ...adapter.info, vendor: adapter.info?.vendor,
    architecture: adapter.info?.architecture, device: adapter.info?.device,
    description: adapter.info?.description }, limits, features: [...adapter.features],
    resolutions: RESOLUTIONS.filter(([w, h]) => fitsResolution(limits, w, h)),
    recommendedProfile: 'portable', reasons: ['Start at 720p; measure the real graph before raising resolution'],
    profiles: { portable: true, 'qualified-latency': true, minimal: false },
  };
}
