/** Python compatibility shim. BinaryTransport remains outside the numerical core.
 * The core retains format: 'r32float', PREFILTER_CLAMP_HALO and prefilterXSafe.
 */
import { createPyramidHost } from '../core/pyramid.js';
const query = new URLSearchParams(window.location.search);
const host = createPyramidHost({
  subsetSize: Number(query.get('subsetSize') ?? 16),
  subsetStride: Number(query.get('subsetStride') ?? query.get('subsetSize') ?? 16),
});
const legacy = ({ roiSelected, ...result }) => ({ ...result,
  parameters: Array.from(result.parameters), status: Array.from(result.status),
  iterations: Array.from(result.iterations),
});
window.ICGNHeadless = {
  info: host.info, configure: host.configure,
  setReference: async (source, options = {}) => {
    const next = { ...options };
    if (next.roiMaskBase64 !== undefined) {
      next.roiMask = Uint8Array.from(atob(next.roiMaskBase64), (c) => c.charCodeAt(0));
      delete next.roiMaskBase64;
    }
    const info = await host.setReference(source, next);
    return { ...info, referenceValid: Array.from(info.referenceValid) };
  },
  register: async (...args) => legacy(await host.register(...args)),
  registerBytes: async (...args) => legacy(await host.registerBytes(...args)),
  benchmark: async (...args) => {
    const result = await host.benchmark(...args);
    return { ...result, registration: legacy(result.registration) };
  },
};
window.ICGNHeadlessReady = host.ensureDevice();
