/** Build a shareable browser/GPU capability report without acquisition data. */
export function createDeviceReport({
  capabilities,
  build,
  navigatorObject = globalThis.navigator,
  secureContext = globalThis.isSecureContext,
  generatedAt = new Date(),
} = {}) {
  const userAgentData = navigatorObject?.userAgentData;
  const timestamp = generatedAt instanceof Date ? generatedAt : new Date(generatedAt);

  return {
    format: 'live-dic-device-report',
    schemaVersion: 1,
    generatedAt: timestamp.toISOString(),
    app: { name: 'live-dic', build: build ?? null },
    browser: {
      userAgent: navigatorObject?.userAgent ?? null,
      platform: navigatorObject?.platform ?? null,
      userAgentData: userAgentData
        ? {
            brands: userAgentData.brands ?? [],
            mobile: Boolean(userAgentData.mobile),
            platform: userAgentData.platform ?? null,
          }
        : null,
      language: navigatorObject?.language ?? null,
      online: navigatorObject?.onLine ?? null,
      secureContext: Boolean(secureContext),
      hardwareConcurrency: navigatorObject?.hardwareConcurrency ?? null,
      deviceMemoryGiB: navigatorObject?.deviceMemory ?? null,
    },
    webgpu: capabilities ?? null,
  };
}
