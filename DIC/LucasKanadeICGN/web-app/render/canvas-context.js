/**
 * Resize a WebGPU presentation canvas and configure its new drawing buffer
 * before the next texture is acquired.
 */
export function resizeAndConfigureCanvas(canvas, context, device, format, width, height, configuredSize) {
  const sizeChanged = configuredSize.width !== width || configuredSize.height !== height;
  const canvasChanged = canvas.width !== width || canvas.height !== height;
  if (!sizeChanged && !canvasChanged) return false;

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  context.configure({ device, format, alphaMode: 'opaque' });
  configuredSize.width = width;
  configuredSize.height = height;
  return true;
}
