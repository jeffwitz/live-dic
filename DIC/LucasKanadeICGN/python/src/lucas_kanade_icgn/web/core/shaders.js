/** Canonical sources are installed/copied beside web/, never generated here. */
export async function loadShader(device, filename) {
  const url = new URL(`../../shaders/${filename}`, import.meta.url);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Shader ${url.pathname}: HTTP ${response.status}`);
  const module = device.createShaderModule({ label: filename, code: await response.text() });
  const errors = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(errors.map((m) => `${filename}:${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
  return module;
}
