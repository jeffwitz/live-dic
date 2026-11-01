/** Supervisor settings are deliberately separate from the numerical engine. */
export function supervisionOptions(options) {
  const defaults = { overlapRecording: false, displayRate: 0, previewMaxWidth: 0, previewEnabled: true };
  if (options === undefined) return defaults;
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('supervision must be an object');
  for (const key of Object.keys(options)) if (!(key in defaults)) throw new Error(`Unknown supervision option: ${key}`);
  const result = { ...defaults, ...options };
  if (typeof result.overlapRecording !== 'boolean') throw new Error('overlapRecording must be boolean');
  if (typeof result.previewEnabled !== 'boolean') throw new Error('previewEnabled must be boolean');
  if (!Number.isFinite(result.displayRate) || result.displayRate < 0 || result.displayRate > 240) throw new Error('displayRate must be 0..240 (0 means every field)');
  if (!Number.isInteger(result.previewMaxWidth) || result.previewMaxWidth < 0 || result.previewMaxWidth > 16384) throw new Error('previewMaxWidth must be an integer 0..16384 (0 means native)');
  return result;
}
export function presentationSize(width, height, maximum = 0) {
  if (![width,height].every(value=>Number.isInteger(value)&&value>0) || !Number.isInteger(maximum) || maximum<0) throw new Error('Invalid presentation dimensions');
  const factor = maximum > 0 ? Math.min(1, maximum / width) : 1;
  return [Math.max(1, Math.round(width * factor)), Math.max(1, Math.round(height * factor))];
}
