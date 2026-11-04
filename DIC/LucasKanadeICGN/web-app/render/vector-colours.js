/** Shared colour contract for sparse GPU vectors and their HTML legend. */
import { SEQUENTIAL_COLOUR_MAPS, sequentialColour } from './colormaps.js';

export const VECTOR_COLOURS = [[21, 57, 99], [31, 132, 138], [241, 206, 95]];
export const VECTOR_COLOUR_MAPS = Object.freeze(['original', ...SEQUENTIAL_COLOUR_MAPS]);
const gradients = new Map();

export function validateVectorColourMap(name) {
  if (!VECTOR_COLOUR_MAPS.includes(name)) throw new Error(`Unknown vector colour map: ${name}`);
  return name;
}

/** Default interpolation stays identical to the historical three-anchor map. */
export function vectorColour(fraction) {
  if (!Number.isFinite(fraction)) throw new Error('Vector colour fraction must be finite');
  const position = Math.max(0, Math.min(1, fraction)) * 2;
  const index = Math.min(1, Math.floor(position)), weight = position - index;
  return VECTOR_COLOURS[index].map((value, channel) => Math.round(value + weight * (VECTOR_COLOURS[index + 1][channel] - value)));
}

/** Named maps use an explicit API; vectorColour stays safe as Array.map callback. */
export function vectorMapColour(name, fraction) {
  validateVectorColourMap(name);
  if (!Number.isFinite(fraction)) throw new Error('Vector colour fraction must be finite');
  return name === 'original' ? vectorColour(fraction) : sequentialColour(name, fraction);
}

/** Build a CSS legend once per map; never rebuild hundreds of stops per frame. */
export function vectorGradient(name = 'original') {
  validateVectorColourMap(name);
  if (!gradients.has(name)) {
    const count = name === 'original' ? 11 : 256;
    const stops = Array.from({ length: count }, (_, index) => {
      const fraction = index / (count - 1);
      return `rgb(${vectorMapColour(name, fraction).join(',')}) ${index * 100 / (count - 1)}%`;
    });
    gradients.set(name, `linear-gradient(90deg,${stops.join(',')})`);
  }
  return gradients.get(name);
}
