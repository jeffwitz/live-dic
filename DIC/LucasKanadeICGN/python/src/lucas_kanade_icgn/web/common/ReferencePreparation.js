/*
 * Shared CPU preparation for the browser WebGPU adapters.
 *
 * This module owns the reference-image convention used by both the Three.js
 * adapter and the raw WGSL adapter. GPU-IO remains an independently validated
 * compatibility implementation.
 */

const mirror = (coordinate, size) => {
  if (coordinate < 0) return -coordinate;
  if (coordinate >= size) return 2 * size - 2 - coordinate;
  return coordinate;
};

export const imageToGray = (image, width, height, intensityScale = 1 / 255) => {
  if (image.width !== width || image.height !== height) {
    throw new Error(`Expected ${width}x${height}, got ${image.width}x${image.height}`);
  }
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const rgba = context.getImageData(0, 0, width, height).data;
  const gray = new Float64Array(width * height);
  for (let index = 0; index < gray.length; ++index) {
    gray[index] = rgba[4 * index] * intensityScale;
  }
  return gray;
};

export const reduceHalfGray = (source, width, height) => {
  const reducedWidth = Math.floor(width / 2);
  const reducedHeight = Math.floor(height / 2);
  const horizontal = new Float64Array(reducedWidth * height);
  const reduced = new Uint8Array(reducedWidth * reducedHeight);
  const weights = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < reducedWidth; ++x) {
      let value = 0;
      for (let tap = -2; tap <= 2; ++tap) {
        value += weights[tap + 2] * source[y * width + mirror(2 * x + tap, width)];
      }
      horizontal[y * reducedWidth + x] = value;
    }
  }
  for (let y = 0; y < reducedHeight; ++y) {
    for (let x = 0; x < reducedWidth; ++x) {
      let value = 0;
      for (let tap = -2; tap <= 2; ++tap) {
        value += weights[tap + 2]
          * horizontal[mirror(2 * y + tap, height) * reducedWidth + x];
      }
      reduced[y * reducedWidth + x] = Math.trunc(Math.max(0, Math.min(255, value)));
    }
  }
  return { values: reduced, width: reducedWidth, height: reducedHeight };
};

const gaussianSigmaOne = (source, width, height) => {
  const radius = 3;
  const weights = new Float64Array(2 * radius + 1);
  let total = 0;
  for (let offset = -radius; offset <= radius; ++offset) {
    const weight = Math.exp(-0.5 * offset * offset);
    weights[offset + radius] = weight;
    total += weight;
  }
  for (let index = 0; index < weights.length; ++index) weights[index] /= total;

  const horizontal = new Float64Array(width * height);
  const filtered = new Float64Array(width * height);
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      let value = 0;
      for (let offset = -radius; offset <= radius; ++offset) {
        value += weights[offset + radius] * source[y * width + mirror(x + offset, width)];
      }
      horizontal[y * width + x] = value;
    }
  }
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      let value = 0;
      for (let offset = -radius; offset <= radius; ++offset) {
        value += weights[offset + radius] * horizontal[mirror(y + offset, height) * width + x];
      }
      filtered[y * width + x] = value;
    }
  }
  return filtered;
};

const cubicBSplinePrefilter = (source, width, height) => {
  const coefficients = new Float64Array(source);
  const pole = Math.sqrt(3) - 2;
  const gain = (1 - pole) * (1 - 1 / pole);
  const filterLine = (offset, count, stride) => {
    for (let index = 0; index < count; ++index) coefficients[offset + index * stride] *= gain;
    const horizon = Math.min(count, Math.ceil(Math.log(1e-12) / Math.log(Math.abs(pole))));
    let sum = coefficients[offset];
    let power = pole;
    for (let index = 1; index < horizon; ++index) {
      sum += power * coefficients[offset + index * stride];
      power *= pole;
    }
    coefficients[offset] = sum;
    for (let index = 1; index < count; ++index) {
      coefficients[offset + index * stride] += pole * coefficients[offset + (index - 1) * stride];
    }
    const last = offset + (count - 1) * stride;
    coefficients[last] = pole / (pole * pole - 1)
      * (pole * coefficients[last - stride] + coefficients[last]);
    for (let index = count - 2; index >= 0; --index) {
      const position = offset + index * stride;
      coefficients[position] = pole * (coefficients[position + stride] - coefficients[position]);
    }
  };
  for (let y = 0; y < height; ++y) filterLine(y * width, width, 1);
  for (let x = 0; x < width; ++x) filterLine(x, height, width);
  return coefficients;
};

const invertSquare = (matrix, size) => {
  const stride = 2 * size;
  const augmented = new Float64Array(size * stride);
  for (let row = 0; row < size; ++row) {
    for (let column = 0; column < size; ++column) {
      augmented[row * stride + column] = matrix[row * size + column];
      augmented[row * stride + size + column] = row === column ? 1 : 0;
    }
  }
  for (let column = 0; column < size; ++column) {
    let pivotRow = column;
    for (let row = column + 1; row < size; ++row) {
      if (Math.abs(augmented[row * stride + column])
          > Math.abs(augmented[pivotRow * stride + column])) {
        pivotRow = row;
      }
    }
    if (Math.abs(augmented[pivotRow * stride + column]) < 1e-14) return null;
    if (pivotRow !== column) {
      for (let entry = 0; entry < stride; ++entry) {
        const saved = augmented[column * stride + entry];
        augmented[column * stride + entry] = augmented[pivotRow * stride + entry];
        augmented[pivotRow * stride + entry] = saved;
      }
    }
    const pivot = augmented[column * stride + column];
    for (let entry = 0; entry < stride; ++entry) augmented[column * stride + entry] /= pivot;
    for (let row = 0; row < size; ++row) {
      if (row === column) continue;
      const factor = augmented[row * stride + column];
      for (let entry = 0; entry < stride; ++entry) {
        augmented[row * stride + entry] -= factor * augmented[column * stride + entry];
      }
    }
  }
  const inverse = new Float64Array(size * size);
  for (let row = 0; row < size; ++row) {
    for (let column = 0; column < size; ++column) {
      inverse[row * size + column] = augmented[row * stride + size + column];
    }
  }
  return inverse;
};

export const prepareReferenceGray = (
  gray,
  {
    width, height, subsetSize = 16, interpolation = 'bspline', motionModel = 'affine',
  },
) => {
  const pixelCount = width * height;
  const gridWidth = Math.floor(width / subsetSize);
  const gridHeight = Math.floor(height / subsetSize);
  const gridCount = gridWidth * gridHeight;
  const halfExtent = (subsetSize - 1) / 2;
  const subsetPixels = subsetSize * subsetSize;
  if (gray.length !== pixelCount) {
    throw new Error(`Expected ${pixelCount} grayscale values, got ${gray.length}`);
  }
  if (!['bspline', 'linear', 'cubic'].includes(interpolation)) {
    throw new Error("interpolation must be 'bspline', 'linear', or 'cubic'");
  }
  if (!['affine', 'translation', 'quadratic'].includes(motionModel)) {
    throw new Error("motionModel must be 'affine', 'translation', or 'quadratic'");
  }
  const intensity = interpolation === 'bspline'
    ? gaussianSigmaOne(gray, width, height)
    : Float64Array.from(gray);
  const gradientSource = interpolation === 'bspline'
    ? cubicBSplinePrefilter(intensity, width, height)
    : intensity;
  const reference = new Float32Array(pixelCount * 4);
  const componentCount = motionModel === 'quadratic' ? 12 : 6;
  const metadataStride = motionModel === 'quadratic' ? 28 : 16;
  const inverseHessian = new Float32Array(gridCount * componentCount * componentCount);
  const metadata = new Float32Array(gridCount * metadataStride);
  const atMirror = (values, x, y) => values[mirror(y, height) * width + mirror(x, width)];

  for (let gridY = 0; gridY < gridHeight; ++gridY) {
    for (let gridX = 0; gridX < gridWidth; ++gridX) {
      const subsetIndex = gridY * gridWidth + gridX;
      const x0 = gridX * subsetSize;
      const y0 = gridY * subsetSize;
      let mean = 0;
      for (let yy = 0; yy < subsetSize; ++yy) {
        for (let xx = 0; xx < subsetSize; ++xx) mean += intensity[(y0 + yy) * width + x0 + xx];
      }
      mean /= subsetPixels;

      const hessian = new Float64Array(componentCount * componentCount);
      const jtReference = new Float64Array(componentCount);
      const sumJ = new Float64Array(componentCount);
      let referenceEnergy = 0;
      for (let yy = 0; yy < subsetSize; ++yy) {
        for (let xx = 0; xx < subsetSize; ++xx) {
          const x = x0 + xx;
          const y = y0 + yy;
          const localX = xx - halfExtent;
          const localY = yy - halfExtent;
          const f = intensity[y * width + x] - mean;
          let gx = 0.5
            * (atMirror(gradientSource, x + 1, y) - atMirror(gradientSource, x - 1, y));
          let gy = 0.5
            * (atMirror(gradientSource, x, y + 1) - atMirror(gradientSource, x, y - 1));
          if (interpolation === 'bspline') {
            const gxSouth = 0.5 * (atMirror(gradientSource, x + 1, y - 1)
              - atMirror(gradientSource, x - 1, y - 1));
            const gxNorth = 0.5 * (atMirror(gradientSource, x + 1, y + 1)
              - atMirror(gradientSource, x - 1, y + 1));
            const gyWest = 0.5 * (atMirror(gradientSource, x - 1, y + 1)
              - atMirror(gradientSource, x - 1, y - 1));
            const gyEast = 0.5 * (atMirror(gradientSource, x + 1, y + 1)
              - atMirror(gradientSource, x + 1, y - 1));
            gx = (gxSouth + 4 * gx + gxNorth) / 6;
            gy = (gyWest + 4 * gy + gyEast) / 6;
          }

          const pixel = 4 * (y * width + x);
          reference[pixel] = f;
          reference[pixel + 1] = gx;
          reference[pixel + 2] = gy;
          referenceEnergy += f * f;
          if (motionModel === 'translation') {
            jtReference[0] += gx * f;
            jtReference[3] += gy * f;
            sumJ[0] += gx;
            sumJ[3] += gy;
            hessian[0] += gx * gx;
            hessian[3] += gx * gy;
            hessian[18] += gx * gy;
            hessian[21] += gy * gy;
          } else {
            const jacobian = [gx, gx * localX, gx * localY, gy, gy * localX, gy * localY];
            if (motionModel === 'quadratic') {
              const halfX2 = 0.5 * localX * localX;
              const xy = localX * localY;
              const halfY2 = 0.5 * localY * localY;
              jacobian.push(
                gx * halfX2, gx * xy, gx * halfY2,
                gy * halfX2, gy * xy, gy * halfY2,
              );
            }
            for (let row = 0; row < componentCount; ++row) {
              jtReference[row] += jacobian[row] * f;
              sumJ[row] += jacobian[row];
              for (let column = 0; column < componentCount; ++column) {
                hessian[componentCount * row + column] += jacobian[row] * jacobian[column];
              }
            }
          }
        }
      }

      let valid = false;
      if (motionModel === 'translation') {
        const determinant = hessian[0] * hessian[21] - hessian[3] * hessian[18];
        const scale = Math.max(Math.abs(hessian[0] * hessian[21]), 1);
        valid = Number.isFinite(determinant)
          && Math.abs(determinant) > Number.EPSILON * scale
          && referenceEnergy > 1e-12;
        if (valid) {
          const inverseOffset = subsetIndex * 36;
          inverseHessian[inverseOffset] = hessian[21] / determinant;
          inverseHessian[inverseOffset + 3] = -hessian[3] / determinant;
          inverseHessian[inverseOffset + 18] = -hessian[18] / determinant;
          inverseHessian[inverseOffset + 21] = hessian[0] / determinant;
        }
      } else {
        const inverse = invertSquare(hessian, componentCount);
        valid = Boolean(inverse) && referenceEnergy > 1e-12;
        if (inverse) inverseHessian.set(inverse, subsetIndex * componentCount * componentCount);
      }
      const metaOffset = subsetIndex * metadataStride;
      metadata[metaOffset] = Math.sqrt(referenceEnergy);
      metadata[metaOffset + 1] = valid ? 1 : 0;
      metadata.set(jtReference, metaOffset + 2);
      metadata.set(sumJ, metaOffset + 2 + componentCount);
    }
  }
  return { reference, inverseHessian, metadata };
};

export const prepareReferenceData = (
  image,
  {
    width,
    height,
    subsetSize = 16,
    intensityScale = 1 / 255,
    interpolation = 'bspline',
    motionModel = 'affine',
  },
) => prepareReferenceGray(
  imageToGray(image, width, height, intensityScale),
  {
    width, height, subsetSize, interpolation, motionModel,
  },
);

export const makeFIRWeights = (radius = 15) => {
  const gaussian = new Float64Array(7);
  let gaussianSum = 0;
  for (let offset = -3; offset <= 3; ++offset) {
    gaussian[offset + 3] = Math.exp(-0.5 * offset * offset);
    gaussianSum += gaussian[offset + 3];
  }
  for (let index = 0; index < gaussian.length; ++index) gaussian[index] /= gaussianSum;
  const pole = Math.sqrt(3) - 2;
  const weights = new Float32Array(2 * radius + 1);
  for (let offset = -radius; offset <= radius; ++offset) {
    let value = 0;
    for (let gaussianOffset = -3; gaussianOffset <= 3; ++gaussianOffset) {
      value += gaussian[gaussianOffset + 3] * Math.sqrt(3)
        * Math.pow(pole, Math.abs(offset - gaussianOffset));
    }
    weights[offset + radius] = value;
  }
  return weights;
};
