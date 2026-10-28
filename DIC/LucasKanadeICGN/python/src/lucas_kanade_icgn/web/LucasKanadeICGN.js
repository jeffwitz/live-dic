'use strict';

/*
 * Affine inverse-compositional Lucas-Kanade registration for GPU-IO.
 *
 * Reference-only quantities are prepared once on the CPU, while target
 * prefiltering, IC-GN iterations, and median replacement run on WebGL2. The
 * same file supports the optional browser demo and the Python headless API;
 * headless mode never creates UI controls or attaches the canvas to the DOM.
 */

const {
  GPUComposer,
  GPULayer,
  GPUProgram,
  copyProgram,
  FLOAT,
  INT,
  CLAMP_TO_EDGE,
  LINEAR,
  NEAREST,
} = GPUIO;

const SUBSET_SIZE = 16;
const SUBSET_HALF_EXTENT = 7.5;
const FIR_RADIUS = 15;
const URL_PARAMETERS = new URLSearchParams(window.location.search);
const frameDimension = (name, fallback) => {
  const value = Number(URL_PARAMETERS.get(name) ?? fallback);
  if (!Number.isInteger(value) || value < SUBSET_SIZE || value % SUBSET_SIZE !== 0) {
    throw new Error(`${name} must be a positive multiple of ${SUBSET_SIZE}; received ${value}.`);
  }
  return value;
};
const FRAME_WIDTH = frameDimension('width', 512);
const FRAME_HEIGHT = frameDimension('height', 512);
const HEADLESS_MODE = URL_PARAMETERS.get('headless') === '1';

const DATASET = {
  name: 'seed2026101102_translation/raw',
  sourceFrame: 0,
  frames: [1, 5, 10, 20],
  velocity: [0.2, -0.1],
  url: (frame) => `./assets/frame_${String(frame).padStart(3, '0')}.png`,
};

const PARAMS = {
  tracking: {
    active: true,
    spacing: SUBSET_SIZE,
    iterations: 30,
    tolerance: 0.001,
    znccMin: 0.8,
    vectorScale: 2,
  },
  dataset: {
    frame: 10,
  },
  benchmark: {
    repeats: 30,
  },
};

const statusElement = document.querySelector('#status') || { textContent: '' };
const resultsElement = document.querySelector('#results') || { textContent: '' };
const setStatus = (message) => { statusElement.textContent = message; };
const setResults = (message) => { resultsElement.textContent = message; };

const canvas = document.createElement('canvas');
if (!HEADLESS_MODE) document.body.appendChild(canvas);
const gl = canvas.getContext('webgl2');
if (!gl) throw new Error('WebGL2 is required to run this implementation.');

const composer = new GPUComposer({ canvas });
const frameDimensions = [FRAME_WIDTH, FRAME_HEIGHT];

// SYCL prototype configuration E: a sigma=1 Gaussian convolved with the
// inverse cubic B-spline response, truncated to 12 coefficients on each side.
const makeFIRWeights = () => {
  const gaussianRadius = 3;
  const z = Math.sqrt(3) - 2;
  const gaussian = [];
  let gaussianSum = 0;
  for (let i = -gaussianRadius; i <= gaussianRadius; ++i) {
    const value = Math.exp(-0.5 * i * i);
    gaussian.push(value);
    gaussianSum += value;
  }
  for (let i = 0; i < gaussian.length; ++i) gaussian[i] /= gaussianSum;
  const weights = new Float64Array(2 * FIR_RADIUS + 1);
  for (let i = -gaussianRadius; i <= gaussianRadius; ++i) {
    for (let b = -12; b <= 12; ++b) {
      weights[i + b + FIR_RADIUS] +=
        gaussian[i + gaussianRadius] * Math.sqrt(3) * z ** Math.abs(b);
    }
  }
  return weights;
};

const FIR_WEIGHTS = makeFIRWeights();
const firHorizontalExpression = Array.from(FIR_WEIGHTS, (weight, index) => {
  const offset = index - FIR_RADIUS;
  return `${weight.toPrecision(17)} * texelFetch(u_input, ivec2(mirror(p.x + ${offset}, width), p.y), 0).x`;
}).join(' +\n        ');
const firVerticalExpression = Array.from(FIR_WEIGHTS, (weight, index) => {
  const offset = index - FIR_RADIUS;
  return `${weight.toPrecision(17)} * texelFetch(u_input, ivec2(p.x, mirror(p.y + ${offset}, height)), 0).x`;
}).join(' +\n        ');

const frameLayerParams = {
  dimensions: frameDimensions,
  type: FLOAT,
  filter: LINEAR,
  wrapX: CLAMP_TO_EDGE,
  wrapY: CLAMP_TO_EDGE,
};

const sourceImage = new GPULayer(composer, {
  ...frameLayerParams,
  name: 'sourceImage',
  numComponents: 3,
  numBuffers: 1,
  filter: NEAREST,
});

const currentFrame = new GPULayer(composer, {
  ...frameLayerParams,
  name: 'currentFrame',
  numComponents: 1,
  numBuffers: 1,
});

const referenceFrame = new GPULayer(composer, {
  ...frameLayerParams,
  name: 'referenceFrame',
  numComponents: 1,
  numBuffers: 1,
});

const prefilterTemporary = new GPULayer(composer, {
  ...frameLayerParams,
  name: 'prefilterTemporary',
  numComponents: 1,
  numBuffers: 1,
  filter: NEAREST,
});

const currentCoefficients = new GPULayer(composer, {
  ...frameLayerParams,
  name: 'currentCoefficients',
  numComponents: 1,
  numBuffers: 1,
  filter: NEAREST,
});

// F, dI/dx, and dI/dy are prepared like SYCL Registrar::prepare/build.
const referenceTemplate = new GPULayer(composer, {
  ...frameLayerParams,
  name: 'referenceTemplate',
  numComponents: 3,
  numBuffers: 1,
  filter: NEAREST,
});

const rgbToMonoProgram = new GPUProgram(composer, {
  name: 'rgbToMono',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_rgb;
    out float out_mono;

    void main() {
      vec3 rgb = texture(u_rgb, v_uv).rgb;
      out_mono = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    }
  `,
  uniforms: [
    { name: 'u_rgb', value: 0, type: INT },
  ],
});

const copyFrameProgram = copyProgram(composer, { type: FLOAT });

const prefilterHorizontalProgram = new GPUProgram(composer, {
  name: 'syclFIRHorizontal',
  fragmentShader: `
    uniform sampler2D u_input;
    uniform vec2 u_frame_size;
    out float out_value;

    int mirror(int coordinate, int size) {
      return coordinate < 0 ? -coordinate :
        (coordinate >= size ? 2 * size - 2 - coordinate : coordinate);
    }

    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int width = int(u_frame_size.x);
      out_value = ${firHorizontalExpression};
    }
  `,
  uniforms: [
    { name: 'u_input', value: 0, type: INT },
    { name: 'u_frame_size', value: frameDimensions, type: FLOAT },
  ],
});

const prefilterVerticalProgram = new GPUProgram(composer, {
  name: 'syclFIRVertical',
  fragmentShader: `
    uniform sampler2D u_input;
    uniform vec2 u_frame_size;
    out float out_value;

    int mirror(int coordinate, int size) {
      return coordinate < 0 ? -coordinate :
        (coordinate >= size ? 2 * size - 2 - coordinate : coordinate);
    }

    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int height = int(u_frame_size.y);
      out_value = ${firVerticalExpression};
    }
  `,
  uniforms: [
    { name: 'u_input', value: 0, type: INT },
    { name: 'u_frame_size', value: frameDimensions, type: FLOAT },
  ],
});

const renderFramesProgram = new GPUProgram(composer, {
  name: 'renderFrames',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_current;
    uniform sampler2D u_reference;
    out vec4 out_color;

    void main() {
      float current = texture(u_current, v_uv).x;
      float reference = texture(u_reference, v_uv).x;
      out_color = vec4(current, 0.5 * (current + reference), reference, 1.0);
    }
  `,
  uniforms: [
    { name: 'u_current', value: 0, type: INT },
    { name: 'u_reference', value: 1, type: INT },
  ],
});

const gridDimensions = () => [
  FRAME_WIDTH / SUBSET_SIZE,
  FRAME_HEIGHT / SUBSET_SIZE,
];

const trackingLayerParams = {
  dimensions: gridDimensions(),
  type: FLOAT,
  filter: NEAREST,
  wrapX: CLAMP_TO_EDGE,
  wrapY: CLAMP_TO_EDGE,
  numBuffers: 2,
};

// p = [tx, du/dx, du/dy, ty, dv/dx, dv/dy].  WebGL textures carry at
// most four components, so translation and the linear part use two layers.
const translation = new GPULayer(composer, {
  ...trackingLayerParams,
  name: 'translation',
  numComponents: 2,
  clearValue: [0, 0],
});

const linearPart = new GPULayer(composer, {
  ...trackingLayerParams,
  name: 'linearPart',
  numComponents: 4,
  clearValue: [0, 0, 0, 0],
});

// x: 1 accepted, 2 median-replaced, 0 active, negative rejected; y: iterations.
const trackingStatus = new GPULayer(composer, {
  ...trackingLayerParams,
  name: 'trackingStatus',
  numComponents: 2,
  clearValue: [0, 0],
});

// H^-1 stores 36 scalars per grid point, packed into nine RGBA texels.
const hessianInverseAtlas = new GPULayer(composer, {
  dimensions: gridDimensions().map((value) => 3 * value),
  type: FLOAT,
  filter: NEAREST,
  wrapX: CLAMP_TO_EDGE,
  wrapY: CLAMP_TO_EDGE,
  numComponents: 4,
  numBuffers: 1,
  name: 'hessianInverseAtlas',
});

// fnorm, validity, J^T F, and sum(J): 14 scalars packed into four RGBA texels.
const solverMetaAtlas = new GPULayer(composer, {
  dimensions: gridDimensions().map((value) => 2 * value),
  type: FLOAT,
  filter: NEAREST,
  wrapX: CLAMP_TO_EDGE,
  wrapY: CLAMP_TO_EDGE,
  numComponents: 4,
  numBuffers: 1,
  name: 'solverMetaAtlas',
});

const affineICGNProgram = new GPUProgram(composer, {
  name: 'affineICGNSYCLEquivalent',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_current_coefficients;
    uniform sampler2D u_reference_template;
    uniform sampler2D u_hessian_inverse;
    uniform sampler2D u_solver_meta;
    uniform sampler2D u_translation;
    uniform sampler2D u_linear;
    uniform sampler2D u_status;
    uniform vec2 u_frame_size;
    uniform vec2 u_grid_size;
    uniform float u_tolerance;
    uniform float u_zncc_min;
    uniform int u_max_iterations;

    layout(location = 0) out vec2 out_translation;
    layout(location = 1) out vec4 out_linear;
    layout(location = 2) out vec2 out_status;

    float hessianValue(in vec4 blocks[9], int index) {
      int block = index / 4;
      return blocks[block][index - 4 * block];
    }

    float metaValue(in vec4 blocks[4], int index) {
      int block = index / 4;
      return blocks[block][index - 4 * block];
    }

    float sampleCubic(vec2 position) {
      vec2 floored = floor(position);
      ivec2 base = ivec2(floored);
      vec2 t = position - floored;
      float tx2 = t.x * t.x;
      float tx3 = tx2 * t.x;
      float ux = 1.0 - t.x;
      float ty2 = t.y * t.y;
      float ty3 = ty2 * t.y;
      float uy = 1.0 - t.y;
      float wx[4];
      float wy[4];
      wx[0] = ux * ux * ux / 6.0;
      wx[1] = (3.0 * tx3 - 6.0 * tx2 + 4.0) / 6.0;
      wx[2] = (-3.0 * tx3 + 3.0 * tx2 + 3.0 * t.x + 1.0) / 6.0;
      wx[3] = tx3 / 6.0;
      wy[0] = uy * uy * uy / 6.0;
      wy[1] = (3.0 * ty3 - 6.0 * ty2 + 4.0) / 6.0;
      wy[2] = (-3.0 * ty3 + 3.0 * ty2 + 3.0 * t.y + 1.0) / 6.0;
      wy[3] = ty3 / 6.0;
      float value = 0.0;
      for (int yy = 0; yy < 4; ++yy) {
        float row = 0.0;
        for (int xx = 0; xx < 4; ++xx) {
          row += wx[xx] * texelFetch(
            u_current_coefficients,
            base + ivec2(xx - 1, yy - 1),
            0
          ).x;
        }
        value += wy[yy] * row;
      }
      return value;
    }

    bool iterateICGN(
      ivec2 gridIndex,
      vec2 center,
      inout vec2 translation,
      inout vec4 linear,
      inout vec2 status,
      in vec4 hessianBlocks[9],
      in vec4 metaBlocks[4]
    ) {
      bool inside = true;
      for (int corner = 0; corner < 4; ++corner) {
        vec2 local = vec2(
          (corner == 1 || corner == 3) ? ${SUBSET_HALF_EXTENT} : -${SUBSET_HALF_EXTENT},
          corner >= 2 ? ${SUBSET_HALF_EXTENT} : -${SUBSET_HALF_EXTENT}
        );
        vec2 position = center - vec2(0.5) + local + translation + vec2(
          linear.x * local.x + linear.y * local.y,
          linear.z * local.x + linear.w * local.y
        );
        inside = inside && position.x >= 2.0 && position.y >= 2.0 &&
          position.x <= u_frame_size.x - 3.0 && position.y <= u_frame_size.y - 3.0;
      }
      if (!inside) {
        status.x = -1.0;
        return false;
      }

      float sumCurrent = 0.0;
      float sumCurrent2 = 0.0;
      float sumReferenceCurrent = 0.0;
      float sumJCurrent[6];
      for (int a = 0; a < 6; ++a) sumJCurrent[a] = 0.0;

      // One cubic read per pixel: statistics and J^T g are accumulated in the
      // same pass. The complete warp remains in registers between iterations.
      for (int yy = 0; yy < ${SUBSET_SIZE}; ++yy) {
        for (int xx = 0; xx < ${SUBSET_SIZE}; ++xx) {
          vec2 local = vec2(float(xx) - ${SUBSET_HALF_EXTENT}, float(yy) - ${SUBSET_HALF_EXTENT});
          vec2 displacement = translation + vec2(
            linear.x * local.x + linear.y * local.y,
            linear.z * local.x + linear.w * local.y
          );
          float g = sampleCubic(center - vec2(0.5) + local + displacement);
          vec3 templateValue = texelFetch(
            u_reference_template,
            gridIndex * ${SUBSET_SIZE} + ivec2(xx, yy),
            0
          ).xyz;
          float f = templateValue.x;
          float gx = templateValue.y;
          float gy = templateValue.z;
          sumCurrent += g;
          sumCurrent2 += g * g;
          sumReferenceCurrent += f * g;
          float gxg = gx * g;
          float gyg = gy * g;
          sumJCurrent[0] += gxg;
          sumJCurrent[1] += gxg * local.x;
          sumJCurrent[2] += gxg * local.y;
          sumJCurrent[3] += gyg;
          sumJCurrent[4] += gyg * local.x;
          sumJCurrent[5] += gyg * local.y;
        }
      }

      float currentMean = sumCurrent / 256.0;
      float currentNorm = sqrt(max(0.0, sumCurrent2 - sumCurrent * currentMean));
      float referenceNorm = metaValue(metaBlocks, 0);
      if (currentNorm < 1e-9 || referenceNorm < 1e-9) {
        status = vec2(-2.0, status.y + 1.0);
        return false;
      }

      float b[6];
      float ratio = referenceNorm / currentNorm;
      for (int a = 0; a < 6; ++a) {
        float jtReference = metaValue(metaBlocks, 2 + a);
        float sumJ = metaValue(metaBlocks, 8 + a);
        b[a] = jtReference - ratio * (sumJCurrent[a] - currentMean * sumJ);
      }

      float dp[6];
      for (int row = 0; row < 6; ++row) {
        float value = 0.0;
        for (int column = 0; column < 6; ++column) {
          value += hessianValue(hessianBlocks, 6 * row + column) * b[column];
        }
        dp[row] = -value;
      }

      float da11 = 1.0 + dp[1];
      float da12 = dp[2];
      float da21 = dp[4];
      float da22 = 1.0 + dp[5];
      float determinant = da11 * da22 - da12 * da21;
      if (abs(determinant) < 1e-8) {
        status = vec2(-1.0, status.y + 1.0);
        return false;
      }

      mat2 inverseDelta = mat2(
        da22 / determinant, -da21 / determinant,
        -da12 / determinant, da11 / determinant
      );
      mat2 warp = mat2(
        1.0 + linear.x,
        linear.z,
        linear.y,
        1.0 + linear.w
      );
      vec2 inverseDeltaTranslation = -(inverseDelta * vec2(dp[0], dp[3]));
      mat2 updatedWarp = warp * inverseDelta;
      translation += warp * inverseDeltaTranslation;
      linear = vec4(
        updatedWarp[0][0] - 1.0,
        updatedWarp[1][0],
        updatedWarp[0][1],
        updatedWarp[1][1] - 1.0
      );

      float cornerChange = 0.0;
      for (int corner = 0; corner < 4; ++corner) {
        float x = (corner == 1 || corner == 3) ? ${SUBSET_HALF_EXTENT} : -${SUBSET_HALF_EXTENT};
        float y = corner >= 2 ? ${SUBSET_HALF_EXTENT} : -${SUBSET_HALF_EXTENT};
        vec2 deltaAtCorner = vec2(
          dp[0] + dp[1] * x + dp[2] * y,
          dp[3] + dp[4] * x + dp[5] * y
        );
        cornerChange = max(cornerChange, length(deltaAtCorner));
      }

      status.y += 1.0;
      float zncc = sumReferenceCurrent / (referenceNorm * currentNorm);
      if (cornerChange < u_tolerance) {
        status.x = zncc >= u_zncc_min ? 1.0 : -2.0;
        return false;
      }
      status.x = 0.0;
      return true;
    }

    void main() {
      vec2 translation = texture(u_translation, v_uv).xy;
      vec4 linear = texture(u_linear, v_uv);
      vec2 status = texture(u_status, v_uv).xy;
      ivec2 gridIndex = ivec2(floor(v_uv * u_grid_size));

      if (status.x <= 0.5 && status.x >= -0.5) {
        vec4 metaBlocks[4];
        for (int block = 0; block < 4; ++block) {
          metaBlocks[block] = texelFetch(
            u_solver_meta,
            gridIndex * 2 + ivec2(block - 2 * (block / 2), block / 2),
            0
          );
        }
        if (metaValue(metaBlocks, 1) < 0.5) {
          status.x = -1.0;
        } else {
          // These invariant blocks stay in registers throughout every IC-GN
          // iteration instead of being fetched repeatedly from the atlases.
          vec4 hessianBlocks[9];
          for (int block = 0; block < 9; ++block) {
            hessianBlocks[block] = texelFetch(
              u_hessian_inverse,
              gridIndex * 3 + ivec2(block - 3 * (block / 3), block / 3),
              0
            );
          }
          vec2 center = (vec2(gridIndex) + vec2(0.5)) * ${SUBSET_SIZE}.0;
          for (int iteration = 0; iteration < 30; ++iteration) {
            if (iteration >= u_max_iterations) break;
            if (!iterateICGN(
              gridIndex,
              center,
              translation,
              linear,
              status,
              hessianBlocks,
              metaBlocks
            )) break;
          }
        }
      }

      out_translation = translation;
      out_linear = linear;
      out_status = status;
    }
  `,
  uniforms: [
    { name: 'u_current_coefficients', value: 0, type: INT },
    { name: 'u_reference_template', value: 1, type: INT },
    { name: 'u_hessian_inverse', value: 2, type: INT },
    { name: 'u_solver_meta', value: 3, type: INT },
    { name: 'u_translation', value: 4, type: INT },
    { name: 'u_linear', value: 5, type: INT },
    { name: 'u_status', value: 6, type: INT },
    { name: 'u_frame_size', value: frameDimensions, type: FLOAT },
    { name: 'u_grid_size', value: gridDimensions(), type: FLOAT },
    { name: 'u_tolerance', value: PARAMS.tracking.tolerance, type: FLOAT },
    { name: 'u_zncc_min', value: PARAMS.tracking.znccMin, type: FLOAT },
    { name: 'u_max_iterations', value: PARAMS.tracking.iterations, type: INT },
  ],
});

// Match the normalized --median-replace test: an outlier with at least four
// accepted neighbors receives their median prediction and affine gradients.
// The analytical translation normally triggers no replacement, but benchmark
// timings intentionally include this pass.
const medianReplaceProgram = new GPUProgram(composer, {
  name: 'syclMedianReplace',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_translation;
    uniform sampler2D u_linear;
    uniform sampler2D u_status;
    uniform vec2 u_grid_size;

    layout(location = 0) out vec2 out_translation;
    layout(location = 1) out vec4 out_linear;
    layout(location = 2) out vec2 out_status;

    float median8(inout float values[8], int count) {
      for (int pass = 0; pass < 8; ++pass) {
        for (int index = 0; index < 7; ++index) {
          if (index + 1 >= count) continue;
          if (values[index] > values[index + 1]) {
            float temporary = values[index];
            values[index] = values[index + 1];
            values[index + 1] = temporary;
          }
        }
      }
      return (count % 2 == 1)
        ? values[count / 2]
        : 0.5 * (values[count / 2 - 1] + values[count / 2]);
    }

    void main() {
      ivec2 point = ivec2(floor(v_uv * u_grid_size));
      vec2 translation = texelFetch(u_translation, point, 0).xy;
      vec4 linear = texelFetch(u_linear, point, 0);
      vec2 status = texelFetch(u_status, point, 0).xy;
      out_translation = translation;
      out_linear = linear;
      out_status = status;
      if (status.x < 0.5) return;

      float predictedX[8];
      float predictedY[8];
      float p1[8];
      float p2[8];
      float p4[8];
      float p5[8];
      int count = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          ivec2 neighbor = point + ivec2(dx, dy);
          if (neighbor.x < 0 || neighbor.y < 0 ||
              neighbor.x >= int(u_grid_size.x) || neighbor.y >= int(u_grid_size.y)) continue;
          if (texelFetch(u_status, neighbor, 0).x < 0.5) continue;
          vec2 nt = texelFetch(u_translation, neighbor, 0).xy;
          vec4 nl = texelFetch(u_linear, neighbor, 0);
          float deltaX = -${SUBSET_SIZE}.0 * float(dx);
          float deltaY = -${SUBSET_SIZE}.0 * float(dy);
          predictedX[count] = nt.x + nl.x * deltaX + nl.y * deltaY;
          predictedY[count] = nt.y + nl.z * deltaX + nl.w * deltaY;
          p1[count] = nl.x;
          p2[count] = nl.y;
          p4[count] = nl.z;
          p5[count] = nl.w;
          count += 1;
        }
      }
      if (count < 4) return;

      float valuesX[8];
      float valuesY[8];
      for (int index = 0; index < 8; ++index) {
        valuesX[index] = predictedX[index];
        valuesY[index] = predictedY[index];
      }
      float medianX = median8(valuesX, count);
      float medianY = median8(valuesY, count);
      for (int index = 0; index < 8; ++index) {
        valuesX[index] = abs(predictedX[index] - medianX);
        valuesY[index] = abs(predictedY[index] - medianY);
      }
      float spreadX = median8(valuesX, count);
      float spreadY = median8(valuesY, count);
      bool flagged = abs(translation.x - medianX) / (spreadX + 0.1) > 2.0 ||
        abs(translation.y - medianY) / (spreadY + 0.1) > 2.0;
      if (!flagged) return;
      out_translation = vec2(medianX, medianY);
      out_linear = vec4(
        median8(p1, count),
        median8(p2, count),
        median8(p4, count),
        median8(p5, count)
      );
      out_status.x = 2.0;
    }
  `,
  uniforms: [
    { name: 'u_translation', value: 0, type: INT },
    { name: 'u_linear', value: 1, type: INT },
    { name: 'u_status', value: 2, type: INT },
    { name: 'u_grid_size', value: gridDimensions(), type: FLOAT },
  ],
});

const resetParameters = () => {
  translation.clear();
  linearPart.clear();
  trackingStatus.clear();
};

if (!HEADLESS_MODE) {
  const pane = new Tweakpane.Pane({ title: 'Lucas–Kanade affine IC-GN' });
  pane.registerPlugin(TweakpaneEssentialsPlugin);

  const trackingFolder = pane.addFolder({ title: 'Registration' });
  trackingFolder.addInput(PARAMS.tracking, 'active', { label: 'active' });
  trackingFolder.addInput(PARAMS.tracking, 'iterations', {
    label: 'max iterations', min: 1, max: 30, step: 1,
  }).on('change', () => {
    affineICGNProgram.setUniform('u_max_iterations', PARAMS.tracking.iterations);
  });
  trackingFolder.addInput(PARAMS.tracking, 'tolerance', {
    label: 'tolerance (px)', min: 0.0001, max: 0.05,
  }).on('change', () => {
    affineICGNProgram.setUniform('u_tolerance', PARAMS.tracking.tolerance);
  });
  trackingFolder.addInput(PARAMS.tracking, 'znccMin', {
    label: 'minimum ZNCC', min: 0, max: 1,
  }).on('change', () => {
    affineICGNProgram.setUniform('u_zncc_min', PARAMS.tracking.znccMin);
  });
  trackingFolder.addInput(PARAMS.tracking, 'vectorScale', {
    label: 'vector scale', min: 0.25, max: 10,
  });

  const datasetFolder = pane.addFolder({ title: 'Raw 512x512 dataset' });
  datasetFolder.addInput(PARAMS.dataset, 'frame', {
    label: 'target frame',
    options: {
      't=1 (+0.2, -0.1 px)': 1,
      't=5 (+1, -0.5 px)': 5,
      't=10 (+2, -1 px)': 10,
      't=20 (+4, -2 px)': 20,
    },
  }).on('change', async () => {
    await selectTarget(PARAMS.dataset.frame);
    solveCurrent();
    const accuracy = readAccuracy(PARAMS.dataset.frame);
    render();
    setResults(formatInteractiveResult(accuracy));
  });

  datasetFolder.addInput(PARAMS.benchmark, 'repeats', {
    label: 'repetitions', min: 10, max: 100, step: 1,
  });
  datasetFolder.addButton({ title: 'Run benchmark again' }).on('click', () => {
    runBenchmark().catch(showFailure);
  });
}

const loadedImages = new Map();

const loadImage = async (frame) => {
  if (loadedImages.has(frame)) return loadedImages.get(frame);
  const image = new Image();
  image.src = DATASET.url(frame);
  await image.decode();
  if (image.naturalWidth !== FRAME_WIDTH || image.naturalHeight !== FRAME_HEIGHT) {
    throw new Error(
      `Unexpected dimensions for t=${frame}: ${image.naturalWidth}x${image.naturalHeight}`,
    );
  }
  loadedImages.set(frame, image);
  return image;
};

const imageToGray = (image) => {
  const extractionCanvas = document.createElement('canvas');
  extractionCanvas.width = FRAME_WIDTH;
  extractionCanvas.height = FRAME_HEIGHT;
  const context = extractionCanvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const rgba = context.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT).data;
  const gray = new Float64Array(FRAME_WIDTH * FRAME_HEIGHT);
  for (let index = 0; index < gray.length; ++index) gray[index] = rgba[4 * index] / 255;
  return gray;
};

const mirror = (coordinate, size) => coordinate < 0
  ? -coordinate
  : (coordinate >= size ? 2 * size - 2 - coordinate : coordinate);

const atMirror = (values, x, y) => values[
  mirror(y, FRAME_HEIGHT) * FRAME_WIDTH + mirror(x, FRAME_WIDTH)
];

const gaussianSigmaOne = (source) => {
  const radius = 3;
  const kernel = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; ++i) {
    kernel[i + radius] = Math.exp(-0.5 * i * i);
    sum += kernel[i + radius];
  }
  for (let i = 0; i < kernel.length; ++i) kernel[i] /= sum;
  const horizontal = new Float64Array(source.length);
  const filtered = new Float64Array(source.length);
  for (let y = 0; y < FRAME_HEIGHT; ++y) {
    for (let x = 0; x < FRAME_WIDTH; ++x) {
      let value = 0;
      for (let tap = -radius; tap <= radius; ++tap) {
        value += kernel[tap + radius] * source[y * FRAME_WIDTH + mirror(x + tap, FRAME_WIDTH)];
      }
      horizontal[y * FRAME_WIDTH + x] = value;
    }
  }
  for (let y = 0; y < FRAME_HEIGHT; ++y) {
    for (let x = 0; x < FRAME_WIDTH; ++x) {
      let value = 0;
      for (let tap = -radius; tap <= radius; ++tap) {
        value += kernel[tap + radius] * horizontal[mirror(y + tap, FRAME_HEIGHT) * FRAME_WIDTH + x];
      }
      filtered[y * FRAME_WIDTH + x] = value;
    }
  }
  return filtered;
};

const cubicBSplinePrefilter = (source) => {
  const coefficients = new Float64Array(source);
  const z = Math.sqrt(3) - 2;
  const lambda = (1 - z) * (1 - 1 / z);
  const filterLine = (offset, count, stride) => {
    for (let k = 0; k < count; ++k) coefficients[offset + k * stride] *= lambda;
    const horizon = Math.min(count, Math.ceil(Math.log(1e-12) / Math.log(Math.abs(z))));
    let sum = coefficients[offset];
    let zk = z;
    for (let k = 1; k < horizon; ++k) {
      sum += zk * coefficients[offset + k * stride];
      zk *= z;
    }
    coefficients[offset] = sum;
    for (let k = 1; k < count; ++k) {
      coefficients[offset + k * stride] += z * coefficients[offset + (k - 1) * stride];
    }
    const last = offset + (count - 1) * stride;
    coefficients[last] = z / (z * z - 1) * (
      z * coefficients[last - stride] + coefficients[last]
    );
    for (let k = count - 2; k >= 0; --k) {
      const index = offset + k * stride;
      coefficients[index] = z * (coefficients[index + stride] - coefficients[index]);
    }
  };
  for (let y = 0; y < FRAME_HEIGHT; ++y) filterLine(y * FRAME_WIDTH, FRAME_WIDTH, 1);
  for (let x = 0; x < FRAME_WIDTH; ++x) filterLine(x, FRAME_HEIGHT, FRAME_WIDTH);
  return coefficients;
};

const invert6 = (matrix) => {
  const augmented = new Float64Array(6 * 12);
  for (let row = 0; row < 6; ++row) {
    for (let column = 0; column < 6; ++column) {
      augmented[row * 12 + column] = matrix[row * 6 + column];
      augmented[row * 12 + 6 + column] = row === column ? 1 : 0;
    }
  }
  for (let column = 0; column < 6; ++column) {
    let pivotRow = column;
    for (let row = column + 1; row < 6; ++row) {
      if (Math.abs(augmented[row * 12 + column]) > Math.abs(augmented[pivotRow * 12 + column])) {
        pivotRow = row;
      }
    }
    if (Math.abs(augmented[pivotRow * 12 + column]) < 1e-14) return null;
    if (pivotRow !== column) {
      for (let entry = 0; entry < 12; ++entry) {
        const temporary = augmented[column * 12 + entry];
        augmented[column * 12 + entry] = augmented[pivotRow * 12 + entry];
        augmented[pivotRow * 12 + entry] = temporary;
      }
    }
    const pivot = augmented[column * 12 + column];
    for (let entry = 0; entry < 12; ++entry) augmented[column * 12 + entry] /= pivot;
    for (let row = 0; row < 6; ++row) {
      if (row === column) continue;
      const factor = augmented[row * 12 + column];
      for (let entry = 0; entry < 12; ++entry) {
        augmented[row * 12 + entry] -= factor * augmented[column * 12 + entry];
      }
    }
  }
  const inverse = new Float64Array(36);
  for (let row = 0; row < 6; ++row) {
    for (let column = 0; column < 6; ++column) {
      inverse[row * 6 + column] = augmented[row * 12 + 6 + column];
    }
  }
  return inverse;
};

const setAtlasPoint = (atlas, atlasWidth, side, pointX, pointY, values) => {
  for (let index = 0; index < values.length; ++index) {
    const block = Math.floor(index / 4);
    const atlasX = pointX * side + block % side;
    const atlasY = pointY * side + Math.floor(block / side);
    atlas[4 * (atlasY * atlasWidth + atlasX) + index % 4] = values[index];
  }
};

const prepareReferenceTemplate = (image) => {
  const filtered = gaussianSigmaOne(imageToGray(image));
  const coefficients = cubicBSplinePrefilter(filtered);
  const template = new Float32Array(FRAME_WIDTH * FRAME_HEIGHT * 3);
  const grid = gridDimensions();
  const hessianWidth = grid[0] * 3;
  const metaWidth = grid[0] * 2;
  const hessianAtlas = new Float32Array(hessianWidth * grid[1] * 3 * 4);
  const metaAtlas = new Float32Array(metaWidth * grid[1] * 2 * 4);

  for (let gridY = 0; gridY < grid[1]; ++gridY) {
    for (let gridX = 0; gridX < grid[0]; ++gridX) {
      const x0 = gridX * SUBSET_SIZE;
      const y0 = gridY * SUBSET_SIZE;
      let mean = 0;
      for (let yy = 0; yy < SUBSET_SIZE; ++yy) {
        for (let xx = 0; xx < SUBSET_SIZE; ++xx) {
          mean += filtered[(y0 + yy) * FRAME_WIDTH + x0 + xx];
        }
      }
      mean /= SUBSET_SIZE * SUBSET_SIZE;
      const hessian = new Float64Array(36);
      const jtReference = new Float64Array(6);
      const sumJ = new Float64Array(6);
      let referenceEnergy = 0;
      for (let yy = 0; yy < SUBSET_SIZE; ++yy) {
        for (let xx = 0; xx < SUBSET_SIZE; ++xx) {
          const x = x0 + xx;
          const y = y0 + yy;
          const localX = xx - SUBSET_HALF_EXTENT;
          const localY = yy - SUBSET_HALF_EXTENT;
          const f = filtered[y * FRAME_WIDTH + x] - mean;
          let gx = 0.5 * (atMirror(coefficients, x + 1, y) - atMirror(coefficients, x - 1, y));
          let gy = 0.5 * (atMirror(coefficients, x, y + 1) - atMirror(coefficients, x, y - 1));
          const gxSouth = 0.5 * (
            atMirror(coefficients, x + 1, y - 1) - atMirror(coefficients, x - 1, y - 1)
          );
          const gxNorth = 0.5 * (
            atMirror(coefficients, x + 1, y + 1) - atMirror(coefficients, x - 1, y + 1)
          );
          const gyWest = 0.5 * (
            atMirror(coefficients, x - 1, y + 1) - atMirror(coefficients, x - 1, y - 1)
          );
          const gyEast = 0.5 * (
            atMirror(coefficients, x + 1, y + 1) - atMirror(coefficients, x + 1, y - 1)
          );
          gx = (gxSouth + 4 * gx + gxNorth) / 6;
          gy = (gyWest + 4 * gy + gyEast) / 6;
          const pixel = 3 * (y * FRAME_WIDTH + x);
          template[pixel] = f;
          template[pixel + 1] = gx;
          template[pixel + 2] = gy;
          const j = [gx, gx * localX, gx * localY, gy, gy * localX, gy * localY];
          referenceEnergy += f * f;
          for (let row = 0; row < 6; ++row) {
            jtReference[row] += j[row] * f;
            sumJ[row] += j[row];
            for (let column = 0; column < 6; ++column) {
              hessian[6 * row + column] += j[row] * j[column];
            }
          }
        }
      }
      const inverse = invert6(hessian);
      setAtlasPoint(
        hessianAtlas,
        hessianWidth,
        3,
        gridX,
        gridY,
        inverse || new Float64Array(36),
      );
      setAtlasPoint(
        metaAtlas,
        metaWidth,
        2,
        gridX,
        gridY,
        [Math.sqrt(referenceEnergy), inverse && referenceEnergy > 1e-12 ? 1 : 0, ...jtReference, ...sumJ],
      );
    }
  }
  referenceTemplate.setFromArray(template);
  hessianInverseAtlas.setFromArray(hessianAtlas);
  solverMetaAtlas.setFromArray(metaAtlas);
};

const uploadImage = (image, output) => {
  gl.bindTexture(gl.TEXTURE_2D, sourceImage._buffers[sourceImage._bufferIndex]);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
  composer.step({ program: rgbToMonoProgram, input: sourceImage, output });
};

const prefilterCurrent = () => {
  composer.step({ program: prefilterHorizontalProgram, input: currentFrame, output: prefilterTemporary });
  composer.step({ program: prefilterVerticalProgram, input: prefilterTemporary, output: currentCoefficients });
};

const setReferenceImage = (image) => {
  if (image.naturalWidth !== FRAME_WIDTH || image.naturalHeight !== FRAME_HEIGHT) {
    throw new Error(`Reference image must be ${FRAME_WIDTH}x${FRAME_HEIGHT} pixels.`);
  }
  uploadImage(image, currentFrame);
  composer.step({ program: copyFrameProgram, input: currentFrame, output: referenceFrame });
  prepareReferenceTemplate(image);
  gl.finish();
};

const setTargetImage = (image) => {
  if (image.naturalWidth !== FRAME_WIDTH || image.naturalHeight !== FRAME_HEIGHT) {
    throw new Error(`Target image must be ${FRAME_WIDTH}x${FRAME_HEIGHT} pixels.`);
  }
  uploadImage(image, currentFrame);
  prefilterCurrent();
  gl.finish();
};

const selectReference = async (frame) => {
  const image = await loadImage(frame);
  setReferenceImage(image);
};

const initializeReference = () => selectReference(DATASET.sourceFrame);

const selectTarget = async (frame) => {
  const image = await loadImage(frame);
  setTargetImage(image);
};

const solveCurrent = () => {
  resetParameters();
  composer.step({
    program: affineICGNProgram,
    input: [
      currentCoefficients,
      referenceTemplate,
      hessianInverseAtlas,
      solverMetaAtlas,
      translation,
      linearPart,
      trackingStatus,
    ],
    output: [translation, linearPart, trackingStatus],
  });
  composer.step({
    program: medianReplaceProgram,
    input: [translation, linearPart, trackingStatus],
    output: [translation, linearPart, trackingStatus],
  });
};

const render = () => {
  composer.step({
    program: renderFramesProgram,
    input: [currentFrame, referenceFrame],
  });
  if (PARAMS.tracking.active) {
    composer.drawLayerAsVectorField({
      layer: translation,
      vectorSpacing: PARAMS.tracking.spacing,
      vectorScale: PARAMS.tracking.vectorScale,
      color: [1, 1, 1],
    });
  }
};

const readAccuracy = (frame) => {
  const start = performance.now();
  const values = translation.getValues();
  const status = trackingStatus.getValues();
  const readbackMs = performance.now() - start;
  const expected = [DATASET.velocity[0] * frame, DATASET.velocity[1] * frame];
  let accepted = 0;
  let sumX = 0;
  let sumY = 0;
  let sumIterations = 0;
  let replaced = 0;
  let squaredError = 0;
  for (let point = 0; point < values.length / 2; ++point) {
    const valid = status[2 * point] > 0.5;
    const x = values[2 * point];
    const y = values[2 * point + 1];
    if (!valid || !Number.isFinite(x) || !Number.isFinite(y)) continue;
    accepted += 1;
    if (status[2 * point] > 1.5) replaced += 1;
    sumIterations += status[2 * point + 1];
    sumX += x;
    sumY += y;
    squaredError += (x - expected[0]) ** 2 + (y - expected[1]) ** 2;
  }
  return {
    frame,
    expected,
    accepted,
    total: values.length / 2,
    replaced,
    iterationsMean: accepted ? sumIterations / accepted : NaN,
    mean: accepted ? [sumX / accepted, sumY / accepted] : [NaN, NaN],
    rmse: accepted ? Math.sqrt(squaredError / accepted) : NaN,
    readbackMs,
  };
};

const readRegistration = () => {
  const start = performance.now();
  const translations = translation.getValues();
  const linear = linearPart.getValues();
  const packedStatus = trackingStatus.getValues();
  const readbackMs = performance.now() - start;
  const points = translations.length / 2;
  const parameters = new Float32Array(points * 6);
  const status = new Float32Array(points);
  const iterations = new Float32Array(points);
  let accepted = 0;
  let replaced = 0;
  let sumX = 0;
  let sumY = 0;
  let sumIterations = 0;
  for (let point = 0; point < points; ++point) {
    const state = packedStatus[2 * point];
    const count = packedStatus[2 * point + 1];
    parameters[6 * point] = translations[2 * point];
    parameters[6 * point + 1] = linear[4 * point];
    parameters[6 * point + 2] = linear[4 * point + 1];
    parameters[6 * point + 3] = translations[2 * point + 1];
    parameters[6 * point + 4] = linear[4 * point + 2];
    parameters[6 * point + 5] = linear[4 * point + 3];
    status[point] = state;
    iterations[point] = count;
    if (state > 0.5) {
      accepted += 1;
      if (state > 1.5) replaced += 1;
      sumX += translations[2 * point];
      sumY += translations[2 * point + 1];
      sumIterations += count;
    }
  }
  return {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    grid: gridDimensions(),
    subsetSize: SUBSET_SIZE,
    parameters: Array.from(parameters),
    status: Array.from(status),
    iterations: Array.from(iterations),
    accepted,
    replaced,
    meanTranslation: accepted ? [sumX / accepted, sumY / accepted] : [null, null],
    meanIterations: accepted ? sumIterations / accepted : null,
    readbackMs,
  };
};

const percentile = (values, fraction) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
};

const rendererName = () => {
  const extension = gl.getExtension('WEBGL_debug_renderer_info');
  return extension
    ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);
};

const timerQueryExtension = gl.getExtension('EXT_disjoint_timer_query_webgl2');

const waitForTimerQuery = (query) => new Promise((resolve) => {
  const deadline = performance.now() + 5000;
  const poll = () => {
    const available = gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE);
    const disjoint = gl.getParameter(timerQueryExtension.GPU_DISJOINT_EXT);
    if (available) {
      const gpuMs = disjoint ? NaN : gl.getQueryParameter(query, gl.QUERY_RESULT) * 1e-6;
      gl.deleteQuery(query);
      resolve(gpuMs);
      return;
    }
    if (performance.now() >= deadline) {
      gl.deleteQuery(query);
      resolve(NaN);
      return;
    }
    setTimeout(poll, 0);
  };
  poll();
});

const timedSolve = async (includePrefilter = false) => {
  const query = timerQueryExtension ? gl.createQuery() : null;
  if (query) gl.beginQuery(timerQueryExtension.TIME_ELAPSED_EXT, query);
  const start = performance.now();
  if (includePrefilter) prefilterCurrent();
  solveCurrent();
  if (query) gl.endQuery(timerQueryExtension.TIME_ELAPSED_EXT);
  let gpuMs = NaN;
  if (query) {
    gl.flush();
    gpuMs = await waitForTimerQuery(query);
  } else gl.finish();
  const wallMs = performance.now() - start;
  return { wallMs, gpuMs };
};

const timedPrefilter = async () => {
  const query = timerQueryExtension ? gl.createQuery() : null;
  if (query) gl.beginQuery(timerQueryExtension.TIME_ELAPSED_EXT, query);
  prefilterCurrent();
  if (query) gl.endQuery(timerQueryExtension.TIME_ELAPSED_EXT);
  if (!query) {
    gl.finish();
    return NaN;
  }
  gl.flush();
  return waitForTimerQuery(query);
};

const timedEndToEnd = (image) => {
  const start = performance.now();
  uploadImage(image, currentFrame);
  prefilterCurrent();
  solveCurrent();
  translation.getValues();
  trackingStatus.getValues();
  return performance.now() - start;
};

const formatNumber = (value, digits = 3) => (
  Number.isFinite(value) ? value.toFixed(digits) : 'n/a'
);

const formatInteractiveResult = (accuracy) => [
  `Dataset: ${DATASET.name}`,
  `Target: t=${accuracy.frame}, truth = (${accuracy.expected[0].toFixed(2)}, ${accuracy.expected[1].toFixed(2)}) px`,
  `Measured mean = (${formatNumber(accuracy.mean[0])}, ${formatNumber(accuracy.mean[1])}) px`,
  `Displacement RMSE = ${formatNumber(accuracy.rmse, 4)} px`,
  `Accepted subsets = ${accuracy.accepted}/${accuracy.total}; ` +
    `mean iterations = ${formatNumber(accuracy.iterationsMean)}`,
  `Median replacements = ${accuracy.replaced}`,
].join('\n');

let benchmarkRunning = false;
const runBenchmark = async () => {
  if (benchmarkRunning) return;
  benchmarkRunning = true;
  setStatus('Synchronous benchmark in progress...');
  setResults('Compiling, warming up, and collecting measurements...');
  try {
    const compileStart = performance.now();
    await selectTarget(1);
    solveCurrent();
    gl.finish();
    const firstSolveMs = performance.now() - compileStart;

    for (let warmup = 0; warmup < 3; ++warmup) {
      solveCurrent();
      gl.finish();
    }

    await selectTarget(PARAMS.dataset.frame);
    const pipelineWallTimes = [];
    const prefilterGpuTimes = [];
    const pipelineGpuTimes = [];
    for (let repeat = 0; repeat < PARAMS.benchmark.repeats; ++repeat) {
      const timing = await timedSolve(true);
      pipelineWallTimes.push(timing.wallMs);
      if (Number.isFinite(timing.gpuMs)) pipelineGpuTimes.push(timing.gpuMs);
    }
    for (let repeat = 0; repeat < PARAMS.benchmark.repeats; ++repeat) {
      const gpuMs = await timedPrefilter();
      if (Number.isFinite(gpuMs)) prefilterGpuTimes.push(gpuMs);
    }
    const endToEndTimes = [];
    const targetImage = await loadImage(PARAMS.dataset.frame);
    for (let repeat = 0; repeat < PARAMS.benchmark.repeats; ++repeat) {
      endToEndTimes.push(timedEndToEnd(targetImage));
    }

    const selectedAccuracy = readAccuracy(PARAMS.dataset.frame);
    const accuracyRows = [];
    for (const frame of DATASET.frames) {
      await selectTarget(frame);
      solveCurrent();
      gl.finish();
      accuracyRows.push(readAccuracy(frame));
    }

    // The direct t=0→20 displacement is outside the local basin.  Check the
    // same final image as the next step of a sequence (t=10→20, +2/−1 px).
    await selectReference(10);
    await selectTarget(20);
    solveCurrent();
    gl.finish();
    const incrementalAccuracy = readAccuracy(10);

    await initializeReference();
    await selectTarget(PARAMS.dataset.frame);
    solveCurrent();
    gl.finish();
    render();

    const pipelineMedianMs = percentile(pipelineWallTimes, 0.5);
    const pipelineP95Ms = percentile(pipelineWallTimes, 0.95);
    const prefilterGpuMedianMs = prefilterGpuTimes.length ? percentile(prefilterGpuTimes, 0.5) : NaN;
    const prefilterGpuP95Ms = prefilterGpuTimes.length ? percentile(prefilterGpuTimes, 0.95) : NaN;
    const pipelineGpuMedianMs = pipelineGpuTimes.length ? percentile(pipelineGpuTimes, 0.5) : NaN;
    const pipelineGpuP95Ms = pipelineGpuTimes.length ? percentile(pipelineGpuTimes, 0.95) : NaN;
    const solveGpuMedianMs = pipelineGpuMedianMs - prefilterGpuMedianMs;
    const endToEndMedianMs = percentile(endToEndTimes, 0.5);
    const endToEndP95Ms = percentile(endToEndTimes, 0.95);
    const grid = gridDimensions();
    const lines = [
      'WEBGL2 IC-GN BENCHMARK — raw YUV images',
      `GPU: ${rendererName()}`,
      `Dataset: ${DATASET.name}, reference t=0`,
      `Size: ${FRAME_WIDTH}x${FRAME_HEIGHT}; grid: ${grid[0]}x${grid[1]} (${grid[0] * grid[1]} subsets)`,
      `Subset: ${SUBSET_SIZE}x${SUBSET_SIZE}; max iterations: ${PARAMS.tracking.iterations}`,
      `First pass: ${formatNumber(firstSolveMs)} ms (includes first-load compilation)`,
      `FIR — GPU median / p95: ${formatNumber(prefilterGpuMedianMs)} / ${formatNumber(prefilterGpuP95Ms)} ms`,
      `Solver estimated by difference — GPU median: ${formatNumber(solveGpuMedianMs)} ms`,
      `FIR + solver — GPU median / p95: ${formatNumber(pipelineGpuMedianMs)} / ${formatNumber(pipelineGpuP95Ms)} ms`,
      `FIR + solver — wall median / p95: ${formatNumber(pipelineMedianMs)} / ${formatNumber(pipelineP95Ms)} ms`,
      `Upload + FIR + solver + readback — wall median / p95: ` +
        `${formatNumber(endToEndMedianMs)} / ${formatNumber(endToEndP95Ms)} ms`,
      `Solver / GPU pipeline throughput: ${formatNumber(1000 / solveGpuMedianMs, 1)} / ` +
        `${formatNumber(1000 / pipelineGpuMedianMs, 1)} pairs/s`,
      '',
      'Accuracy (mean x/y, RMSE, accepted, iterations, median replacements):',
    ];
    for (const row of accuracyRows) {
      lines.push(
        `t=${String(row.frame).padStart(2)} truth=(${row.expected[0].toFixed(2)},${row.expected[1].toFixed(2)}) ` +
        `measured=(${formatNumber(row.mean[0])},${formatNumber(row.mean[1])}) ` +
        `RMSE=${formatNumber(row.rmse, 4)} px ` +
        `accepted=${row.accepted}/${row.total} iter=${formatNumber(row.iterationsMean)} median=${row.replaced}`,
      );
    }
    lines.push(
      `t=10->20 incremental truth=(2.00,-1.00) ` +
      `measured=(${formatNumber(incrementalAccuracy.mean[0])},${formatNumber(incrementalAccuracy.mean[1])}) ` +
      `RMSE=${formatNumber(incrementalAccuracy.rmse, 4)} px ` +
      `accepted=${incrementalAccuracy.accepted}/${incrementalAccuracy.total} ` +
      `iter=${formatNumber(incrementalAccuracy.iterationsMean)} median=${incrementalAccuracy.replaced}`,
    );
    lines.push('', `GPU-to-CPU readback: ${formatNumber(selectedAccuracy.readbackMs)} ms`);
    setResults(lines.join('\n'));
    setStatus('Benchmark complete. Timings include gl.finish(), with no H.264 decoding.');
    document.body.dataset.benchmark = 'complete';
  } finally {
    benchmarkRunning = false;
  }
};

const showFailure = (error) => {
  console.error(error);
  setStatus('Benchmark failed.');
  setResults(`ERROR\n${error.stack || error.message || error}`);
  document.body.dataset.benchmark = 'failed';
};

const boot = async () => {
  setStatus('Loading five frames extracted from raw YUV...');
  await Promise.all([DATASET.sourceFrame, ...DATASET.frames].map(loadImage));
  await initializeReference();
  await selectTarget(PARAMS.dataset.frame);
  render();
  await runBenchmark();
};

const decodeExternalImage = async (dataUrl) => {
  if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/')) {
    throw new Error('Expected an image encoded as a data URL.');
  }
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  return image;
};

const configureHeadless = (options = {}) => {
  if (options.iterations !== undefined) {
    const iterations = Number(options.iterations);
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 30) {
      throw new Error('iterations must be an integer between 1 and 30.');
    }
    PARAMS.tracking.iterations = iterations;
    affineICGNProgram.setUniform('u_max_iterations', iterations);
  }
  if (options.tolerance !== undefined) {
    const tolerance = Number(options.tolerance);
    if (!(tolerance > 0)) throw new Error('tolerance must be greater than zero.');
    PARAMS.tracking.tolerance = tolerance;
    affineICGNProgram.setUniform('u_tolerance', tolerance);
  }
  if (options.znccMin !== undefined) {
    const znccMin = Number(options.znccMin);
    if (!(znccMin >= 0 && znccMin <= 1)) throw new Error('znccMin must be in [0, 1].');
    PARAMS.tracking.znccMin = znccMin;
    affineICGNProgram.setUniform('u_zncc_min', znccMin);
  }
  return {
    iterations: PARAMS.tracking.iterations,
    tolerance: PARAMS.tracking.tolerance,
    znccMin: PARAMS.tracking.znccMin,
  };
};

const headlessSetReference = async (dataUrl, options = {}) => {
  const config = configureHeadless(options);
  const image = await decodeExternalImage(dataUrl);
  const start = performance.now();
  setReferenceImage(image);
  const preparationWallMs = performance.now() - start;
  return {
    preparationWallMs,
    runtimeBuildWallMs: 0,
    referenceBuildWallMs: preparationWallMs,
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    subsetSize: SUBSET_SIZE,
    config,
    device: rendererName(),
    grid: gridDimensions(),
  };
};

const headlessRegister = async (dataUrl, options = {}) => {
  const config = configureHeadless(options);
  const image = await decodeExternalImage(dataUrl);
  const totalStart = performance.now();
  const prefilterStart = performance.now();
  setTargetImage(image);
  const prefilterWallMs = performance.now() - prefilterStart;
  const solveTiming = await timedSolve(false);
  const registration = readRegistration();
  registration.timing = {
    prefilterWallMs,
    solveWallMs: solveTiming.wallMs,
    solveGpuMs: solveTiming.gpuMs,
    readbackMs: registration.readbackMs,
    totalWallMs: performance.now() - totalStart,
  };
  registration.config = config;
  return registration;
};

const headlessBenchmark = async (dataUrl, repeats = 30, warmups = 3, options = {}) => {
  const config = configureHeadless(options);
  repeats = Number(repeats);
  warmups = Number(warmups);
  if (!Number.isInteger(repeats) || repeats < 1 || repeats > 1000) {
    throw new Error('repeats must be an integer between 1 and 1000.');
  }
  if (!Number.isInteger(warmups) || warmups < 0 || warmups > 100) {
    throw new Error('warmups must be an integer between 0 and 100.');
  }
  const image = await decodeExternalImage(dataUrl);
  setTargetImage(image);
  for (let repeat = 0; repeat < warmups; ++repeat) {
    prefilterCurrent();
    solveCurrent();
    gl.finish();
  }
  const gpuTimes = [];
  const wallTimes = [];
  for (let repeat = 0; repeat < repeats; ++repeat) {
    const timing = await timedSolve(true);
    wallTimes.push(timing.wallMs);
    if (Number.isFinite(timing.gpuMs)) gpuTimes.push(timing.gpuMs);
  }
  const gpuMedianMs = gpuTimes.length ? percentile(gpuTimes, 0.5) : null;
  const gpuP95Ms = gpuTimes.length ? percentile(gpuTimes, 0.95) : null;
  const wallMedianMs = percentile(wallTimes, 0.5);
  const wallP95Ms = percentile(wallTimes, 0.95);
  const registration = readRegistration();
  registration.config = config;
  registration.timing = {gpuMedianMs, gpuP95Ms, wallMedianMs, wallP95Ms};
  return {
    repeats,
    warmups,
    gpuTimes,
    wallTimes,
    gpuMedianMs,
    gpuP95Ms,
    wallMedianMs,
    wallP95Ms,
    device: rendererName(),
    registration,
  };
};

if (HEADLESS_MODE) {
  window.ICGNHeadless = Object.freeze({
    version: 1,
    info: () => ({
      width: FRAME_WIDTH,
      height: FRAME_HEIGHT,
      grid: gridDimensions(),
      subsetSize: SUBSET_SIZE,
      device: rendererName(),
      timerQuery: Boolean(timerQueryExtension),
      solver: 'fused-fragment',
      subgroupWidth: 0,
    }),
    configure: configureHeadless,
    setReference: headlessSetReference,
    register: headlessRegister,
    benchmark: headlessBenchmark,
  });
  document.body.dataset.engine = 'ready';
} else {
  boot().catch(showFailure);
}
