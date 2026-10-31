/** Dense packed field decoding; no DOM or GPU dependencies. */
export function decodePacked(packed, { width: FRAME_WIDTH, height: FRAME_HEIGHT,
  gridWidth: GRID_WIDTH, gridHeight: GRID_HEIGHT, subsetSize: SUBSET_SIZE,
  subsetStride: SUBSET_STRIDE, configuration, timing }) {
  const GRID_COUNT = GRID_WIDTH * GRID_HEIGHT;
  const parameterCount = configuration.motionModel === 'quadratic' ? 12 : 6;
  const stateStride = configuration.motionModel === 'quadratic' ? 16 : 8;
  const parameters = new Float32Array(GRID_COUNT * parameterCount);
  const status = new Float32Array(GRID_COUNT);
  const iterations = new Float32Array(GRID_COUNT);
  let accepted = 0;
  let replaced = 0;
  let meanX = 0;
  let meanY = 0;
  let iterationTotal = 0;
  for (let index = 0; index < GRID_COUNT; ++index) {
    const source = stateStride * index;
    const target = parameterCount * index;
    for (let component = 0; component < parameterCount; ++component) {
      parameters[target + component] = packed[source + component];
    }
    status[index] = packed[source + parameterCount];
    iterations[index] = packed[source + parameterCount + 1];
    if (status[index] > 1.5) replaced += 1;
    if (status[index] >= 0.5) {
      accepted += 1;
      meanX += parameters[target];
      meanY += parameters[target + 3];
      iterationTotal += iterations[index];
    }
  }
  return {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    subsetSize: SUBSET_SIZE,
    subsetStride: SUBSET_STRIDE,
    grid: [GRID_WIDTH, GRID_HEIGHT],
    stateStride,
    parameters,
    status,
    iterations,
    accepted,
    replaced,
    meanTranslation: accepted ? [meanX / accepted, meanY / accepted] : [null, null],
    meanIterations: accepted ? iterationTotal / accepted : null,
    timing,
    config: { ...configuration },
  };
}
