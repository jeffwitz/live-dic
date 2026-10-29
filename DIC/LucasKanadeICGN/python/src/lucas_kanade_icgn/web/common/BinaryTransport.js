/* Persistent same-origin binary transport for the headless Python adapters. */

const token = new URLSearchParams(window.location.search).get('binaryToken');
const textEncoder = new TextEncoder();

const endpoint = (operation, ticket = null) => {
  const suffix = ticket === null ? '' : `/${encodeURIComponent(ticket)}`;
  return `/__icgn_binary__/${encodeURIComponent(token)}/${operation}${suffix}`;
};

const packState = (result) => {
  const gridCount = result.grid[0] * result.grid[1];
  const parameterCount = result.config.motionModel === 'quadratic' ? 12 : 6;
  const stateStride = parameterCount === 12 ? 16 : 8;
  const packed = new Float32Array(gridCount * stateStride);
  for (let index = 0; index < gridCount; ++index) {
    const source = parameterCount * index;
    const destination = stateStride * index;
    for (let component = 0; component < parameterCount; ++component) {
      packed[destination + component] = result.parameters[source + component];
    }
    packed[destination + parameterCount] = result.status[index];
    packed[destination + parameterCount + 1] = result.iterations[index];
  }
  return { packed, stateStride };
};

const frameResult = (result, inputWallMs) => {
  const packingStarted = performance.now();
  const { packed, stateStride } = packState(result);
  const packingWallMs = performance.now() - packingStarted;
  const { parameters, status, iterations, ...metadata } = result;
  metadata.stateStride = stateStride;
  metadata.timing = {
    ...metadata.timing,
    binaryInputWallMs: inputWallMs,
    binaryPackingWallMs: packingWallMs,
  };
  const encodedMetadata = textEncoder.encode(JSON.stringify(metadata));
  const body = new Uint8Array(4 + encodedMetadata.byteLength + packed.byteLength);
  new DataView(body.buffer).setUint32(0, encodedMetadata.byteLength, true);
  body.set(encodedMetadata, 4);
  body.set(new Uint8Array(packed.buffer), 4 + encodedMetadata.byteLength);
  return body;
};

const requireHeader = (response, name) => {
  const value = response.headers.get(name);
  if (value === null) throw new Error(`Binary input is missing the ${name} header`);
  return value;
};

const processNextFrame = async () => {
  const response = await fetch(endpoint('next'), { cache: 'no-store' });
  if (response.status === 204) return false;
  if (!response.ok) throw new Error(`Binary input request failed with HTTP ${response.status}`);
  const inputStarted = performance.now();
  const ticket = requireHeader(response, 'X-ICGN-Ticket');
  try {
    const width = Number(requireHeader(response, 'X-ICGN-Width'));
    const height = Number(requireHeader(response, 'X-ICGN-Height'));
    const options = JSON.parse(requireHeader(response, 'X-ICGN-Options'));
    const pixels = new Uint8Array(await response.arrayBuffer());
    const inputWallMs = performance.now() - inputStarted;
    const result = await window.ICGNHeadless.registerBytes(pixels, width, height, options);
    const body = frameResult(result, inputWallMs);
    const outputResponse = await fetch(endpoint('result', ticket), {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body,
    });
    if (!outputResponse.ok) {
      throw new Error(`Binary result upload failed with HTTP ${outputResponse.status}`);
    }
  } catch (error) {
    const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    await fetch(endpoint('failure', ticket), {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: message,
    });
  }
  return true;
};

export const installBinaryTransport = () => {
  if (!token) throw new Error('The binary transport token is missing');
  if (!window.ICGNHeadless?.registerBytes) {
    throw new Error('The selected backend does not expose registerBytes()');
  }

  const run = async () => {
    while (await processNextFrame()) {
      // The next long poll starts only after the current result is delivered.
    }
  };
  const stopped = run();
  window.ICGNBinaryTransport = { stopped };
};
