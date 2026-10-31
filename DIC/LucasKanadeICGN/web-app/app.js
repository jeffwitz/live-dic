import { installRoi } from './ui/roi.js';
const $ = (id) => document.getElementById(id);
const query = new URLSearchParams(location.search), smoke = query.has('smoke');
let transport, stream, reader, manifest, source = null, paused = true, inFlight = 0, dropped = 0;
let generation = 0, previousFrame = -1, referenceInfo, capabilities, sourceSettings = {}, requests = 0;
const pending = new Map(), video = $('video');
$('demo').disabled = $('camera').disabled = true;
const status = globalThis.liveDicStatus = { ready: false, completed: 0, errors: [], results: [] };
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function send(type, data = {}, transfer = []) {
  const id = ++requests;
  const result = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
  transport.postMessage({ type, id, ...data }, transfer);
  return result;
}
const options = () => ({ profile: $('profile').value });
function details() { $('details').textContent = JSON.stringify({ capabilities, acquisition: sourceSettings,
  execution: transport.isMain ? 'Main-thread fallback: OffscreenCanvas unavailable' : 'Dedicated worker',
  reference: referenceInfo, profile: $('profile').value,
  requestedResolution: $('resolution').value }, null, 2); }
function updateScale(scale) {
  if (!Number.isFinite(scale)) return;
  const mode = Number($('component').value);
  document.querySelector('.legend').classList.toggle('magnitude', mode === 2);
  document.querySelector('.legend').classList.toggle('status', mode === 6);
  if (mode === 6) { $('low').textContent = 'Accepted'; $('high').textContent = 'Flagged / replaced'; return; }
  const strain = Number($('component').value) >= 3 && Number($('component').value) <= 5;
  $('low').textContent = mode === 2 ? '0' : `−${scale.toPrecision(3)}`;
  $('high').textContent = `+${scale.toPrecision(3)}${strain ? ' (gradient)' : ' px'}`;
}
function receive({ data }) {
  if (data.bitmap) {
    $('field').width = data.width; $('field').height = data.height;
    $('field').getContext('bitmaprenderer').transferFromImageBitmap(data.bitmap);
    delete data.bitmap;
  }
  if (data.type === 'error') {
    status.errors.push(data.message); message(data.message, true);
    if (pending.has(data.id)) { pending.get(data.id).reject(new Error(data.message)); pending.delete(data.id); }
    paused = true; return;
  }
  if (data.type === 'result') {
    status.completed++; status.results.push(data); if (status.results.length > 18000) status.results.shift();
    $('fps').textContent = data.fps.toFixed(1);
    $('age').textContent = `${data.ageMedian.toFixed(1)} / ${data.ageP95.toFixed(1)} ms`;
    $('acceptance').textContent = `${data.accepted} / ${data.selected}`;
    $('iterations').textContent = data.meanIterations?.toFixed(2) ?? '—';
    $('error').textContent = data.truthRmse == null ? '—' : `${data.truthRmse.toFixed(4)} px`;
    updateScale(data.scale);
  }
  if (data.type === 'scale') updateScale(data.scale);
  if (data.type === 'dropped') { dropped++; $('drops').textContent = dropped; }
  const request = pending.get(data.id);
  if (request) { pending.delete(data.id); request.resolve(data); }
}
async function init() {
  const canvas = $('field');
  if (typeof OffscreenCanvas !== 'undefined' && typeof Worker !== 'undefined') {
    transport = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    transport.onmessage = receive;
    transport.onerror = (error) => { status.errors.push(error.message); message(error.message, true); paused = true; for (const request of pending.values()) request.reject(new Error(error.message)); pending.clear(); };
    capabilities = (await send('boot', { presentation: 'bitmap', measure: !smoke })).capabilities;
  } else {
    const { createSession } = await import('./ui/session.js');
    const session = createSession((data) => queueMicrotask(() => receive({ data })));
    transport = { isMain: true, postMessage: (data) => { session.handle(data); } };
    capabilities = (await send('boot', { canvas, measure: !smoke })).capabilities;
  }
  details();
  if (!capabilities.supported) {
    message(`WebGPU is unavailable. ${capabilities.reasons.join('. ')} Use a supported secure browser; this app has no WebGL fallback.`, true);
    $('camera').disabled = $('demo').disabled = true; return;
  }
  for (const option of $('resolution').options) {
    const [w, h] = option.value.split('x').map(Number);
    option.disabled = !capabilities.resolutions.some(([x, y]) => x === w && y === h);
  }
  $('demo').disabled = $('camera').disabled = false;
  $('profile').value = capabilities.recommendedProfile;
  message(`Ready. ${capabilities.reasons.join('. ')}`); status.ready = true;
  if (smoke) await startDemo();
}
async function drain() {
  while (inFlight) await new Promise((resolve) => setTimeout(resolve, 10));
}
async function stop() {
  paused = true; generation++; await drain();
  await reader?.cancel(); reader = null;
  stream?.getTracks().forEach((track) => track.stop()); stream = null;
  video.pause(); video.srcObject = null; source = null;
}
async function frameFromVideo() {
  const [width, height] = sourceSettings.outputSize ?? [video.videoWidth, video.videoHeight];
  if (typeof VideoFrame === 'function' && width === video.videoWidth && height === video.videoHeight) {
    try { return new VideoFrame(video, { timestamp: Math.round(video.currentTime * 1e6) }); } catch { /* ImageBitmap fallback below. */ }
  }
  return createImageBitmap(video, { resizeWidth: width, resizeHeight: height });
}
async function captureReference(frame = null) {
  paused = true; await drain();
  frame ??= await frameFromVideo();
  referenceInfo = (await send('reference', { frame, options: options() }, [frame])).info;
  $('placeholder').hidden = true;
  $('roi').width = referenceInfo.width; $('roi').height = referenceInfo.height;
  for (const id of ['reference', 'pause', 'reset', 'draw', 'clear-roi', 'export']) $(id).disabled = false;
  previousFrame = -1; details(); message('Reference captured. Camera and results stay on this device.');
}
async function processFrame(frame, frameIndex = null) {
  if (paused || inFlight >= 1) { frame.close(); dropped++; $('drops').textContent = dropped; return; }
  inFlight++;
  const reset = frameIndex != null && frameIndex < previousFrame;
  if (frameIndex != null) previousFrame = frameIndex;
  const stretch = manifest?.stretch[frameIndex];
  try { await send('frame', { frame, frameIndex, reset,
    truth: source === 'demo' && stretch ? [stretch[0] - 1, stretch[1] - 1] : null }, [frame]); }
  catch (error) { paused = true; message(error.message, true); }
  finally { inFlight--; }
}
function scheduleVideo(token) {
  let lastPresented = 0, lastMediaTime = -1;
  const callback = async (_, metadata) => {
    if (!paused && lastPresented && metadata?.presentedFrames) { dropped += Math.max(0, metadata.presentedFrames - lastPresented - 1); $('drops').textContent = dropped; }
    if (metadata?.presentedFrames) lastPresented = metadata.presentedFrames;
    if (generation !== token) return;
    const mediaTime = metadata?.mediaTime ?? video.currentTime;
    if (!video.requestVideoFrameCallback && mediaTime === lastMediaTime) {
      requestAnimationFrame((time) => callback(time, null)); return;
    }
    lastMediaTime = mediaTime;
    if (!paused && inFlight < 1) {
      inFlight++; // Reserve admission before asynchronous ImageBitmap creation.
      try {
        const frame = await frameFromVideo(); inFlight--;
        if (generation !== token) { frame.close(); return; }
        const index = source === 'demo' ? Math.min(manifest.frames - 1, Math.round((metadata?.mediaTime ?? video.currentTime) * manifest.frame_rate)) : null;
        processFrame(frame, index);
      } catch (error) { inFlight--; message(error.message, true); }
    } else if (!paused) { dropped++; $('drops').textContent = dropped; }
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(callback);
    else requestAnimationFrame((time) => callback(time, null));
  };
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(callback);
  else requestAnimationFrame((time) => callback(time, null));
}
async function startDemo() {
  await stop(); source = 'demo'; $('resolution').disabled = true;
  manifest = await fetch(new URL('./demo/manifest.json', import.meta.url)).then((r) => r.json());
  const reference = await createImageBitmap(await fetch(new URL('./demo/reference.png', import.meta.url)).then((r) => r.blob()));
  sourceSettings = { source: 'derived analytic tensile demo', encoding: manifest.encoding, outputSize: [manifest.width, manifest.height] };
  $('resolution').value = `${manifest.width}x${manifest.height}`;
  if (!capabilities.resolutions.some(([w, h]) => w === manifest.width && h === manifest.height)) throw new Error('The shipped demo requires 1080p capability');
  await captureReference(reference);
  video.src = new URL('./demo/tensile.webm', import.meta.url); video.loop = true;
  if (smoke && query.has('rate')) video.playbackRate = Number(query.get('rate'));
  await video.play(); paused = false; $('pause').textContent = 'Pause'; scheduleVideo(generation);
  message('Analytic tensile demo · compressed 1080p replay. Live error includes image formation and video decoding.');
}
async function lockCamera(track) {
  const caps = track.getCapabilities?.() ?? {}, settings = track.getSettings(), locks = {};
  for (const key of ['exposureMode', 'focusMode', 'whiteBalanceMode']) {
    if (Array.isArray(caps[key]) && caps[key].includes('manual')) {
      try { await track.applyConstraints({ advanced: [{ [key]: 'manual' }] }); locks[key] = 'manual requested'; }
      catch (error) { locks[key] = `not locked: ${error.message}`; }
    } else locks[key] = 'manual lock not exposed';
  }
  return { requested: settings, actual: track.getSettings(), capabilities: caps, locks };
}
async function startCamera() {
  await stop(); manifest = null; source = 'camera'; $('resolution').disabled = false;
  const [width, height] = $('resolution').value.split('x').map(Number);
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'environment', width, height, frameRate: { ideal: 60 } } });
  video.src = ''; video.srcObject = stream; await video.play();
  const track = stream.getVideoTracks()[0];
  sourceSettings = { source: 'camera', requested: { width, height, frameRate: 60 }, ...await lockCamera(track), outputSize: [video.videoWidth, video.videoHeight] };
  await captureReference(); paused = false; $('pause').textContent = 'Pause';
  const token = generation;
  if (typeof MediaStreamTrackProcessor === 'function') {
    try { reader = new MediaStreamTrackProcessor({ track }).readable.getReader(); }
    catch (error) { sourceSettings.trackProcessor = String(error); scheduleVideo(token); details(); return; }
    const consume = async () => {
      while (generation === token) { const item = await reader.read(); if (item.done) break; processFrame(item.value); }
    }; consume().catch((error) => message(error.message, true));
  } else scheduleVideo(token);
  details(); message('Camera active. Actual settings and available manual locks are shown below.');
}
const roi = installRoi($('roi'), $('shape'), (ready) => { $('apply-roi').disabled = !ready; });
$('demo').onclick = () => startDemo().catch((error) => message(error.message, true));
$('camera').onclick = () => startCamera().catch((error) => message(error.message, true));
$('reference').onclick = async () => { await captureReference(); paused = false; $('pause').textContent = 'Pause'; };
$('pause').onclick = () => { paused = !paused; $('pause').textContent = paused ? 'Resume' : 'Pause'; };
$('reset').onclick = () => send('reset');
$('draw').onclick = async () => { paused = true; await drain(); $('pause').textContent = 'Resume'; roi.start(referenceInfo.width, referenceInfo.height); $('apply-roi').disabled = false; };
$('apply-roi').onclick = async () => {
  try { roi.finish(); const mask = roi.mask(); referenceInfo = (await send('roi', { mask, options: options() }, [mask.buffer])).info;
    message('Full-window ROI applied. Resume to continue.'); details(); }
  catch (error) { message(error.message, true); }
};
$('clear-roi').onclick = async () => { paused = true; await drain(); roi.clear(); referenceInfo = (await send('roi', { mask: null, options: options() })).info; $('pause').textContent = 'Resume'; details(); };
for (const id of ['component', 'scale']) $(id).onchange = () => send('display', { display: { mode: Number($('component').value), scale: $('scale').value === 'auto' ? null : Number($('scale').value) } });
function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type })), link = document.createElement('a');
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('export').onclick = async () => {
  paused = true; await drain(); $('pause').textContent = 'Resume';
  const { field, metadata } = await send('export');
  const parameters = field.config.motionModel === 'quadratic' ? 12 : 6;
  const lines = [`x,y,status,iterations,${Array.from({ length: parameters }, (_, i) => `p${i}`).join(',')}`];
  for (let i = 0; i < field.status.length; i++) lines.push([
    (i % field.grid[0]) * field.subsetStride + (field.subsetSize - 1) / 2,
    Math.floor(i / field.grid[0]) * field.subsetStride + (field.subsetSize - 1) / 2,
    field.status[i], field.iterations[i], ...field.parameters.slice(i * parameters, (i + 1) * parameters),
  ].join(','));
  metadata.acquisition = sourceSettings;
  download('live-dic-field.csv', lines.join('\n'), 'text/csv');
  download('live-dic-metadata.json', JSON.stringify(metadata, null, 2), 'application/json');
};
$('resolution').onchange = () => { if (source === 'camera') startCamera().catch((error) => message(error.message, true)); };
$('profile').onchange = () => { if (referenceInfo) message('Profile change takes effect when the reference is captured again.'); };
window.addEventListener('pagehide', () => { stream?.getTracks().forEach((t) => t.stop()); transport?.postMessage({ type: 'destroy' }); });
window.addEventListener('error', (event) => { status.errors.push(event.message); message(event.message, true); });
window.addEventListener('unhandledrejection', (event) => { const text = String(event.reason); status.errors.push(text); message(text, true); });
init().catch((error) => { status.errors.push(error.message); message(error.message, true); });
if ('serviceWorker' in navigator && (!smoke || query.has('offline'))) navigator.serviceWorker.register(new URL('./sw.js', import.meta.url)).catch((error) => message(`Offline installation unavailable: ${error.message}`, true));
