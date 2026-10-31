/** Local file admission. Files stay browser-owned; only one image is decoded ahead. */
const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|avif|gif)$/i;
const collator = new Intl.Collator('en', { numeric: true, sensitivity: 'variant' });
export const fileName = file => file.webkitRelativePath || file.name;
export const fileInfo = file => ({ name: fileName(file), size: file.size, type: file.type, lastModified: file.lastModified });
export function imageFiles(files) {
  const images = [], ignored = [];
  for (const file of files) (IMAGE_EXT.test(file.name) ? images : ignored).push(file);
  images.sort((a,b) => collator.compare(fileName(a), fileName(b)) || (fileName(a) < fileName(b) ? -1 : fileName(a) > fileName(b) ? 1 : 0));
  if (!images.length) throw new Error('No supported image files. Choose PNG, JPEG, WebP, BMP, AVIF or GIF; decoding depends on the browser.');
  return { images: Object.freeze(images), ignored: ignored.map(fileName) };
}
function aborted(signal) { if (signal?.aborted) throw new DOMException('Source changed', 'AbortError'); }
export async function decodeImage(file, expected = null, signal = null, decode = createImageBitmap) {
  aborted(signal);
  let bitmap;
  try { bitmap = await decode(file); }
  catch (error) { throw new Error(`Cannot decode ${fileName(file)}: ${error.message}`); }
  if (signal?.aborted) { bitmap.close(); aborted(signal); }
  if (expected && (bitmap.width !== expected[0] || bitmap.height !== expected[1])) {
    const actual = `${bitmap.width}×${bitmap.height}`; bitmap.close();
    throw new Error(`${fileName(file)} is ${actual}; the reference is ${expected.join('×')}. Images are never resized implicitly.`);
  }
  return bitmap;
}
export function waitForMedia(video, event, signal, timeout = 30000) {
  aborted(signal);
  return new Promise((resolve, reject) => {
    let timer;
    const clean = () => { clearTimeout(timer); video.removeEventListener(event, done); video.removeEventListener('error', fail); signal?.removeEventListener('abort', cancel); };
    const done = () => { clean(); resolve(); };
    const fail = () => { clean(); reject(new Error(`The browser cannot decode this video${video.error ? ` (media error ${video.error.code})` : ''}. Try a supported MP4/WebM codec.`)); };
    const cancel = () => { clean(); reject(new DOMException('Source changed', 'AbortError')); };
    video.addEventListener(event, done, { once: true }); video.addEventListener('error', fail, { once: true });
    signal?.addEventListener('abort', cancel, { once: true });
    timer = setTimeout(() => { clean(); reject(new Error(`Timed out waiting for video ${event}`)); }, timeout);
  });
}
export async function seekVideo(video, time, signal) {
  if (!Number.isFinite(time) || time < 0 || (Number.isFinite(video.duration) && time > video.duration)) throw new Error('Invalid video seek time');
  if (Math.abs(video.currentTime - time) < .00001 && video.readyState >= 2) return;
  const ready = waitForMedia(video, 'seeked', signal); video.currentTime = time; await ready;
}
