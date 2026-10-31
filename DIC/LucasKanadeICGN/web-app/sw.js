// Replaced by tools/build_pages.py; one cache version per source revision.
const VERSION = '__COMMIT__';
const PREFIX = `live-dic-${encodeURIComponent(new URL(self.registration.scope).pathname)}-`;
const CACHE = `${PREFIX}${VERSION}`;
const ASSETS = __PRECACHE__;
self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(ASSETS)));
});
self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys().then((names) => Promise.all(names.filter((n) => n.startsWith(PREFIX) && n !== CACHE).map((n) => caches.delete(n)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET' || new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cached = await (await caches.open(CACHE)).match(event.request, { ignoreVary: true, ignoreSearch: event.request.mode === 'navigate' });
    if (!cached) return fetch(event.request);
    const range = event.request.headers.get('Range');
    if (range && new URL(event.request.url).pathname.endsWith('.webm')) {
      const match = /^bytes=(\d+)-(\d*)$/.exec(range);
      if (match) {
        const blob = await cached.blob(), start = Number(match[1]), end = match[2] ? Math.min(Number(match[2]), blob.size - 1) : blob.size - 1;
        if (start >= blob.size || start > end) return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${blob.size}` } });
        return new Response(blob.slice(start, end + 1), { status: 206, headers: {
          'Content-Type': 'video/webm', 'Accept-Ranges': 'bytes', 'Content-Length': String(end - start + 1), 'Content-Range': `bytes ${start}-${end}/${blob.size}`,
        } });
      }
    }
    return cached;
  })());
});
