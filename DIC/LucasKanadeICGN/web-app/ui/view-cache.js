/** Derived values/statistics belong to a completed analysis, never a live frame.
 * Weak keys release cached views with their analysis. Palette, range and vector
 * styling reuse arrays; a different field or processing recipe has a new key.
 */
export function createViewCache(derive, summarize) {
  const sources = new WeakMap();
  let computations = 0, hits = 0;
  return {
    get(analysis, component) {
      let views = sources.get(analysis);
      if (!views) { views = new Map(); sources.set(analysis, views); }
      if (views.has(component)) { hits++; return views.get(component); }
      const values = derive(analysis, component);
      const view = { ...values, statistics: summarize(values.values, values.valid) };
      views.set(component, view); computations++;
      return view;
    },
    report() { return { computations, hits }; },
  };
}

/** Small LRU of derived arrays only: never retain raw fields, masks or rasters.
 * Keys include the field identity and complete recipe. Long recordings remain
 * disk-backed; this separate cache has both a byte and an entry limit.
 */
export function createAnalysisCache({ maxBytes = 32 * 1024 * 1024, maxEntries = 8 } = {}) {
  const entries = new Map();
  let bytes = 0, hits = 0, misses = 0;
  function remove(key) {
    const entry = entries.get(key);
    if (entry) { bytes -= entry.bytes; entries.delete(key); }
  }
  return {
    get(key) {
      const entry = entries.get(key);
      if (!entry) { misses++; return null; }
      entries.delete(key); entries.set(key, entry); hits++;
      return entry.value;
    },
    set(key, result) {
      remove(key);
      const { field, ...value } = result;
      const buffers = new Set([...Object.values(value.fields), value.valid, value.strainValid].map(array => array.buffer));
      const size = [...buffers].reduce((sum, buffer) => sum + buffer.byteLength, 0);
      if (size > maxBytes || maxEntries < 1) return;
      entries.set(key, { value, bytes: size }); bytes += size;
      while (bytes > maxBytes || entries.size > maxEntries) remove(entries.keys().next().value);
    },
    clear() { entries.clear(); bytes = 0; },
    report() { return { entries: entries.size, bytes, maxBytes, maxEntries, hits, misses }; },
  };
}
