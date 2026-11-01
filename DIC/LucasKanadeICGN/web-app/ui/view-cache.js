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
