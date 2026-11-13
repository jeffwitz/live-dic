let contractPromise;

export function loadMatcherContract() {
  contractPromise ??= fetch(new URL("./python/src/wgpu_matching/matcher-contracts.json", import.meta.url))
    .then(response => {
      if (!response.ok) throw new Error(`Unable to load matcher contract (${response.status}).`);
      return response.json();
    });
  return contractPromise;
}

export async function browserDefaults(algorithm) {
  const contract = await loadMatcherContract();
  const spec = contract.algorithms[algorithm];
  if (!spec) throw new Error(`No shared contract for matcher ${algorithm}.`);
  return Object.fromEntries(Object.entries(spec.defaults).map(([name, value]) => {
    const key = spec.option_keys[name]?.javascript;
    if (!key) throw new Error(`Missing JavaScript option mapping for ${algorithm}.${name}.`);
    return [key, value];
  }));
}

function validateBrowserOptions(options, algorithm, spec) {
  for (const [parameter, rule] of Object.entries(spec.constraints ?? {})) {
    const key = spec.option_keys[parameter]?.javascript;
    const value = options[key];
    if (rule.type === "integer") {
      if (!Number.isSafeInteger(value) || value < rule.minimum
          || (rule.maximum !== undefined && value > rule.maximum)) {
        throw new RangeError(`${algorithm}.${parameter} must be an integer in the configured range.`);
      }
    } else if (rule.type === "number" || rule.type === "optional_number") {
      if (rule.type === "optional_number" && value == null) continue;
      if (!Number.isFinite(value) || value < rule.minimum
          || (rule.maximum !== undefined && value > rule.maximum)) {
        throw new RangeError(`${algorithm}.${parameter} must be a finite number in the configured range.`);
      }
    } else if (rule.type === "enum" && !rule.values.includes(value)) {
      throw new RangeError(`${algorithm}.${parameter} must be one of ${rule.values.join(", ")}.`);
    } else if (rule.type === "points" && value != null) {
      if (!Array.isArray(value) || value.length > rule.maximum_count) {
        throw new RangeError(`${algorithm}.seed_points must contain at most ${rule.maximum_count} points.`);
      }
      for (const [index, point] of value.entries()) {
        const x = Array.isArray(point) ? point[0] : point?.x;
        const y = Array.isArray(point) ? point[1] : point?.y;
        if (typeof x !== "number" || typeof y !== "number"
            || !Number.isFinite(x) || !Number.isFinite(y)) {
          throw new RangeError(`${algorithm}.seed_points[${index}] must contain finite x and y coordinates.`);
        }
      }
    }
  }
  return options;
}

export async function applyBrowserDefaults(target, algorithm) {
  const contract = await loadMatcherContract();
  const spec = contract.algorithms[algorithm];
  if (!spec) throw new Error(`No shared contract for matcher ${algorithm}.`);
  const defaults = Object.fromEntries(Object.entries(spec.defaults).map(([name, value]) => {
    const key = spec.option_keys[name]?.javascript;
    if (!key) throw new Error(`Missing JavaScript option mapping for ${algorithm}.${name}.`);
    return [key, value];
  }));
  return validateBrowserOptions({ ...defaults, ...target }, algorithm, spec);
}
