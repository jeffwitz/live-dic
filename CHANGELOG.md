# Changelog

Notable user-facing changes are recorded here. Unreleased entries move under a
version heading when a release is published.

## Unreleased

### Naming

- Use `live-dic` as the Python distribution and browser-control command,
  `live_dic` as the public import package, and `ICGNBrowser` for the
  Python-controlled browser engine. The former `lucas_kanade_icgn` import,
  `ICGNWebGL` class and `icgn-webgl` command remain compatibility aliases.
- The distribution rename is made before the first PyPI release; no old
  distribution name is published as a second package.

### Licensing and citation

- Licensed first-party live-dic source, documentation, examples and analytic
  qualification fixtures under GPL-2.0-or-later. Added SPDX package metadata,
  shipped license text and a `CITATION.cff`; third-party components retain
  their own terms.

### Breaking changes

- Removed the `ICGNWebGL(backend="gpuio")` backend and the corresponding
  `--backend gpuio` CLI option.
- Removed the Three.js WebGL2 fallback. The browser-control engine's Three.js/TSL backend
  requires browser WebGPU and now reports an explicit error when it is
  unavailable.
