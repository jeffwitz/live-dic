# Changelog

Notable user-facing changes are recorded here. Unreleased entries move under a
version heading when a release is published.

## Unreleased

### Breaking changes

- Removed the `ICGNWebGL(backend="gpuio")` backend and the corresponding
  `--backend gpuio` CLI option.
- Removed the Three.js WebGL2 fallback. The `ICGNWebGL` Three.js/TSL backend
  requires browser WebGPU and now reports an explicit error when it is
  unavailable.
