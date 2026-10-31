# Three.js/TSL browser compatibility backend

> **Built on Pierre Margerit's foundational work.** His original
> [`tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio) project,
> GPU-IO experiments and repository structure made this work possible.

This directory is a checkout entry point for live-dic's browser backend.
The canonical implementation is packaged at
[`../python/src/lucas_kanade_icgn/web/three/`](../python/src/lucas_kanade_icgn/web/three/).
The checkout JavaScript module delegates there, so numerical changes belong
in the packaged source rather than in a second implementation.

## Scope and execution

The backend supports translation and affine IC-GN, B-spline/bilinear/Keys-cubic
interpolation and square widths 8, 16 and 32 with stride equal to size.
It is single-scale and rejects ROI and pyramid options it cannot implement.
Quadratic solving and native temporal-pyramid features belong to the shared
WGSL paths, not this TSL implementation.

On WebGPU it uses a subgroup solver when qualified, otherwise a portable
workgroup-memory solver. A discovery probe nominates the subgroup width;
reference preparation verifies that width inside the actual solver pipeline.
A mismatch selects the portable fallback. Solver diagnostics report the
observed mapping. Native `solver_workgroup_size` choices do not configure
this independent browser solver.

Without WebGPU, Three.js selects a serial WebGL2 transform-feedback solver.
It shares the TSL mathematical helpers and result layout, but does not emulate
workgroup barriers and is not qualified as a 4K60 path. The GPU target
prefilter and optional UOD/median pass remain separate from the local fused
IC-GN iteration kernel. Direct interpolators omit target prefiltering.

## Python compatibility entry point

```python
from lucas_kanade_icgn import ICGNWebGL

with ICGNWebGL() as engine:
    result = engine.register_pair(reference, target)
```

The application supplies `reference` and `target`. Force WebGL2 with
`ICGNWebGL(force_three_webgl=True)` when qualifying that compatibility path.
Chromium runs headlessly. Targets and packed fields use binary transport,
but Python and Chromium still have separate memory. For resident NumPy
streams use the native full-pyramid engine; a JavaScript streaming supervisor
is not provided here.

Three.js r186 and its MIT license are vendored under `vendor/` for offline
use. Other notices are in [THIRD_PARTY.md](../THIRD_PARTY.md).
See [backend selection](../docs/explanation/backend-selection.rst),
[GPU organization](../docs/explanation/gpu-optimization.rst), and
[performance evidence](../docs/reference/performance.rst).
