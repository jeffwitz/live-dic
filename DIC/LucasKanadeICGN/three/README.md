# Three.js/TSL browser compatibility backend

> **Methodology acknowledgment — [Pierre Margerit](https://github.com/MargeritPierre).**
> His earlier [GPU-DIC prototype](https://github.com/MargeritPierre/tests_gpuio)
> and [Three.js/WebGPU camera prototype](https://github.com/MargeritPierre/Strobox)
> informed the methodological and browser-GPU exploration. The Three.js/TSL
> solver maintained here is a separate live-dic implementation, not a copy of
> Strobox source code.

This directory is a checkout entry point for live-dic's browser backend.
The canonical implementation is packaged at
[`../python/src/live_dic/web/three/`](../python/src/live_dic/web/three/).
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

Three.js requires WebGPU; there is no WebGL2 fallback. On WebGPU it uses a
subgroup solver when qualified, otherwise a portable workgroup-memory solver.
The GPU target prefilter and optional UOD/median pass remain separate from the
local fused IC-GN iteration kernel. Direct interpolators omit target
prefiltering.

## Python compatibility entry point

```python
from live_dic import ICGNBrowser

with ICGNBrowser() as engine:
    result = engine.register_pair(reference, target)
```

The application supplies `reference` and `target`. Chromium runs headlessly.
Targets and packed fields use binary transport, but Python and Chromium still
have separate memory. For resident NumPy streams use the native full-pyramid
engine; a JavaScript streaming supervisor is not provided here.

Three.js r186 and its MIT license are vendored under `vendor/` for offline
use. Other notices are in [THIRD_PARTY.md](../THIRD_PARTY.md).
See [backend selection](../docs/explanation/backend-selection.rst),
[GPU organization](../docs/explanation/gpu-optimization.rst), and
[performance evidence](../docs/reference/performance.rst).
