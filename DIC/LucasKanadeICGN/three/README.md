# Three.js/TSL affine IC-GN backend

> **Built on Pierre Margerit's foundational work.** This backend extends
> [Pierre Margerit's original `tests_gpuio`
> project](https://github.com/MargeritPierre/tests_gpuio); its GPU-IO
> experiments and repository structure made this work possible.

This directory contains an independent WebGPU implementation of DIC-RT's
six-parameter affine inverse-compositional Gauss-Newton algorithm. It does not
contain camera, video-codec, or graphical-interface code, and it does not copy
the DIC implementation from Strobox.

The numerical pipeline is:

1. prepare the centered reference, gradients, inverse Hessian, `J^T F`, and
   `sum(J)` once on the CPU;
2. apply the radius-15 separable Gaussian/B-spline FIR to each target on the
   GPU;
3. run the affine IC-GN iterations in one fused compute dispatch, with 16
   cooperating lanes per subset, convergence, and ZNCC rejection;
4. apply the eight-neighbour median consistency pass on the GPU;
5. read the six affine parameters, status, and iteration count.

`headless.html` exposes the same `window.ICGNHeadless` contract as the GPU-IO
runtime. The supported entry points are `info`, `configure`, `setReference`,
`register`, and `benchmark`. Python users should select this runtime through:

```python
from lucas_kanade_icgn import ICGNWebGL

with ICGNWebGL() as engine:
    result = engine.register_pair(reference, target)
```

Three.js r186 is vendored under `vendor/` so the backend runs offline. The
corresponding MIT licence is `vendor/THREE-LICENSE.txt`. Accuracy and timing
results are recorded in [`../BENCHMARK.md`](../BENCHMARK.md).

At startup, the runtime measures the subgroup width selected for a 16-thread
workgroup. It uses hardware subgroup reductions when all 16 active lanes fit in
one subgroup and otherwise selects the portable workgroup-memory solver.
