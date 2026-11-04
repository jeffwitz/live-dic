# Browser propagation workgroup controls — 2026-11-04

The explicit width of 32 improves this browser workload without changing status
arrays or analytical coverage. **It does not achieve 4K60 on this boundary.**
Default width 16 and the fixed Live preset remain unchanged.

Original UHD tensile reference/frame 40; translation 32/16, B-spline, uniform,
two fixed levels (8/16), tolerance 0.005, ZNCC 0.8, raw full readback and two
failed-only propagation passes with empty-work stopping. Two engines share one
injected Intel LNL device and alternate every frame. They retain independent
reference/state resources, unlike the native pipeline-swap control. Default
WebGPU limits are used (8 storage buffers, 16 KiB workgroup storage, 256 maximum
invocations). No optional feature is required by this width selection.

| Width | Median | p95 | p99 | Fields / sum of call durations |
|---|---:|---:|---:|---:|
| 16 | 38.70 ms | 40.60 ms | 41.40 ms | 26.05/s |
| 32 | 33.65 ms | 35.40 ms | 36.30 ms | 30.09/s |

The boundary is a resident decoded ImageBitmap target to a full typed field and
statistics inside the browser. It includes external-image upload/luminance,
solving and readback; excludes decoding, reference preparation, display,
storage and per-frame Python/DevTools round trips. This is not the native
NumPy boundary or the PWA's GPU-only/temporal Live preset. Do not compare these
absolute values to capability-probe zero-motion timings or label them camera FPS.

Twelve 100-pair AB/BA blocks, eight warm-ups per variant/block. A separate 16/16
reference run precedes the candidate; its margin is 0.31750 ms. The median
per-block paired saving is 5.100 ms with block-bootstrap 95% interval
[4.975, 5.200] ms (10,000 samples, seed 1729), beyond that fixed margin.
Commands, exact build revision/clean state, limits, inputs and every timing are
retained. No project test, build or other GPU campaign ran concurrently;
unrelated desktop processes were not controlled.

Both variants accept 31,284 grid positions and cover all 30,916 analytical
interior centres. Vector RMSE is about 0.0101684 px and p95 error 0.0172273 px.
Status arrays agree exactly, maximum parameter difference is 0.0002412 px under
the predeclared 0.001 bound, and two iteration values differ. Translation
gradients remain exactly zero. No scoring, convergence or accuracy gate changed.

A separate one-block, 20-pair gray-input diagnostic uses a cached Uint8Array
target with the same prepared reference. Its descriptive medians are 24.20 ms
(width 16) and 18.85 ms (32). It excludes the one-time gray extraction and is
not a qualified percentile campaign or isolated estimate of upload cost.
Different input routes, timing boundaries and campaign conditions must remain
visible; the ImageBitmap result cannot be substituted for a resident NumPy rate.
Last-result wall timing snapshots are retained and are not GPU timestamps.

Original reports are gzipped without rewriting. criteria.json was written
before candidate evaluation; summary.json records the block analysis.
sources.tar.gz contains the exact scripts, with a machine-specific ROOT path
to adjust for another checkout. The raw browser core and canonical WGSL come
from the clean build revision recorded in each report. No fixture regeneration,
new runtime dependency, width autotuning or default change is involved.
