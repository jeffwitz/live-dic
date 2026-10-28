# Affine IC-GN backend benchmarks

Date: October 27, 2026. Three.js/TSL comparison added October 28, 2026.

## Common configuration

- GPU: Intel Lunar Lake `[8086:64a0]`, `xe` kernel driver;
- WebGL: Chromium 155.0.8059.79, ANGLE/Mesa Intel LNL OpenGL 4.6;
- SYCL: DPC++ `-O2`, Level Zero backend, same GPU;
- images: raw 512 x 512 luma frames from `seed2026101102_translation`;
- measured pair: `t=0 -> 10`, true translation `(2, -1)` pixels;
- grid: 32 x 32, or 1,024 subsets at 16-pixel spacing;
- subsets: 16 x 16 pixels, with 900 usable interior subsets;
- prefilter: `sigma=1` Gaussian and cubic B-spline coefficients;
- target interpolation: 4 x 4 cubic B-spline;
- six-parameter affine IC-GN with a constant inverse Hessian;
- `1e-3` tolerance, minimum ZNCC `0.8`, at most 30 iterations;
- median consistency test and replacement enabled;
- no H.264 decoding or processing.

## Ported SYCL operations

The reference is prepared once using the same Gaussian filter and cubic
B-spline IIR prefilter. The centered reference, nodal gradients, `H^-1`,
`J^T F`, and `sum(J)` are packed into three WebGL textures. Two GPU passes apply
the same separable radius-15 FIR used by the optimized SYCL configuration to
each target.

The registration shader reads each interpolated target pixel once per
iteration. It accumulates the mean, norm, ZNCC, and `J^T g` together, then
applies `-H^-1 b` and the same inverse-compositional update. A final pass runs
the normalized 3 x 3 median consistency test and replaces flagged subsets.

## WebGL2 performance

Each campaign uses three warm-up runs followed by 30 measured runs. GPU timing
comes from `EXT_disjoint_timer_query_webgl2`. The FIR is measured separately;
solver time is estimated by subtracting it from the complete GPU pipeline to
avoid inserting a timer-query barrier between the two stages.

| Warm campaign | Median FIR GPU | Estimated solver | Median FIR + solver | FIR + solver p95 |
|---|---:|---:|---:|---:|
| 1 | 0.221 ms | 3.556 ms | 3.777 ms | 4.238 ms |
| 2 | 0.222 ms | 3.543 ms | 3.765 ms | 6.810 ms |
| 3 | 0.229 ms | 3.565 ms | 3.795 ms | 4.376 ms |

The representative warm result is **3.78 ms of GPU time per pair**, or about
**265 pairs/s**. Two campaigns have a 4.24-4.38 ms p95; one contains a 6.81 ms
excursion. Compilation plus the first pass takes approximately 79-105 ms with
a populated Chromium cache and 329 ms on the first compilation of the shader.

An additional campaign covering target upload, FIR, solver, and result readback
reports a **17.6 ms median wall time** and **19.1 ms p95**. Reading the two result
textures alone takes approximately 1.4-2.8 ms. Transfers, WebGL/JavaScript
synchronization, and timer-query waits explain most of the difference from GPU
time.

The headless Python interface was validated independently with three warm-ups
and 30 repetitions on the same pair. It measured a **3.908 ms median GPU time**
(**5.702 ms p95**) and a **16.9 ms median wall time** (**19.5 ms p95**). It
returned 900 accepted subsets, a mean translation of
`(1.999993, -0.999995)` pixels, and a `3.819e-5` pixel RMSE. DevTools transport
is outside the GPU timer query.

After the API cleanup, English documentation pass, and added validation, two
30-run non-regression campaigns measured **3.967 ms** and **3.859 ms** median GPU
time, with **16.8 ms** and **16.9 ms** median wall time. Their midpoint is
3.913 ms, effectively unchanged from the 3.908 ms baseline. The displacement,
accepted-subset count, and iteration count remained identical.

## Three.js/TSL WebGPU backend

The independent Three.js r186 backend implements the same radius-15 target
prefilter, affine IC-GN iterations, and eight-neighbour median-replacement pass
with TSL storage-buffer compute kernels. On this adapter, all iterations are
fused into one dispatch and 16 active lanes cooperate through WebGPU subgroup
reductions. A startup probe selects this path only when one subgroup contains
all 16 lanes; otherwise a workgroup-memory implementation is used. The
reference-dependent quantities are prepared with the same double-precision
JavaScript equations as the GPU-IO implementation. No code from Strobox is
used.

Three paired campaigns ran both browser backends on the same Intel Lunar Lake
GPU, image pair, Python driver, and process-launch environment. Each campaign
uses five warm-ups followed by 30 measured runs. The table reports the median
of the three campaign summaries. GPU times include prefiltering, IC-GN, and
median replacement; image decoding, result readback, and DevTools transport
are excluded from both samples.

| Backend | Native device path | GPU median | GPU p95 | Browser wall median | Browser wall p95 |
|---|---|---:|---:|---:|---:|
| GPU-IO | ANGLE / Mesa Intel OpenGL 4.6 | 3.768 ms | 4.362 ms | 16.6 ms | 18.4 ms |
| Three.js/TSL | WebGPU / Intel Xe2-LPG Vulkan | **0.672 ms** | **1.225 ms** | **2.7 ms** | **3.5 ms** |

For this workload, Three.js/TSL uses **17.8% of the GPU time** of GPU-IO, an
82.2% reduction. The three per-campaign ratios were 23.1%, 17.4%, and 17.2%.
Its browser-side synchronized median is 83.7% lower. Three.js GPU p95 ranged
from 1.174 to 1.227 ms across the campaigns; GPU-IO ranged from 4.253 to
4.613 ms.
That wall-time difference is backend synchronization overhead, not an
acquisition-to-result measurement; complete Python calls still include
DevTools transport and result readback.

Numerical agreement on the 1,024-subset field is:

- identical status values for all 1,024 subsets;
- 900 accepted subsets with both backends and no median replacements;
- translation difference RMS: `1.90e-5 px`;
- largest absolute translation difference: `1.42e-4 px`;
- Three.js mean translation: `(1.9999928, -0.9999963) px`;
- Three.js truth RMSE: `2.75e-5 px` versus `3.82e-5 px` for GPU-IO.

All four offline targets were also checked after one reference preparation:

| Target | Accepted GPU-IO / Three.js | Replaced | Equal statuses | Translation difference RMS | Maximum difference |
|---|---:|---:|---:|---:|---:|
| t=1 | 900 / 900 | 0 / 0 | 1,024 / 1,024 | `1.72e-5 px` | `1.49e-4 px` |
| t=5 | 900 / 900 | 0 / 0 | 1,024 / 1,024 | `1.73e-5 px` | `2.80e-4 px` |
| t=10 | 900 / 900 | 0 / 0 | 1,024 / 1,024 | `1.70e-5 px` | `1.92e-4 px` |
| t=20 | 634 / 634 | 1 / 1 | 1,023 / 1,024 | `1.51e-5 px` | `2.16e-4 px` |

The sole status mismatch at `t=20` is a convergence-boundary classification;
it does not change the accepted or median-replaced counts. The four comparisons
therefore exercise small motion, nominal motion, large motion, rejection, and
the median-replacement path.

Chromium must use the native Vulkan adapter for a meaningful comparison. With
only `--enable-unsafe-webgpu`, Chromium selected Google SwiftShader and the
same Three.js pipeline took 29.38 ms median GPU time. The Python engine supplies
the required headless Vulkan flags automatically and reports the selected
adapter through `ICGNWebGL.info()`.

## UHD Three.js validation and SYCL comparison

The dynamic Three.js path was validated at 3840 × 2160 with a reproducible
offline pair generated by `benchmarks/benchmark_4k.py`. The shipped reference
texture is tiled to UHD and shifted by exactly `(2, -1)` pixels. Reference
construction is performed once. Each of three campaigns used five warm-ups
followed by 30 measured optical-flow runs; the table reports the median of the
three campaign summaries.

| Phase | Three.js/TSL result |
|---|---:|
| Runtime and TSL graph construction | 5.1 ms |
| Reference-dependent construction | 996.3 ms |
| Complete browser preparation | 1,003.1 ms |
| Complete Python preparation call | 1,291.3 ms |
| Optical-flow GPU median / p95 | **13.341 / 14.405 ms** |
| Optical-flow browser wall median / p95 | **14.7 / 16.0 ms** |

The 240 × 135 grid contains 32,400 subsets. The subgroup solver accepted
31,654, recovered `(1.999991, -0.999997)` pixels, required 5.943 iterations on
average, and produced a vector RMSE of `3.076e-5 px`. The startup probe measured
a 32-lane hardware subgroup, so the 16-lane fast path was active. Compared with
the earlier one-dispatch-per-iteration TSL implementation (81.817 ms median),
the fused cooperative kernel is **6.13× faster**.

The current SYCL source was rebuilt and run on the same pair, GPU, tolerance,
mixed-precision mode, column layout, FIR prefilter, inverse Hessian, and median
replacement settings. Thirty independent measured processes followed five
warm-ups.

| UHD measurement | Three.js/TSL | SYCL mixed/double reduction | SYCL `--float-reduce` |
|---|---:|---:|---:|
| Prefilter + registration GPU median | **13.341 ms** | 19.099 ms | 18.111 ms |
| Prefilter + registration p95 | **14.405 ms** | 19.863 ms | 19.062 ms |
| Upload + compute + download median | not included | 19.573 ms | 18.509 ms |
| Accepted subsets | 31,654 | 31,654 | 31,654 |
| Mean iterations | 5.943 | 5.944 | 5.940 |
| Truth RMSE | `3.076e-5 px` | `2.555e-5 px` | `2.712e-5 px` |

At the comparable compute boundary, optimized Three.js is **30.1% faster**
than the mixed/double-reduction SYCL configuration (1.43× the throughput). It
is still **26.3% faster** than the numerically closer float-reduction variant
(1.36× the throughput). Three.js GPU timing excludes input upload and output
readback; the separate SYCL total row includes both. This is a backend compute
comparison, not an acquisition-to-result or 4K60 claim. WebGPU upload and
pipeline compilation are lazy and are absorbed by the five unreported warm-ups
rather than the steady optical-flow distribution.

## Direct comparison with SYCL

The SYCL binary was rebuilt from `icgn_sycl.cpp`. Thirty independent processes
were measured after three warm-ups, resetting initialization for every
`t=0 -> 10` pair.

| Measurement | Equivalent WebGL2 | SYCL | WebGL/SYCL ratio |
|---|---:|---:|---:|
| GPU registration | 3.55 ms estimated | 2.573 ms | 1.38x |
| GPU prefilter + registration | 3.78 ms | 3.651 ms total SYCL | 1.04x |
| Pipeline p95 | 4.24-4.38 ms typical | 4.280 ms | approximately 1.00x |
| End-to-end with CPU readback | 17.6 ms | 3.651 ms | 4.82x |
| Mean iterations, accepted subsets | 5.957 | 5.763 | 1.03x |
| Accepted subsets | 900/1,024 | 900/1,024 | identical |
| Vector RMSE | `3.82e-5 px` | `1.12e-5 px` | both subpixel |

GPU compute cost is now comparable to SYCL for this case. The browser
end-to-end path remains slower because of synchronization and WebGL readback.
The SYCL `total_ms` includes overlapped upload, prefiltering, registration, and
readback. WebGL GPU timing excludes the initial transfer and final readback, so
the table reports end-to-end time separately.

## WebGL2 accuracy

| Target from t=0 | Truth (px) | Measured mean (px) | RMSE | Accepted | Mean iterations | Replaced |
|---|---:|---:|---:|---:|---:|---:|
| t=1 | (0.20, -0.10) | (0.191, -0.093) | 0.0186 px | 900/1,024 | 3.994 | 0 |
| t=5 | (1.00, -0.50) | (1.000, -0.500) | 0.0170 px | 900/1,024 | 5.197 | 0 |
| t=10 | (2.00, -1.00) | (1.999993, -0.999995) | `3.82e-5 px` | 900/1,024 | 5.957 | 0 |
| t=20 | (4.00, -2.00) | (3.998, -1.997) | 0.0906 px | 634/1,024 | 11.855 | 1 |

The incremental `t=10 -> 20` step, a `(2, -1)` pixel displacement, accepts
900/1,024 subsets in 5.993 mean iterations with a displayed error below
`1e-4 px`.

## Equivalence limits

The implementations are not bit-for-bit equivalent. The Three.js path uses
single-precision WebGPU arithmetic and WebGPU subgroup collectives, while SYCL
uses mixed precision, Intel's larger-register-file extension, and Level Zero
queues. The portable Three.js fallback uses workgroup memory when a subgroup of
at least 16 lanes is unavailable. This port does not implement Aitken
acceleration, predictive tolerance, temporal/extrapolated initialization,
retry from the last accepted state, a pyramid, or the H.264 pipeline. These
features do not change the uniform `t=0 -> 10` result used here, but matter for
noisy sequences and larger motions.
