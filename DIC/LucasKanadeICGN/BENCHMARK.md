# Affine IC-GN backend benchmarks

Date: October 28, 2026.

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

## Native wgpu-py and PyO3 milestone

The new `ICGNWGPU` engine executes the same FIR, affine IC-GN, and median
replacement stages directly through wgpu-native/Vulkan. It accepts a
contiguous NumPy `uint8` frame, uploads it to an `r8unorm` texture, and reads
one packed binary subset-state buffer. No PNG, Chromium, DevTools, JSON, or
JavaScript-array conversion is present in this path.

The storage buffers now use aligned `vec4` records for reference values,
inverse Hessians, metadata, and output state. When native subgroup support is
available, reference preparation times both the portable WGSL reduction and a
SPIR-V subgroup reduction on a representative shifted frame. The faster
pipeline becomes the per-frame default; either path can still be forced for
qualification. On this Intel driver the portable path won decisively.

Three five-warm-up, 30-run campaigns on the 512 x 512 `t=0 -> 10` pair
measured the following median campaign:

| Native complete-call measurement | Median | p95 |
|---|---:|---:|
| complete `ICGNWGPU.register` | **1.812 ms** | **2.187 ms** |
| direct target upload | 0.044 ms | — |
| compute plus packed readback | 1.647 ms | — |

The result accepted 900 subsets, recovered
`(1.9999933, -0.9999965) px`, and produced a `2.696e-5 px` truth RMSE.
The automatic preparation probe measured 1.993 ms for the portable pipeline
and 3.109 ms for the subgroup pipeline, so subsequent calls used the portable
solver. The same browser campaign measured 56.203 ms median / 59.117 ms p95
for its complete Python call because that boundary includes PNG and DevTools
transport.

The reproducible UHD pair was then measured in three campaigns of five
warm-ups and 30 complete native Python calls. The table reports the median
campaign summary:

| Native UHD measurement | Median | p95 |
|---|---:|---:|
| complete `ICGNWGPU.register` | **14.621 ms** | **15.748 ms** |
| direct target upload | 0.403 ms | — |
| compute plus packed readback | 13.876 ms | — |

Reference preparation measured 1,036.5 ms and is outside the per-target loop.
The portable solver was forced so this campaign directly qualifies the source
shared with the browser adapter. The UHD result accepted 31,654 subsets and
produced a `3.075e-5 px` truth RMSE. Both the median and p95 remain inside one
60 Hz period, with 0.919 ms of p95 margin. This excludes acquisition and
downstream consumers and is not an end-to-end 4K60 claim.

Forced portable and subgroup runs were compared on all four shipped targets.
All 1,024 statuses were identical for every target, including the t=20 median
replacement. Their jointly accepted displacement fields differed by only
`5.71e-6` to `8.76e-6 px` RMS; the largest component difference was
`1.57e-4 px`.

A follow-up Three.js UHD campaign on the same machine measured 14.462 ms GPU
median and 15.9 ms synchronized browser-wall median. The native 14.621 ms
complete-call median is 8.0% below that browser-wall value while also including
a fresh target upload, packed result readback, and NumPy statistics on every
sample. The boundaries are therefore informative but not identical; the
native compute-plus-readback median alone was 13.876 ms.

The Rust wgpu/PyO3 crate is a parallel host milestone that embeds the exact
same WGSL source and borrows the NumPy frame without a Python byte-string
copy. Its initial 30-run UHD texture-upload probe measured 1.783 ms median and
6.029 ms p95 at the synchronized boundary. The full solver is not hosted in
Rust yet, so this number is an integration baseline rather than a performance
claim. The next migration should be retained only if it improves the complete
call, not merely a microbenchmark.

## Native bounded sequence pipeline

The synchronous `register()` path waits for one packed readback before Python
submits the next target. The new `register_stream()` path maintains a bounded
ring of per-frame target textures, FIR intermediates, solver state, and mapped
readback buffers while sharing the reference data. This removes avoidable GPU
idle gaps without permitting an unbounded queue.

Four balanced UHD campaigns rotated the order of synchronous registration and
pipeline depths 1, 2, and 3. Each method used five warm-ups and 30 measured
targets. Period percentiles refer to the four campaign-average periods.

| Method | Period median | Period p95 | Throughput | Result latency median | Result latency p95 |
|---|---:|---:|---:|---:|---:|
| synchronous `register()` | 14.792 ms | 15.149 ms | 67.6 FPS | 14.671 ms | 15.647 ms |
| mapped pipeline, depth 1 | 14.493 ms | 14.802 ms | 69.0 FPS | 14.447 ms | 14.950 ms |
| **bounded pipeline, depth 2** | **13.380 ms** | **14.203 ms** | **74.7 FPS** | 26.617 ms | 27.883 ms |
| bounded pipeline, depth 3 | 13.077 ms | 14.008 ms | 76.5 FPS | 38.921 ms | 39.939 ms |

All paths accepted 31,654 subsets, produced the same `3.075e-5 px` truth RMSE,
and returned bit-for-bit identical parameter and status fields. Depth two is
the recommended throughput setting. A third slot improves throughput by only
2.3% while adding another 12.3 ms of median latency.

This establishes offline 4K60 processing throughput, not an end-to-end camera
qualification. The two-slot median period leaves 3.287 ms inside a 16.667 ms
frame interval for acquisition handoff and downstream work, but its result is
available roughly one processed frame later.

## Selective native pyramid recovery

The wgpu-py engine now accepts `pyramid_levels=1`, `2`, or `3`. One scale is
the default and retains the original command graph. Additional scales are
prepared once; at run time a five-tap reduction and coarse solve are restricted
to the interpolation closure of fine subsets that failed. The final merge
preserves parameters, status, and iteration count for every initially accepted
fine subset bit for bit.

The deterministic `(4, -2)` pixel recovery campaign measured:

| Resolution / scales | Median | p95 | Accepted | Recovered failures | Truth RMSE |
|---|---:|---:|---:|---:|---:|
| 512 x 512 / 1 | 3.189 ms | 3.682 ms | 634 | 0 | 0.0906 px |
| 512 x 512 / 2 | 5.834 ms | 6.396 ms | 900 | 266 | 0.0760 px |
| 3840 x 2160 / 1 | 36.247 ms | 38.285 ms | 22,401 | 0 | 0.0964 px |
| 3840 x 2160 / 2 | 90.035 ms | 91.359 ms | 31,654 | 9,253 | 0.0811 px |

The 512 x 512 one-scale output hash matches the pre-pyramid wheel exactly.
On the nominal `(2, -1)` UHD throughput case, the new one-scale two-slot run
measured 13.469 ms versus the earlier 13.380 ms, while returning the same
31,654 accepted subsets and `3.075e-5 px` RMSE. Active 4K recovery is therefore
not a 60 FPS path; it remains the optional exception path for preserving
already accepted fine results.
Fixed border subsets with no valid interpolation support are excluded from the
recovery trigger.

## GPU-resident full pyramid

`ICGNWGPUFullPyramid` is the recommended native backend when coarse-to-fine
registration is part of every frame. It uploads only the finest target, builds
all reduced levels with the five-tap filter on the GPU, retains each level's
spline coefficients and solver state, transfers affine seeds on the GPU, and
reads back only the finest packed field. All passes are recorded into one
command buffer and use one queue submission.

The default is two levels with iteration ceilings `(12, 30)` from coarse to
fine. Three levels are opt-in and use `(6, 8, 20)` by default. The same exact
`(4, -2)` validation pair measured:

| Resolution / path | Median | p95 | Accepted | Truth RMSE |
|---|---:|---:|---:|---:|
| 512 x 512 / one scale | 3.309 ms | 3.642 ms | 634 | 0.0906 px |
| 512 x 512 / full pyramid 2 | **1.973 ms** | **2.371 ms** | 900 | 0.000027 px |
| 512 x 512 / full pyramid 3 | 2.420 ms | 3.229 ms | 897 | 0.000025 px |
| 3840 x 2160 / one scale | 32.177 ms | 32.679 ms | 22,401 | 0.0964 px |
| 3840 x 2160 / full pyramid 2 | **14.811 ms** | **15.163 ms** | 31,692 | 0.00123 px |
| 3840 x 2160 / full pyramid 3 | **15.458 ms** | **16.210 ms** | 31,681 | 0.00124 px |

The published campaign placed both full-pyramid UHD p95 values inside the
16.667 ms period. A later two-campaign audit reproduced the two-level result at
14.699--14.939 ms median and 14.917--15.562 ms p95. The same audit measured
16.331--16.353 ms median and 17.286--17.665 ms p95 for three levels. Two levels
are therefore the only configuration with a repeated p95 4K60 compute margin.
Three levels remain above 60 FPS by median and useful for a larger expected
capture range, but are not documented as a p95 60 Hz path. A one-level
full-pyramid run matched the portable `ICGNWGPU` packed output bit for bit.

Reducing the three-level ceilings from `(8, 12, 30)` to `(6, 8, 20)` preserved
all 31,681 accepted subsets and the measured RMSE. A more aggressive descending
schedule `(12, 8, 6)` reduced an alternating campaign median from 15.895 to
15.345 ms, but accepted only 31,260 subsets. It is therefore available through
`iterations_per_level` but is not the robust default. Accepted fine subsets
already converged in 2.45 iterations on average, so fixed reduction,
prefiltering, seed-transfer, and dispatch costs dominate further ceiling cuts.

Reproduce the UHD comparison with:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_full_pyramid.py \
  --width 3840 --height 2160 --levels 2 3 --warmups 2 --repeats 7
```

## Ported SYCL operations

The reference is prepared once using the same Gaussian filter and cubic
B-spline IIR prefilter. The centered reference, nodal gradients, `H^-1`,
`J^T F`, and `sum(J)` are packed into three WebGL textures. Two GPU passes apply
the same separable radius-15 FIR used by the optimized SYCL configuration to
each target.

The registration shader reads each interpolated target pixel once per
iteration. It accumulates the mean, norm, ZNCC, and `J^T g` together, then
applies `-H^-1 b` and the same inverse-compositional update. All iterations are
fused in one fragment draw; invariant Hessian and metadata blocks remain in
registers until the subset finishes. A final pass runs the normalized 3 x 3
median consistency test and replaces flagged subsets.

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

The published fused-iteration shader result was measured in three campaigns of five
warm-ups and 30 samples. The median campaign result is **3.301 ms GPU**
(**4.210 ms p95**) and **10.3 ms browser wall** (**11.9 ms p95**). This is a
12.4% reduction from the previous 3.768 ms paired-backend GPU median and a 38.0%
reduction from its 16.6 ms synchronized browser median. The numerical result is
unchanged within single-precision rounding: 900 accepted subsets and a
`3.873e-5 px` truth RMSE.

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
| GPU-IO | ANGLE / Mesa Intel OpenGL 4.6 | 3.301 ms | 4.210 ms | 10.3 ms | 11.9 ms |
| Three.js/TSL | WebGPU / Intel Xe2-LPG Vulkan | **0.675 ms** | **1.227 ms** | **2.7 ms** | **3.4 ms** |

For this workload, Three.js/TSL uses **20.4% of the GPU time** of GPU-IO, a
79.6% reduction and 4.89 times the throughput. Its browser-side synchronized
median is 73.8% lower.
That wall-time difference is backend synchronization overhead, not an
acquisition-to-result measurement; complete Python calls still include
DevTools transport and result readback.

A later three-campaign audit measured 3.293 ms GPU median for GPU-IO and
0.668 ms for Three.js, with the same 900 accepted subsets and the same RMSE
values shown below. The small timing shift changes the corresponding reduction
to 79.7% and the throughput ratio to 4.93, confirming rather than changing the
published conclusion.

Numerical agreement on the 1,024-subset field is:

- identical status values for all 1,024 subsets;
- 900 accepted subsets with both backends and no median replacements;
- translation difference RMS: `2.04e-5 px`;
- largest absolute translation difference: `2.00e-4 px`;
- Three.js mean translation: `(1.9999928, -0.9999963) px`;
- Three.js truth RMSE: `2.75e-5 px` versus `3.87e-5 px` for GPU-IO.

All four offline targets were also checked after one reference preparation:

| Target | Accepted GPU-IO / Three.js | Replaced | Equal statuses | Translation difference RMS | Maximum difference |
|---|---:|---:|---:|---:|---:|
| t=1 | 900 / 900 | 0 / 0 | 1,024 / 1,024 | `2.33e-5 px` | `1.72e-4 px` |
| t=5 | 900 / 900 | 0 / 0 | 1,024 / 1,024 | `2.35e-5 px` | `2.93e-4 px` |
| t=10 | 900 / 900 | 0 / 0 | 1,024 / 1,024 | `2.04e-5 px` | `2.00e-4 px` |
| t=20 | 634 / 634 | 1 / 1 | 1,024 / 1,024 | `1.87e-5 px` | `1.11e-4 px` |

The four comparisons have identical status fields and exercise small motion,
nominal motion, large motion, rejection, and the median-replacement path.

## Shared WGSL browser backend

The `webgpu-wgsl` adapter is a host-only browser WebGPU implementation. It
loads the exact `shaders/icgn.wgsl` file used by wgpu-py, while Three.js keeps
its faster subgroup-specialized TSL kernel. Both browser WebGPU paths use the
same `ReferencePreparation.js` module and return the same packed result
contract.

Three five-warm-up, 30-run campaigns at 512 × 512 measured a median campaign
of **0.817 ms GPU** (**1.298 ms p95**) and **2.6 ms synchronized browser wall**
(**3.1 ms p95**). The result accepted 900 subsets with a `2.700e-5 px` truth
RMSE. Against wgpu-py on all jointly accepted `t=0 -> 10` subsets, the
translation disagreement was `7.36e-6 px` RMS with a `6.46e-5 px` maximum
component difference; all 1,024 statuses were identical.

All four shipped targets were then compared with the final wheel. Shared WGSL
and wgpu-py had identical status fields for every target, including 634
accepted subsets and one median replacement at `t=20`. Translation differences
ranged from `6.22e-6` to `9.02e-6 px` RMS; the largest component difference
over all targets was `1.39e-4 px`.

At UHD the shared shader measured **16.490 ms GPU median**, **17.068 ms GPU
p95**, **17.9 ms browser-wall median**, and **18.8 ms browser-wall p95**. It
accepted 31,654 subsets with a `3.078e-5 px` truth RMSE. The raw portable
kernel is 14.0% slower than the Three.js subgroup kernel at the median GPU
boundary on this adapter. Source sharing therefore works and remains close to
the 60 Hz compute budget, but Three.js remains the default performance backend.

Chromium must use the native Vulkan adapter for a meaningful comparison. With
only `--enable-unsafe-webgpu`, Chromium selected Google SwiftShader and the
same Three.js pipeline took 29.38 ms median GPU time. The Python engine supplies
the required headless Vulkan flags automatically and reports the selected
adapter through `ICGNWebGL.info()`.

## UHD browser validation and SYCL comparison

All three browser paths were validated at 3840 × 2160 with a reproducible
offline pair generated by `benchmarks/benchmark_4k.py`. The shipped reference
texture is tiled to UHD and shifted by exactly `(2, -1)` pixels. Reference
construction is performed once. Each of three campaigns used five warm-ups
followed by 30 measured optical-flow runs; the table reports the median of the
three campaign summaries.

| Phase | Three.js/TSL | Shared WGSL | GPU-IO |
|---|---:|---:|---:|
| Runtime graph/shader construction | 4.3 ms | 37.9 ms | page startup |
| Reference-dependent construction | 1,144.3 ms | 1,094.7 ms | 1,261.1 ms |
| Complete browser preparation | 1,148.8 ms | 1,132.6 ms | 1,261.1 ms |
| Complete Python preparation call | 1,456.9 ms | 1,451.9 ms | 2,583.3 ms |
| Optical-flow GPU median / p95 | **14.462 / 17.097 ms** | 16.490 / 17.068 ms | 80.738 / 82.522 ms |
| Optical-flow browser wall median / p95 | **15.9 / 18.1 ms** | 17.9 / 18.8 ms | 85.9 / 90.2 ms |

The 240 × 135 grid contains 32,400 subsets. All three solvers accepted 31,654.
Three.js recovered `(1.999991, -0.999997)` pixels in 5.943 iterations with a
`3.076e-5 px` RMSE; shared WGSL returned the same rounded translation in 5.942
iterations with a `3.078e-5 px` RMSE. The startup probe measured a 32-lane
hardware subgroup, so the Three.js 16-lane fast path was active. Compared with
GPU-IO, the fused cooperative Three.js kernel is **5.58× faster**. GPU-IO
recovered `(1.999991, -0.999996)` with 5.947 iterations and a `4.182e-5 px`
RMSE, establishing functional UHD support but not a 4K60 performance path.

The current SYCL source was rebuilt and run on the same pair, GPU, tolerance,
mixed-precision mode, column layout, FIR prefilter, inverse Hessian, and median
replacement settings. Thirty independent measured processes followed five
warm-ups.

| UHD measurement | Three.js/TSL | Shared WGSL | GPU-IO | SYCL mixed/double reduction | SYCL `--float-reduce` |
|---|---:|---:|---:|---:|---:|
| Prefilter + registration GPU median | **14.462 ms** | 16.490 ms | 80.738 ms | 19.099 ms | 18.111 ms |
| Prefilter + registration p95 | **17.097 ms** | 17.068 ms | 82.522 ms | 19.863 ms | 19.062 ms |
| Upload + compute + download median | not included | not included | not included | 19.573 ms | 18.509 ms |
| Accepted subsets | 31,654 | 31,654 | 31,654 | 31,654 | 31,654 |
| Mean iterations | 5.943 | 5.942 | 5.947 | 5.944 | 5.940 |
| Truth RMSE | `3.076e-5 px` | `3.078e-5 px` | `4.182e-5 px` | `2.555e-5 px` | `2.712e-5 px` |

At the comparable compute boundary, optimized Three.js uses **24.3% less GPU
time** than the mixed/double-reduction SYCL configuration (1.32× the
throughput). It uses **20.1% less GPU time** than the numerically closer
float-reduction variant (1.25× the throughput). Browser GPU timing excludes
input upload and output readback; the separate SYCL total row includes both.
This is a backend compute comparison, not an acquisition-to-result or 4K60
claim. WebGPU upload and
pipeline compilation are lazy and are absorbed by the five unreported warm-ups
rather than the steady optical-flow distribution.

## Direct comparison with SYCL

The SYCL binary was rebuilt from `icgn_sycl.cpp`. Thirty independent processes
were measured after three warm-ups, resetting initialization for every
`t=0 -> 10` pair.

| Measurement | Current WebGL2 | SYCL | WebGL/SYCL ratio |
|---|---:|---:|---:|
| GPU prefilter + registration + median | 3.301 ms | 3.651 ms total SYCL | 0.90x |
| Pipeline p95 | 4.210 ms | 4.280 ms | 0.98x |
| Browser synchronization / SYCL total | 10.3 ms | 3.651 ms | 2.82x |
| Mean iterations, accepted subsets | 5.957 | 5.763 | 1.03x |
| Accepted subsets | 900/1,024 | 900/1,024 | identical |
| Vector RMSE | `3.87e-5 px` | `1.12e-5 px` | both subpixel |

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
| t=10 | (2.00, -1.00) | (1.999994, -0.999995) | `3.87e-5 px` | 900/1,024 | 5.957 | 0 |
| t=20 | (4.00, -2.00) | (3.998, -1.997) | 0.0906 px | 634/1,024 | 11.855 | 1 |

The incremental `t=10 -> 20` step, a `(2, -1)` pixel displacement, accepts
900/1,024 subsets in 5.993 mean iterations with a displayed error below
`1e-4 px`.

## Equivalence limits

The implementations are not bit-for-bit equivalent. The Three.js path uses
single-precision WebGPU arithmetic and WebGPU subgroup collectives, while SYCL
uses mixed precision, Intel's larger-register-file extension, and Level Zero
queues. The portable Three.js fallback uses workgroup memory when a subgroup of
at least 16 lanes is unavailable. Raw browser WebGPU and wgpu-py instead load
one canonical portable WGSL file, so formula corrections propagate directly
between those two hosts. This port does not implement Aitken
acceleration, predictive tolerance, temporal/extrapolated initialization,
or retry from the last accepted state. The browser paths do not yet implement
the native image pyramid, and no path includes the H.264 pipeline. These
features do not change the uniform `t=0 -> 10` result used here, but matter for
noisy sequences and larger motions.
