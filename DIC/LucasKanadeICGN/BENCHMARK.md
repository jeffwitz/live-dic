# Current IC-GN performance

Last updated: October 29, 2026.

This file contains only measurements for the current kernels. Earlier campaign
snapshots were removed because their layouts, filter costs, and subgroup
occupancy no longer describe the implementation. Re-run the commands below
when a kernel, driver, browser, or measurement boundary changes.

## Current backend policy

- `ICGNWGPUFullPyramid()` with two levels is the recommended native
  coarse-to-fine path.
- `ICGNWebGL(backend="webgpu-wgsl")` runs the same two-level full-pyramid WGSL
  graph in Chromium.
- `ICGNWebGL()` remains the default browser entry point and uses the optimized
  Three.js/TSL single-scale solver with automatic WebGL2 fallback.
- Both native engines expose bounded streaming. `ICGNWGPU()` is the
  single-scale engine; its optional selective recovery is not the normal
  pyramid path.
- GPU-IO is deprecated and has no current performance row.

## Validation machine

- CPU: Intel Core Ultra 5 228V;
- GPU: Intel Lunar Lake integrated graphics `[8086:64a0]`;
- kernel driver: `xe`;
- operating system: Ubuntu 24.04.5 LTS, Linux 7.0.0-38-generic;
- browser: Chromium 155.0.8059.79 snap;
- browser WebGPU: Three.js r186, Vulkan;
- native WebGPU: wgpu-py 0.32.0, wgpu-native Vulkan.

The offline validation texture is tiled to 3840 × 2160, producing 240 × 135,
or 32,400, 16 × 16 subsets. No camera or H.264 stage is included.

## Primary UHD image-to-field stream boundary

This is the deployable Python boundary used for performance claims. A
persistent engine receives an already acquired, contiguous `uint8` NumPy image
and the timer stops only when the complete `(135, 240, 6)` NumPy parameter
field, status, iterations, and statistics are available. Reference preparation,
camera acquisition, and decoding are excluded.

The sequence cycles the four distinct qualified `frame_001`, `frame_005`,
`frame_010`, and `frame_020` fixtures. Every row uses three changing-frame
warm-ups followed by 12 measured frames. Execution is synchronous, so the
reported latency is also the result age and the inverse stream period. The
browser rows report the median of three campaigns after the binary transport
change.

| Backend and model | Median / p95 image-to-field latency | Sustained throughput |
|---|---:|---:|
| Native full pyramid, 2 levels, affine | **11.920 / 13.072 ms** | **83.37 FPS** |
| Native full pyramid, 2 levels, translation | **10.526 / 11.170 ms** | **94.03 FPS** |
| Raw browser WGSL, 2 levels, affine | **40.873 / 44.782 ms** | **24.03 FPS** |
| Raw browser WGSL, 2 levels, translation | **40.213 / 45.024 ms** | **24.21 FPS** |
| Three.js/TSL, 1 scale, affine | **43.297 / 59.107 ms** | **21.74 FPS** |
| Three.js/TSL, 1 scale, translation | **39.966 / 45.706 ms** | **24.54 FPS** |

The native and raw-browser WGSL rows execute the same two-level algorithm. The
browser API now keeps the persistent Chromium process but removes per-frame
PNG, data URLs, DevTools payloads, image decoding, and JSON fields. A
same-origin long poll transfers the contiguous grayscale bytes unchanged and
one packed eight-float state returns as binary. Three.js uploads four pixels
per 32-bit word and expands them on the GPU, avoiding its former 33 MB
JavaScript float conversion at UHD. The one-time reference setup still uses
the image path and remains excluded from this boundary.

This does not make Chromium memory and NumPy memory identical. The UHD frame
still crosses into Chromium, then into a GPU resource, and the result crosses
back. In the affine campaign the browser body copy took about 8.5--9.1 ms
median and the synchronized GPU graph about 16--19 ms. The complete browser
paths therefore sustain about 22--24 FPS, not 60 FPS. Native wgpu remains the
real-time NumPy path.

The native path uploads the NumPy bytes directly to an `r8unorm` texture and
reads one packed result buffer. The current timing contract separately reports
input normalization, upload, encoding, queue submission, host completion wait,
mapped-buffer copy, and NumPy result construction. The completion wait is not
an isolated GPU timestamp.

Reproduce this boundary with:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
  --width 3840 --height 2160 --frames 12 --warmups 3 --campaigns 3 \
  --backend three-tsl --backend webgpu-wgsl \
  --motion-model translation
```

## Native full-pyramid bounded stream

`ICGNWGPUFullPyramid.register_stream()` keeps all mutable resources independent
for every in-flight target: target textures, pyramid reductions, solver states,
coarse-to-fine seed buffers, and final readbacks. Reference data and compiled
pipelines remain shared. Contiguous NumPy bytes are uploaded directly; this
path contains no PNG, Chromium, DevTools, or JSON conversion.

The current campaign used four runs, each with eight warm-ups and 50 changing
UHD frames. The table reports the median campaign. Result age starts when the
iterator accepts an image and ends after the complete NumPy field and
statistics exist. Stream period is total wall time divided by 50.

| Model and targets in flight | Median / p95 result age | Stream period | Throughput |
|---|---:|---:|---:|
| Affine, depth 1 | **11.478 / 12.400 ms** | 11.561 ms | 86.53 FPS |
| Affine, depth 2 | 21.408 / 22.644 ms | **10.735 ms** | **93.16 FPS** |
| Affine, depth 3 | 31.894 / 39.452 ms | 10.530 ms | 94.97 FPS |
| Translation, depth 1 | **10.331 / 11.164 ms** | 10.373 ms | 96.42 FPS |
| Translation, depth 2 | 18.888 / 19.827 ms | **9.474 ms** | **105.58 FPS** |
| Translation, depth 3 | 29.705 / 35.718 ms | 9.420 ms | 106.16 FPS |

Depth one minimizes result age. Depth two is the recommended throughput
setting: it gains 7.7% affine throughput and 9.5% translation throughput while
approximately doubling result age. The third slot adds little throughput and
raises median age beyond 32 ms.

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
  --backend native-pyramid --width 3840 --height 2160 \
  --campaigns 4 --frames 50 --warmups 8 \
  --pipeline-depth 2 --motion-model affine
```

## Diagnostic UHD kernel audit

These rows isolate kernels or repeat a target for optimization work; they are
not interchangeable image-to-field performance claims. Each row uses eight
warm-ups and 50 measured registrations. Native rows are complete Python
`register()` calls. Browser rows are GPU timestamps and therefore exclude
binary host transport and result readback.

| Boundary and translation | Current median / p95 | Current numerical result |
|---|---:|---:|
| Native affine full pyramid, 2 levels, `(4, -2)` | **10.684 / 13.238 ms** | 31,724 accepted; 0.001873 px RMSE |
| Native affine full pyramid, 3 levels, `(4, -2)` | **10.891 / 13.161 ms** | 31,734 accepted; 0.001937 px RMSE |
| Raw browser WGSL affine, 2 levels, `(4, -2)` | **8.681 / 11.698 ms** | 31,724 accepted; 0.001804 px RMSE |
| Three.js/TSL affine subgroup, `(4, -2)` | **26.657 / 28.441 ms** | 22,401 accepted; 0.096354 px RMSE |
| Three.js/TSL affine subgroup, `(2, -1)` | **12.568 / 14.552 ms** | 31,654 accepted; 0.0000305 px RMSE |

The native two-level result corresponds to 93.6 calls/s from the median and
75.5 calls/s from p95. Its complete-call boundary includes target upload, GPU
pyramid construction, all coarse-to-fine solves, packed readback, and NumPy
result construction. It excludes acquisition and downstream consumers.

The raw-browser two-level GPU boundary corresponds to 115.2 frames/s from the
median. Its synchronized browser wall time was 10.8 ms median and 14.2 ms p95.
The Three.js nominal browser wall time was 14.3 / 16.0 ms; the difficult
single-scale pair measured 28.7 / 30.7 ms.

## Translation-only A/B audit

The same 8-warm-up, 50-measurement protocol was repeated with
`motion_model="translation"`. The public six-component result remains
unchanged, with the four deformation terms fixed to zero.

| Boundary | Affine median / p95 | Translation median / p95 | Median change |
|---|---:|---:|---:|
| Native one scale, `(4, -2)`, complete call | 34.683 / 35.766 ms | **14.948 / 17.696 ms** | **-56.9%** |
| Native full pyramid, 2 levels, `(4, -2)`, complete call | 10.684 / 13.238 ms | **10.121 / 12.352 ms** | **-5.3%** |
| Raw browser WGSL, 2 levels, `(4, -2)`, GPU timestamp | **8.681 / 11.698 ms** | 8.777 / 11.099 ms | +1.1% |
| Three.js subgroup, one scale, `(2, -1)`, GPU timestamp | 12.568 / 14.552 ms | **9.087 / 10.560 ms** | **-27.7%** |

The translation model is substantially faster when iterative solving dominates.
The two-level paths already average about one iteration per level, so fixed
reduction and B-spline filtering costs dominate. The raw WGSL median difference
is within run-to-run variation and is not evidence of a speed-up there; its p95
was 5.1% lower in this campaign.

Translation mode accepted 30,185 subsets at 0.0000642 px RMSE in the native
one-scale difficult case, 31,781 at 0.002110 px in the native pyramid, 31,783 at
0.002141 px in raw WGSL, and the same 31,654 subsets at 0.0000654 px in Three.js.
These translation-only results must not be used to infer strain accuracy.

## Current native sequence throughput

The bounded single-scale sequence API was measured in four campaigns. Each
campaign used five warm-ups and 30 UHD targets with an exact `(2, -1)`
translation.

| Execution | Median period | Throughput | Median result latency |
|---|---:|---:|---:|
| synchronous `register()` | 11.539 ms | 86.7 FPS | 11.398 ms |
| bounded pipeline, depth 1 | 11.805 ms | 84.7 FPS | — |
| **bounded pipeline, depth 2** | **11.026 ms** | **90.7 FPS** | **21.963 ms** |
| bounded pipeline, depth 3 | 10.780 ms | 92.8 FPS | 32.093 ms |

All configurations accepted 31,654 subsets and reported `3.066e-5 px` RMSE.
Two slots remain the recommended throughput setting: the third slot adds only
2.1 FPS while adding another processed frame of result latency. This pipeline
depth is unrelated to image-pyramid depth.

## Optimizations represented by these measurements

- the canonical WGSL FIR stages 32 × 8 horizontal and 8 × 32 vertical output
  tiles in workgroup memory;
- it evaluates the central 25 coefficients of the compatible 31-value FIR
  record; the six omitted tails have a total absolute weight below `7e-8`;
- the Three.js FIR uses a conservative 16 × 16 output tile with all 31 taps;
- the Three.js solver fills the measured 32-lane hardware subgroup, so each
  lane handles eight subset pixels;
- the portable shared WGSL solver retains one 16-lane workgroup per subset;
- the two-level full pyramid keeps reduction, coefficients, affine seed
  transfer, solves, and final readback on one GPU command graph.

The opt-in GPU regressions validate partial FIR tiles, exact integer pyramid
reduction, fresh seeded state, selective-retry neighbour preservation, and the
translation-only fractional-displacement contract. All 80 tests pass when GPU
execution is enabled.

## Reproduce the current measurements

Native full pyramid:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_full_pyramid.py \
  --width 3840 --height 2160 --levels 2 3 --warmups 8 --repeats 50 \
  --motion-model affine
```

Raw browser WGSL:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
  --backend webgpu-wgsl --pyramid-levels 2 \
  --translation 4 -2 --warmups 8 --repeats 50 --motion-model affine
```

Three.js/TSL nominal and difficult pairs:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
  --backend three-tsl --translation 2 -1 --warmups 8 --repeats 50

PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
  --backend three-tsl --translation 4 -2 --warmups 8 --repeats 50
```

Native sequence pipeline:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_pipeline.py \
  --repeats 30 --warmups 5 --campaigns 4 --depths 1 2 3
```

Run the hardware regressions before publishing a new campaign:

```console
ICGN_RUN_GPU_TESTS=1 PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests \
  -p 'test_*.py' -v
```

## External SYSCL comparison

The related SYSCL project remains referenced for algorithmic provenance, but
there is currently no qualified apples-to-apples timing row. Add one only after
both projects are run on the same machine, input, iteration schedule, and
explicitly equivalent timing boundary.

## Interpretation

These results demonstrate offline 4K60 processing margin on one GPU and one
controlled dataset. They do not include camera acquisition, frame-arrival
jitter, a codec, downstream processing, or operating-system deadline
guarantees. Median and p95, accepted subsets, RMSE, queue depth, and result
latency must all be reported together when qualifying a deployment.
