# Current affine IC-GN performance

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
- `ICGNWGPU()` is the native single-scale and bounded-streaming engine. Its
  optional selective recovery is not the normal pyramid path.
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

## Current UHD kernel audit

Each row uses eight warm-ups and 50 measured registrations. Native rows are
complete Python `register()` calls. Browser rows are GPU timestamps and
therefore exclude PNG/DevTools transport and result readback.

| Boundary and translation | Current median / p95 | Current numerical result |
|---|---:|---:|
| Native full pyramid, 2 levels, `(4, -2)` | **10.194 / 11.073 ms** | 31,724 accepted; 0.001874 px RMSE |
| Native full pyramid, 3 levels, `(4, -2)` | **10.280 / 11.151 ms** | 31,734 accepted; 0.001939 px RMSE |
| Raw browser WGSL, 2 levels, `(4, -2)` | **8.789 / 9.375 ms** | 31,724 accepted; 0.001804 px RMSE |
| Three.js/TSL subgroup, `(4, -2)` | **24.300 / 26.627 ms** | 22,401 accepted; 0.096354 px RMSE |
| Three.js/TSL subgroup, `(2, -1)` | **11.893 / 12.798 ms** | 31,654 accepted; 0.0000305 px RMSE |

The native two-level result corresponds to 98.1 calls/s from the median and
90.3 calls/s from p95. Its complete-call boundary includes target upload, GPU
pyramid construction, all coarse-to-fine solves, packed readback, and NumPy
result construction. It excludes acquisition and downstream consumers.

The raw-browser two-level GPU boundary corresponds to 113.8 frames/s from the
median. Its synchronized browser wall time was 10.4 ms median and 11.5 ms p95.
The Three.js nominal browser wall time was 13.5 / 14.7 ms; the difficult
single-scale pair measured 26.2 / 28.7 ms.

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
reduction, fresh seeded state, and selective-retry neighbour preservation. All
67 tests pass when GPU execution is enabled.

## Reproduce the current measurements

Native full pyramid:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_full_pyramid.py \
  --width 3840 --height 2160 --levels 2 3 --warmups 8 --repeats 50
```

Raw browser WGSL:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
  --backend webgpu-wgsl --pyramid-levels 2 \
  --translation 4 -2 --warmups 8 --repeats 50
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
