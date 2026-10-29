# Current IC-GN performance

Last updated: October 29, 2026.

This file contains only measurements for the current kernels. Earlier campaign
snapshots were removed because their layouts, filter costs, and subgroup
occupancy no longer describe the implementation. Re-run the commands below
when a kernel, driver, browser, or measurement boundary changes.

## Current backend policy

- DIC-RT is measured first as a Python-supervised NumPy pipeline.
- `ICGNWGPUFullPyramid()` with two levels is the recommended product
  coarse-to-fine path for that boundary.
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
or 32,400, 16 × 16 subsets with the default 16-pixel stride. No published row
below uses overlapping subsets. No camera or H.264 stage is included.

## Primary UHD image-to-field stream boundary

Primary backend performance measurements use the default
`outlier_policy="replace"`. The UOD pass tests only recentered `tx` and `ty`;
after a trigger it replaces the complete local motion state. Use `"flag"` to
preserve raw coefficients with status 2, or `"disabled"` to omit the pass for
raw solver studies and isolated timing. Earlier all-coefficient policies were
removed because they modified 66--81% of a healthy noisy field and behaved as
derivative smoothing. See the
[displacement outlier explanation](docs/explanation/displacement-outliers.rst).

This is the deployable Python boundary used for performance claims. A
persistent engine receives an already acquired, contiguous `uint8` NumPy image
and the timer stops only when the complete `(135, 240, 6)` NumPy parameter
field, status, iterations, and statistics are available. Reference preparation,
camera acquisition, and decoding are excluded.

This boundary reflects the product's intended Python supervisor. Browser rows
measure Python controlling a separate Chromium worker; they are not estimates
of a future JavaScript-supervised Three.js stream. Such a supervisor would keep
browser-native input, scheduling, and result consumers in Chromium and needs a
separate implementation and campaign.

The sequence cycles the four distinct qualified `frame_001`, `frame_005`,
`frame_010`, and `frame_020` fixtures. Every row uses three changing-frame
warm-ups followed by 12 measured frames. Execution is synchronous, so the
reported latency is also the result age and the inverse stream period. The
browser rows report the median of three campaigns after the binary transport
change.

| Backend and model | Median / p95 image-to-field latency | Sustained throughput |
|---|---:|---:|
| Native full pyramid, 2 levels, affine | **11.571 / 12.185 ms** | **85.37 FPS** |
| Native full pyramid, 2 levels, translation | **10.992 / 11.397 ms** | **90.72 FPS** |
| Raw browser WGSL, 2 levels, affine | **41.094 / 44.423 ms** | **24.27 FPS** |
| Raw browser WGSL, 2 levels, translation | **39.996 / 43.995 ms** | **24.82 FPS** |
| Three.js/TSL, 1 scale, affine | **43.203 / 56.336 ms** | **21.91 FPS** |
| Three.js/TSL, 1 scale, translation | **39.666 / 44.002 ms** | **25.14 FPS** |

The native policy control was measured with the same three-campaign protocol:

| Model and UOD policy | Median / p95 | Throughput | Median / maximum RMSE |
|---|---:|---:|---:|
| Affine, `replace` | 11.571 / 12.185 ms | 85.37 FPS | 0.04161 / 0.06274 px |
| Affine, `disabled` | **10.851 / 11.436 ms** | **92.24 FPS** | 0.04161 / 0.07810 px |
| Translation, `replace` | 10.992 / 11.397 ms | 90.72 FPS | **0.03590 / 0.04063 px** |
| Translation, `disabled` | **10.095 / 10.539 ms** | **98.63 FPS** | 0.04487 / 0.06548 px |

Skipping UOD saved about 0.72 ms affine and 0.90 ms translation in these
campaign medians. It is not a numerically equivalent fast mode: the raw field
has no wrong-basin replacement, and its maximum error increased on this
sequence. Production tables therefore retain the default `replace` policy;
`disabled` is provided for raw metrology studies and isolated solver timing.

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
back. In the affine campaign the browser body copy took about 9.8 ms median
and the synchronized GPU graph about 17--20 ms. The complete browser paths
therefore sustain about 20--22 FPS, not 60 FPS. Native wgpu remains the
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
| Affine, depth 1 | **11.354 / 11.899 ms** | 11.368 ms | 88.00 FPS |
| Affine, depth 2 | 21.008 / 21.683 ms | **10.471 ms** | **95.51 FPS** |
| Affine, depth 3 | 31.761 / 35.991 ms | 10.523 ms | 95.03 FPS |
| Translation, depth 1 | **10.496 / 11.229 ms** | 10.708 ms | 93.43 FPS |
| Translation, depth 2 | 19.131 / 19.665 ms | **9.607 ms** | **104.09 FPS** |
| Translation, depth 3 | 29.854 / 36.126 ms | 9.655 ms | 103.57 FPS |

Depth one minimizes result age. Depth two is the recommended throughput
setting: it gains 8.5% affine throughput and 11.4% translation throughput while
approximately doubling result age. The third slot does not improve throughput
and raises median result age above 29 ms.

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
  --backend native-pyramid --width 3840 --height 2160 \
  --campaigns 4 --frames 50 --warmups 8 \
  --pipeline-depth 2 --motion-model affine
```

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
translation-only fractional-displacement contract. They also verify complete-
state replacement and flag-only preservation for an injected displacement
outlier, at most 1% false positives on a healthy noisy field, and preservation
of a one-subset-wide affine-gradient band. All 105 tests pass when GPU
execution is enabled.

## Reproduce the current measurements

Current image-to-field campaign:

```console
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
  --backend native-pyramid --backend webgpu-wgsl --backend three-tsl \
  --campaigns 3 --frames 12 --warmups 3 --motion-model affine \
  --outlier-policy replace
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
