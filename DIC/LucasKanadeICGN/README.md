# Python-supervised IC-GN GPU backends toward 4K60

> **Foundational acknowledgement — Pierre Margerit.** This work would not exist
> without [Pierre Margerit's original `tests_gpuio`
> project](https://github.com/MargeritPierre/tests_gpuio). Its GPU-IO
> experiments, repository structure, and WebGL groundwork made this Python
> real-time DIC implementation possible. The implementation presented here is a
> continuation of that foundation.

[![Documentation](https://github.com/jeffwitz/tests_gpuio/actions/workflows/docs.yml/badge.svg)](https://github.com/jeffwitz/tests_gpuio/actions/workflows/docs.yml)

## Project purpose: a Python-supervised numerical engine

DIC-RT is built to sit inside a Python measurement application. Python is the
supervisor: it prepares and replaces references, configures the algorithm,
connects acquisition, bounds queues, records timing, handles failures, and
consumes NumPy fields. The recommended resident-NumPy path is
``ICGNWGPUFullPyramid`` because it runs WebGPU in-process and keeps the complete
image-to-field boundary under Python control.

Both native engines accept a boolean reference ROI with
`set_reference(reference, roi_mask=mask)`. The default `full` policy retains
only complete nominal windows, including checks for interior holes. The
[ROI guide](./docs/how-to/use-roi.rst) explains conservative pyramids,
neighbour permissions, streaming, and self-contained result exports.

The Three.js and raw-WGSL browser backends serve portability, browser-native
deployment, and numerical qualification. ``ICGNWebGL`` makes them callable
from Python, but it necessarily copies each target into Chromium and returns
the result to Python. Binary transport has reduced that complete UHD boundary
to roughly 45--47 ms median, but it does not make the browser process share
NumPy memory.

A competitive Three.js streaming deployment therefore needs a persistent
JavaScript supervisor in the browser process. That supervisor would own frame
ingestion, reference and GPU-resource lifetime, bounded scheduling, and local
result consumption; Python would be a configuration and monitoring plane, not
the per-frame image and full-field transport. This repository does not yet
provide that JavaScript supervisor. When NumPy is the per-frame source and the
complete field is required in Python, use native wgpu.

## 4K60 engineering objective

The long-term objective is sustained 3840 x 2160 DIC at 60 FPS over the full
acquisition-to-result path. On a changing four-image sequence, the recommended
two-level native full pyramid measured 11.571 ms median / 12.185 ms p95 in
affine mode and 10.992 / 11.397 ms in translation mode. The boundary starts
with a resident NumPy image and ends with the materialized NumPy field; direct
upload, GPU pyramid construction and solving, packed readback, and result
construction are included. Acquisition and downstream work are excluded, so
this repository does not yet claim a qualified camera-to-consumer 4K60
application.

For sustained native input, the same two-level graph now supports a bounded
full-pyramid stream. With two targets in flight it sustained **95.51 FPS** in
affine mode and **104.09 FPS** in translation mode on this machine. Median
result age was 21.008 and 19.131 ms; use one slot for the lower 11.354 and
10.496 ms median ages. Throughput and result age are reported separately.

## Browser backend policy

`three-tsl` is the supported browser backend for browser applications. It uses
WebGPU when available and automatically falls back to Three.js' WebGL2
transform-feedback implementation when WebGPU is unavailable. Applications
therefore no longer need GPU-IO to retain WebGL2 compatibility.

The `gpuio` backend is deprecated. It remains packaged temporarily so that
existing results can be reproduced and the replacement can be qualified on
additional browsers and GPU vendors. It receives no new algorithmic features
and should not be selected by new code. See the
[GPU-IO migration guide](./docs/how-to/migrate-from-gpuio.rst).

This implementation ports and extends the inverse-compositional Gauss-Newton
(IC-GN) registration core from the SYCL prototype in
`DIC_CODEC/DIC_FROM_H264` to the original WebGL2/GPU-IO path, an independent
Three.js/TSL WebGPU path, a raw browser WebGPU adapter, and an experimental
native wgpu-py path. It runs on offline test images and does not require a
camera or an H.264 pipeline.

The implementation retains the main algorithmic choices of the SYCL version:

- a two-parameter translation, the default six-parameter affine
  transformation, or a twelve-parameter quadratic transformation per subset;
- reference-image gradients and a reference-dependent Hessian;
- inverse-compositional warp updates;
- convergence based on the maximum update at the four subset corners;
- normalized correlation to reduce sensitivity to affine brightness changes;
- selectable historical B-spline, direct bilinear, and direct Keys-cubic
  target reconstruction;
- one constant inverse Hessian prepared with the reference image;
- ZNCC rejection and displacement-only universal outlier detection (UOD).
  The default ``outlier_policy="replace"`` replaces the complete motion state
  when `tx` or `ty` is inconsistent with recentered neighbours. ``"flag"``
  preserves the raw state with status 2, and ``"disabled"`` skips the pass.
  ``"median"`` is a separate, explicit centred 3×3 component median applied
  to every valid subset; it preserves the solver status and is smoothing, not
  outlier detection. Coefficient-triggered derivative filtering was removed
  because it smoothed healthy noisy fields rather than demonstrating outlier
  robustness. See
  [displacement outlier detection](./docs/explanation/displacement-outliers.rst).

The GPU-IO backend requires a different memory layout from C++/SYCL. Translation and the
four linear affine terms are stored in two floating-point textures. Reference
quantities (`F`, gradients, `H^-1`, `J^T F`, and `sum(J)`) are prepared once on
the CPU and uploaded to three GPU textures. Each target is filtered on the GPU
with the same radius-15 FIR as the optimized SYCL configuration. The IC-GN
shader accumulates ZNSSD statistics and `J^T g` in one pass before applying the
inverse-compositional update.

The Three.js backend implements those same operations as TSL storage-buffer
compute kernels: two target prefilter passes, the selected IC-GN motion model,
solver, and the final displacement UOD pass. A startup probe nominates a
fused subgroup solver sized to the measured hardware subgroup. Reference
preparation then checks the subgroup width reported by that actual solver
pipeline; a mismatch selects the portable workgroup-memory solver
automatically. When WebGPU is not
available, the same TSL implementation now falls back to Three.js' WebGL2
transform-feedback backend and selects a serial one-invocation-per-subset
solver. The WebGL2 path shares the mathematical helpers and result layout but
does not attempt to emulate workgroup barriers. It is an
implementation of the DIC-RT equations in this repository, not a copy of
Strobox. Three.js r186 and its licence are vendored locally under
[`three/vendor/`](./three/vendor/) for offline execution.

The raw browser adapter and wgpu-py load the same canonical
[`icgn.wgsl`](./python/src/lucas_kanade_icgn/shaders/icgn.wgsl) and
[`full_pyramid.wgsl`](./python/src/lucas_kanade_icgn/shaders/full_pyramid.wgsl)
sources. They own the FIR prefilter, increment solve, inverse warp
composition, convergence, displacement UOD rules, image reduction, and
polynomial seed transfer. Their host adapters only allocate resources, transfer frames,
dispatch kernels, and convert the common packed result. The
optimized Three.js path keeps its TSL implementation because subgroup
collectives are faster on the validation GPU, but it now shares named
mathematical helpers and browser reference preparation instead of duplicating
those operations between its workgroup and subgroup variants.

The shared WGSL prefilter stages axis-oriented tiles in workgroup memory. It
keeps the 31-value radius-15 weight buffer for compatibility and evaluates the
central 25 taps; the omitted tails total less than ``7e-8`` in absolute weight.
Three.js retains all 31 taps and fills the measured subgroup width when its
specialized WebGPU path is available. Both tiled implementations clamp
speculative partial-tile halo loads after border reflection. This leaves every
sample used by a valid output unchanged while keeping the extra invocations
that participate in a workgroup barrier inside the image buffers.

The interpolation mode is fixed when an engine is created. ``bspline`` keeps
the historical Gaussian/FIR coefficient path. ``linear`` and ``cubic`` use
the target texture or source buffer directly, so their per-frame graphs omit
both prefilter dispatches. Direct modes also prepare the reference from raw
samples with centered gradients. See
[`Interpolation modes`](./docs/explanation/interpolation.rst) for equations,
backend coverage, fractional-displacement checks, and current measurements.

The motion model is also fixed at engine creation. ``motion_model="affine"``
is the six-parameter default. ``motion_model="translation"`` estimates only
``tx`` and ``ty`` using an analytic 2 by 2 inverse and specialized GPU kernels;
the existing six-component result layout is preserved with exact zeros for the
four deformation terms. ``motion_model="quadratic"`` uses twelve parameters
and adds ``x^2/2``, ``xy``, and ``y^2/2`` terms for both displacement
components. It is implemented by the shared WGSL native and raw-browser paths;
Three.js and GPU-IO do not yet support it. Use translation for rigid
displacement-only work, affine for ordinary strain measurement, and quadratic
when within-subset displacement curvature is required. See
[`Motion models`](./docs/explanation/motion-models.rst) for the equations,
backend coverage, and current A/B results.

The spatial post-pass has one UOD detection scope and one explicit smoothing
mode. UOD tests only recentered ``tx`` and ``ty``. ``replace`` is the default
and replaces the complete local motion state after a trigger; ``flag`` leaves
raw coefficients untouched and sets status 2; ``disabled`` omits the pass.
``median`` bypasses UOD and replaces every valid state with the component-wise
median of its centred 3×3 neighbourhood. See
[`Displacement outlier detection`](./docs/explanation/displacement-outliers.rst).

Quadratic mode is currently a correctness-first path, not the 4K60 default.
Its previous timing campaign predates the current outlier policy and is not
retained. The affine and translation figures above must not be used for
quadratic mode; publish a new complete image-to-field campaign before making a
quadratic throughput claim.

## Backend and pyramid policy

The word *default* is scoped to an engine class:

| Use case | Construction | Default graph |
|---|---|---|
| **Python-supervised NumPy stream** | `ICGNWGPUFullPyramid()` | two-level GPU-resident full pyramid |
| Native single-scale/selective path | `ICGNWGPU()` | one scale |
| Python-controlled browser compatibility | `ICGNWebGL()` | Three.js/TSL, one scale, automatic WebGL2 fallback |
| Python-controlled shared WGSL in Chromium | `ICGNWebGL(backend="webgpu-wgsl")` | two-level GPU-resident full pyramid |
| Historical comparison | `ICGNWebGL(backend="gpuio")` | deprecated one-scale GPU-IO |

The `pyramid_levels` option on `ICGNWGPU` controls an older selective recovery
path. It is disabled by that class's one-scale default and is not the native
recommendation for normal coarse-to-fine processing. See
[`Backend selection and defaults`](./docs/explanation/backend-selection.rst)
for the distinction and complete execution graphs.

All runtimes derive their buffers or textures and subset grid from the
reference dimensions. They have been validated at both 512 × 512 and UHD 3840
× 2160; GPU-IO dimensions must be multiples of 16. The one-time reference
construction is exposed by `set_reference()`, while `register()` and
`benchmark_target()` operate against that prepared state so reference
construction and optical-flow timings remain separate.

The native ``ICGNWGPU`` host submits vectorized WGSL kernels in-process through
wgpu-native. A contiguous NumPy ``uint8`` frame is uploaded directly to an
``r8unorm`` texture and one packed state buffer is read back. At reference
preparation it benchmarks the portable workgroup solver against an optional
WGSL subgroup solver and retains the faster valid path for the current adapter.
This removes PNG, Chromium, DevTools, JSON, and JavaScript-array conversions. The
portable and specialized shaders are compiled directly by the wgpu-native
library shipped in the wgpu-py wheel. The project contains no PyO3 extension or
GLSL-to-SPIR-V build chain and needs no Cargo, Maturin, `glslc`, or Vulkan SDK.

For sustained native sequences, both native engines expose
``register_stream`` with a bounded ring of independent mutable resources. For
the recommended full pyramid, each slot owns the finest target, reduced
targets, reduction intermediates, per-level solver states and seeds, and final
readback; immutable reference data and pipelines remain shared. Two slots are
the default throughput compromise. On the current 50-frame changing UHD
campaign, affine throughput increased from 88.00 FPS at depth one to 95.51 FPS
at depth two, while translation increased from 93.43 to 104.09 FPS. Median
result ages increased from 11.354 to 21.008 ms affine and from 10.496 to 19.131
ms translation. The third slot did not improve throughput and raised median age
above 29 ms. Slot depth is unrelated to image-pyramid depth.

The remaining differences are platform constraints: the browser backends use
single precision and do not expose the same register controls or queue model as
SYCL/Level Zero. This version does not implement Aitken acceleration,
predictive tolerance, temporal/extrapolated initialization, or the H.264
pipeline. The low-latency ``ICGNWGPU`` engine retains optional selective image
recovery for compatibility: ``pyramid_levels=1`` is that class's unchanged
single-scale default, while ``pyramid_levels=2`` retries only failed fine
subsets from half-resolution seeds and preserves every previously accepted
result bit for bit. This is not the recommended normal pyramid.

For workloads that need a pyramid on every frame, the separate
``ICGNWGPUFullPyramid`` backend keeps reduction, per-level target coefficients,
coarse-to-fine polynomial seeds, and solver states on one GPU. It encodes all
levels in one command buffer and performs one final readback. Iterations and
tolerances are configurable per scale from coarse to fine. Two levels are the
recommended native default. Three levels remain available explicitly for
larger expected displacements. On the changing-frame, resident-image to
materialized-field campaign, the two-level graph measured **11.571 / 12.185
ms** in affine mode and **10.992 / 11.397 ms** in translation mode. Both p95
values are below the 16.667 ms 60 FPS period on this machine. Repeated-target
and GPU-only figures are retained only as diagnostic kernel measurements. See
the performance reference for the exact boundaries, protocol, and caveats.

Contiguous grayscale ``uint8`` NumPy arrays are passed directly to wgpu-py's
texture upload; a conforming camera buffer does not undergo PNG, Pillow, or
browser conversion. A CPU-to-GPU copy remains necessary because ordinary
NumPy memory is not GPU texture memory. The Three.js and raw WGSL browser
adapters now use a persistent same-origin binary transport: target bytes cross
unchanged and one packed float field returns. There is no per-frame PNG,
DevTools image payload, decode, or JSON field. Chromium nevertheless owns
separate memory, so the loopback copy and GPU upload remain. Current complete
UHD calls are about 45--47 ms median; native wgpu remains the 60 Hz NumPy path.

The raw browser ``webgpu-wgsl`` backend now runs that same GPU-resident
full-pyramid graph. It defaults to two levels with schedules ``(8, 16)`` and
``(5e-3, 5e-3)`` from coarse to fine. Set ``pyramid_levels=1`` to select the
unchanged single-scale dispatch path, or choose three levels for a larger
capture range. Three.js/TSL and GPU-IO remain single-scale implementations.
The deprecated GPU-IO backend is retained temporarily as a historical WebGL2
reference while the Three.js/WebGL2 fallback is qualified on additional
adapters.

## Test dataset

The five PNG files in `assets/` were extracted directly from the raw YUV file
`validation_inputs/seed2026101102_translation/master.yuv` in the source
repository. The selected frames are `t = 0, 1, 5, 10, 20`. The analytical
motion is `(+0.2, -0.1)` pixel per frame, producing displacements up to
`(+4, -2)` pixels. No H.264 stream is used when loading or preparing these
images. [`assets/source.json`](./assets/source.json) records the source path,
hashes, format, and motion parameters.

## Headless Python interface

The [`python/`](./python/) package is the primary programmatic interface. It
provides:

- a persistent engine for image sequences;
- a one-shot function for a single image pair;
- complete affine fields, status, and iteration arrays as NumPy arrays;
- GPU and wall-clock benchmarking;
- `.npz` export and the `icgn-webgl` command-line tool;
- compatibility with scripts and Jupyter notebooks.

Chromium is used only as an invisible GPU runtime. The headless entry points
do not load Tweakpane, attach a canvas to the DOM, or render a visible UI.
All HTML and JavaScript runtime assets are embedded in the Python wheel, so an
installed package does not need this repository or network access at runtime.
See [`python/README.md`](./python/README.md) for installation and API examples.

Three.js/TSL is the default and recommended **browser** backend exposed by the
Python adapter and command-line tool. For the primary Python-supervised NumPy
pipeline, ``ICGNWGPUFullPyramid`` is the recommended backend.
The deprecated GPU-IO implementation can still be selected explicitly with
`ICGNWebGL(backend="gpuio")` or `--backend gpuio` for regression and
reproducibility work only. Select the canonical shared shaders and default
two-level pyramid in Chromium with `ICGNWebGL(backend="webgpu-wgsl")`, or the
recommended browser-free path with `ICGNWGPUFullPyramid()`.
All return the same NumPy result schema.

Three.js chooses WebGPU first and falls back to WebGL2 when necessary. Use
`ICGNWebGL(backend="three-tsl", force_three_webgl=True)` or
`--force-three-webgl` to exercise and qualify the compatibility path explicitly.

## Optional browser demo

Serve the repository root over HTTP and open:

```text
/DIC/LucasKanadeICGN/
```

The page loads `t = 0` as the reference, evaluates all four targets, and runs a
synchronous benchmark. Its controls can select another target or rerun the
campaign. Timings use `gl.finish()` or WebGL timer queries, so GPU completion is
observed without including video decoding.

The legacy graphical demo still loads GPU-IO, Tweakpane 3.1.2, and Tweakpane
Essentials 0.1.0 from local files. It is retained as an historical example and
is not the recommended Python execution path. The demo makes no CDN request.
Versions, hashes, and licence locations are recorded in
[`THIRD_PARTY.md`](./THIRD_PARTY.md).

Default parameters match the SYCL geometry: a 16-pixel grid spacing, 16 x 16
subsets centered at `(8 + 16i, 8 + 16j)` and a minimum ZNCC of `0.8`.
Native `wgpu-py` and browser `webgpu-wgsl` accept subset widths 8, 15, 16, 17,
31, 32, and 33; odd widths place the control point on an acquired centre
pixel. The chosen width remains the grid spacing unless `subset_stride` is
provided. These shared-WGSL paths accept a stride from 1 through the subset
size, so `subset_size=32, subset_stride=16` produces overlapping 32 x 32
windows. Native engines additionally expose centred Gaussian integration with
`spatial_weighting="gaussian"` and `gaussian_sigma` in pixels. Uniform
integration remains the default. Three.js/TSL, WebGL2, and GPU-IO still use
widths 8, 16, or 32 and require stride equal to size.
The published performance campaign remains a qualification of the default
16-pixel size and stride.
Single-scale engines use a `1e-3` pixel tolerance and at most 30 iterations;
the qualified two-level full pyramid uses `(5e-3, 5e-3)` tolerances and
`(8, 16)` iteration ceilings from coarse to fine.

Measured accuracy and performance are recorded in
[`BENCHMARK.md`](./BENCHMARK.md). The Sphinx
[`performance reference`](./docs/reference/performance.rst) contains only the
current native, raw-WGSL, Three.js, and bounded-stream measurements. The
related [`DIC_FROM_H264` SYSCL implementation](https://github.com/jeffwitz/DIC_FROM_H264)
remains referenced for provenance, but no stale timing row is retained.

[`GPU tiling and grouping`](./docs/explanation/gpu-optimization.rst) explains
the axis-oriented FIR tiles, the 16-lane portable WGSL reduction, the
hardware-width Three.js subgroup reduction, and the fused one-fragment-per-
subset WebGL2 strategy with execution diagrams.

[`Native WebGPU with wgpu-py`](./docs/explanation/native-backend.rst) explains
the direct NumPy transfer path, WGSL subgroup qualification, solver selection,
and current measurements.

[`Shared kernel and adapter contract`](./docs/explanation/shared-kernel.rst)
defines the canonical WGSL functions, buffer bindings, parameter order, status
values, and the boundary between common mathematics and backend-specific host
code.

The reproducible offline UHD campaign is
[`benchmarks/benchmark_4k.py`](./benchmarks/benchmark_4k.py). It constructs a
known `(2, -1)` pixel translation, reports preparation separately, and measures
the warmed optical-flow phase. Use `--backend wgpu-native` for the complete
native Python-call boundary.

[`benchmarks/benchmark_wgpu_pipeline.py`](./benchmarks/benchmark_wgpu_pipeline.py)
compares synchronous native registration with bounded pipeline depths and
reports sequence period separately from result latency.

## Documentation

Select `device_type="cpu"` on either native engine to run the same WGSL on an
installed software adapter, or `"gpu"` to require hardware. The `"auto"`
default prefers hardware. See [CPU execution](./docs/how-to/run-on-cpu.rst).

Engineers discovering GPU programming can start with the
[WebGPU introduction](./docs/explanation/webgpu-fundamentals.rst), then run
the [first kernel tutorial](./docs/tutorials/first-gpu-kernel.rst).
The [GPU vocabulary](./docs/reference/gpu-vocabulary.rst) explains the terms;
the [optimization guide](./docs/how-to/optimize-webgpu.rst) connects them to
efficient kernel development and numerical qualification.

The Sphinx documentation follows the Diátaxis structure and focuses on using
the Python implementation for real-time DIC. It starts at
[`docs/index.rst`](./docs/index.rst) and includes tutorials, operational
how-to guides, API reference, and explanations of IC-GN and latency boundaries.
Every documentation-related push and pull request runs a strict Sphinx build;
the generated HTML site is retained as a downloadable GitHub Actions artifact.

Build it locally with:

```bash
python3 -m sphinx -W --keep-going -b html \
  DIC/LucasKanadeICGN/docs \
  DIC/LucasKanadeICGN/docs/_build/html
```
