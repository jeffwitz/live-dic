# Affine IC-GN GPU backends toward 4K60

> **Foundational acknowledgement — Pierre Margerit.** This work would not exist
> without [Pierre Margerit's original `tests_gpuio`
> project](https://github.com/MargeritPierre/tests_gpuio). Its GPU-IO
> experiments, repository structure, and WebGL groundwork made this Python
> real-time DIC implementation possible. The implementation presented here is a
> continuation of that foundation.

## 4K60 engineering objective

The long-term objective is sustained 3840 x 2160 DIC at 60 FPS over the full
acquisition-to-result path. The experimental native wgpu-py backend currently
measures 14.621 ms median and 15.748 ms p95 for the complete Python registration
call on the validation machine. Direct upload, computation, packed readback,
and NumPy result construction are included. Acquisition and downstream work
are excluded, so this repository does not yet claim a qualified 4K60
application.

This implementation ports the affine inverse-compositional Gauss-Newton
(IC-GN) registration core from the SYCL prototype in
`DIC_CODEC/DIC_FROM_H264` to the original WebGL2/GPU-IO path, an independent
Three.js/TSL WebGPU path, a raw browser WebGPU adapter, and an experimental
native wgpu-py path. It runs on offline test images and does not require a
camera or an H.264 pipeline.

The implementation retains the main algorithmic choices of the SYCL version:

- one six-parameter affine transformation per subset;
- reference-image gradients and a reference-dependent Hessian;
- inverse-compositional warp updates;
- convergence based on the maximum update at the four subset corners;
- normalized correlation to reduce sensitivity to affine brightness changes;
- Gaussian filtering and cubic B-spline coefficients;
- one constant inverse Hessian prepared with the reference image;
- ZNCC rejection and median replacement of spatial outliers.

The GPU-IO backend requires a different memory layout from C++/SYCL. Translation and the
four linear affine terms are stored in two floating-point textures. Reference
quantities (`F`, gradients, `H^-1`, `J^T F`, and `sum(J)`) are prepared once on
the CPU and uploaded to three GPU textures. Each target is filtered on the GPU
with the same radius-15 FIR as the optimized SYCL configuration. The IC-GN
shader accumulates ZNSSD statistics and `J^T g` in one pass before applying the
inverse-compositional update.

The Three.js backend implements those same operations as TSL storage-buffer
compute kernels: two target prefilter passes, the six-parameter affine IC-GN
solver, and the final median consistency pass. A startup probe selects a fused
16-lane subgroup solver when the adapter supports it; a portable
workgroup-memory solver remains available automatically. It is an
implementation of the DIC-RT equations in this repository, not a copy of
Strobox. Three.js r186 and its licence are vendored locally under
[`three/vendor/`](./three/vendor/) for offline execution.

The raw browser adapter and wgpu-py load the same canonical
[`icgn.wgsl`](./python/src/lucas_kanade_icgn/shaders/icgn.wgsl) source. It owns
the FIR prefilter, increment solve, inverse affine composition, convergence,
and spatial median rules. Their host adapters only allocate resources,
transfer frames, dispatch kernels, and convert the common packed result. The
optimized Three.js path keeps its TSL implementation because subgroup
collectives are faster on the validation GPU, but it now shares named
mathematical helpers and browser reference preparation instead of duplicating
those operations between its workgroup and subgroup variants.

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
SPIR-V subgroup solver and retains the faster path for the current adapter.
This removes PNG, Chromium, DevTools, JSON, and JavaScript-array conversions. The
``native/`` Rust wgpu/PyO3 crate already qualifies adapter creation, shared
shader compilation, and the same direct texture-upload boundary; full solver
orchestration remains in wgpu-py for this milestone.

For sustained native sequences, ``ICGNWGPU.register_stream`` uses a bounded
ring of independent target, solver-state, and readback resources. Two slots
are the default and allow Python to retire one packed result while the GPU
processes the next frame. This slot count is unrelated to an image pyramid. On
the validation machine this reduced the median UHD sequence period
from 14.792 ms to 13.380 ms (74.7 FPS), while preserving bit-for-bit output;
median result latency increased to 26.617 ms. The ordinary ``register`` method
remains the lowest-latency path.

The remaining differences are platform constraints: the browser backends use
single precision and do not expose the same register controls or queue model as
SYCL/Level Zero. This version does not implement Aitken acceleration,
predictive tolerance, temporal/extrapolated initialization, or the H.264
pipeline. The native wgpu-py engine now provides an optional selective image
pyramid: ``pyramid_levels=1`` is the unchanged default fast path, while
``pyramid_levels=2`` retries only failed fine subsets from half-resolution
seeds and preserves every previously accepted result bit for bit.

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

Three.js/TSL is the default Python and command-line backend. Select the previous
implementation explicitly with `ICGNWebGL(backend="gpuio")` or
`--backend gpuio`. Select the canonical shared shader in Chromium with
`ICGNWebGL(backend="webgpu-wgsl")`, or the browser-free path with `ICGNWGPU()`.
All return the same NumPy result schema.

## Optional browser demo

Serve the repository root over HTTP and open:

```text
/DIC/LucasKanadeICGN/
```

The page loads `t = 0` as the reference, evaluates all four targets, and runs a
synchronous benchmark. Its controls can select another target or rerun the
campaign. Timings use `gl.finish()` or WebGL timer queries, so GPU completion is
observed without including video decoding.

GPU-IO, Tweakpane 3.1.2, and Tweakpane Essentials 0.1.0 are loaded from local
files. The demo makes no CDN request. Versions, hashes, and licence locations
are recorded in [`THIRD_PARTY.md`](./THIRD_PARTY.md).

Default parameters match the SYCL geometry: a 16-pixel grid spacing, 16 x 16
subsets centered at `(8 + 16i, 8 + 16j)`, a `1e-3` pixel tolerance, a minimum
ZNCC of `0.8`, and at most 30 iterations with per-subset early convergence.

Measured accuracy and performance are recorded in
[`BENCHMARK.md`](./BENCHMARK.md). The Sphinx
[`performance reference`](./docs/reference/performance.rst) compares Three.js,
GPU-IO, and the external
[`DIC_FROM_H264` SYCL implementation](https://github.com/jeffwitz/DIC_FROM_H264)
on the validation machine.

[`GPU tiling and grouping`](./docs/explanation/gpu-optimization.rst) explains
the 16-lane WebGPU subgroup/workgroup reductions and the fused one-fragment-per-
subset WebGL2 strategy with execution diagrams.

[`Native WebGPU and PyO3`](./docs/explanation/native-backend.rst) explains the
direct NumPy transfer path, shared WGSL layout, current measurements, and the
remaining Rust migration work.

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

The Sphinx documentation follows the Diátaxis structure and focuses on using
the Python implementation for real-time DIC. It starts at
[`docs/index.rst`](./docs/index.rst) and includes tutorials, operational
how-to guides, API reference, and explanations of IC-GN and latency boundaries.

Build it locally with:

```bash
python3 -m sphinx -W --keep-going -b html \
  DIC/LucasKanadeICGN/docs \
  DIC/LucasKanadeICGN/docs/_build/html
```
