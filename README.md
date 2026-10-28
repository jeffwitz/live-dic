# Toward 4K60 Real-Time DIC in Python with WebGPU

> **Foundational work by [Pierre Margerit](https://github.com/MargeritPierre).**
> This fork and its real-time Python DIC work would not exist without Pierre's
> original [`tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio)
> project. His GPU-IO experiments, repository structure, and WebGL groundwork
> are the foundation on which the work presented here is built.

## Long-term target: qualified 4K at 60 FPS

The project goal is sustained digital image correlation on 3840 x 2160 frames
at 60 frames per second, measured across the complete acquisition-to-result
pipeline. The GPU-qualified consolidation campaign processed the reproducible
UHD test pair with a two-level GPU-resident pyramid in a **14.580 ms median
complete Python call** with a **15.478 ms p95**.
This boundary includes direct target upload,
GPU pyramid construction, coarse-to-fine IC-GN computation, packed readback,
and NumPy result construction. Both measurements are below the 16.667 ms
period on the validation machine. The reduction and selective-recovery shaders
were compiled and exercised by the opt-in hardware regressions before this
campaign.

This is an encouraging compute result, not yet a 4K60 system qualification.
Camera acquisition and downstream consumers are still excluded. The browser
GPU timer also excludes input upload, Python/DevTools transport, and readback.
Those stages must all be integrated and measured without growing queues before
the 4K60 objective can be claimed.

This fork adds a Python-accessible, GPU-accelerated digital image correlation
(DIC) implementation to `tests_gpuio`. It performs affine
inverse-compositional Gauss-Newton registration through a default Three.js/TSL
WebGPU backend, an optional GPU-IO/WebGL2 backend, and an experimental native
wgpu-py backend. All paths accept dynamic dimensions and have been validated
from 512 x 512 through UHD 3840 x 2160. Three.js remains the browser-facing API
default. For native NumPy processing, the recommended default is the two-level
`ICGNWGPUFullPyramid` path; the previous single-scale and selective-recovery
`ICGNWGPU` implementation remains available.

The current implementation deliberately starts with deterministic images from
the validation dataset. It does **not** include camera acquisition or the H.264
pipeline. This isolates the DIC algorithm, numerical equivalence, accuracy, and
performance before live acquisition is connected.

## What this fork implements

The implementation in [`DIC/LucasKanadeICGN/`](./DIC/LucasKanadeICGN/) adds:

- a six-parameter affine warp for every DIC subset;
- inverse-compositional Gauss-Newton updates;
- Gaussian filtering and cubic B-spline interpolation;
- reference gradients and a constant inverse Hessian prepared once;
- ZNCC validation and spatial median replacement;
- a dynamic 16-pixel grid of 16 x 16 subsets, including 32 x 32 at 512 pixels
  and 240 x 135 at UHD;
- target prefiltering and registration on Three.js/TSL WebGPU by default;
- a fused, dynamically sized GPU-IO/WebGL2 implementation as an optional
  backend;
- an experimental in-process wgpu-py backend with direct NumPy texture upload,
  shared WGSL compute kernels, and one packed binary readback;
- a two-level default native full pyramid whose five-tap reductions, affine
  seed transfers, per-level solvers, and final readback remain on one GPU and
  one command submission;
- a compiling Rust wgpu/PyO3 host milestone that shares the WGSL source and
  measures the NumPy-to-texture boundary;
- a persistent, headless Python API with no visible browser interface;
- NumPy displacement, affine-parameter, status, and iteration fields;
- sequence processing, benchmarking, `.npz` export, and a command-line tool;
- Sphinx documentation organized according to Diátaxis and configured for
  Read the Docs.

The parameter order is compatible with the SYCL prototype:

```text
[tx, du/dx, du/dy, ty, dv/dx, dv/dy]
```

## Architecture

The Python package exposes the browser implementations through one headless
contract and the native implementation through the same result schema.
Three.js/TSL is the default; GPU-IO remains selectable for regression
comparisons, and `ICGNWGPU` selects the browser-free path:

```text
Python reference image
        |
        v
grayscale conversion + PNG transport
        |
        v
headless Chromium / WebGPU or WebGL2
        |
        +--> one-time reference preparation on CPU
        |    F, gradients, H^-1, J^T F, sum(J)
        |
        +--> persistent reference textures on GPU

Python target frame
        |
        v
GPU target FIR prefilter
        |
        v
affine IC-GN iterations
        |
        v
median consistency pass
        |
        v
GPU readback --> NumPy arrays
```

The native path is shorter: `NumPy uint8 → r8unorm texture → WGSL FIR/IC-GN/
median → packed readback → NumPy views`. Its detailed resource and workgroup
layout is documented in the [native backend explanation](./DIC/LucasKanadeICGN/docs/explanation/native-backend.rst).

The Three.js solver fuses the IC-GN iterations into one compute dispatch and
uses 16 cooperating lanes per subset. A startup probe selects hardware subgroup
reductions when all lanes fit in one subgroup and otherwise selects the
portable workgroup-memory solver.

The GPU-IO solver uses the WebGL2 execution model: one fragment owns one subset
and scans its 16 x 16 pixels serially. Its iterations are fused into one draw,
the affine state and reference atlases stay in registers, and three result
attachments are written once. The complete tiling and grouping diagrams are in
the [GPU optimization documentation](./DIC/LucasKanadeICGN/docs/explanation/gpu-optimization.rst).

Chromium is used only as an invisible WebGPU or WebGL2 execution environment
for browser backends. The native backend does not start it. The browser Python
entry point does not load Tweakpane, attach the canvas to the visible page, or
open a browser window. One persistent engine should be reused for a sequence
so that runtime startup and reference preparation are not repeated for every
frame.

## Validated numerical result

The shipped test images are direct luma-frame extractions from the raw YUV
sequence `seed2026101102_translation`. The analytical motion is
`(+0.2, -0.1)` pixel per frame.

For the `t=0 -> 10` pair:

| Metric | Result |
|---|---:|
| True translation | `(2.0, -1.0)` px |
| Measured mean translation | `(1.999993, -0.999996)` px |
| Vector RMSE | `2.749e-5` px |
| Accepted subsets | `900 / 1,024` |
| Median-replaced subsets | `0` |
| Mean iterations over accepted subsets | `5.959` |

The images, source hashes, format, and motion parameters are recorded in
[`assets/source.json`](./DIC/LucasKanadeICGN/assets/source.json).

## Performance

Measurements on an Intel Lunar Lake GPU use three campaigns of 30 runs after
five warm-ups. At 512 x 512, the median campaign summaries are:

| Backend | GPU median | GPU p95 | Browser wall median |
|---|---:|---:|---:|
| Three.js/TSL WebGPU | **0.675 ms** | **1.227 ms** | **2.7 ms** |
| GPU-IO/WebGL2 | 3.301 ms | 4.210 ms | 10.3 ms |

At the complete Python-call boundary on the same 512 x 512 pair, native
wgpu-py measured **1.882 ms median / 2.382 ms p95**, versus 56.203 / 59.117 ms
for Three.js through Chromium and 61.877 / 65.655 ms for GPU-IO. The browser
calls include their PNG and DevTools transport, so this comparison measures the
architectural transport change rather than kernel speed alone.

The reproducible UHD pair contains 32,400 subsets and an exact `(2, -1)` pixel
translation:

| UHD compute boundary | Median | p95 | Accepted | RMSE |
|---|---:|---:|---:|---:|
| Three.js/TSL | **13.629 ms** | **15.352 ms** | 31,654 | `3.076e-5` px |
| GPU-IO/WebGL2 | 80.738 ms | 82.522 ms | 31,654 | `4.182e-5` px |
| SYCL mixed/double reduction | 19.099 ms | 19.863 ms | 31,654 | `2.555e-5` px |
| SYCL `--float-reduce` | 18.111 ms | 19.062 ms | 31,654 | `2.712e-5` px |

The native wgpu-py **complete Python call** measured 16.160 ms median and
16.569 ms p95, with 31,654 accepted subsets and `3.080e-5 px` RMSE. It uses a
stricter boundary than the browser GPU rows because upload and packed readback
are included.

At the comparable UHD compute boundary, Three.js is 28.6% faster than the
mixed/double-reduction SYCL configuration and 24.7% faster than the numerically
closer float-reduction variant on this machine. Three.js GPU timestamps exclude
input upload and output readback; the documented SYCL totals include them in a
separate timing boundary.

These values do **not** prove a sustained 60 Hz Python system:
the browser benchmark excludes Python PNG encoding, DevTools transport, image
decoding, result transfer, and camera acquisition. Real-time qualification must
measure the complete `engine.register(frame)` call and the acquisition-to-result
latency on the target machine.

The timing protocol, preparation measurements, and full SYCL comparison are documented in
[`DIC/LucasKanadeICGN/BENCHMARK.md`](./DIC/LucasKanadeICGN/BENCHMARK.md).

## Installation

Requirements:

- Python 3.10 or newer;
- Chromium or Google Chrome with WebGPU and WebGL2 support;
- a compatible GPU and driver.

The optional native path requires Python 3.11 or newer and does not require
Chromium.

Install the local Python package:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python
```

Install wgpu-py and the packaged WGSL runtime with:

```bash
.venv-icgn/bin/pip install -e 'DIC/LucasKanadeICGN/python[native]'
```

The Python package embeds GPU-IO, Three.js r186, both browser solvers, the
native WGSL kernels, and the headless HTML entry points. After dependencies
have been installed, the runtime performs no network requests and does not
require the repository checkout. The optional browser demonstration also uses
only vendored scripts; see
[`THIRD_PARTY.md`](./DIC/LucasKanadeICGN/THIRD_PARTY.md).

## First Python registration

```python
from lucas_kanade_icgn import ICGNWebGL

reference = "DIC/LucasKanadeICGN/assets/frame_000.png"
target = "DIC/LucasKanadeICGN/assets/frame_010.png"

with ICGNWebGL() as engine:
    print(engine.info())
    engine.set_reference(reference)
    result = engine.register(target)

print(result.mean_translation)
print(result.parameters.shape)      # (32, 32, 6)
print(result.displacement.shape)    # (32, 32, 2)
print(result.accepted_mask.sum())   # 900
print(result.rmse((2.0, -1.0)))
```

For a sequence, keep the same context open and call `register` for every target,
or use `engine.register_many(reference, targets)`.

Select the experimental browser-free path explicitly:

```python
from lucas_kanade_icgn import ICGNWGPU

with ICGNWGPU() as engine:
    engine.set_reference(reference)
    result = engine.register(target)
```

## Command-line usage

Register one pair and export all numeric fields:

```bash
.venv-icgn/bin/icgn-webgl \
  DIC/LucasKanadeICGN/assets/frame_000.png \
  DIC/LucasKanadeICGN/assets/frame_010.png \
  --truth 2 -1 \
  --output result.npz
```

Run a 30-sample GPU benchmark:

```bash
.venv-icgn/bin/icgn-webgl \
  DIC/LucasKanadeICGN/assets/frame_000.png \
  DIC/LucasKanadeICGN/assets/frame_010.png \
  --benchmark --repeats 30 --warmups 3 --truth 2 -1
```

Standard output is JSON. Complete arrays and benchmark samples are written to
the optional compressed NumPy archive.

## Documentation

The documentation focuses exclusively on real-time DIC from Python. It contains
tutorials, operational guides, API reference, the IC-GN formulation, timing
boundaries, limitations, and scientific provenance.

- source: [`DIC/LucasKanadeICGN/docs/`](./DIC/LucasKanadeICGN/docs/);
- entry page: [`docs/index.rst`](./DIC/LucasKanadeICGN/docs/index.rst);
- acknowledgement of Pierre's work:
  [`acknowledgements.rst`](./DIC/LucasKanadeICGN/docs/explanation/acknowledgements.rst);
- Read the Docs configuration: [`.readthedocs.yaml`](./.readthedocs.yaml).

Build the HTML documentation locally:

```bash
python3 -m pip install -r DIC/LucasKanadeICGN/docs/requirements.txt
python3 -m sphinx -W --keep-going -b html \
  DIC/LucasKanadeICGN/docs \
  DIC/LucasKanadeICGN/docs/_build/html
```

## Repository map

```text
DIC/LucasKanadeICGN/
├── LucasKanadeICGN.js      GPU-IO/WebGL2 affine IC-GN implementation
├── three/                  Three.js/TSL WebGPU implementation and vendor files
├── headless.html           GPU-IO headless runtime
├── index.html              optional interactive browser benchmark
├── assets/                 traceable raw-YUV validation frames
├── python/                 typed package, CLI, and tests
├── native/                 experimental Rust wgpu/PyO3 host
├── benchmarks/             reproducible backend and UHD campaigns
├── docs/                   Sphinx/Diátaxis documentation
├── BENCHMARK.md            measurement protocol and SYCL comparison
└── README.md               detailed implementation overview
```

The original GPU-IO demonstrations elsewhere in this repository remain
available from [`index.html`](./index.html).

## Current limitations

- dynamic grayscale frame dimensions in all backends; GPU-IO requires width
  and height to be multiples of 16;
- fixed 16 x 16 subsets on a 16-pixel grid;
- Three.js/TSL and historical GPU-IO remain single-scale; the raw WGSL browser
  backend and native full-pyramid backend default to two levels and support
  three explicitly;
- no temporal or extrapolated initialization;
- no Aitken acceleration or predictive-tolerance guard;
- no camera, trigger, synchronization, or codec ownership;
- PNG and DevTools transport between Python and browser GPU runtimes; the
  native path removes both;
- full result-field readback for every `register` call;
- soft real-time behavior only, with no operating-system deadline guarantee.

The direct `t=0 -> 20` displacement of `(4, -2)` pixels is outside the robust
local basin for many single-scale subsets. The current two-level native
full-pyramid campaign returned 31,724 accepted subsets out of 32,400 at UHD
with a translation RMSE of 0.00182 pixel. These values refer to that campaign's
accepted population. Larger motions can select three levels
or provide a better initial estimate.

## Verification

Development checks and the recorded GPU campaigns include:

- hardware-independent Python regression tests for API and numerical helpers;
- Ruff linting and formatting checks;
- offline Cargo check, rustfmt, Clippy with warnings denied, release wheel
  build, and installed PyO3 smoke test;
- JavaScript syntax checking;
- wheel build and installation;
- real GPU execution through the installed package in the recorded campaigns;
- strict Sphinx builds with warnings treated as errors;
- visual inspection of the generated landing page and API reference.

Run the local checks:

```bash
python -m pip install -e 'DIC/LucasKanadeICGN/python[dev]'
ruff check DIC/LucasKanadeICGN/python
ruff format --check DIC/LucasKanadeICGN/python
python -m unittest discover -s DIC/LucasKanadeICGN/python/tests -v
cargo fmt --check --manifest-path DIC/LucasKanadeICGN/native/Cargo.toml
cargo check --release --offline \
  --manifest-path DIC/LucasKanadeICGN/native/Cargo.toml
cargo clippy --release --offline \
  --manifest-path DIC/LucasKanadeICGN/native/Cargo.toml -- -D warnings
node --check DIC/LucasKanadeICGN/LucasKanadeICGN.js
node --check DIC/LucasKanadeICGN/three/LucasKanadeICGNThree.js
node --check DIC/LucasKanadeICGN/python/src/lucas_kanade_icgn/web/three/LucasKanadeICGNThree.js
node --check DIC/LucasKanadeICGN/python/src/lucas_kanade_icgn/web/webgpu/LucasKanadeICGNWebGPU.js
```

The default unit tests do not require Chromium or a GPU. Platform-specific browser
discovery and launch arguments are covered with deterministic Linux, Windows,
and macOS simulations. The separate GitHub Actions matrix is prepared to run
the same installed-package suite on all three operating-system families;
hardware-backed execution and performance still require qualification on each
target machine and architecture.

The native GPU regression tests are explicit opt-in tests. On a GPU host with
Python 3.11 or newer, install the native extra and run:

```bash
python -m pip install -e 'DIC/LucasKanadeICGN/python[native]'
ICGN_RUN_GPU_TESTS=1 python -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p 'test_gpu_regressions.py' -v
```

These tests compare GPU reductions against the CPU reference and check that
seeded retries preserve the states needed by the median filter. When enabled,
a missing native dependency or adapter is an error, not a successful skip.
The portable CI matrix skips these tests; its green status alone does not
validate shader execution or performance.

## Acknowledgements and provenance

This fork is a continuation of
[Pierre Margerit's `tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio),
whose experiments and WebGL structure made the present work possible.
The original demonstrations remain published at
[margeritpierre.github.io/tests_gpuio](https://margeritpierre.github.io/tests_gpuio/).

The repository uses [GPU-IO](https://github.com/amandaghassaei/gpu-io), created
by Amanda Ghassaei, as its WebGL computation framework.
