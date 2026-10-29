# Toward 4K60 Real-Time DIC in Python with WebGPU

> **Foundational work by [Pierre Margerit](https://github.com/MargeritPierre).**
> This fork and its real-time Python DIC work would not exist without Pierre's
> original [`tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio)
> project. His GPU-IO experiments, repository structure, and WebGL groundwork
> are the foundation on which the work presented here is built.

## Product boundary: Python-supervised DIC

DIC-RT is designed to be supervised by a Python application. Python owns the
reference lifecycle, solver configuration, acquisition integration, bounded
queues, timing, error handling, and consumption of the NumPy displacement and
parameter fields. For resident NumPy streams, the recommended production path
is therefore `ICGNWGPUFullPyramid()`: it keeps WebGPU in-process and avoids a
browser data-plane boundary.

The browser backends remain important for WebGPU/WebGL2 portability, numerical
cross-checks, and browser deployments. Their Python adapter is not the intended
4K60 data plane. Even with the new binary transport, every UHD target crosses
from Python into Chromium and every full field crosses back; current complete
calls measure about 40--43 ms median. To turn the faster Three.js GPU graph
into a competitive streaming application, a persistent **JavaScript
supervisor** would need to own frame ingestion, GPU submission, backpressure,
and result consumption in the browser process. Python could configure and
monitor that supervisor, but it should not marshal every 4K frame and full
field across the process boundary.

That JavaScript supervisor is an integration direction, not a component
provided by this repository today. If every input is a NumPy array and every
full result must be materialized in Python, the native wgpu backend remains the
correct architecture.

## Long-term target: qualified 4K at 60 FPS

The project goal is sustained digital image correlation on 3840 x 2160 frames
at 60 frames per second, measured across the complete acquisition-to-result
pipeline. On the changing UHD sequence, the recommended two-level
GPU-resident pyramid measures **11.920 / 13.072 ms median / p95** in affine
mode and **10.526 / 11.170 ms** in translation mode at the complete
resident-NumPy-image to materialized-field boundary. A bounded two-slot stream
sustains **93.16 FPS affine** and **105.58 FPS translation**; its median result
ages are 21.408 and 18.888 ms, so throughput is not presented as latency. The
boundary includes direct target upload, GPU pyramid construction,
coarse-to-fine IC-GN computation, packed readback, and NumPy result
construction.

This is an encouraging compute result, not yet a 4K60 system qualification.
Camera acquisition and downstream consumers are still excluded. The browser
GPU timer also excludes input upload, binary Python/Chromium transport, and readback.
Those stages must all be integrated and measured without growing queues before
the 4K60 objective can be claimed.

This fork adds a Python-supervised, GPU-accelerated digital image correlation
(DIC) implementation to `tests_gpuio`. It performs affine
inverse-compositional Gauss-Newton registration through a native wgpu-py path,
a Three.js/TSL browser path with automatic WebGPU/WebGL2 selection, and an
optional historical GPU-IO/WebGL2 backend. All paths accept dynamic dimensions and have been validated
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
- selectable target interpolation: historical Gaussian/cubic B-spline,
  direct bilinear, or direct Keys cubic convolution;
- reference gradients and a constant inverse Hessian prepared once;
- ZNCC validation and spatial median replacement;
- configurable square subsets including centred odd widths 15, 17, 31, and
  33, plus optional centred Gaussian integration on native wgpu, and an
  independent ``subset_stride`` for overlapping shared-WGSL fields;
  stride defaults to size, with 16 x 16 retained as the qualified geometry;
- target prefiltering and registration on Three.js/TSL WebGPU by default;
- a fused, dynamically sized GPU-IO/WebGL2 implementation as an optional
  backend;
- an experimental in-process wgpu-py backend with direct NumPy texture upload,
  shared WGSL compute kernels, and one packed binary readback;
- a two-level default native full pyramid whose five-tap reductions, affine
  seed transfers, per-level solvers, and final readback remain on one GPU and
  one command submission;
- a bounded full-pyramid sequence API with independent mutable GPU resources
  per in-flight NumPy frame and ordered result delivery;
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

## Which backend is the current default?

There is no global default independent of the selected class:

| Intended use | Construction | Pyramid policy |
|---|---|---|
| **Python-supervised NumPy DIC and streaming** | `ICGNWGPUFullPyramid()` | Full pyramid, **2 levels by default and recommended** |
| Native single-scale/selective path | `ICGNWGPU()` | Single scale by default |
| Python-controlled browser compatibility | `ICGNWebGL()` | Three.js/TSL, single scale |
| Python-controlled browser using canonical WGSL | `ICGNWebGL(backend="webgpu-wgsl")` | Full pyramid, **2 levels by default** |
| Historical reproduction only | `ICGNWebGL(backend="gpuio")` | Deprecated single scale |

`ICGNWGPU(pyramid_levels=2 or 3)` is the older selective-recovery mechanism.
It first executes the fine solve and invokes CPU-orchestrated coarse recovery
only for failures. It remains available as an optional exception mode, but it
is not the recommended normal pyramid. The full decision table and execution
graphs are in the [backend-selection explanation](./DIC/LucasKanadeICGN/docs/explanation/backend-selection.rst).

All active backends accept `interpolation="bspline"`, `"linear"`, or
`"cubic"` at engine construction. `bspline` remains the compatibility default.
The direct modes skip both per-frame FIR passes; `linear` prioritizes speed,
while `cubic` retains substantially more subpixel accuracy. The deprecated
GPU-IO backend supports only `bspline`. See the
[interpolation explanation](./DIC/LucasKanadeICGN/docs/explanation/interpolation.rst).

## Architecture

The Python package exposes the browser implementations through one headless
contract and the native implementation through the same result schema.
Three.js/TSL is the default inside the browser-control class; GPU-IO remains
selectable for regression comparisons, and `ICGNWGPUFullPyramid` selects the
recommended browser-free Python path:

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
same-origin binary uint8 transfer
        |
        v
packed GPU upload / texture write
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
GPU readback --> packed binary field --> NumPy views
```

The reference remains a one-time encoded setup operation. Steady-state targets
do not use PNG, data URLs, DevTools payloads, image decoding, or JSON result
arrays. Browser memory is still separate from NumPy memory, so the binary
loopback transfer and GPU upload remain measurable.

The native path is shorter: `NumPy uint8 → r8unorm texture → WGSL FIR/IC-GN/
median → packed readback → NumPy views`. Its detailed resource and workgroup
layout is documented in the [native backend explanation](./DIC/LucasKanadeICGN/docs/explanation/native-backend.rst).

The specialized Three.js/WebGPU solver fuses the IC-GN iterations into one
compute dispatch and fills the measured hardware subgroup. On the validation
GPU, 32 lanes each process eight pixels of a default 16 x 16 subset. Its portable
fallback instead uses one 16-lane workgroup per subset and shared-memory
reductions. These solver groups are independent of the FIR tiles: the shared
WGSL prefilter uses 32 x 8 horizontal and 8 x 32 vertical output tiles.

The GPU-IO solver uses the WebGL2 execution model: one fragment owns one subset
and scans the selected square serially. Its iterations are fused into one draw,
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

The current Intel Lunar Lake audit uses eight warm-ups, 50 measured UHD
registrations, and 32,400 subsets. Native values are complete Python
calls; browser values are GPU timestamps and therefore exclude Python
transport and result readback:

| UHD path | Translation | Median | p95 | Accepted | RMSE |
|---|---:|---:|---:|---:|---:|
| Native full pyramid, 2 levels | `(4, -2)` | **10.863 ms** | **12.915 ms** | 31,724 | `1.873e-3` px |
| Raw browser WGSL, 2 levels | `(4, -2)` | **9.277 ms** | **10.681 ms** | 31,724 | `1.804e-3` px |
| Three.js/TSL subgroup | `(4, -2)` | 24.300 ms | 26.627 ms | 22,401 | `9.635e-2` px |
| Three.js/TSL subgroup | `(2, -1)` | **11.893 ms** | **12.798 ms** | 31,654 | `3.05e-5` px |

The difficult `(4, -2)` pair demonstrates why the full pyramid is the native
recommendation: it both recovers substantially more subsets and avoids failed
single-scale subsets consuming their iteration ceiling. GPU-IO is intentionally
absent, and no SYSCL row is currently claimed without a new equivalent run.

These values do **not** prove a sustained 60 Hz Python system:
the browser kernel benchmark excludes binary process transfer, result transfer,
and camera acquisition. Real-time qualification must
measure the complete `engine.register(frame)` call and the acquisition-to-result
latency on the target machine.

The timing protocol, current measurements, and reproduction commands are documented in
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

with ICGNWGPU(interpolation="linear") as engine:
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
  --benchmark --repeats 30 --warmups 3 --truth 2 -1 \
  --interpolation linear
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
├── BENCHMARK.md            current measurements and reproduction protocol
└── README.md               detailed implementation overview
```

The original GPU-IO demonstrations elsewhere in this repository remain
available from [`index.html`](./index.html).

## Current limitations

- dynamic grayscale frame dimensions in all backends; GPU-IO requires width
  and height to be multiples of the selected subset size;
- widths 8, 15, 16, 17, 31, 32, and 33 on native and raw browser WGSL, with an
  independent stride from 1 through the subset width; Three.js/TSL, WebGL2,
  and GPU-IO remain limited to widths 8, 16, and 32 with stride equal to size;
- Three.js/TSL and historical GPU-IO remain single-scale; the raw WGSL browser
  backend and native full-pyramid backend default to two levels and support
  three explicitly;
- no temporal or extrapolated initialization;
- no Aitken acceleration or predictive-tolerance guard;
- no camera, trigger, synchronization, or codec ownership;
- binary loopback transfer between NumPy and the separate Chromium process;
  per-frame PNG, DevTools payloads, decoding, and JSON fields have been removed,
  while the native path also removes the browser process copy;
- full result-field readback for every `register` call;
- soft real-time behavior only, with no operating-system deadline guarantee.

The direct `t=0 -> 20` displacement of `(4, -2)` pixels is outside the robust
local basin for many single-scale subsets. The current two-level native
full-pyramid campaign returned 31,724 accepted subsets out of 32,400 at UHD
with a translation RMSE of 0.001873 pixel. These values refer to that campaign's
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
