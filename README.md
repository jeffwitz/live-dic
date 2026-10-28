# Real-Time DIC in Python with WebGPU and GPU-IO

> **Foundational work by [Pierre Margerit](https://github.com/MargeritPierre).**
> This fork and its real-time Python DIC work would not exist without Pierre's
> original [`tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio)
> project. His GPU-IO experiments, repository structure, and WebGL groundwork
> are the foundation on which the work presented here is built.

This fork adds a Python-accessible, GPU-accelerated digital image correlation
(DIC) implementation to `tests_gpuio`. It performs affine
inverse-compositional Gauss-Newton registration through a default Three.js/TSL
WebGPU backend and an optional GPU-IO/WebGL2 backend. The dynamic Three.js path
has been validated from 512 x 512 through UHD 3840 x 2160 and is designed as
the numerical core of a soft real-time Python processing pipeline.

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
- the previous fixed-512 GPU-IO/WebGL2 implementation as an optional backend;
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

The Python package drives either browser implementation through the same
headless contract. Three.js/TSL is the default; GPU-IO remains selectable for
regression comparisons:

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

The Three.js solver fuses the IC-GN iterations into one compute dispatch and
uses 16 cooperating lanes per subset. A startup probe selects hardware subgroup
reductions when all lanes fit in one subgroup and otherwise selects the
portable workgroup-memory solver.

Chromium is used only as an invisible WebGPU or WebGL2 execution environment.
The Python entry point does not load Tweakpane, attach the canvas to the visible
page, or open a browser window. One persistent engine should be reused for a
sequence so that browser startup and reference preparation are not repeated
for every frame.

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
| Three.js/TSL WebGPU | **0.672 ms** | **1.225 ms** | **2.7 ms** |
| GPU-IO/WebGL2 | 3.768 ms | 4.362 ms | 16.6 ms |

The reproducible UHD pair contains 32,400 subsets and an exact `(2, -1)` pixel
translation:

| UHD compute boundary | Median | p95 | Accepted | RMSE |
|---|---:|---:|---:|---:|
| Three.js/TSL | **13.341 ms** | **14.405 ms** | 31,654 | `3.076e-5` px |
| SYCL mixed/double reduction | 19.099 ms | 19.863 ms | 31,654 | `2.555e-5` px |
| SYCL `--float-reduce` | 18.111 ms | 19.062 ms | 31,654 | `2.712e-5` px |

At the comparable UHD compute boundary, Three.js is 30.1% faster than the
mixed/double-reduction SYCL configuration and 26.3% faster than the numerically
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

Install the local Python package:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python
```

The Python package embeds GPU-IO, Three.js r186, both IC-GN solvers, and their
headless HTML entry points. After Python dependencies and Chromium have been
installed, the runtime performs no network requests and does not require the
repository checkout. The optional browser demonstration also uses only
vendored scripts; see
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
├── benchmarks/             reproducible backend and UHD campaigns
├── docs/                   Sphinx/Diátaxis documentation
├── BENCHMARK.md            measurement protocol and SYCL comparison
└── README.md               detailed implementation overview
```

The original GPU-IO demonstrations elsewhere in this repository remain
available from [`index.html`](./index.html).

## Current limitations

- dynamic grayscale frame dimensions in the default Three.js backend; the
  optional GPU-IO backend remains fixed at 512 x 512;
- fixed 16 x 16 subsets on a 16-pixel grid;
- single-scale registration with no pyramid;
- no temporal or extrapolated initialization;
- no Aitken acceleration or predictive-tolerance guard;
- no camera, trigger, synchronization, or codec ownership;
- PNG and DevTools transport between Python and the browser GPU runtime;
- full result-field readback for every `register` call;
- soft real-time behavior only, with no operating-system deadline guarantee.

The direct `t=0 -> 20` displacement of `(4, -2)` pixels is outside the robust
local basin for many subsets. The incremental `t=10 -> 20` step of `(2, -1)`
pixels returns 900 accepted subsets. Larger motions will require a pyramid or a
better initial estimate.

## Verification

The current code has been verified with:

- 15 Python unit tests;
- Ruff linting and formatting checks;
- JavaScript syntax checking;
- wheel build and installation;
- real GPU execution through the installed package;
- strict Sphinx builds with warnings treated as errors;
- visual inspection of the generated landing page and API reference.

Run the local checks:

```bash
ruff check DIC/LucasKanadeICGN/python
ruff format --check DIC/LucasKanadeICGN/python
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 -m unittest discover -s DIC/LucasKanadeICGN/python/tests -v
node --check DIC/LucasKanadeICGN/LucasKanadeICGN.js
node --check DIC/LucasKanadeICGN/three/LucasKanadeICGNThree.js
```

## Acknowledgements and provenance

This fork is a continuation of
[Pierre Margerit's `tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio),
whose experiments and WebGL structure made the present work possible.
The original demonstrations remain published at
[margeritpierre.github.io/tests_gpuio](https://margeritpierre.github.io/tests_gpuio/).

The repository uses [GPU-IO](https://github.com/amandaghassaei/gpu-io), created
by Amanda Ghassaei, as its WebGL computation framework.
