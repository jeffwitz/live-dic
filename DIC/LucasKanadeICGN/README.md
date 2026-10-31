# live-dic numerical engine

> **Foundational work by [Pierre Margerit](https://github.com/MargeritPierre).**
> His original [`tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio)
> project, GPU-IO experiments and WebGL groundwork made this work possible.

This directory contains live-dic's Python-supervised digital image correlation
engine, its browser compatibility runtimes, analytic image fixtures and
Sphinx/Diátaxis documentation. The goal is qualified **4K at 60 FPS**, measured
from resident NumPy image to materialized NumPy field, with acquisition and
result consumption accounted for separately in the application.

Start with the [project README](../../README.md) for installation and a native
Python example, or the [package README](python/README.md) for API usage.

The [browser companion](docs/how-to/run-in-browser.rst) runs a local camera, video, image folder or
analytic demo through the shared WebGPU core, with worker backpressure, full
ROI, temporal history, GPU display and offline assets. Its results lab preserves
computed fields while adding explicit smoothing, strain fitting, point/line/gauge
measurements and NPZ export/import. Recording is optional and adds full readback. Pages deployment is
prepared but requires the repository owner's setup. Hand-held camera use is
not qualified for metrology. Python remains the primary NumPy supervisor.

## Algorithm and execution

The native `ICGNWGPUFullPyramid` engine runs every configured pyramid level on
every target. It uploads contiguous uint8 NumPy bytes directly, retains all
intermediate images and states on one WebGPU device, and reads back the final
field. Reference gradients and inverse Hessians are prepared once. There is
no Chromium process, per-frame PNG encoding or external shader compiler.

The core estimates translation, affine or quadratic motion using normalized
inverse-compositional Gauss-Newton updates. Interpolation, subset geometry,
spatial weighting and result policies are explicit choices:

- native subset widths: 8, 15, 16, 17, 31, 32, 33; independent stride;
- B-spline, bilinear and Keys cubic target interpolation;
- uniform or centered Gaussian integration;
- reference ROI for translation and affine: complete windows by default,
  optional partial supports at boundaries;
- displacement-only UOD replacement by default, flag-only and raw modes, plus
  an explicit component median smoothing mode;
- optional temporal history, velocity prediction, fine correction and
  finest-level stagnation stopping; two fixed levels remain the default.

The default six-component order is
`[tx, du/dx, du/dy, ty, dv/dx, dv/dy]`. Quadratic adds six second derivatives;
use `result.parameter_order`. Do not interpret excluded or rejected states
as displacement measurements.

## Source responsibilities

| Path | Responsibility |
|---|---|
| `python/src/lucas_kanade_icgn/` | Public API, native engines, ROI, CLI and result types |
| `python/src/lucas_kanade_icgn/shaders/` | Canonical portable WGSL and native subgroup specialization |
| `python/src/lucas_kanade_icgn/web/` | Packaged shared JS core, compatibility adapters and local vendor modules |
| `web-app/` | Standalone camera/demo PWA; assembled with the canonical core and WGSL |
| `three/` | Checkout entry point delegating to the packaged Three.js source |
| `LucasKanadeICGN.js`, `headless.html` | Historical GPU-IO checkout runtime |
| `python/tests/` | Portable, GPU and analytic-UHD regression tests |
| `assets/` | Traceable fixtures, masks, analytical truth and hashes |
| `benchmarks/` | Active measurement tools |
| `benchmarks/results/` | Immutable campaign evidence and reproduction scripts |
| `docs/` | Sphinx/Diátaxis documentation |

Native portable wgpu-py and raw browser WebGPU load the same WGSL equations.
Three.js/TSL and the native subgroup specialization are separate implementations
that require numerical comparison when formulas change. The native workgroup
width is 16 by default; 32 and experimental 128 are opt-in. These widths are
unrelated to subset size, stream depth or pyramid levels.

## Backend boundaries

`ICGNWGPUFullPyramid` is recommended for Python NumPy streams.
`ICGNWGPU` is single-scale by default, with the older selective-recovery path
available explicitly. `ICGNWebGL` retains Three.js as its browser default,
including a WebGL2 fallback. Raw browser WGSL has a full pyramid; Three.js and
historical GPU-IO are single-scale. Raw WGSL supports full ROI; Three.js/GPU-IO
still reject masks.

The browser adapter transports raw target bytes and packed fields, but still
crosses the Python/Chromium process boundary. The standalone PWA supplies a
JavaScript supervisor for browser-local raw WebGPU processing and GPU display. The existing CLI and
one-shot helper are browser interfaces; their defaults have not been changed.

Use native engines as context managers or call `close()`. Native close is
permanent, idempotent, and releases the owning device and its GPU references.
Shared pyramid levels never destroy the parent's device. Keep one engine alive
for a sequence and finish a stream before changing reference or temporal state.

## Documentation and validation

The [documentation index](docs/index.rst) links tutorials, how-to guides,
reference contracts and technical explanations. The
[WebGPU introduction](docs/explanation/webgpu-fundamentals.rst) assumes
engineering/Python knowledge rather than graphics-programming experience.

The [performance reference](docs/reference/performance.rst) is the canonical
location for numerical timing tables and the GPU coverage matrix. It separates
Linux/Vulkan executions on AMD, Intel and NVIDIA from portable CI, software
Vulkan checks and unqualified deployment platforms. Read
[BENCHMARK.md](BENCHMARK.md) for the protocol and
[the campaign index](benchmarks/results/README.md) for raw evidence.

[CONTRIBUTING.md](../../CONTRIBUTING.md) defines local checks, wheel validation
and the distinction between portable tests, GPU correctness and performance.
No single timing or test count in a landing page is a substitute for a
revision-specific report.

Third-party browser assets are vendored for offline operation; see
[THIRD_PARTY.md](THIRD_PARTY.md). The mathematical port also draws on the
related SYCL DIC prototype; that provenance is documented in
[acknowledgements](docs/explanation/acknowledgements.rst).

The browser companion also includes an editable multi-domain mask (rectangles,
polygons, ellipses, circles and nested holes/islands), displacement arrows and
plane-strain von Mises strain display. The original UHD plate-with-hole case
loads its editable rectangle/circle preset and both analytical load states.
See the [browser guide](docs/how-to/run-in-browser.rst) for the scientific
conventions and the distinction between static scoring and video timing.
