# live-dic — Python-supervised DIC toward 4K60

> **Foundational work by [Pierre Margerit](https://github.com/MargeritPierre).**
> live-dic would not exist without his original
> [`tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio) project,
> GPU-IO experiments and WebGL groundwork. This project continues that work.

[![Python tests](https://github.com/jeffwitz/live-dic/actions/workflows/python-tests.yml/badge.svg)](https://github.com/jeffwitz/live-dic/actions/workflows/python-tests.yml)
[![Documentation](https://github.com/jeffwitz/live-dic/actions/workflows/docs.yml/badge.svg)](https://github.com/jeffwitz/live-dic/actions/workflows/docs.yml)

live-dic is a GPU digital image correlation engine for **Python measurement
applications**. A persistent Python supervisor provides NumPy images,
prepares the reference, schedules bounded work and consumes NumPy displacement,
gradient, curvature, status and iteration fields.

The engineering goal is **3840 × 2160 at 60 FPS**, with both throughput and
result age measured. Native 4K runs have completed on tested AMD, Intel and
NVIDIA configurations under Linux/Vulkan. Coverage differs by GPU: see the
[hardware matrix and performance reference](DIC/LucasKanadeICGN/docs/reference/performance.rst).
This is evidence for the recorded configurations, not certification of every
GPU, OS or architecture. Camera acquisition and downstream processing remain
outside the numerical engine's timing boundary.

## Browser acquisition and results workbench

A standalone **WebGPU application** now provides browser-local camera, image-folder and video
supervision, a bounded worker, editable multi-domain/full ROI with holes,
temporal prediction, GPU field display with displacement arrows/equivalent strain
and on-demand CSV/JSON export. A results workbench adds bounded opt-in field
recording, point/line/extensometer extraction, reversible displacement smoothing,
VSG strain, principal/equivalent strain, length units and NumPy NPZ interchange.
Fixed-reference and real consecutive-pair comparisons are distinct options; the
latter rebuilds each reference and does not inherit fixed-reference FPS claims. It imports the same portable WGSL core as the
Python raw-WebGPU adapter. It has no Three.js/CDN dependency and can run offline
after its first installation. The original UHD plate-with-hole case loads
with an editable rectangle/circle preset and both analytical load states.

From a checkout, run `python3 tools/run_browser_app.py` to build, serve and open
the workbench in your usual browser. It uses the browser’s existing GPU settings.

[Run the browser application locally or prepare Pages](DIC/LucasKanadeICGN/docs/how-to/run-in-browser.rst).
The Pages workflow is ready; repository-owner activation/deployment is separate.
Phone capture is a demonstration and field-screening tool, **not qualified for
metrology**. Native NumPy-to-NumPy timings and browser worker/display timings
have different boundaries; consult the performance reference.

## Start with the native Python engine

From the repository root, with Python 3.11 or newer:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/python -m pip install -e DIC/LucasKanadeICGN/python
```

The standard installation includes **wgpu-py**. Native processing requires a
working GPU driver, but **does not require Chromium, Cargo, a project-specific
binary extension or an external shader compiler**. The project's wheel is
pure Python plus WGSL and browser resources; dependency wheels provide their
own platform runtimes. See the [installation guide](DIC/LucasKanadeICGN/docs/how-to/install.rst)
for offline installation and software CPU adapters.

```python
import numpy as np
from PIL import Image
from lucas_kanade_icgn import ICGNWGPUFullPyramid

reference = np.array(Image.open("DIC/LucasKanadeICGN/assets/frame_000.png").convert("L"))
target = np.array(Image.open("DIC/LucasKanadeICGN/assets/frame_010.png").convert("L"))

with ICGNWGPUFullPyramid() as engine:
    preparation = engine.set_reference(reference)
    result = engine.register(target)
    print(engine.adapter_info)

print(result.mean_translation)       # Close to (2, -1) pixels for this fixture.
print(result.parameters.shape)       # (32, 32, 6)
print(result.displacement.shape)     # (32, 32, 2)
print(result.accepted_mask.sum())
result.save_npz("result.npz")
```

Reuse the engine for a sequence. Its default two-level full pyramid runs
coarse-to-fine on every target, with GPU-resident intermediate data:

```python
with ICGNWGPUFullPyramid() as engine:
    engine.set_reference(reference)
    for result in engine.register_stream(numpy_frames, pipeline_depth=1):
        consume(result)
```

`numpy_frames` and `consume` belong to the application. One slot minimizes
result age; the API default of two slots prioritizes throughput. Slot depth
and pyramid depth are independent. Native `close()` permanently releases the
owning GPU resources; repeated close is harmless, and returned NumPy fields
remain usable.

## Models, ROI and result policies

- **Motion:** translation, six-parameter affine (default), or twelve-parameter
  quadratic. Quadratic adds the displacement's second derivatives.
- **Sampling:** B-spline (default), bilinear or Keys cubic interpolation.
- **Geometry:** native square widths 8, 15, 16, 17, 31, 32 and 33; independent
  subset stride; uniform or Gaussian spatial integration.
- **ROI:** native translation and affine accept a reference-sized boolean
  mask. `True` means retain. Default `roi_policy="full"` selects only complete
  nominal windows; explicit `"partial"` uses qualified connected boundary
  supports. Raw browser WGSL also supports full ROI; Three.js/GPU-IO reject it.
- **Spatial policy:** default `outlier_policy="replace"` detects displacement
  outliers and replaces the complete local state. `"flag"` preserves raw
  values, `"disabled"` omits the pass, and `"median"` explicitly smooths every
  valid state. Smoothing is not outlier detection.
- **Initialization:** zero by default; temporal history, velocity prediction,
  fine correction and finest-level stagnation stopping are opt-in. The
  qualified temporal profile is documented separately from API defaults.

```python
mask = np.ones(reference.shape, dtype=bool)
mask[200:256, 200:256] = False
with ICGNWGPUFullPyramid(motion_model="affine") as engine:
    engine.set_reference(reference, roi_mask=mask)
    result = engine.register(target)
```

Excluded points are not zero-displacement measurements. Use `roi_selected`,
`reference_valid` and `accepted_mask`; exports preserve the ROI and strict JSON
configuration. See the [ROI guide](DIC/LucasKanadeICGN/docs/how-to/use-roi.rst)
and [result contract](DIC/LucasKanadeICGN/docs/reference/result-layout.rst).

## Select the appropriate execution path

| Use | Entry point | Default behavior |
|---|---|---|
| Python NumPy images and full fields | **`ICGNWGPUFullPyramid()`** | Native WebGPU, two fixed levels |
| Native single-scale studies | `ICGNWGPU()` | Single scale; optional selective recovery |
| Browser compatibility | `ICGNWebGL()` | Three.js/TSL, WebGPU with WebGL2 fallback |
| Shared-WGSL browser qualification | `ICGNWebGL(backend="webgpu-wgsl")` | Two-level full pyramid |
| Historical reproduction | `ICGNWebGL(backend="gpuio")` | Deprecated GPU-IO |

The **CLI `icgn-webgl` and one-shot `register_pair()` retain their browser
behavior**. For native production streams use the classes above. The Python
distribution is still `lucas-kanade-icgn-webgl`, imported as
`lucas_kanade_icgn`; the project name is live-dic.

Browser calls transfer images and fields between Python and Chromium. Binary
transport removes per-frame PNG/JSON encoding, but does not share NumPy memory.
The standalone app now supplies that supervisor for the portable raw-WebGPU
core. Three.js remains a separate Python compatibility backend. See
[supervision](DIC/LucasKanadeICGN/docs/explanation/supervision.rst).

## Documentation and evidence

- [First native registration](DIC/LucasKanadeICGN/docs/tutorials/first-registration.rst)
  and [live NumPy frames](DIC/LucasKanadeICGN/docs/how-to/process-live-frames.rst).
- [WebGPU explained for engineers](DIC/LucasKanadeICGN/docs/explanation/webgpu-fundamentals.rst),
  [GPU vocabulary](DIC/LucasKanadeICGN/docs/reference/gpu-vocabulary.rst), and
  [kernel optimization](DIC/LucasKanadeICGN/docs/how-to/optimize-webgpu.rst).
- [API](DIC/LucasKanadeICGN/docs/reference/api.rst),
  [backend selection](DIC/LucasKanadeICGN/docs/explanation/backend-selection.rst),
  and [limitations](DIC/LucasKanadeICGN/docs/explanation/limitations.rst).
- [Performance tables and hardware coverage](DIC/LucasKanadeICGN/docs/reference/performance.rst),
  [measurement protocol](DIC/LucasKanadeICGN/BENCHMARK.md), and
  [archived campaign index](DIC/LucasKanadeICGN/benchmarks/results/README.md).
- Checked-in native 4K [tensile](DIC/LucasKanadeICGN/assets/uhd_tensile/README.md)
  and [plate-hole](DIC/LucasKanadeICGN/assets/uhd_plate_hole/README.md) fixtures
  include ground truth, hashes and provenance.

Performance tables name their workload, configuration, GPU, revision and
population. Resident NumPy image → materialized NumPy field is the primary
Python boundary. GPU timestamps, throughput, p95 result age and end-to-end
camera latency answer different questions; do not compare them interchangeably.

## Development and project layout

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests, packaging, documentation and
validation outside the checkout. The maintained source is under
[`DIC/LucasKanadeICGN/python`](DIC/LucasKanadeICGN/python/); Sphinx/Diátaxis
sources are under [`docs`](DIC/LucasKanadeICGN/docs/).
The original GPU-IO demonstrations remain accessible from [index.html](index.html).
Third-party attributions are in
[THIRD_PARTY.md](DIC/LucasKanadeICGN/THIRD_PARTY.md).
