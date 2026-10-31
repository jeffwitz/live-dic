# live-dic Python package

> **Built on Pierre Margerit's foundational work.** His original
> [`tests_gpuio`](https://github.com/MargeritPierre/tests_gpuio) project,
> GPU-IO experiments and WebGL groundwork made this implementation possible.

live-dic provides Python-supervised digital image correlation on NumPy images.
The recommended engine, `ICGNWGPUFullPyramid`, uses **native wgpu-py**, keeps
intermediate pyramid data on the GPU and returns NumPy fields. Chromium is
needed only for browser compatibility backends.

The engineering goal is qualified **4K60**, with throughput and result age
reported separately. See the [performance and hardware coverage reference](https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/docs/reference/performance.rst)
for measured workloads and limitations.

## Installation and stable names

With Python 3.11 or newer, from this `python/` directory:

```bash
python3 -m pip install .
```

Or from the repository root:

```bash
python3 -m pip install -e DIC/LucasKanadeICGN/python
```

| Name | Value |
|---|---|
| Project | live-dic |
| Python distribution | `lucas-kanade-icgn-webgl` |
| Import | `lucas_kanade_icgn` |
| Existing browser CLI | `icgn-webgl` |

The standard dependencies are NumPy, Pillow, websockets and wgpu-py. The
project wheel contains Python, WGSL and vendored browser resources. It has no
first-party compiled extension and needs no Cargo, Maturin or external shader
compiler. Dependency wheels supply their platform runtimes; GPU/software
adapter drivers and optional Chromium are system prerequisites.

After installing dependencies, execution is offline and independent of a
repository checkout. The large scientific image fixtures are in the repository,
not embedded in the wheel. See the [installation guide](https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/docs/how-to/install.rst)
for offline wheelhouses and CPU adapter setup.

## First native registration

Run this example from the repository root:

```python
import numpy as np
from PIL import Image
from lucas_kanade_icgn import ICGNWGPUFullPyramid

reference = np.array(Image.open("DIC/LucasKanadeICGN/assets/frame_000.png").convert("L"))
target = np.array(Image.open("DIC/LucasKanadeICGN/assets/frame_010.png").convert("L"))

with ICGNWGPUFullPyramid() as engine:
    reference_info = engine.set_reference(reference)
    result = engine.register(target)
    print(engine.adapter_info)

print(result.mean_translation)  # Approximately (2, -1) pixels.
print(result.parameters.shape)  # (32, 32, 6)
print(result.accepted_mask.sum())
result.save_npz("result.npz")
```

`parameters` contains `[tx, du/dx, du/dy, ty, dv/dx, dv/dy]` for affine
motion. `displacement` is a memory-sharing view selecting `tx` and `ty`.
Quadratic motion appends six second derivatives; use `parameter_order`.
States also include `status` and `iterations`. Always use `accepted_mask`
when computing measurement statistics.

Use a context manager or call `close()`. Native close permanently releases
the owning device and GPU references; repeated close is harmless. Create a
new engine after close. Result arrays are CPU data and remain usable.

## Persistent NumPy streams

```python
with ICGNWGPUFullPyramid() as engine:
    engine.set_reference(reference)
    for field in engine.register_stream(numpy_frames, pipeline_depth=1):
        consume(field)
```

The application supplies `numpy_frames` and `consume`. Every slot owns its
mutable target/seed/readback storage; references and pipelines are shared.
Results are yielded in order. One slot minimizes result age; the API default
is two slots for throughput. `pyramid_levels` counts image scales and is
independent of `pipeline_depth`.

For cumulative motion, explicitly enable `initialization="temporal"`.
Chronological inputs are required, and every configured level still runs
under the default fixed policy. Velocity prediction and fine correction are
additional opt-in controls; see the [temporal guide](https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/docs/how-to/use-temporal-initialization.rst).
Finish consuming a stream before resetting history or replacing its reference.

## ROI and result policies

Native translation and affine accept a boolean mask matching the reference:

```python
mask = np.ones(reference.shape, dtype=bool)
mask[200:256, 200:256] = False
with ICGNWGPUFullPyramid() as engine:
    engine.set_reference(reference, roi_mask=mask, roi_policy="full")
    result = engine.register(target)
```

`True` retains pixels. The default `full` policy selects a subset only when
all pixels of its nominal square are allowed. Explicit `partial` uses fixed
connected boundary supports. Browser engines reject ROI options.
`roi_selected`, `reference_valid` and `accepted_mask` represent different
populations; excluded points are not measurements of zero motion. NPZ exports
include mask data, selection and strict `config_json`, readable with
`allow_pickle=False`. See the [ROI guide](https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/docs/how-to/use-roi.rst).

| `outlier_policy` | Meaning |
|---|---|
| `replace` (default) | Detect displacement outliers; replace the complete local state |
| `flag` | Detect the same outliers; retain raw coefficients with status 2 |
| `disabled` | Omit the spatial pass |
| `median` | Explicitly smooth each valid state; preserve solver status |

The median mode is smoothing, not an outlier detector. Status 2 therefore
means detection, not necessarily replacement. `result.flagged` and
`outlier_mask` express this; older `replaced` aliases remain for compatibility.

## Supported choices and defaults

- `motion_model`: `affine` (default), `translation`, `quadratic`.
- `interpolation`: `bspline` (default), `linear`, `cubic`.
- Native `subset_size`: 8, 15, 16 (default), 17, 31, 32, 33; independent
  `subset_stride` defaults to size. Spatial integration is uniform by default,
  with optional Gaussian weighting.
- `device_type`: `auto` (default), `gpu`, `cpu`. CPU requires an installed
  software WebGPU adapter; no driver is downloaded at runtime.
- `solver_workgroup_size`: 16 (default), 32, experimental 128. Non-default
  widths use the portable solver; no automatic width selection is performed.

Native full pyramid defaults to two levels. The single-scale `ICGNWGPU`
engine instead defaults to one; its optional selective retry is a separate
compatibility strategy. The [API reference](https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/docs/reference/api.rst)
defines schedules, validation, model coverage and all temporal controls.

Inputs can be image paths, encoded image bytes, Pillow images or supported
NumPy arrays. Inputs normalize to uint8 grayscale. Higher-bit-depth scientific
images need an explicit shared intensity conversion for reference and targets;
the API does not silently map an arbitrary 12/16-bit range.

## Browser compatibility and CLI

`ICGNWebGL()` selects Three.js/TSL with WebGPU and automatic WebGL2 fallback.
`backend="webgpu-wgsl"` selects the shared WGSL full pyramid. Historical
`backend="gpuio"` is deprecated. Browser targets use binary transport, but
images and full fields still cross between Python and Chromium. A separate
JavaScript stream supervisor is not provided by live-dic.

The one-shot `register_pair()` and existing `icgn-webgl` CLI retain their
browser defaults. Native applications should use the persistent classes.

```bash
icgn-webgl --help
icgn-webgl DIC/LucasKanadeICGN/assets/frame_000.png \
  DIC/LucasKanadeICGN/assets/frame_010.png --truth 2 -1 --output result.npz
```

See the [CLI reference](https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/docs/reference/command-line.rst).
Development and release validation are described in
[CONTRIBUTING.md](https://github.com/jeffwitz/live-dic/blob/main/CONTRIBUTING.md).
