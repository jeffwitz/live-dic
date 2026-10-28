# Headless Python interface for affine IC-GN

> **Built on Pierre Margerit's foundational work.** This Python interface is an
> extension of [Pierre Margerit's original `tests_gpuio`
> project](https://github.com/MargeritPierre/tests_gpuio). His GPU-IO
> experiments and WebGL foundation made this real-time DIC path possible.

This package exposes the affine IC-GN solver in the parent directory to Python
without opening a browser window or loading graphical controls. Chromium is
used as an invisible runtime for either the WebGL2/GPU-IO backend or the
Three.js/TSL WebGPU backend.

## Requirements and installation

- Python 3.10 or newer;
- Chromium or Google Chrome with WebGL2 and WebGPU support;
- a compatible GPU and driver.

From the `tests_gpuio` repository root:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python
```

The wheel embeds the GPU-IO runtime and Three.js r186 runtime. The
installed runtime therefore needs neither a source checkout nor network access.
`repo_root=...` remains available as an explicit development override for a
backend-specific runtime directory or for a complete `tests_gpuio` checkout.

For a fully offline installation on another machine, prepare a wheelhouse on a
connected machine with the same operating system and Python version:

```bash
python3 -m pip wheel DIC/LucasKanadeICGN/python --wheel-dir wheelhouse
```

Copy `wheelhouse/` to the offline machine, where Chromium must already be
installed, then run:

```bash
python3 -m pip install --no-index --find-links wheelhouse \
  lucas-kanade-icgn-webgl
```

## Persistent engine

Reuse one engine for an image sequence. This keeps Chromium alive and prepares
the reference only once:

```python
from lucas_kanade_icgn import ICGNWebGL

reference = "DIC/LucasKanadeICGN/assets/frame_000.png"
targets = [
    "DIC/LucasKanadeICGN/assets/frame_001.png",
    "DIC/LucasKanadeICGN/assets/frame_010.png",
]

with ICGNWebGL() as engine:
    engine_info = engine.info()
    reference_info = engine.set_reference(reference)
    results = [engine.register(target) for target in targets]

result = results[-1]
print(engine_info.device)
print(reference_info.reference_build_wall_ms)
print(reference_info.total_wall_ms)
print(result.parameters.shape)       # (32, 32, 6)
print(result.displacement.shape)     # (32, 32, 2)
print(result.mean_translation)
print(result.accepted, result.replaced)
print(result.timing)
print(result.rmse((2.0, -1.0)))
```

The affine parameter order is:

```text
[tx, du/dx, du/dy, ty, dv/dx, dv/dy]
```

`status` and `iterations` both have shape `(32, 32)`. Status values are:

- `1`: accepted;
- `2`: replaced by the spatial median filter;
- `0`: did not converge within the iteration limit;
- negative: rejected.

Accepted image inputs are file paths, encoded PNG/JPEG bytes, Pillow images,
and 2D/RGB/RGBA NumPy-compatible arrays. The default Three.js backend derives
its dimensions from the reference; all targets must match. The optional GPU-IO
backend remains fixed at 512 x 512 pixels. Floating-point arrays with a maximum
value at or below 1 are interpreted as normalized images and scaled to 8-bit
grayscale.

`engine.register_many(reference, targets)` is a convenience method that also
prepares the reference once. The synchronous API works inside Jupyter notebooks
that already have an active `asyncio` event loop. An engine instance must not be
used concurrently from multiple threads.

`backend="three-tsl"` is the default. Use `backend="gpuio"` to select the
previous WebGL2 implementation. On Linux the default engine supplies Chromium's
native headless Vulkan flags and `engine.info().device` identifies the selected
adapter; check that it does not report SwiftShader before interpreting GPU
benchmarks.

## One-shot function

```python
from lucas_kanade_icgn import register_pair

result = register_pair(
    "DIC/LucasKanadeICGN/assets/frame_000.png",
    "DIC/LucasKanadeICGN/assets/frame_010.png",
)
```

The one-shot function starts and stops the runtime for each pair. Prefer the
persistent engine when processing more than one target.

## Algorithm parameters

`set_reference`, `register`, `register_pair`, `register_many`, `benchmark`, and
`benchmark_target` accept these keyword arguments:

- `iterations`: integer from 1 to 30, default 30;
- `tolerance`: positive convergence threshold in pixels, default `1e-3`;
- `zncc_min`: accepted ZNCC range threshold in `[0, 1]`, default `0.8`.

The runtime validates these values before launching the GPU calculation.

## Benchmarking

```python
with ICGNWebGL() as engine:
    preparation = engine.set_reference(reference)
    benchmark = engine.benchmark_target(targets[-1], repeats=30, warmups=3)

print(preparation.reference_build_wall_ms, preparation.total_wall_ms)
print(benchmark.gpu_median_ms, benchmark.gpu_p95_ms)
print(benchmark.wall_median_ms, benchmark.wall_p95_ms)
```

GPU time uses WebGL timer queries for GPU-IO and WebGPU timestamp queries for
Three.js/TSL. It includes target prefiltering, IC-GN, and median replacement.
Wall time excludes PNG conversion in Python, DevTools transport, and result
readback.

This explicit two-step form keeps reference construction out of the measured
optical-flow runs. `benchmark(reference, target, ...)` remains a convenience
wrapper when the preparation breakdown is not needed.

## Command-line interface

Register one pair and export the complete arrays:

```bash
icgn-webgl \
  DIC/LucasKanadeICGN/assets/frame_000.png \
  DIC/LucasKanadeICGN/assets/frame_010.png \
  --truth 2 -1 \
  --output /tmp/icgn-t10.npz
```

Benchmark the same pair:

```bash
icgn-webgl \
  DIC/LucasKanadeICGN/assets/frame_000.png \
  DIC/LucasKanadeICGN/assets/frame_010.png \
  --benchmark --repeats 30 --warmups 3 --truth 2 -1
```

Standard output is a compact JSON summary. `--output` writes the complete
`parameters`, `displacement`, `status`, `iterations`, and benchmark samples to
a compressed NumPy `.npz` archive. Run `icgn-webgl --help` for browser runtime
options and additional solver controls.

## Development checks

From the repository root:

```bash
ruff check DIC/LucasKanadeICGN/python
ruff format --check DIC/LucasKanadeICGN/python
PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 -m unittest discover -s DIC/LucasKanadeICGN/python/tests -v
node --check DIC/LucasKanadeICGN/LucasKanadeICGN.js
```
