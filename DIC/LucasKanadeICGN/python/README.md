# Headless Python interface for IC-GN WebGL2

> **Built on Pierre Margerit's foundational work.** This Python interface is an
> extension of [Pierre Margerit's original `tests_gpuio`
> project](https://github.com/MargeritPierre/tests_gpuio). His GPU-IO
> experiments and WebGL foundation made this real-time DIC path possible.

This package exposes the affine IC-GN solver in the parent directory to Python
without opening a browser window or loading the graphical controls. Chromium is
still used as an invisible WebGL2 runtime so the Python path executes the same
GPU shaders as the browser implementation.

## Requirements and installation

- Python 3.10 or newer;
- Chromium or Google Chrome;
- a checkout containing `gpu-io.js` and
  `DIC/LucasKanadeICGN/headless.html`.

From the `tests_gpuio` repository root:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python
```

The repository root is discovered automatically from the working directory and
package location. Pass `repo_root=...` to `ICGNWebGL` when using a different
layout.

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
print(reference_info.preparation_wall_ms)
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
and 2D/RGB/RGBA NumPy-compatible arrays. Every input must be exactly 512 x 512
pixels. Floating-point arrays with a maximum value at or below 1 are interpreted
as normalized images and scaled to 8-bit grayscale.

`engine.register_many(reference, targets)` is a convenience method that also
prepares the reference once. The synchronous API works inside Jupyter notebooks
that already have an active `asyncio` event loop. An engine instance must not be
used concurrently from multiple threads.

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

`set_reference`, `register`, `register_pair`, `register_many`, and `benchmark`
accept these keyword arguments:

- `iterations`: integer from 1 to 30, default 30;
- `tolerance`: positive convergence threshold in pixels, default `1e-3`;
- `zncc_min`: accepted ZNCC range threshold in `[0, 1]`, default `0.8`.

The runtime validates these values before launching the GPU calculation.

## Benchmarking

```python
with ICGNWebGL() as engine:
    benchmark = engine.benchmark(reference, targets[-1], repeats=30, warmups=3)

print(benchmark.gpu_median_ms, benchmark.gpu_p95_ms)
print(benchmark.wall_median_ms, benchmark.wall_p95_ms)
```

GPU time is returned when `EXT_disjoint_timer_query_webgl2` is available. Wall
time includes target prefiltering and the solver, but excludes PNG conversion in
Python and DevTools transport.

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
