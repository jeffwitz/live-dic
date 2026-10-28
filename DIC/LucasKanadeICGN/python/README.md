# Headless Python interface for affine IC-GN

> **Built on Pierre Margerit's foundational work.** This Python interface is an
> extension of [Pierre Margerit's original `tests_gpuio`
> project](https://github.com/MargeritPierre/tests_gpuio). His GPU-IO
> experiments and WebGL foundation made this real-time DIC path possible.

This package exposes the affine IC-GN solver in the parent directory to Python
without opening a visible browser window or loading graphical controls.
Chromium is used as an invisible runtime for WebGL2/GPU-IO, Three.js/TSL, and
the raw WebGPU shared-WGSL backend. The optional `ICGNWGPU` backend runs
in-process and does not start Chromium.

## Requirements and installation

- Python 3.10 or newer;
- Chromium or Google Chrome with WebGL2 support; WebGPU is preferred;
- a compatible GPU and driver.

From the `tests_gpuio` repository root:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python
```

For the experimental native WebGPU path, use Python 3.11 or newer and install
the optional dependency group:

```bash
.venv-icgn/bin/pip install -e 'DIC/LucasKanadeICGN/python[native]'
```

The wheel embeds the GPU-IO runtime, Three.js r186 runtime, raw WebGPU adapter,
canonical WGSL, and first-party subgroup GLSL/SPIR-V shaders. The installed
runtime therefore needs neither a source checkout, shader compiler, nor
network access.
`repo_root=...` remains available as an explicit development override for a
backend-specific runtime directory or for a complete `tests_gpuio` checkout.

For a fully offline installation on another machine, prepare a wheelhouse on a
connected machine with the same operating system and Python version:

```bash
python3 -m pip wheel 'DIC/LucasKanadeICGN/python[native]' --wheel-dir wheelhouse
```

Copy `wheelhouse/` to the offline machine, where Chromium is required only for
the browser backends, then run:

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
print(engine_info.solver, engine_info.subgroup_width)
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
backend also derives its allocation from the reference, with width and height
restricted to multiples of 16. Floating-point arrays with a maximum
value at or below 1 are interpreted as normalized images and scaled to 8-bit
grayscale.

`engine.register_many(reference, targets)` is a convenience method that also
prepares the reference once. The synchronous API works inside Jupyter notebooks
that already have an active `asyncio` event loop. An engine instance must not be
used concurrently from multiple threads.

`backend="three-tsl"` is the default. It selects WebGPU when available and
falls back to a serial Three.js/TSL transform-feedback solver on WebGL2. Use
`force_three_webgl=True` to qualify that fallback even on a WebGPU-capable
machine. Use `backend="gpuio"` to select the historical fused WebGL2
implementation. Use `backend="webgpu-wgsl"` to run the
same portable WGSL file as wgpu-py through raw browser WebGPU. On Linux the
default engine supplies Chromium's native headless Vulkan flags and
`engine.info().device` identifies the selected adapter; check that it does not
report SwiftShader before interpreting GPU benchmarks. `engine.info().solver`
reports either `subgroup` or `portable-workgroup`, and `subgroup_width` records
the measured hardware width.

The raw ``webgpu-wgsl`` backend uses the shared GPU-resident full pyramid and
defaults to two levels. Configure its coarse-to-fine schedules on the engine:

```python
with ICGNWebGL(
    backend="webgpu-wgsl",
    pyramid_levels=3,
    iterations_per_level=(6, 8, 20),
    tolerance_per_level=(1e-2, 5e-3, 1e-3),
) as engine:
    engine.set_reference(reference)
    result = engine.register(target)
```

Use ``pyramid_levels=1`` for the original single-scale execution path. It does
not allocate reduced levels or dispatch reduction and seed-transfer kernels.
The Three.js/TSL and GPU-IO backends are still single-scale and reject pyramid
options explicitly.

The explicit WebGL2 compatibility form is:

```python
with ICGNWebGL(backend="three-tsl", force_three_webgl=True) as engine:
    engine.set_reference(reference)
    result = engine.register(target)

assert engine.info().solver == "serial-webgl2"
```

## Native WebGPU engine

Use `ICGNWGPU` when the input already exists as a NumPy array and browser
transport would dominate the latency:

```python
from lucas_kanade_icgn import ICGNWGPU

with ICGNWGPU() as engine:
    reference_info = engine.set_reference(reference_frame)
    result = engine.register(target_frame)

print(reference_info.total_wall_ms)
print(engine.solver, engine.subgroup_width, engine.solver_tuning_ms)
print(result.timing)
print(result.displacement)
```

The engine uploads contiguous grayscale bytes directly to an `r8unorm`
texture and returns NumPy views backed by one packed result readback. The FIR,
affine IC-GN solver, and median replacement all run on the native GPU. The
result schema and parameter order are identical to `ICGNWebGL`. This backend
is experimental and must be selected explicitly. At reference preparation,
the default `prefer_subgroups=None` policy times both available native solvers
and retains the faster one. Use `prefer_subgroups=False` to force portable
WGSL or `prefer_subgroups=True` to force SPIR-V subgroup reductions.

For a sustained sequence, keep two frames bounded in flight so Python can
retire one packed readback while the GPU processes the next target:

```python
with ICGNWGPU(prefer_subgroups=False) as engine:
    engine.set_reference(reference_frame)
    for result in engine.register_stream(frame_source):
        consume(result)
```

The sequence APIs keep at most two images in flight by default. This is an
execution pipeline, not an image pyramid. Results are yielded in input order.
The default improves throughput but adds approximately one processed frame of latency.
Keep ``register()`` for the lowest latency on isolated pairs.
``register_many()`` uses the same default and returns a list when all results
must be retained.

For displacements outside the local single-scale basin, enable selective
pyramid recovery explicitly:

```python
with ICGNWGPU(prefer_subgroups=False, pyramid_levels=2) as engine:
    engine.set_reference(reference_frame)
    result = engine.register(target_frame)
```

``pyramid_levels`` counts the fine image, so ``1`` means no pyramid and remains
the default. With two or three scales, reduced references are prepared once.
For each target, the ordinary fine solve runs first; a five-tap pyramid and a
coarse-to-fine seeded retry are executed only when fine subsets fail. The
coarse dependency mask contains only the interpolation support required by
those failures. Parameters, statuses, and iteration counts already accepted by
the fine solve are preserved bit for bit. Two scales are the recommended
recovery setting; a third scale should be enabled only after qualification on
the expected displacement field.

When coarse-to-fine registration is expected on every frame, use the separate
GPU-resident backend:

```python
from lucas_kanade_icgn import ICGNWGPUFullPyramid

with ICGNWGPUFullPyramid(
    pyramid_levels=2,
    iterations_per_level=(12, 30),
    tolerance_per_level=(5e-3, 1e-3),
) as engine:
    engine.set_reference(reference_frame)
    result = engine.register(target_frame)
```

The schedules are ordered from coarse to fine. The finest target is uploaded
once; reductions, prefilters, affine seed transfers, and all solves stay on one
WebGPU device in one command submission. Only the finest packed field is read
back. The selective `ICGNWGPU` recovery path remains useful when large motion
is exceptional, while `ICGNWGPUFullPyramid` avoids its CPU reductions and
intermediate readbacks when a pyramid is the normal case.

Two levels are the default because they provided the best measured 4K60 margin
on the validation GPU. Select three levels explicitly for a larger expected
capture range; its robust default iteration schedule is `(6, 8, 20)`.

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
Three.js/TSL and raw WGSL. It includes target prefiltering, IC-GN, and median replacement.
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
  --backend webgpu-wgsl --pyramid-levels 2 \
  --benchmark --repeats 30 --warmups 3 --truth 2 -1
```

Use ``--iterations-per-level`` and ``--tolerance-per-level`` to override the
coarse-to-fine schedules; each list must contain exactly one value per level.
Use ``--backend three-tsl --force-three-webgl`` to exercise the WebGL2 fallback.

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
