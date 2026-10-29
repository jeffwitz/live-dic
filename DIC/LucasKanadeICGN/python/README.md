# Headless Python interface for GPU IC-GN

> **Built on Pierre Margerit's foundational work.** This Python interface is an
> extension of [Pierre Margerit's original `tests_gpuio`
> project](https://github.com/MargeritPierre/tests_gpuio). His GPU-IO
> experiments and WebGL foundation made this real-time DIC path possible.

This package exposes the IC-GN solvers in the parent directory to Python
without opening a visible browser window or loading graphical controls.
Chromium is used as an invisible runtime for Three.js/TSL on WebGPU or WebGL2,
the raw WebGPU shared-WGSL backend, and the deprecated GPU-IO compatibility
backend. The optional `ICGNWGPU` and `ICGNWGPUFullPyramid` backends run
in-process and do not start Chromium.

## Intended supervision model

The package is primarily a Python-supervised numerical engine. A long-running
Python application is expected to own acquisition integration, reference
changes, algorithm configuration, bounded queues, timing, error handling, and
the downstream use of NumPy result fields. For that deployment model,
`ICGNWGPUFullPyramid` is the recommended coarse-to-fine backend: it submits
WebGPU work in-process and does not route each frame through Chromium.

`ICGNWebGL` is a Python control adapter for the Three.js, raw-WGSL, and legacy
GPU-IO browser runtimes. Three.js itself can execute the GPU graph quickly, but
Python-supervised browser calls still cross a process boundary in both
directions. The binary transport removes PNG and JSON serialization; it does
not provide shared memory. Current complete UHD calls are about 45--47 ms
median.

Reaching competitive browser-stream performance would require a persistent
JavaScript supervisor that acquires or receives browser-native frames, owns the
GPU resources and queue, applies backpressure, and consumes results without
returning every full field to Python. Python may remain the configuration and
monitoring plane. That JavaScript supervisor is not currently part of this
package. If the source is NumPy and full fields are required in Python, select
the native backend.

## Requirements and installation

- Python 3.10 or newer;
- Chromium or Google Chrome with WebGL2 support; WebGPU is preferred;
- a compatible GPU and driver.

From the `tests_gpuio` repository root:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python
```

For the native WebGPU path, use Python 3.11 or newer and install
the optional dependency group:

```bash
.venv-icgn/bin/pip install -e 'DIC/LucasKanadeICGN/python[native]'
```

The wheel embeds the Three.js r186 runtime, raw WebGPU adapter, portable and
subgroup-specialized WGSL shaders, and the deprecated GPU-IO runtime. The
installed runtime therefore needs neither a source checkout, external shader
compiler, Rust toolchain, nor network access.
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

## Backend selection

Defaults are scoped to the selected engine, not to the whole package:

| Use case | Construction | Default pyramid |
|---|---|---|
| **Python-supervised NumPy processing and streaming** | `ICGNWGPUFullPyramid()` | full pyramid, two levels |
| Native single-scale/selective processing | `ICGNWGPU()` | one scale |
| Python-controlled browser with automatic WebGPU/WebGL2 fallback | `ICGNWebGL()` | Three.js/TSL, one scale |
| Python-controlled browser shared WGSL | `ICGNWebGL(backend="webgpu-wgsl")` | full pyramid, two levels |

`ICGNWGPU(pyramid_levels=2 or 3)` selects the older selective-recovery path.
It is retained for exceptional failures and compatibility, but it is not the
recommended normal pyramid.

## Interpolation choice

Every active engine accepts an interpolation mode at construction:

```python
with ICGNWGPUFullPyramid(interpolation="linear") as engine:
    engine.set_reference(reference)
    result = engine.register(target)
```

The square subset width and the result-grid spacing can be selected
independently. Stride defaults to size, so existing code retains the qualified
16-pixel geometry:

```python
with ICGNWGPUFullPyramid(subset_size=32, subset_stride=16) as engine:
    info = engine.set_reference(reference)
    result = engine.register(target)

assert info.subset_size == 32
assert info.subset_stride == 16
assert result.parameters.shape[:2] == (134, 239)  # for a 3840 x 2160 frame
```

The shared-WGSL sizes are 8, 15, 16, 17, 31, 32, and 33; odd widths provide an
exact centre pixel. Stride is an integer from 1 through the selected size.
Independent stride is available in native ``wgpu-py`` and browser
``webgpu-wgsl``. Three.js/TSL, WebGL2, and GPU-IO use widths 8, 16, or 32 and
require stride equal to size. Any native size or stride other than 16 uses the
portable WGSL solver because the optional subgroup WGSL source is specialized
for 16 x 16.

Native engines can use a centred Gaussian integration window:

```python
with ICGNWGPUFullPyramid(
    subset_size=31,
    subset_stride=16,
    spatial_weighting="gaussian",
    gaussian_sigma=8.0,
) as engine:
    engine.set_reference(reference)
    result = engine.register(target)
```

The same weights are applied to all photometric statistics, Hessian terms,
and residual projections. Uniform integration remains the default. Gaussian
weighting is not yet exposed by the browser hosts.

The choices are `"bspline"` (historical default), `"linear"`, and `"cubic"`
(Keys cubic convolution). The two direct modes skip target FIR prefiltering.
Changing the mode requires preparing the reference again. GPU-IO is deprecated
and accepts only `"bspline"`.

## Motion model

The shared-WGSL engines also accept `motion_model="quadratic"` in addition to
`"affine"` (the default) and `"translation"`:

```python
with ICGNWGPUFullPyramid(motion_model="translation") as engine:
    engine.set_reference(reference)
    result = engine.register(target)
```

Translation mode estimates only `tx` and `ty` with an analytic 2 × 2 Hessian
inverse. It keeps the public six-parameter result layout and writes exact zeros
to `du/dx`, `du/dy`, `dv/dx`, and `dv/dy`. Use it for rigid displacement-only
work, not for strain measurement. Quadratic mode returns twelve components: the
six affine terms followed by the six second displacement derivatives. It is
available in native wgpu and `backend="webgpu-wgsl"`; Three.js and deprecated
GPU-IO reject it.
Changing the model requires a new engine and new reference preparation.

## Displacement outlier policy

The default tests only recentered `tx` and `ty`, then replaces the complete
motion state if either is a UOD outlier:

```python
with ICGNWGPUFullPyramid(
    outlier_policy="replace",
) as engine:
    engine.set_reference(reference)
    result = engine.register(target)
```

Use `outlier_policy="flag"` to keep raw coefficients and mark detected
subsets with status 2. Use `outlier_policy="disabled"` to omit the post-pass
entirely. Use `outlier_policy="median"` for an explicit centred 3×3 component
median on every valid subset. Median smoothing preserves the original solver
status and does not claim that a subset is an outlier. The selected value is
returned as `config["outlierPolicy"]`. Derivative coefficients are not used
for UOD detection: earlier coefficient-triggered modes were removed because
they behaved as implicit smoothing on healthy noisy data.

## Persistent Python supervisor

Reuse one native engine for an image sequence. This keeps the wgpu device,
pipelines, and reference pyramid alive while Python supervises the stream:

```python
from lucas_kanade_icgn import ICGNWGPUFullPyramid

reference = "DIC/LucasKanadeICGN/assets/frame_000.png"
targets = [
    "DIC/LucasKanadeICGN/assets/frame_001.png",
    "DIC/LucasKanadeICGN/assets/frame_010.png",
]

with ICGNWGPUFullPyramid() as engine:
    reference_info = engine.set_reference(reference)
    results = [engine.register(target) for target in targets]

result = results[-1]
print(reference_info.total_wall_ms)
print(result.parameters.shape)  # (32, 32, 6)
print(result.displacement.shape)  # (32, 32, 2)
print(result.mean_translation)
print(result.accepted, result.flagged)
print(result.timing)
print(result.rmse((2.0, -1.0)))
```

Use ``ICGNWebGL`` in the same persistent pattern when browser compatibility is
required. It keeps Chromium alive and the reference prepared, but each frame
and full result still crosses the process boundary. It is not the recommended
low-latency NumPy stream.

The affine parameter order is:

```text
[tx, du/dx, du/dy, ty, dv/dx, dv/dy]
```

Quadratic results append:

```text
[d2u/dx2, d2u/dxdy, d2u/dy2, d2v/dx2, d2v/dxdy, d2v/dy2]
```

Use `result.parameter_order` instead of assuming a fixed last-axis size.

`result.flagged` and `result.outlier_mask` describe status-2 UOD detections.
The older `replaced` and `replaced_mask` names remain compatibility aliases;
under `outlier_policy="flag"` they select values that were flagged but not
replaced.

`status` and `iterations` both have shape `(32, 32)`. Status values are:

- `1`: accepted;
- `2`: displacement outlier detected; coefficients are replaced only under
  `outlier_policy="replace"`;
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

`backend="three-tsl"` is the default and supported browser path. It selects
WebGPU when available and falls back to a serial Three.js/TSL
transform-feedback solver on WebGL2. Use
`force_three_webgl=True` to qualify that fallback even on a WebGPU-capable
machine. The `backend="gpuio"` option is deprecated and remains available only
for historical comparisons and migration validation. Use
`backend="webgpu-wgsl"` to run the
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
    iterations_per_level=(6, 8, 16),
    tolerance_per_level=(1e-2, 5e-3, 5e-3),
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

Existing GPU-IO applications should normally remove the explicit
`backend="gpuio"` selection. Use the form above only when deployment must be
qualified specifically without WebGPU. The result schema and numerical
parameter order do not change.

## Native WebGPU engine

Use `ICGNWGPU` when the input already exists as a NumPy array and the remaining
browser process transfer would dominate the latency. Browser targets already
use binary rather than PNG, but native wgpu also removes the Chromium copy:

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
selected IC-GN solver and optional UOD pass all run on the native GPU. The
result schema and parameter order are identical to `ICGNWebGL`. This backend
is experimental and must be selected explicitly. At reference preparation,
the default `prefer_subgroups=None` policy times both available native solvers
and retains the faster one. Use `prefer_subgroups=False` to force portable
WGSL or `prefer_subgroups=True` to force WGSL subgroup reductions. The native
probe requests wgpu-py's `subgroup` feature and executes a vector reduction
before the specialized solver is made available.

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

The low-latency engine retains optional selective recovery for compatibility
and experiments in which large motion is exceptional:

```python
with ICGNWGPU(prefer_subgroups=False, pyramid_levels=2) as engine:
    engine.set_reference(reference_frame)
    result = engine.register(target_frame)
```

``pyramid_levels`` counts the fine image, so ``1`` means no pyramid and remains
the default for ``ICGNWGPU``. With two or three scales, reduced references are prepared once.
For each target, the ordinary fine solve runs first; a five-tap pyramid and a
coarse-to-fine seeded retry are executed only when fine subsets fail. The
coarse dependency mask contains only the interpolation support required by
those failures. Parameters, statuses, and iteration counts already accepted by
the fine solve are preserved bit for bit. Two scales are the recommended
recovery setting; a third scale should be enabled only after qualification on
the expected displacement field. For normal coarse-to-fine processing, use the
full-pyramid engine below instead.

When coarse-to-fine registration is expected on every frame, use the separate
GPU-resident backend:

```python
from lucas_kanade_icgn import ICGNWGPUFullPyramid

with ICGNWGPUFullPyramid(
    pyramid_levels=2,
    iterations_per_level=(8, 16),
    tolerance_per_level=(5e-3, 5e-3),
) as engine:
    engine.set_reference(reference_frame)
    result = engine.register(target_frame)
```

The schedules are ordered from coarse to fine. The finest target is uploaded
once; reductions, prefilters, polynomial seed transfers, and all solves stay on one
WebGPU device in one command submission. Only the finest packed field is read
back. The selective `ICGNWGPU` recovery path remains available when large
motion is exceptional, while `ICGNWGPUFullPyramid` is the recommended native
path and avoids its CPU reductions and intermediate readbacks.

The recommended backend also supports a bounded changing-frame stream:

```python
with ICGNWGPUFullPyramid(motion_model="affine") as engine:
    engine.set_reference(reference_frame)
    for result in engine.register_stream(frame_source, pipeline_depth=2):
        print(result.timing["resultAgeWallMs"])
        consume(result)
```

Every in-flight slot owns independent target-pyramid, solver-state, seed, and
readback resources. Reference data and pipelines stay shared, and results are
yielded in input order. Depth one minimizes result age. The default depth of
two improves sustained throughput; a third slot is supported but added little
throughput on the validation machine. The timing dictionary separates input
preparation, direct upload, command encoding, submission, completion wait,
readback copy, result construction, and total result age.

For an already contiguous two-dimensional `uint8` array, input normalization
keeps the same NumPy storage and wgpu-py uploads those bytes directly to an
`r8unorm` texture. There is no PNG, browser, DevTools, or JSON stage. One
CPU-to-GPU transfer is still required because a normal NumPy allocation is not
GPU texture memory.

Two levels are the default because they provided the best measured 4K60 margin
on the validation GPU. Select three levels explicitly for a larger expected
capture range; its default iteration schedule is `(6, 8, 16)`.

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
- `tolerance`: positive convergence threshold in pixels; single-scale default
  `1e-3`, while full-pyramid engines use their per-level schedule;
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
Three.js/TSL and raw WGSL. It includes target prefiltering, IC-GN, and the
optional displacement UOD pass.
Wall time excludes Python-to-browser binary transport and result readback.

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
python -m pip install -e 'DIC/LucasKanadeICGN/python[dev]'
ruff check DIC/LucasKanadeICGN/python
ruff format --check DIC/LucasKanadeICGN/python
python -m unittest discover -s DIC/LucasKanadeICGN/python/tests -v
node --check DIC/LucasKanadeICGN/LucasKanadeICGN.js
```

The Python unit suite is intentionally hardware-independent: constructing and
validating an engine does not discover Chromium or create a GPU device. It also
checks the browser-discovery and GPU-argument policies for Linux, Windows, and
macOS without pretending to exercise those systems. Real browser and native
GPU registrations remain separate integration checks on each target machine.
