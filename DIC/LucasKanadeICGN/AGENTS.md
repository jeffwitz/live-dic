# live-dic development conventions

Instructions for work in `DIC/LucasKanadeICGN/`. See the root
[CONTRIBUTING.md](../../CONTRIBUTING.md) for the complete workflow.

## Project conventions

- The Python package lives in `python/src/lucas_kanade_icgn/`. Canonical WGSL
  is in `python/src/lucas_kanade_icgn/shaders/`. It is shared by native wgpu-py
  and the raw `webgpu-wgsl` browser backend, so a shader change affects both.
- `python/build/` and `python/.venv/` are generated. Never edit them.
- Write code, docs, and commit messages in English. Match the surrounding style:
  dataclasses with `slots=True`, explicit validation that raises `ValueError`,
  and timing keys in camelCase `...WallMs`.
- Documentation is Sphinx with the Diátaxis layout under `docs/`. Every new
  option needs API reference, how-to, explanation, and an update to
  `docs/explanation/limitations.rst`.
- **Do not change any default.** Existing defaults must stay bit-for-bit
  identical. New behaviour is opt-in.
- **Report honestly.** Publish measured numbers with their exact timing boundary
  (resident NumPy image → materialized NumPy field). Keep throughput and result
  age separate. Show failures and limitations instead of hiding them behind a
  passing test. Do not claim CI results you did not observe.
- GPU work by `register_stream` is bounded by `pipeline_depth` slots. Slot depth
  is unrelated to `pyramid_levels`. Never conflate the two.

### Commands

```bash
cd DIC/LucasKanadeICGN/python
.venv/bin/python -m unittest discover -s tests                 # CPU-only tests
ICGN_RUN_GPU_TESTS=1 .venv/bin/python -m unittest discover -s tests   # + GPU tests
.venv/bin/ruff check . ../benchmarks ../../../tools ../docs
.venv/bin/ruff format --check . ../benchmarks ../../../tools ../docs
cd ../../.. && python3 -m sphinx -W --keep-going -b html \
  DIC/LucasKanadeICGN/docs DIC/LucasKanadeICGN/docs/_build/html
```

Run these checks before finishing and report their actual outcomes.
Include the complete package directory in style checks, as CI does: checking
only `src` and `tests` omits Python examples in the package README.
GPU tests must be explicitly enabled on available hardware or a software
adapter; skipped GPU tests do not validate numerical execution.


## Architecture and public contracts

- The product is a Python-supervised NumPy image-to-field engine. Native
  `ICGNWGPUFullPyramid` is recommended and has two fixed levels by default.
  `ICGNWGPU` is single-scale by default; selective retry is a distinct optional
  compatibility path. Do not silently substitute one policy for another.
- `ICGNWebGL` and the existing CLI retain their browser defaults. Raw
  `webgpu-wgsl` supports full ROI; Three.js/GPU-IO reject masks. Browser partial
  supports/truncation/Gaussian integration remain unsupported. The standalone
  PWA implements a JavaScript supervisor around `web/core`; it does not add a
  Three.js stream layer or WebGL fallback. Keep both the Python shim and PWA
  on the same canonical core and WGSL sources.
- The distribution is `lucas-kanade-icgn-webgl`, the import is
  `lucas_kanade_icgn`, and the CLI is `icgn-webgl`. Keep these stable unless a
  separate migration is requested. `wgpu` is a standard dependency.
- No first-party native extension or external shader compiler is required.
  Preserve the pure-Python wheel with packaged WGSL and local browser assets.
- Use context managers or `close()`. Native close is permanent and idempotent;
  it releases GPU references and destroys only an owning device. Shared levels
  must not destroy their parent's device. Cleanup must stay outside live loops.
- Portable native widths are 16 (default), 32, and experimental 128 invocations
  per workgroup. Width, subset size, subset stride and stream depth are
  independent. Never infer the fastest width solely from a vendor name or limit.
- Temporal initialization, velocity prediction, fine correction and stagnation
  stopping are opt-in. Fixed pyramids still solve all levels on every frame.
  Stagnation stopping applies only to the finest level. Adaptive pyramids and
  direct interpolation have failed some qualified trajectory cases; do not
  present their speed alone as a successful optimization.

## Numerical and performance evidence

- Preserve existing accuracy gates, model conventions, parameter ordering and
  explicit distinctions between ROI selection, reference validity and status.
  Do not relax tolerances or drop difficult points to make a change pass.
- Use small cases for unit tests and native 3840 x 2160 fixtures for performance
  claims. The archived tensile and plate-hole images must not be downloaded or
  regenerated as a substitute for the checked-in fixtures.
- Measure resident NumPy input to materialized NumPy field and statistics.
  Separate preparation/compilation from steady state, GPU time from full calls,
  throughput from result age, and acquisition from the numerical engine.
- Keep warmups, input hashes, exact configuration, source revision/hashes,
  hardware/driver, selected variant, timing samples and quality populations.
  Alternate repeated A/B runs and do not run tests/builds during timed campaigns.
- Preserve immutable reports in `benchmarks/results/`, including negative
  controls. Do not reformat their scripts, alter their numbers, overwrite
  baselines or call historical measurements a fresh validation.
- Keep numerical tables in `docs/reference/performance.rst`; landing pages
  should link to the relevant campaign rather than duplicate stale numbers.
  Hardware execution on Linux/Vulkan does not qualify every OS or architecture.
- After packaging/resource changes, build a wheel, install it in a fresh
  environment outside the checkout, and exercise its packaged resources.
- No automatic push, PR creation, dependency upgrade or default change is part
  of routine cleanup. The user's explicit instructions take precedence.

## Current task: portable WebGPU front-end published on GitHub Pages

### Goal

Ship a static, installable web application (PWA) that runs live DIC on a phone,
tablet or laptop camera, and on archived fixtures, from GitHub Pages. Reuse
the qualified numerical engine instead of creating a third one. **The canonical
WGSL in `python/src/lucas_kanade_icgn/shaders/` stays the single numerical
source.** Only host orchestration is ported to JavaScript.

### Non-goals

- No change to Python defaults, the native engine's numerical behaviour, or
  public Python names.
- No GPU-IO, and no Three.js/TSL or WebGL2 path in the new app. Without WebGPU,
  the app shows a clear unsupported-browser message. Three.js stays a separate
  legacy browser backend for Python.
- No metrology claim for hand-held phone capture. The app is a demonstration,
  education and field-screening tool until a separate qualification says
  otherwise. Say so in the UI and the docs.
- No server and no remote compute. Everything runs client-side.
- No `SharedArrayBuffer`. GitHub Pages cannot set COOP/COEP headers.
- No CDN at runtime. Everything is served from the Pages origin.

### Existing assets to reuse (read them first)

| Asset | Role in the new design |
|---|---|
| `shaders/icgn.wgsl`, `shaders/full_pyramid.wgsl` | Unchanged math. All entry points, including temporal projection, merge, `predict_temporal_state`, `save_previous_temporal_state`, `refine_temporal_seed`, `reduce_fused`. |
| `web/webgpu/LucasKanadeICGNWebGPU.js` (~1070 lines) | Starting point for the JS host. It already runs the full pyramid in Chromium, but it is a page-global script (`window.ICGNHeadless`) coupled to the Python `BinaryTransport`, with no temporal mode and no ROI. |
| `web/common/ReferencePreparation.js` | Browser reference preparation (gradients, inverse Hessian, metadata). Must match native preparation. |
| `full_pyramid_backend.py` | **Reference specification** for the host orchestration to port: slot ring, temporal buffers and reset/expiry, `_create_temporal_pipelines`, `_prepare_temporal_state`, `reset_temporal_state`, `_create_temporal_slot`, velocity prediction, fine correction, divergence stopping, counter readback, encode order. |
| `roi.py`, `partial.py` | ROI validation, `reduce_mask`, `seed_permissions`, ROI seed links. Port `full` policy only in this task. |
| `assets/uhd_tensile/`, `assets/uhd_plate_hole/` | Demo mode and parity fixtures. |
| `engine.py` `ICGNWebGL` (Chromium driven via subprocess + DevTools) | Existing headless-browser harness. Reuse it for JS parity tests; do not add Playwright unless the harness cannot do the job, and justify it if you do. |

### Target architecture

```
DIC/LucasKanadeICGN/
  python/src/lucas_kanade_icgn/shaders/*.wgsl   # single source (unchanged location)
  python/src/lucas_kanade_icgn/web/
    core/                      # NEW: dic-rt-core, dependency-free ES modules
      device.js                #   adapter/device acquisition, limits/feature probe, capability report
      shaders.js               #   load WGSL by URL relative to import.meta.url, compile, report errors
      reference.js             #   reference preparation (moved from common/ReferencePreparation.js)
      pyramid.js               #   level resources, reduction, seed transfer, encode order
      temporal.js              #   history buffers, projection, prediction, merge, reset, expiry
      engine.js                #   public class DICEngine (see API)
      ring.js                  #   bounded slot ring, readback, result ordering
      roi.js                   #   full-policy ROI: validation, reduction, seed permissions
      input.js                 #   frame sources: Uint8Array, ImageBitmap, VideoFrame, HTMLVideoElement
      result.js                #   packed result decoding, typed-array result object
    webgpu/                    # existing adapter, rewritten as a thin shim over core/ for Python
  web-app/                     # NEW: the PWA (not packaged in the Python wheel)
    index.html, app.js, worker.js, ui/, render/, manifest.webmanifest, sw.js
```

- `core/` is plain ES modules with no bundler requirement and no npm runtime
  dependency. It must run in a page **and** in a dedicated Worker.
- The Python `ICGNWebGL(backend="webgpu-wgsl")` page must be rebuilt on top of
  `core/`. **The browser code tested by Python is then the code shipped on
  Pages.** This is the main reuse lever. Keep `BinaryTransport` as the Python
  adapter around `core/`, outside it.
- `core/` must not read `window`, the DOM or `navigator.mediaDevices`. Inputs,
  outputs and the device are injected or acquired through `navigator.gpu` only.

### Public JS API (adjust names only with a reason)

```js
import { probeCapabilities, DICEngine } from './core/engine.js';

const caps = await probeCapabilities();     // adapter info, limits, features, recommended profile, reasons
const engine = await DICEngine.create({
  device,                                   // optional: inject an existing GPUDevice
  width, height,
  subsetSize: 16, subsetStride: 16,
  motionModel: 'affine', interpolation: 'bspline',
  pyramidLevels: 2, iterationsPerLevel, tolerancePerLevel,
  outlierPolicy: 'median',
  initialization: 'temporal', temporalPrediction: 'velocity',
  temporalFineCorrection: true, divergencePatience: 3, divergenceMinIterations: 4,
  temporalMaxAge: null,
  pipelineDepth: 1,
  readback: 'full' | 'decimated' | 'none',  // 'none' = GPU-only display
});
await engine.setReference(source, { roiMask });   // source: Uint8Array | ImageBitmap | VideoFrame
const result = await engine.register(source);      // ordered; returns typed arrays + timing + counters
engine.resetTemporalState();
engine.fieldBuffer;                                // GPUBuffer of the latest fine state, for rendering
engine.destroy();                                  // idempotent
```

- Option names mirror the Python ones in camelCase. Defaults must equal the
  Python `ICGNWGPUFullPyramid` defaults. The app chooses the qualified
  latency profile from `benchmarks/results/latency-reduction-20261030/README.md`
  explicitly; it does not become a core default.
- Validation errors must match the Python error conditions. Unsupported options
  (partial ROI, quadratic if not ported, Gaussian weighting if not ported)
  must throw, never be silently ignored.
- The result layout must equal `docs/reference/result-layout.rst`: parameter
  order, status values, iterations, `-3` for excluded ROI windows.

### Phase 0 — capability and binding audit (do this first, write it down)

1. Count storage buffers, uniforms, textures and storage textures **per
   pipeline layout, per shader stage, across all bind groups**, for every
   pipeline in the default, temporal and qualified-latency graphs. The seeded
   solver already declares 10 storage buffers in `icgn.wgsl` group 0. The
   temporal pipelines combine group 2 and group 3 bindings in
   `full_pyramid.wgsl`.
2. The WebGPU guaranteed minimum `maxStorageBuffersPerShaderStage` is **8**,
   and many mobile adapters expose exactly 8. Also check
   `maxBindGroups` (4 guaranteed), `maxStorageBufferBindingSize`,
   `maxBufferSize`, `maxComputeWorkgroupStorageSize`,
   `maxComputeInvocationsPerWorkgroup` and `maxTextureDimension2D` against
   UHD, 1080p and 720p grids.
3. Record the table in `docs/explanation/browser-portability.rst` (new).

### Phase 1 — binding budget ≤ 8 (shared WGSL change, highest risk)

- Reduce every pipeline to at most 8 storage buffers per stage, by packing
  buffers rather than forking the math. Candidates: merge read-only
  `reference_data`, `inverse_hessian`, `metadata` into one buffer with offsets
  in `Config`; move `fir_weights` to a uniform or constant array; merge history
  state and ages; pack counters into one atomic buffer.
- **This changes the native path too.** Required evidence:
  - bit-for-bit parity of parameters, status and iterations against the
    pre-change commit, for the three motion models, levels 1/2/3, full and
    partial ROI, and zero/temporal/qualified-latency profiles, on the archived
    UHD frames (reuse `verify_default_parity.py` style fingerprints);
  - a native UHD no-regression timing campaign at one slot, alternating A/B,
    in `benchmarks/results/binding-budget-<date>/`. A regression must be
    reported, not hidden; if it exceeds noise, discuss whether to keep a
    native-only layout.
- Request device limits explicitly: only what the graph needs, never
  `adapter.limits` wholesale.

### Phase 2 — extract `core/` without behaviour change

- Move the logic of `LucasKanadeICGNWebGPU.js` and `ReferencePreparation.js`
  into `core/` modules. Turn the old file into a shim that keeps
  `window.ICGNHeadless` for Python.
- All existing Python browser tests must pass unchanged, including raw WGSL
  native-UHD frame 40 (99.981% interior coverage, 0.02420 px RMSE recorded).
- Package `core/` in the wheel (`[tool.setuptools.package-data]`). Verify with a
  fresh-environment wheel install, as the conventions require.

### Phase 3 — port temporal and qualified-latency host logic to `core/`

- Port from `full_pyramid_backend.py` exactly: buffer allocation, reset on
  `setReference`, `temporalMaxAge`, velocity prediction from raw history only,
  fine correction, divergence stopping on the finest level only, encode order,
  and counters.
- Ordering contract: frame n+1 reads frame n's history through in-order queue
  submission, at every `pipelineDepth`.
- **Parity tests** (Python harness, Chromium; software adapter in CI). Native
  `ICGNWGPUFullPyramid` vs JS `DICEngine` on a cropped tensile sequence
  (≥ 60 ordered frames), plus the black-frame injection, reset and expiry.
  Compare per frame: status equality, iteration equality, parameter max abs
  difference. Prefer bit equality. If driver differences prevent it, set
  tolerances *before* looking at results, write them in a `criteria.json`, and
  report every exceedance.
- Port `full` ROI (validation, reduction, seed permissions and links) with
  parity against `roi.py` on the plate-hole fixture. `partial` throws
  "not supported in the browser engine".

### Phase 4 — input, worker and rendering (the app)

- **Camera.** Use `getUserMedia({ video: { facingMode: 'environment', width, height, frameRate } })`.
  Read the actual track settings; do not assume the requested ones.
  Try to lock exposure, focus and white balance through `applyConstraints`
  where `getCapabilities()` allows it. Report what could and could not be
  locked. A drifting exposure must be visible in the UI (ZNCC/acceptance).
- **GPU luminance.** Add a small `input.wgsl` kernel, separate from the solver
  shaders: `copyExternalImageToTexture` (`ImageBitmap`/`VideoFrame`/video
  element) to rgba8, then BT.709 luma rounded to the same `r8unorm`/`uint8`
  convention the engine expects. Document the rounding. Test it against a CPU
  reference on synthetic colour images. Never read the frame back to the CPU
  on the hot path.
- **Worker.** Run `DICEngine` in a dedicated Worker. Transfer `VideoFrame`s
  (`postMessage` with transfer) or use `MediaStreamTrackProcessor` where it
  exists, with a fallback to `requestVideoFrameCallback` plus `createImageBitmap`
  on the main thread. Close every `VideoFrame`. Use backpressure: at most
  `pipelineDepth` frames in flight, and drop and **count** frames otherwise,
  never queue them unboundedly. Rendering uses `OffscreenCanvas` in the worker
  when available, and the main thread otherwise.
- **Rendering.** A render pipeline reads `engine.fieldBuffer` directly (no
  readback): camera image plus a colour map of u, v, |u|, an affine strain
  component, or status, with a legend and fixed or auto scale. Grey means
  rejected/unavailable, as in the existing figures. Strains are direct affine
  gradients; label them as such.
- **ROI.** Touch/pointer polygon or rectangle drawn on the reference, converted
  to a boolean mask at full resolution, passed to `setReference`.
- **Controls.** Capture or replace the reference, reset history, choose a
  resolution (720p/1080p/UHD within the probed limits), choose a profile, and
  pause. Show FPS, result age (median/p95 over a sliding window), acceptance,
  mean iterations, dropped frames, adapter name and limits.
- **Export.** On demand: readback of the current field, then CSV and a small
  JSON with configuration, adapter, timestamps and statuses. Use a download
  link, with no server.
- **Demo mode.** Without a camera, or by choice: replay the archived tensile
  sequence. Ship a reduced 1080p crop or downsample generated by a checked-in
  script, with hashes in a manifest. Do not regenerate the original fixtures.
  Show the analytic truth error live. This mode is also the app's CI test.

### Phase 5 — profile selection by capability

- `probeCapabilities()` returns a recommended profile and the reasons behind
  it, using measured limits and an on-device micro-benchmark (a few frames at
  each candidate resolution). Do not use a vendor-name heuristic.
- Profiles: `qualified-latency` (the native qualified configuration),
  `portable` (same math, one slot, lower resolution), and `minimal` (one level
  + temporal, only if it passed the native trajectory gates; cite the campaign).
  Adaptive pyramid and `cubic` failed qualified trajectory gates: they may
  appear only as labelled experimental options.
- Subgroups are never required. Use the portable 16-lane solver.

### Phase 6 — PWA and GitHub Pages deployment

- New workflow `.github/workflows/pages.yml`: on push to `main` and on
  `workflow_dispatch`, assemble `web-app/`, `core/`, the WGSL files (copied from
  `python/src/.../shaders/` at build time, never committed twice) and the demo
  fixtures into an artifact. Deploy with `actions/upload-pages-artifact` and
  `actions/deploy-pages`, pinned to exact versions or SHAs. Add a job step that
  fails if a served `.wgsl` differs from the canonical file (hash check).
- All URLs are relative (`new URL(..., import.meta.url)`). The site works under
  the `/<repo>/` subpath, not just at a domain root.
- `manifest.webmanifest` with icons; `sw.js` precaches the app shell, WGSL and
  demo data with a versioned cache name derived from the commit SHA, and
  removes old caches on activate. Bypass the service worker for camera data
  (there is none on the network anyway).
- Serve over HTTPS (Pages does), so `navigator.gpu` and `getUserMedia` are in a
  secure context. Also provide `python -m http.server`-based local instructions.
- Strict Content-Security-Policy via `<meta>`: `script-src 'self'`, no
  `unsafe-eval`. Avoid inline scripts.
- No analytics or third-party requests.
- Do not enable or deploy Pages yourself. Prepare the workflow and write in the
  docs the one-time repository setting the user must change
  (Settings → Pages → Source: GitHub Actions).

### Testing and CI

- Python harness: new `python/tests/test_browser_core.py` drives `core/` in
  headless Chromium and checks parity against native (Phase 3), the input
  kernel, the ROI port, and validation errors. GPU-gated like the other tests.
  Add a CI job with Chromium's software WebGPU adapter if it is available in
  the runner; otherwise state clearly that browser parity runs only locally.
- App smoke test: headless Chromium loads the built site from a local static
  server under a subpath, runs demo mode for N frames, and asserts finite
  results, acceptance above a threshold and zero console errors.
- Lint the JS with whatever the repo already uses. If nothing exists, do not
  add a toolchain without asking; keep code ES2022 and dependency-free.
- Manual matrix to report (do not claim untested devices): desktop Chrome,
  Android Chrome, iOS/iPadOS Safari, desktop Safari, Firefox. Record, for each
  device: browser version, adapter, limits, chosen profile, resolution, FPS,
  result age and demo-mode error. List missing devices as untested.

### Measurement rules for the browser

- Boundary for app timings: camera frame available in the worker → field
  available on the GPU (for display) and, separately, → field read back (for
  export). Report both; do not compare them with the native NumPy boundary
  without saying so.
- Use timestamp queries only if they pass a wall-time sanity check (a previous
  probe failed one). Otherwise report wall time only.
- Mobile thermal throttling is expected. Report runs ≥ 60 s with the age
  distribution over time, not only short bursts.

### Documentation

- `docs/how-to/run-in-browser.rst`: open the Pages URL, permissions, demo mode,
  profiles, export, limitations.
- `docs/explanation/browser-portability.rst`: architecture (core vs app vs
  Python shim), single-WGSL-source rule, binding budget, input conversion,
  worker model, ordering contract and capability-based profiles.
- `docs/reference/js-api.rst`: the `DICEngine` API, options mapped to Python
  names, result layout and errors.
- Update `limitations.rst` (browser ROI is now `full` only, not rejected; still
  no partial ROI; phone capture not qualified), the README landing section, and
  the AGENTS.md contract line "A JavaScript stream supervisor is a future
  integration" once it is implemented.

### Commits and definition of done

One focused commit per phase, at least: binding audit, binding budget plus
parity campaign, core extraction, temporal port plus parity, input/worker,
rendering/UI, demo fixtures, Pages/PWA, docs. Each commit leaves the tests
green. Do not push and do not open a PR.

Done when:
- Phases 0–6 are implemented.
- The native bit-for-bit parity and the no-regression campaign are recorded.
- JS/native parity passes on the agreed criteria.
- Demo mode runs in headless Chromium from a subpath.
- The Pages workflow is ready, but not enabled by you.
- The docs are written.
- The device matrix states what ran, what didn't and why.


### Implementation record — 2026-10-31

The phases above are implemented locally, starting from `b9d23a2`. Keep this
specification and its acceptance gates; future work should extend the recorded
qualification rather than infer mobile coverage from desktop results.

- Shared layouts use six storage bindings, eight when seeded. The metadata /
  inverse packing and FIR plane packing preserve solver binding bounds. The
  binding-budget campaign verifies 441 native arrays bit for bit across 49
  UHD configurations and alternating native A/B within declared repeatability.
- `web/core` is packaged and used by both the Python raw-WGSL shim and PWA.
  Temporal history/velocity/fine correction, bounded slots, GPU input,
  full ROI and small/decimated/full readback are implemented. Partial ROI,
  truncation, Gaussian integration and adaptive browser pyramids reject
  explicitly. Three.js and its WebGL fallback remain independent and unchanged.
- Hardware and software browser parity each cover 240 frame rows, with no
  predeclared-tolerance exceedance. Actual UHD tensile/plate-hole full-ROI
  checks and external-image conversion are recorded. The native defaults and
  scientific thresholds were not changed.
- The worker PWA includes local camera control, direct GPU rendering, fixed
  reference ROI drawing, analytic demo, explicit exports and a scoped offline
  service worker. A 70.5-second 1080p/30 Hz demo and server-stopped reload pass
  locally. This is not 60 Hz camera, phone or all-browser qualification.
- A fresh outside-checkout wheel install executes all 202 tests with native
  GPU and browser flags, no skips. Ruff, Node syntax, strict Sphinx, resources,
  editable installation and no-pickle ROI export are checked locally.
- See `benchmarks/results/browser-core-20261031/README.md` and
  `docs/reference/performance.rst` for raw records, timing boundaries, retained
  negative controls and the device matrix. In particular, whole-image demo
  errors near an unobservable boundary remain visible and are reproduced by
  the native solver; do not hide them by changing the evaluated population.
- `.github/workflows/pages.yml` and the software-browser CI job are prepared.
  No new remote CI result, push, PR, Pages setting or deployment is asserted.
  Publication remains a separate explicit action by the repository owner.


### Browser mask/display follow-up — 2026-10-31

- The PWA editor keeps multiple vector shapes with exact pixel-centre
  containment, domain/hole/island depth, union of outer domains, hole subtraction
  and island restoration. Equal footprints are duplicates. Keep binary mask
  and applied vector geometry together in exports. Preview pixels exist only
  during editing; never rasterize vector geometry in the live frame loop.
- The original UHD plate preset is an editable rectangle/circle pair. Its
  half-pixel vector offset preserves the archived integer-index mask exactly.
  The assembler copies original plate PNGs with hash verification, not new
  generated or resized scientific fixtures. Both loads are static image pairs.
- VM display uses direct infinitesimal affine strain and the archived ezz=0
  equivalent-strain convention, not stress. GPU vector arrows use valid reference
  origins, explicit spacing and magnification. Rendering must not modify fields.
- `tools/qualify_browser_editor.py` exercises actual pointer input and GPU
  display oracles on hardware/SwiftShader; it is prepared in browser CI. Retain
  the original/edited population distinction and do not interpret static scores
  as FPS qualification. `docs/how-to/run-in-browser.rst` documents all controls.

### Local source and consecutive comparison follow-up — 2026-10-31

- Local image folders decode lazily in natural relative-filename order and
  await every image; local video uses a Blob URL and browser codecs. Never
  resize an incompatible input silently or upload local data.
- Fixed reference stays the default. Consecutive mode performs a real solve
  against the immediately preceding input, including after numerical failure;
  it rebuilds the reference and includes that preparation in result age.
  Do not describe this cold-per-pair mode using fixed-reference FPS claims.
- The geometric ROI remains requested independently of numerical validity.
  Status view covers the entire grid. A mask in consecutive mode remains in
  image coordinates; it is not material tracking or cumulative integration.
- `tools/qualify_local_sources.py` exercises real file inputs, UHD pair parity,
  failure recovery, dimension rejection, and local video seek/play/pause. Source
  replacement drains work, aborts pending decodes and closes frame ownership.

### Browser workbench follow-up — 2026-10-31

- The Input/Mask/Solve/Results workflow offers custom solver edits only for a
  new reference. Preserve native/core defaults. Translation has no direct strain;
  its analysis must use displacement-neighbourhood fitting.
- Saved computed fields include the solver's configured spatial policy. Never
  call median solver output raw/unfiltered. Postprocessing is explicit,
  reversible, separately masked and executed outside the acquisition worker.
- Results recording is opt-in and adds full readback. The bounded in-tab library
  stops recording/acquisition at capacity; it must not evict scientific results
  silently. Save NPZ before tab close; raw images are not recorded.
- Keep computed parameters/status/iterations separate from processed arrays and
  validity masks. Preserve strict no-pickle NPZ, explicit dimensions/ROI/recipe,
  and conservative visibility across image/grid exclusions.
- `tools/qualify_browser_workbench.py` exercises the actual UHD GUI, pointer
  measurements, independent NumPy strain oracle, recording and NPZ reimport.
  Its timing is correctness evidence, not live performance qualification.

- History extraction requires an analysis epoch or native archive hash, preserves
  null gaps, and uses saved order rather than inferred acquisition time. It is
  not cumulative incremental tracking. Native sequence NPZ import selects one
  field and explicitly confirms any missing image geometry. Keep provenance and
  stored ROI data; never guess original image extent from grid dimensions.

### Mask-image and input admission follow-up — 2026-10-31

- Exact mask-image input requires opaque black/white pixels at reference size.
  Threshold conversion is explicit and records file hash/convention. Imported
  pixels form an immutable base intersected with vector geometry. Undo shares
  that base; never copy it per live frame or per undo checkpoint.
- Keep edited and applied mask provenance distinct. PNG export may contain
  current edits; a numerical field must retain its applied binary ROI and shapes.
- Cold source/reference operations use one admission gate, drain work and
  invalidate stale seeks. Comparison mode restores applied options, not pending
  custom controls. Temporal fine correction is admissible only at two levels.
- The mask-image hardware/SwiftShader reports and exact PNG round trip are in
  `benchmarks/results/browser-mask-images-20261031/`. The complete local suite
  reached 238 passing tests with native GPU/browser flags. This is local evidence,
  not a claim that a later remote CI is green or physical cameras were tested.
- Software video qualification uses the actual slow-replay control so a finite
  movie does not end before two slow solves finish. Do not turn this into a
  software-adapter FPS claim or weaken numerical tolerances.

### Results analysis and camera controls — 2026-10-31

- Region statistics use equal-weight actual subset centres. Keep total grid,
  ROI-selected and valid counts separate; status codes are categories. Histories
  preserve changing counts and null gaps, not inferred area integrals or time.
- Projects restore measurement geometry and applied recipes. The unload guard
  tracks changes since a requested project download; it cannot prove a file was
  saved or replace deliberate export. Keyboard shortcuts must not intercept
  ordinary focused controls. Responsive charts resize outside observer callbacks.
- The camera retains requested, initially negotiated and post-lock settings
  separately. Fake-camera UHD records cover permission failure/recovery, both
  frame-delivery paths and track closure; they qualify neither physical cameras
  nor live FPS. Keep failed short observation controls with that context.
- Local validation reached 242 tests with native GPU and browser flags, no skips.
  The complete remote Python/browser/lavapipe CI passed on `7f6fe65`. Later commits
  require their own observed CI result. The one-command local launcher is
  `python3 tools/run_browser_app.py`; it retains the user's browser GPU settings.

### Saved-result navigation and review without WebGPU

- Result replay and batch recipes use the same epoch/ROI/grid/reference-mode
  predicate as histories. Replay waits for one CPU analysis and uses a display
  rate, never an inferred acquisition clock. It stops on manual inspection.
- Without WebGPU, expose NPZ/project import and CPU post-processing; disable
  new correlations. Ordinary HTML canvas and main-thread fallbacks are tested.
  This is not a second IC-GN solver. Keep these roles explicit.
- A prepared reference preview is not a measured target. Clear stale readiness
  and HUD values on reference/ROI/comparison changes. Label camera timestamp
  origins instead of subtracting video media time from track-processor clocks.

### Computed coefficient inspection

- ``computed-pN`` reads the stored public coefficient, including the original
  solver spatial policy. Workbench displacement smoothing/VSG must not change
  it. Translation exposes only indices 0 and 3; native quadratic imports can
  expose all twelve. This does not add quadratic browser acquisition.
- Isotropic display scale multiplies displacement, leaves first derivatives
  unchanged and divides second derivatives. Raw NPZ parameters stay in pixels.
  Preserve checked parameter-order metadata and reject declared conflicts.
- The native UHD quadratic/NumPy/UI round trip and fresh outside-checkout wheel
  suite passed locally (247 tests, native/browser enabled, no skips). This
  qualifies interchange and installed resources, not a new accuracy/performance
  campaign. The wheel contains all six canonical WGSL files and no first-party
  native binary.

### Ended inputs and project snapshots

- Camera frame readers re-check generation after await, close late frames and
  mark ended/failed delivery explicitly. Restart is a new reference and mask
  preparation; keep the last completed result available until then. Fake-track
  ended-event recovery is tested, not physical camera disconnection.
- Results-project export waits for the last requested derived analysis before
  snapshotting applied recipes. This pause/readback/analysis barrier belongs to
  explicit export, never the live frame loop. Both hardware/SwiftShader UHD
  tests cover an edit followed immediately by export.
- The complete remote CI on `57ef273` passed, including macOS launcher tests,
  lavapipe and the software browser plus no-WebGPU review checks. Later revisions
  still need their own observed result.
