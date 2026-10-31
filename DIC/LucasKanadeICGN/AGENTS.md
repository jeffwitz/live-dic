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
.venv/bin/ruff check src tests ../benchmarks
.venv/bin/ruff format --check src tests ../benchmarks
cd ../../.. && python3 -m sphinx -W --keep-going -b html \
  DIC/LucasKanadeICGN/docs DIC/LucasKanadeICGN/docs/_build/html
```

Run these checks before finishing and report their actual outcomes.
GPU tests must be explicitly enabled on available hardware or a software
adapter; skipped GPU tests do not validate numerical execution.


## Architecture and public contracts

- The product is a Python-supervised NumPy image-to-field engine. Native
  `ICGNWGPUFullPyramid` is recommended and has two fixed levels by default.
  `ICGNWGPU` is single-scale by default; selective retry is a distinct optional
  compatibility path. Do not silently substitute one policy for another.
- `ICGNWebGL` and the existing CLI retain their browser defaults. Browser ROI
  requests must be rejected, never ignored. A JavaScript stream supervisor is
  a future integration, not an implemented component.
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
