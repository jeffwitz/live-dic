# AGENTS.md — DIC-RT (LucasKanadeICGN)

Instructions for coding agents working in `DIC/LucasKanadeICGN/`.

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
cd ../../.. && python3 -m sphinx -W --keep-going -b html \
  DIC/LucasKanadeICGN/docs DIC/LucasKanadeICGN/docs/_build/html
```

All four must pass before you finish. Say which of them you could actually run.

---

## Current task: reduce result age in temporal mode

Temporal initialization is done (see
`benchmarks/results/temporal-init-20261030/` and
`realtime-optimization-20261030/`). Affine p95 result age is still not
consistently below 16.67 ms. The goal is a qualified p95 below 16.67 ms at one
slot, affine, native UHD, on the 200-frame tensile trajectory.

### Measured cost model (indicative, not qualified)

Archived `uhd_tensile/frame_040.png`, repeated target, one slot, 60 samples,
Intel LNL. Total ms is resident NumPy → NumPy field.

| Config | Total | GPU wait |
|---|---:|---:|
| bspline, 2 levels, default | 12.00 | 10.37 |
| bspline, 2 levels, iterations (1,1) | 8.76 | 7.45 |
| bspline, 1 level, 1 iteration | 7.00 | 5.68 |
| cubic, 1 level, 1 iteration | 4.49 | 3.19 |

Derived: about 1.4 ms per mean solver iteration, about 2.5 ms for the fine
B-spline prefilter, about 1.8 ms for the coarse level, and about 1.5 ms of host
time. The CPU is idle during `map_sync` (no busy wait). At frame 199, 2,060
fine windows reach the 16-iteration ceiling without converging: 28% of fine
iterations. Reproduce this table first and record it in the campaign.

### Work items (each opt-in, defaults bit-for-bit unchanged)

1. **Constant-velocity seed.** `initialization="temporal"` with
   `temporal_prediction="velocity"`. The seed is 2·uₙ − uₙ₋₁ for each subset
   when both states are accepted and within `temporal_max_age`; otherwise use
   the current seed. Use a second GPU-resident history buffer. History must
   keep storing raw solver states, never predictions. Measure the reduction in
   mean fine iterations.
2. **Early divergence abort** in `shaders/icgn.wgsl`, as a pipeline constant so
   the default shader path is unchanged. Stop when the corner-update norm has
   not decreased for k consecutive iterations after a minimum count. The final
   status is 0, as at the ceiling. Report how many windows would have converged
   late (status 1) without the abort; that is the cost of this option.
3. **Adaptive pyramid.** For example `pyramid_policy="adaptive"`: run one level
   while the previous frame's temporal acceptance is ≥ a threshold, and two
   levels after a reset, expiry, or an acceptance drop. Decide on the host from
   counters already read back; do not add a readback or a CPU/GPU sync. Report
   the level used for each frame. Results must stay deterministic across
   `pipeline_depth` 1/2/3 (the decision may lag by depth−1 frames; document
   this).
4. **Interpolation choice.** Qualify `interpolation="cubic"` against `bspline`
   on the tensile trajectory, measuring both displacement and strain errors.
   Do not change the recommendation unless the error is shown to be acceptable.

Do not retry 32/64-lane solvers, wider FIR tiles, or extra slots for latency;
they were already rejected.

### Campaign

`benchmarks/benchmark_latency_reduction.py`, with results in
`benchmarks/results/latency-reduction-<date>/`, following
`realtime-optimization-20261030/README.md`: the same trajectory, three
alternating repeats, one slot, plus depth 2 for reference. Run an ablation
adding each item cumulatively to the current temporal/2-level baseline. For
each step report: FPS, age median/p95/p99 (per run and pooled), mean
fine/coarse iterations over all windows (not only accepted ones), and coverage
plus RMSE on the fixed 26,924 support and on the whole grid. State plainly
whether p95 < 16.67 ms is reached. Negative results are kept.

### Tests

Defaults keep bit-for-bit parity. Velocity prediction falls back correctly
after reset, expiry and corrupted frames. The divergence abort never changes a
window that converges before k stalled iterations. The adaptive policy switches
levels as specified and is deterministic across depths. A cropped tensile
sequence goes in the unit tests; native UHD is used only in the campaign.

### Done when

Commands pass (state which ran), docs are updated (API, how-to, temporal
explanation, limitations, performance), there are focused commits per item,
and nothing is pushed.
