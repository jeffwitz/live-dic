# Shared binding-budget qualification

Baseline: `b9d23a2`. The native source change is `3c78d2d`; source SHA-256
values in the reports identify the actual files when unrelated browser work
made the checkout dirty. All native sources stayed unchanged during A/B runs.

`before.npz` contains the pre-change arrays. `after.json` verifies exact equality
of all 441 parameter/status/iteration arrays in 49 native UHD configurations:
translation, affine and quadratic; levels 1/2/3; zero and temporal initialization,
plus the two-level qualified latency profile; full/partial ROI for supported
translation/affine models. Quadratic ROI remains explicitly unsupported.
Late tensile frames test layout parity, not capture range from zero.

Packing combines the FIR planes and metadata/Hessians; it does not change the
LK equations. FIR bindings see both planes; solver bindings retain the original
first-plane bounds, including for divergent quadratic warps. Explicit layouts
now use six storage buffers normally and eight for seeded/partial solving.
Temporal pipelines remain at seven or fewer. Native devices explicitly request
eight, and Chromium executes with exactly that device limit.

## Timing boundary and protocol

Resident NumPy UHD input to materialized NumPy field/statistics; hardware Intel
Lunar Lake, Vulkan, Mesa 25.2.8, performance power profile. Same 200-frame
trajectory, 20 warmups, 180 measured frames; affine 16/16, B-spline, median,
fixed two levels, temporal velocity/fine correction, finest stagnation 3/4,
one slot. `baseline-aa-*` established the repeatability margin in `criteria.json`
before changing the layout. A/B order: before/after, after/before, before/after.
No tests, builds or other qualification scoring ran during the timed pairs.
Existing user desktop applications remained open.

| Pair | Before FPS | After FPS | Before age median / p95 (ms) | After age median / p95 (ms) |
|---|---:|---:|---:|---:|
| 0 | 83.62 | 86.96 | 12.095 / 13.755 | 11.809 / 12.857 |
| 1 | 86.25 | 88.46 | 11.840 / 12.772 | 11.307 / 13.010 |
| 2 | 79.85 | 87.22 | 12.688 / 14.916 | 11.620 / 12.509 |

No observed throughput regression; the only p95 increase (+1.87%, pair 1) is
inside the predeclared 12.68% baseline p95 repeatability range. Quality reports
are identical in every pair. The candidate maintains 60 Hz on this trajectory
with one slot. These measurements establish no regression on this machine;
they do not isolate a universal speedup or qualify mobile hardware.

Reports retain every age/completion-period sample, source/image hashes,
configuration, adapter, driver, versions, quality population and preparation
cost. NPZ files retain the final complete fields. The baseline worktree is
only a local comparison aid, not a runtime dependency.

Reproduction (provide the previously qualified target cache):

```bash
python benchmarks/qualify_binding_budget.py --baseline benchmarks/results/binding-budget-20261031/before.npz --output /tmp/layout-verification.json
python benchmarks/benchmark_latency_reduction.py --target-cache /path/to/trajectory.npy --variant abort --pipeline-depth 1 --output /tmp/after.json
```

The driver of the paired runs set `PYTHONPATH` alternately to the baseline
worktree's `python/src` and the current package source. No tolerances changed.
