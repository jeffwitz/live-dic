# Matched native-UHD workgroup-width campaign

## Decision

Keep **16 invocations per workgroup** on the Intel Lunar Lake validation machine.
The explicit `solver_workgroup_size=32` option works numerically, but reduces
throughput and exceeds the one-slot p95 age budget on this trajectory.
This is a workgroup **width**, not a change to the number of subsets or their size.
It does not invalidate the separate RTX 3080 Laptop / Tiger Lake fixed-pair
experiment documented in the performance reference: hardware and workload differ.

| Width | Slots | Sustained FPS | Age p50 | Age p95 | Age p99 |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | **82.26** | **12.13 ms** | **13.82 ms** | **14.86 ms** |
| 32 | 1 | 69.74 | 14.65 ms | 18.66 ms | 19.73 ms |
| 16 | 2 | 88.62 | 22.60 ms | 25.44 ms | 26.04 ms |
| 32 | 2 | 74.61 | 27.15 ms | 35.57 ms | 36.26 ms |

FPS is the median of three campaign throughputs; age quantiles pool all 540
timed fields per cell. The individual one-slot p95 ranges are 13.754–13.888 ms
for width 16 and 18.355–18.993 ms for width 32. The slowdown is present in all
three repeats. Width 32 loses 15.2% throughput at one slot. Two slots improve
throughput while increasing result age; they do not solve the latency problem.
These finite runs are not a hard real-time or cross-device guarantee.

## Configuration and provenance

- Code commit: `66f23acf0167c5cfb7e49beb96711d6368c7c7a1`.
- Intel Graphics LNL, integrated GPU, Vulkan, Mesa `25.2.8-0ubuntu0.24.04.4`;
  Linux `7.0.0-38-generic-x86_64`, NumPy 2.4.6, wgpu-py 0.32.0, Pillow 13.0.0.
- Power profile `performance` before and after every run.
- Same analytic native 3840×2160 tensile trajectory: 200 cached uint8 images,
  verified pixel hashes; frames 20–199 timed. No cropping or target changes.
- Affine, subsets 16×16, stride 16, B-spline, uniform spatial weighting,
  explicit median postprocessing, two fixed pyramid levels, caps `(8, 16)`,
  tolerances `(0.005, 0.005)`, ZNCC minimum 0.8.
- Temporal initialization with velocity prediction and fine correction;
  conservative stagnation stop (`patience=3`, minimum 4), finest level only.
  Both pyramid levels execute for every image. No selective retry.
- Twenty warmups through the actual requested slots, history reset, twenty
  chronological priming frames, then 180 timed fields per run. Width order is
  16/32, 32/16, 16/32 over the three repetitions, each at depths 1 and 2.
- Boundary: resident NumPy image to materialized NumPy field and statistics.
  Preparation, decoding, trajectory rendering, truth scoring, extra diagnostic
  solves and serialization are outside the timed stream. This excludes camera
  acquisition and measures an unpaced producer rather than camera arrival age.
- The four runtime source hashes, target hashes, effective configuration,
  adapter, versions, raw timing components and per-frame quality are archived
  in each report. All runtime source hashes match across all twelve runs.
  `git_dirty=true` records the untracked output archive and pre-existing
  benchmark `__pycache__`; no runtime source was edited during the campaign.
- The original driver ran from `/tmp/compare_workgroup_widths.py` with the same
  loop now retained in `run_campaign.py`. Exact worker commands are in reports.
  No competing tests or benchmark jobs were started during timed runs. Ambient
  desktop/service load and clocks were not controlled or sampled per frame.

## Numerical checks

The unchanged qualification gates apply to the fixed interior of **26,924**
positions: coverage ≥90%, displacement RMSE <0.05 px, displacement error p95
<0.10 px, median-output strain aggregate RMSE <0.002. Both widths pass on every
timed frame at both depths. Per-width quality is identical across repetitions
and depths; widths are not bit-identical because reduction order changes.

| Worst over frames | Width 16 | Width 32 |
|---|---:|---:|
| Minimum fixed-support coverage | 99.9851% | 99.9851% |
| Displacement RMSE | 0.0285780 px | 0.0285779 px |
| Displacement error p95 | 0.0530653 px | 0.0530654 px |
| Strain aggregate RMSE | 0.00144605 | 0.00144593 |
| Mean fine iterations, all windows | 2.579266 | 2.579271 |
| Mean coarse iterations, all windows | 2.008018 | 2.008009 |

Iteration diagnostics are a separate untimed replay. The same-seed no-stop
shadow audit finds 46 late-convergence events lost across the whole grid, and
zero on the fixed support, for either width. No coarse solve is cut short.
The virtually unchanged iteration counts exclude additional convergence work
as the explanation for this slowdown; shader occupancy, barriers and scheduling
have not been profiled separately and are not claimed as a measured cause.

Final-frame width-to-width displacement-vector RMSE on the jointly accepted
fixed support is 0.0001928 px, with one differing status. All six exported
parameter components, their maximum differences, and whole-grid comparisons
are in `summary.json`; all numeric field/state buffers are finite.

**The whole grid is not qualified by these interior gates.** Its worst
displacement RMSE is about 3.2865 px for both widths. At frame 199 it is about
2.5815 px with only 89.99% accepted, including known edge failures. Those
failures remain in each report and final NPZ; they were not removed to improve
the displayed interior precision.

## Reproduction

From the repository root, using the project environment:

```bash
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/benchmark_realtime_optimization.py \
  --target-cache /tmp/live-dic-rt-trajectory.npy --prepare-cache
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/results/workgroup-width-20261030/run_campaign.py
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/results/workgroup-width-20261030/summarize.py
```

The first command is unnecessary when the verified cache exists; it was not
rerun for this campaign. The driver overwrites this directory's run files.
`summarize.py` recomputes the statistics, asserts matched source/input/config
provenance and checks the final NPZ files using `allow_pickle=False`.

## Validation after the campaign

- Portable suite: 191 discovered, 121 passed, 70 GPU tests explicitly disabled.
- GPU-enabled suite: all 191 passed on this hardware (104.101 s).
- Ruff over `src`, `tests`, and `../benchmarks`: passed.
- Sphinx HTML with warnings treated as errors: passed.

These are local checks, not a claim about an unobserved remote CI run. No
runtime code or default was changed for this measurement; only the benchmark
width argument, this archive and the performance documentation were added.
