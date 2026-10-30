# Repeat under a verified performance profile

The system was explicitly set to performance using `powerprofilesctl set
performance`, then the unchanged native-UHD campaign was repeated. Importantly,
`powerprofilesctl get` already reported `performance` before this command.
This is **not** an established balanced-versus-performance comparison. The
earlier campaign did not log its power profile.

At the start, during the run and at completion the profile reported
`performance`. The Intel CPU energy-performance preference also reported
`performance`; the scaling governor's name remained `powersave`. Those two
observations coexist on this machine and the governor name alone does not
establish that the power-saving profile was active. AC power was connected.
No other CPU/GPU frequency, power or solver settings were changed. The
requested system profile was left at performance.

## Reproduce

From the repository root, with the installed project environment:

```bash
powerprofilesctl set performance
powerprofilesctl get
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/benchmark_temporal_init.py \
  --campaigns 3 \
  --output DIC/LucasKanadeICGN/benchmarks/results/temporal-init-20261030/performance-retest
```

Computation revision: `52815b7`, unchanged native host and shader code from the
preceding qualification. Intel Graphics LNL, Vulkan, Mesa 25.2.8, wgpu-py
0.32.0, NumPy 2.4.6 and Python 3.11. The same 200 native 3840 × 2160 tensile
images up to stretch 1.1 and 200 pure translations are reconstructed from the
archived reference and analytic truth. No ROI, B-spline interpolation, affine
or translation motion, size/stride 16/16, uniform weights and explicit median
smoothing. Ceilings remain `(30,)` at one level and `(8,16)` at two levels.
Every configured scale is solved on every frame; no conditional retry.

As before, slots are warmed on 20 chronological frames, history is reset,
frames 0–19 establish history and frames 20–199 are timed. All images are
resident NumPy arrays before timing. Rendering, scientific scoring, plotting
and serialization are outside the measured loops. Campaigns run serially;
no concurrent tests, builds or GPU profiling ran during timing.

## Results

Numbers are medians of the three campaign statistics, not pooled quantiles.
Boundary: **resident NumPy image → materialized NumPy field and statistics**.
Result age and sustained throughput are different metrics.

| Motion | Initialization / levels | Slots | FPS | Age median | Age p95 | Age p99 |
|---|---|---:|---:|---:|---:|---:|
| tensile affine | zero / 2 | 1 | 15.73 | 73.49 ms | 81.88 ms | 84.75 ms |
| tensile affine | zero / 2 | 2 | 16.52 | 141.97 ms | 153.11 ms | 155.25 ms |
| tensile affine | temporal / 1 | 1 | 66.21 | 15.91 ms | 17.97 ms | 18.51 ms |
| tensile affine | temporal / 1 | 2 | 69.08 | 29.97 ms | 35.54 ms | 36.31 ms |
| tensile affine | temporal / 2 | 1 | 59.23 | 16.90 ms | 20.36 ms | 21.28 ms |
| tensile affine | temporal / 2 | 2 | 61.34 | 33.80 ms | 39.02 ms | 40.13 ms |
| pure translation | zero / 2 | 1 | 29.30 | 37.29 ms | 44.29 ms | 48.72 ms |
| pure translation | zero / 2 | 2 | 30.20 | 73.33 ms | 78.52 ms | 81.06 ms |
| pure translation | temporal / 1 | 1 | 83.07 | 11.99 ms | 14.75 ms | 15.28 ms |
| pure translation | temporal / 1 | 2 | 91.85 | 22.24 ms | 25.68 ms | 26.94 ms |
| pure translation | temporal / 2 | 1 | 80.83 | 12.10 ms | 15.05 ms | 16.56 ms |
| pure translation | temporal / 2 | 2 | 78.57 | 23.70 ms | 30.29 ms | 30.84 ms |

For temporal affine/two levels/one slot, the previous median age was
16.62 ms and the repeat is 16.90 ms; the corresponding median campaign p95
is 20.15 → 20.36 ms. Throughput is 59.92 → 59.23 FPS. This shows **no gain**
from reapplying the already active performance profile. It is not evidence
of a causal regression from changing profiles.

Affine does not reach age p95 <16.67 ms at either level count. For translation
with one slot, all three campaign p95 values are below that limit at one level
(14.00–15.16 ms) and two levels (14.77–15.32 ms). Two slots do not meet the
age criterion. Zero initialization still fails the large-displacement cases;
its times describe those failed solves, not an equivalent successful method.

## Numerical checks and controls

Accepted counts match the previous campaign at every measured frame in every
configuration. Final tensile displacement RMSE and fixed-support coverage are
unchanged: temporal/two levels retains 26,915 / 26,924 positions (99.9666%)
with 0.0283764 px RMSE and 0.00143084 strain-component RMSE. The full-grid
accepted population is 29,150 / 32,400 and its displacement RMSE is 2.58179 px;
boundary errors must not be hidden by the fixed interior score. The black
frame accepts no subsets and the next normal frame recovers in one step.

The independent repeated-image UHD frame-40 controls give 12.69 ms median
without history and 14.05 ms with history, with 31,648 accepted windows.
These controls do not replace changing-sequence qualification.

Raw evidence:

- `report.json`: all original timing samples, per-frame quality and provenance;
- `metrics.csv`: all tensile capture-range results;
- `power-profile.json`: observed system state and comparison limitation;
- `summary.json`: medians, p95 ranges and acceptance comparisons to the previous run;
- `frame40-controls.json`: both repeated-image controls and component timings.

No implementation or default was changed for this repeat. The earlier
small-motion timings and the hard cumulative-motion sequence remain distinct
workloads; reapplying a profile does not make their results interchangeable.
