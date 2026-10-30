# Native-UHD 16 versus 128 invocation campaign

This is a matched changing-frame experiment on the Intel Lunar Lake GPU.
`solver_workgroup_size=128` is an explicit experimental option; the default
remains 16. No shader equations, tolerances, iteration caps, or spatial
populations were changed to obtain these measurements.

## Results and decision

Keep width **16** on this machine. Width 128 loses **34.1%** throughput at
one slot and does not sustain 60 FPS, even with two slots.

| Width | Slots | Sustained FPS | Age p50 | Age p95 | Age p99 |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | **81.30** | **12.35 ms** | **13.85 ms** | **14.35 ms** |
| 128 | 1 | 53.61 | 18.80 ms | 23.05 ms | 24.06 ms |
| 16 | 2 | 88.63 | 22.59 ms | 25.39 ms | 26.13 ms |
| 128 | 2 | 57.50 | 34.25 ms | 43.60 ms | 45.27 ms |

FPS is the median of three run throughputs. Age quantiles pool 540 samples
per cell; they are not medians of three quantiles. Individual one-slot FPS:
16 = 82.191 / 80.679 / 81.297; 128 = 53.608 / 54.098 / 53.502. Individual p95
ranges are 13.678–14.052 ms and 22.660–23.442 ms respectively. All repeats
show the slowdown. This campaign remeasures the width-16 control rather
than treating a historical throughput as its baseline.

Both widths pass the unchanged fixed-interior quality gates in every frame:

| Worst over frames | Width 16 | Width 128 |
|---|---:|---:|
| Minimum coverage | 99.9851% | 99.9851% |
| Displacement RMSE | 0.0285780 px | 0.0285786 px |
| Displacement error p95 | 0.0530653 px | 0.0530719 px |
| Strain aggregate RMSE | 0.00144605 | 0.00144597 |
| Mean fine iterations, all windows | 2.579266 | 2.579235 |
| Mean coarse iterations, all windows | 2.008018 | 2.008037 |

Per-width quality is identical across repeats and depths. The final jointly
accepted interior fields differ by 0.0001257 px vector RMSE with no status
differences. Same-seed shadow diagnostics find 46 late-convergence events
lost over the full grid and zero on the fixed support, for either width.

The maximum whole-grid displacement RMSE remains about **3.2865 px**, and
minimum whole-grid coverage is **89.99%**. These edge failures remain in the
reports and NPZ files; passing interior gates does not qualify the whole grid.

Iteration counts are essentially unchanged. The slowdown is therefore not
explained by more GN iterations. Width 128 uses seven reduction stages rather
than four and expands the three partial-sum arrays from 768 to 6,144 bytes.
These resource changes are real; their separate contributions to elapsed time
were not profiled, so no measured causal attribution to one of them is made.

## Protocol

- Runtime commit `71578b6` and the four runtime source hashes are recorded in
  every run. Runtime sources were unchanged throughout the campaign.
- Intel Graphics LNL, Vulkan, Mesa 25.2.8; performance power profile checked
  before and after every run. NumPy 2.4.6, wgpu-py 0.32.0, Pillow 13.0.0.
- Analytic tensile trajectory, 200 native 3840×2160 uint8 images. Cached target
  hashes must match the qualified temporal manifest before execution.
- Affine 16×16 subsets, stride 16, B-spline, uniform integration, explicit
  median postprocessing, two fixed levels. Both levels execute every frame;
  there is no selective retry. Iteration caps `(8, 16)`, tolerances
  `(0.005, 0.005)`, ZNCC minimum 0.8.
- Temporal velocity prediction and fine correction; conservative stopping
  (`patience=3`, minimum 4) on the finest level only.
- Actual-stream warmup of 20 frames at the requested slot depth, history
  reset, chronological priming of 20 frames; frames 20–199 are timed.
- Width order alternates 16/128, 128/16, 16/128. Each width is measured with
  one slot and two slots in each repetition: twelve runs total.
- Boundary: resident NumPy image to materialized NumPy field and statistics.
  Preparation, rendering, decoding, scoring, serialization and extra
  diagnostic replays are outside timing. This is an unpaced producer;
  acquisition time and camera-arrival age are excluded.
- No other tests or benchmark jobs ran concurrently with timed campaigns.
  Desktop/service load, clocks and temperatures were not controlled or
  sampled per frame. No services were reconfigured.
- Dirty state includes the output archive, pre-existing Python cache and
  documentation edits. The runtime hash comparison detects source changes
  independently of that state.

## Scientific scope

The unchanged fixed-interior gates cover 26,924 positions: coverage ≥90%,
displacement RMSE <0.05 px, displacement error p95 <0.10 px, median-output
strain aggregate RMSE <0.002. Whole-grid results are retained separately;
edge failures must not be hidden by the interior qualification.

Iteration counts include all windows, including rejected ones. They come
from a separate untimed replay. Same-seed shadow solves measure how many
early-stopped windows would otherwise have converged later; these solves
are diagnostics, not part of the production pipeline.

The width changes the floating-point summation order. Final NPZ comparisons
therefore report all six parameter-component RMSE/maxima, displacement-vector
RMSE and status differences, rather than asserting bitwise identity between
widths. Each NPZ retains the effective `solverWorkgroupSize`.

## Reproduction

From the repository root, first prepare the trajectory cache if absent:

```bash
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/benchmark_realtime_optimization.py \
  --target-cache /tmp/live-dic-rt-trajectory.npy --prepare-cache
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/results/workgroup-128-20261030/run_campaign.py
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/results/workgroup-128-20261030/summarize.py
```

The existing verified cache was reused for this campaign. The driver overwrites
this directory's run files; each worker's exact command is recorded in its JSON.
The summarizer checks matching runtime/input/configuration provenance,
recomputes age quantiles, checks quality gates and loads NPZ without pickle.
Raw timings, per-frame errors, iteration histograms and final fields remain
in `runs/`. These are finite local observations, not a hard real-time or
cross-device guarantee.

## Local validation after timing

- Portable suite: 193 discovered, 122 passed, 71 GPU tests explicitly disabled.
- GPU-enabled suite: all 193 passed on this hardware (103.389 s).
- Targeted 128-lane regression: translation and affine, single-level and
  full-pyramid engines, synchronous and two-slot stream results passed the
  existing numerical comparison thresholds and fractional-motion bounds.
- Default parity oracle: all 30 parameter/status/iteration arrays identical,
  including native-UHD frame 40 and three models at one to three levels.
- Ruff over `src`, `tests`, and `../benchmarks`: passed.
- Sphinx HTML with warnings treated as errors: passed.

These are observed local validations, not claims about remote CI or other GPUs.
