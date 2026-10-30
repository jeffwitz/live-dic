# Native UHD performance investigation, 30 October 2026

The previous 59 FPS run did not establish a general code regression. In this matched control the old code itself reaches a median 71 FPS at depth 1. Scheduling optimizations increase median throughput, and an optional coarse-change correction reduces fine iterations. Neither improvement establishes consistently lower tail latency.

## Measurement boundary and configuration

- Resident uint8 NumPy target → materialized NumPy field and statistics. Acquisition, image formation, truth scoring, plotting and serialization are outside timing.
- Native 3840 × 2160 archived tensile reference; all 200 inverse-warped targets match the SHA-256 hashes in the earlier temporal report. No tiling, resizing or reduced subset population.
- Fixed reference, stretch up to 1.1; affine, B-spline, uniform, size/stride 16/16, explicit `median`, two scales, unchanged caps `(8,16)` and tolerances. No ROI.
- Intel Graphics LNL, Vulkan/Mesa 25.2.8, Python 3.11.14, NumPy 2.4.6, wgpu-py 0.32.0, Pillow 13.0.0; performance profile verified before and after every run.
- Three repeats, separate serial processes, order reversed for repeat 1. Warm 20 frames through the actual slots, reset, replay 20 to restore history, then time frames 20–199. Each timed run is approximately 2–3 seconds, not an overnight thermal soak.
- Baseline runtime: `93ceb37c9d103f9400630a785b63feb7b369b719`. Candidate runtime: `43fbb1774590a24548c82ec824cab4050862bf4a`. Reports were measured before that candidate commit, with a dirty working tree; the four recorded runtime source hashes match the committed files exactly. Benchmark cache preparation support was added after timing and does not alter runtime code.

## All repetitions retained

The central columns below are **medians of three campaign statistics**, not pooled quantiles. The ranges retain the slow first candidate repeat. `summary.json` also provides pooled quantiles and per-frame quality checks.

| Variant | Slots | FPS median [min–max] | Age median (ms) | Per-run p95 range (ms) | Per-run p99 range (ms) |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | 71.00 [69.46–72.12] | 14.06 | 16.14–16.48 | 16.64–17.83 |
| baseline | 2 | 77.29 [76.88–77.50] | 25.69 | 29.14–29.50 | 29.48–29.92 |
| batched | 1 | 75.05 [66.30–75.19] | 13.48 | 15.26–17.95 | 15.87–18.99 |
| batched | 2 | 80.94 [71.61–81.19] | 24.55 | 27.84–37.17 | 28.71–39.53 |
| corrected | 1 | 76.17 [70.15–77.04] | 13.17 | 14.98–17.84 | 15.51–18.85 |
| corrected | 2 | 82.42 [69.55–83.88] | 24.04 | 27.10–37.81 | 27.78–38.94 |

`batched` uses one compute pass containing ordered dispatches and an exact integer 5 × 5 target reduction without an intermediate texture. `corrected` additionally enables `temporal_fine_correction=True`.

Every candidate repeat exceeds 60 FPS. One slot is preferable when result age matters. Two slots improve throughput while returning older results. Candidate p95 is **not consistently below 16.67 ms**. Across all 540 one-slot ages, pooled p95 is 16.38 ms for baseline, 17.32 ms for batching/fusion and 17.01 ms for correction. This is not proof of improved worst-case latency or absence of a tail regression.

Median throughput gains relative to the old code in this campaign are about 5.7% (batching, one slot) and 7.3% (correction, one slot). Individual paired differences change sign; the slow first repeats prevent a strong low-variance non-regression conclusion. Repeating the power-profile command is not a controlled balanced/performance comparison: performance was already active. Earlier telemetry observed power limiting, but a single cause of the larger campaign-to-campaign drift was not isolated.

## Numerical result and remaining failures

- Default scheduling/fusion: all 180 per-frame quality rows match baseline exactly, for both depths and all repeats. Final parameter, status and iteration arrays match bit for bit (18 comparisons). GPU regression tests independently check fused reduction against CPU pixels.
- Optional correction: every timed frame stays above 99.985% coverage on the fixed 26,924-position interior support, below 0.05 px displacement RMSE and below 0.10 px p95 error. Final coverage is 26,923/26,924 (99.996%), RMSE 0.02858 px, p95 0.05305 px, and strain-component RMSE 0.001446. The old result is 26,915/26,924 with RMSE 0.02838 px. Median smoothing is explicit; these do not qualify raw gradients.
- Mean accepted fine iterations over timed frames: 2.649 → 2.290 (−13.6%). Every configured scale still solves every image. The fine seed combines previous accepted fine detail with the change in accepted coarse history; it does not predict velocity or accumulate output fields.
- Whole-grid final accepted displacement RMSE is approximately **2.58 px** for both variants, versus approximately 0.029 px on the fixed interior support. This reflected image-formation/deformed-boundary failure is retained in the reports and is not represented as a successful measurement. All windows remain dispatched; no difficult windows were removed to improve FPS.
- At baseline frame 199, 2,060 fine windows reach the 16-iteration ceiling without convergence. They account for 32,960 iterations, approximately 28% of all fine iterations, even though accepted windows average only 2.73 iterations. Accepted-only statistics concealed that cost.
- Quality rows match between depths 1 and 2 and between all repeats of a variant. The test suite also checks depth 3, history reset/expiry, corrupted images and full/partial ROI ordering. A separate native-UHD depth-3 black-frame injection produces zero accepted updates and recovers on the first normal image with 100% fixed-support coverage and 0.02185 px RMSE (`uhd-corruption.json`).

## Rejected experiments

`exploratory-prototypes.json` retains observations and the original exploratory runner. Wider multi-output FIR tiles and 32/64-lane solvers did not improve the real trajectory reliably; 32 lanes reduced full-sequence throughput to roughly 46–55 FPS despite favorable static-image controls. The public 32-lane option from intermediate commits was removed. Explicit cubic unrolling and barrier removal did not provide a reliable gain. The canonical solver arithmetic remains unchanged.

## Reproduce

Run from the repository root with the project environment. Cache formation requires about 1.66 GB on disk; measurement copies all targets into resident arrays before timing. Do not run tests, rendering or other GPU work concurrently.

```bash
rtk proxy DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/benchmark_realtime_optimization.py \
  --target-cache /tmp/live-dic-rt-trajectory.npy --prepare-cache

rtk proxy DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/benchmark_realtime_optimization.py \
  --target-cache /tmp/live-dic-rt-trajectory.npy --pipeline-depth 1 \
  --temporal-fine-correction --iteration-diagnostics --output /tmp/corrected.json
```

Omit `--temporal-fine-correction` for the default transfer. For the old runtime, create a detached worktree at the baseline commit and set `PYTHONPATH` to its `DIC/LucasKanadeICGN/python/src`, while running this benchmark script with the same interpreter. Alternate baseline/batched/corrected ordering, use both depths and retain every run. Extra coarse readbacks requested by `--iteration-diagnostics` run in a separate untimed replay.

`qualified/` contains all 18 strict JSON reports and final-field NPZ files (`allow_pickle=False`). `summary.json` records provenance and aggregate checks. `exploratory-layouts/` preserves earlier negative layout controls without redundant final-field files. Local validation commands and their actual outcomes are in `validation.json`; no remote CI result is asserted.

## Decision

Keep exact scheduling/reduction fusion and leave coarse-change correction opt-in. Use one slot for the lowest age. The current evidence supports 60 FPS throughput on this sequence and GPU, with limited margin and unresolved tail variability. It does not qualify a hard 16.67 ms result-age deadline, other hardware, arbitrary motion, acquisition latency or an indefinitely sustained thermal load.
