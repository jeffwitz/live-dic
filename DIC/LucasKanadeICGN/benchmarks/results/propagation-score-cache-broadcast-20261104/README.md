# Score reuse through the existing candidate broadcast — 2026-11-04

Clean source `35a8fab199912b0052113f7bd3202a9152ddfc55`. Same two original-UHD
protocols as `../propagation-score-cache-pilot-20261104/`, but cache records now
travel in the existing candidate workgroup broadcast, avoiding a synchronization
per candidate. Both variants retain exact parameter/status/iteration fingerprints.

| Case | Recompute median / p95 | Cache median / p95 | Paired median saving (95% CI) |
|---|---:|---:|---:|
| Original frame 40 | 17.493 / 19.726 ms | 17.517 / 19.715 ms | −0.015 ms [−0.047, 0.017] |
| Translation with static coarse seed defects | 16.956 / 18.867 ms | 17.035 / 18.818 ms | −0.090 ms [−0.110, −0.079] |

Reference/reference campaigns fixed margins before candidate evaluation:
0.09454 ms and 0.08075 ms. The easy case is within its margin; the active case's
interval crosses the negative acceptance boundary. **No useful improvement is
established; non-regression of the active case is not established either.**
Automatic score caching was therefore removed from the browser. The explicit
native research option remains for further controlled experiments.

All runs use twelve 100-frame blocks per variant, AB/BA after each frame, eight
warm-ups and one prepared Intel LNL/Vulkan device (Mesa 25.2.8). Translation
32/16, B-spline, fixed two levels (8/16), tolerance 0.005, raw fields, two
failed-only passes and empty-work elimination are unchanged. The timing boundary
is resident NumPy target to materialized field/statistics; preparation, decoding,
display and storage are excluded. No other project tests, builds or GPU
benchmarks ran concurrently. Recorded power profile/EPP are performance; the
Intel pstate governor reports powersave. These labels alone do not prove a
constant clock or power budget.

The original pair retains 100% analytical interior coverage and 0.01017 px vector
RMSE. The manufactured translation retains 99.8706% and 0.00000650 px under the
same error gates. Its recipe, seed-defect mask rule, derived-array hash and
work counts are in the reports; it is an initialization stress control, not a
natural discontinuity or universal convergence proof. No private data are used.

The reference itself varies substantially between campaigns; even recomputation
is slower here than in the preceding pilot. Do not attribute that absolute drift
to a cache speedup/regression or renew an unconditional 4K60 claim from these
runs. The paired comparisons and their uncertainty are the relevant evidence.
A separate monitored soak investigates process/driver memory and frequency.

Four original JSON reports are compressed without reformatting. `criteria.json`
authenticates the reference reports; `summary.json` uses the median of per-block
paired savings and a 10,000-resample block bootstrap (seed 1729). Commands,
source/input hashes, device, versions, power snapshot and raw samples remain
available. Numerical caches never change the scoring/acceptance criteria, but
removing repeated arithmetic is not automatically faster on every workload.
