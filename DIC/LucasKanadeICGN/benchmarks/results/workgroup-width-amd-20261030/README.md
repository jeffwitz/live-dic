# Native UHD workgroup comparison on AMD Radeon 780M

## Measured results

Measured locally on 2026-10-30, on the Radeon 780M integrated in an AMD
Ryzen 7 PRO 7840U. These are new measurements on AMD, independent of the
Intel campaign in `../workgroup-width-20261030/`.

| Solver invocations per workgroup | In-flight images | Sustained FPS | Result age p50 | Result age p95 | Result age p99 |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | 74.81 | 13.105 ms | 16.109 ms | 16.513 ms |
| 32 | 1 | 77.25 | 12.859 ms | 16.404 ms | 16.998 ms |
| 16 | 2 | 79.70 | 25.001 ms | 29.079 ms | 29.963 ms |
| 32 | 2 | 84.04 | 23.872 ms | 28.580 ms | 29.212 ms |

FPS is the median of three complete runs. Age percentiles pool the 540
timed results per cell. Individual one-image p95 values are
14.542 / 16.208 / 16.221 ms for width 16, and
16.415 / 16.529 / 16.133 ms for width 32.

Width 32 improves median throughput by 3.26% with one image in flight and
5.44% with two. With one image, width 16 has a slightly better pooled p95;
the observed run-to-run variation does not establish a universal latency
winner. Both widths exceed 60 FPS and remain below a 16.67 ms p95 in each
one-image run. Two images improve throughput but increase result age beyond
16.67 ms. These finite, unpaced streams do not establish a hard deadline.

The timing boundary is **resident uint8 NumPy image to materialized NumPy
field and statistics**. Upload, GPU pyramid/solver and packed readback are
included. Acquisition, decoding, target generation, reference preparation,
quality scoring and diagnostic replays are excluded.

## Matched configuration

- Code commit: `c0c072c24c7bf52a93aaee58b4e754a954ef7d36`.
- Adapter: `AMD Radeon Graphics (RADV PHOENIX)`, integrated GPU, Vulkan;
  Mesa `25.0.7-0ubuntu0.24.04.2`, Linux `6.17.0-29-generic`.
- Python 3.12.3, NumPy 2.5.4, wgpu-py 0.32.0, Pillow 13.0.0.
- Power profile `performance` before and after all twelve runs.
- 200 analytic tensile images, **3840 x 2160**, with every pixel hash
  checked against the existing temporal qualification manifest. Frames
  20–199 are timed, with no crop or downsampling of the input.
- Affine model, subsets 16 x 16, stride 16, 32,400 fine-grid subsets,
  uniform spatial weighting, B-spline interpolation, median postprocessing.
- Two fixed pyramid levels, iteration caps (8, 16), tolerances
  (0.005, 0.005), ZNCC minimum 0.8.
- Temporal initialization, velocity prediction, fine correction, conservative
  stagnation stopping on the finest level (patience 3, minimum 4 iterations).
- Twenty warmups at the requested pipeline depth, reset temporal history,
  twenty chronological priming frames, then 180 timed fields.
- Three repetitions, widths ordered 16/32, 32/16, 16/32; depths 1 and 2
  run serially. No competing tests were launched during the timed campaign.
  Desktop activity, clocks and temperatures were not controlled per frame.

All four runtime source hashes, adapter information, versions, target hashes
and effective configurations match across the twelve reports except for the
requested workgroup width. No runtime code or default was changed. The dirty
working-tree flag records generated benchmark output/cache files.

## Numerical quality

The fixed interior population is **26,924 points**. Every timed frame passes
the existing gates (coverage >=90%, displacement RMSE <0.05 px,
displacement error p95 <0.1 px, strain aggregate RMSE <0.002).

| Worst over timed frames | Width 16 | Width 32 |
|---|---:|---:|
| Fixed-support coverage minimum | 99.9851% | 99.9851% |
| Displacement RMSE maximum | 0.0285778 px | 0.0285785 px |
| Displacement error p95 maximum | 0.0530670 px | 0.0530707 px |
| Strain aggregate RMSE maximum | 0.00144604 | 0.00144603 |

Quality is identical across repeats and depths for each width. Widths are
not bit-identical: final-frame displacement-vector RMSE between widths on
jointly accepted interior points is 0.00010093 px, with zero differing
interior statuses. All exported numeric field buffers are finite.

Untimed replays find mean fine/coarse iterations of 2.57927/2.00801 (16)
and 2.57924/2.00802 (32). For either width, conservative stopping loses
46 late-convergence events over the whole timed grid and zero on the fixed
interior, compared with a same-seed no-stop shadow solve.

**The whole grid is not qualified by the interior gates.** Its worst
displacement RMSE is approximately 3.2865 px for both widths and its minimum
accepted coverage is 89.99%. Known edge failures are retained in the reports
and NPZ fields.

## Reproduction

From the repository root:

```bash
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/benchmark_realtime_optimization.py \
  --target-cache /tmp/live-dic-amd-trajectory.npy --prepare-cache
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/results/workgroup-width-amd-20261030/run_campaign.py \
  --target-cache /tmp/live-dic-amd-trajectory.npy
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/results/workgroup-width-amd-20261030/summarize.py
```

Skip cache preparation when that cache already exists. The campaign driver
overwrites this directory's run files. It and the summarizer are unchanged
copies of the existing Intel campaign scripts, with output paths relative
to their own directory. `summary.json` contains the recomputed statistics
and provenance/field checks; raw JSON and final NPZ fields are in `runs/`.

## Initial validation after measurement

These results describe the original campaign before the subsequent lifecycle
correction. See the [device-loss investigation](device-lifecycle/README.md)
for the diagnosed resource exhaustion, correction and later validation.

- Portable suite: 191 discovered, 121 passed, 70 GPU tests explicitly skipped.
- Ruff over `src`, `tests`, and `../benchmarks`: passed.
- Sphinx HTML build with warnings treated as errors: passed.
- Campaign summarizer: all provenance, quality determinism, timing and finite
  NPZ field assertions passed for all twelve runs.
- **Full GPU-enabled suite did not pass.** It aborted twice with exit code 134:
  `radv/amdgpu: The CS has been rejected ... (-28)` followed by
  `Parent device is lost` in `wgpuQueueSubmit`. The verbose replay located
  the abort in
  `CorrectedTemporalTests.test_zero_default_and_first_temporal_frame_match_bitwise`.
  Running that test alone also aborted with the same error. The preceding
  five GPU tests in `CorrectedTemporalTests` passed in the verbose run.
  Kernel log access was denied, so the underlying cause is undiagnosed.

All twelve timed benchmark processes and the two untimed diagnostic replays
completed before these test failures. Those results remain finite measured
observations, but this machine cannot be described as passing the full GPU
regression suite. No runtime fix was attempted as part of this measurement.
Validation output and the complete campaign console log are in `validation/`.

## Subsequent lifecycle correction

The device-loss cause was subsequently reproduced and traced to closed
engines retaining native GPU resources until the 4 GiB GTT budget was
exhausted. Releasing GPU references and collecting cyclic wrappers at owner
close resolves that test; shared levels do not destroy their parent's device.
The corrected working tree passed **all 193 GPU-enabled tests** on this AMD
host, as well as the portable suite, Ruff and strict Sphinx build.
See the [investigation and validation logs](device-lifecycle/README.md).
The initial failures and all original timing reports above are preserved.
