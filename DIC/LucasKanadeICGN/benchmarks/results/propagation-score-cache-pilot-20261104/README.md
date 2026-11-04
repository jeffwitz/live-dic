# Score-cache pilot with extra broadcasts — 2026-11-04

This is a **negative performance control**, not evidence of a UHD speedup.
Clean source `2c8939aa263e2569ec0db8316a47665b5ec3dd6c`, Intel LNL/Vulkan,
Mesa 25.2.8. The initial implementation broadcasts the cached value with an
additional `workgroupUniformLoad` for every candidate in later passes. The
subsequent version moves those records into the already-existing candidate
broadcast; these reports must not be attributed to that later implementation.

| UHD case | Recompute median / p95 | Cache median / p95 | Paired median saving (95% CI) |
|---|---:|---:|---:|
| Original tensile frame 40 | 15.244 / 16.207 ms | 15.251 / 16.209 ms | −0.017 ms [−0.090, 0.050] |
| Manufactured translation, coarse seed defects | 15.797 / 17.027 ms | 15.851 / 17.089 ms | −0.039 ms [−0.098, 0.012] |

Separate reference/reference campaigns fixed non-regression margins **before**
the candidates: 0.10140 ms and 0.13060 ms respectively (p95 absolute per-block
median paired differences). The intervals are within those margins and include
zero; no useful gain is established. Twelve blocks contain one hundred frames
per variant, per-frame AB/BA alternation, eight warm-ups, one prepared device.
Confidence intervals bootstrap the twelve block medians, 10,000 resamples,
seed 1729. Parameters, statuses and iteration fingerprints agree.

All cases use translation 32/16, B-spline, two fixed levels (8/16), tolerance
0.005, raw fields and two failed-only propagation passes. Empty-work elimination
is enabled for both variants. Resident NumPy target to materialized
field/statistics is timed; decoding, preparation, display and storage are
excluded. No other project tests, builds or GPU benchmarks ran during timing.
Power settings were not captured by this pilot driver, so these reports do not
provide a complete machine-power provenance record.

The second case manufactures `np.roll(reference, (-6,14), axis=(0,1))` from the
original UHD reference, without modifying fixture files. A fixed 3×3 pattern of
coarse seed errors (`x%8<3`, `y%8<3`, x/y>3) replaces that level's initial
translation with (25, −18) coarse pixels. The frame resets this prior on every
call. This tests propagation under controlled initialization errors, not a
natural discontinuity or a claim that all large deformations converge. Its
accepted interior coverage is 99.8706%, vector RMSE 0.00000650 px; the original
case retains 100% coverage and about 0.01017 px RMSE. Both meet the driver's
unchanged analytical gates. The active case queues 928/112 coarse and 217/43
fine re-solves across its two passes. The easy case queues none.

`criteria.json`, four compressed original reports and `summary.json` retain
commands, source/input/derived-array hashes, samples, fields' fingerprints and
work counts. The manufactured run also lists the unused frame-40 file hash;
its explicit target recipe and array hash identify the actual target. Later
driver versions omit that unused input. Sources are recoverable at the recorded
commit. Private images and their reports are not included.
