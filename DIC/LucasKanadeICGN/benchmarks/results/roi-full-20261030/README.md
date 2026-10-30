# Full-window ROI qualification

Starting revision: `4a435841eeda09a5611ef1f93573dd5370ac10e7`.
Candidate reports correctly record a dirty checkout; their source SHA256 maps
identify the Python/WGSL actually loaded. Baseline reports load an isolated
worktree at the starting revision. No performance tolerances or solver defaults
were changed to obtain these measurements.

## Functional scope

Lot A implements `set_reference(..., roi_mask=mask, roi_policy="full")` in both
native engines for translation and affine models. `partial` is unavailable and
raises an explicit error. Quadratic processing without a mask is preserved;
quadratic ROI and browser ROI are explicitly refused.

The CPU oracle tests compare nominal squares directly, including interior holes,
odd/even sizes and overlap. GPU tests cover retained reference quantities, raw
single-level equivalence, immutable ownership, excluded statuses, empty-mask
execution, conservative reduction, crack permissions, all-true pipeline identity,
seed fallback, selective recovery, stream slots, reconfiguration, subgroup
admissibility, numerical rejection and NPZ export. Original UHD images and their
provenance are read unchanged from `assets/uhd_tensile`.

The small smooth-texture affine test uses the existing full-pyramid fine
tolerance 0.005 for both engines and an explicit fine budget of 30 iterations.
This separates model accuracy from a short iteration budget when coarse seeds
are unavailable. The UHD tests and all performance tables retain the normal
two-level `(8, 16)` schedule and existing tolerances.

## Measured boundary and reproduction

Intel Core Ultra 5 228V / Intel Lunar Lake integrated GPU, Vulkan,
Mesa 25.2.8, wgpu-py 0.32.0. Reports include actual adapter information,
Python/platform versions, commands, image and mask hashes, source hashes, raw
latencies, completion intervals, iteration histograms, acceptance and errors.
Iteration histograms use accepted positions; geometric exclusions do not
contribute. The initial predeclared reference campaigns predate the addition of
source-hash fields to the reporting script; their immutable starting commit and
input hashes remain recorded, and later baseline rechecks include the fields.

Each campaign warms the actual `register_stream` path for 12 changing images,
then measures 160 images, with two frames in flight. Each report contains three
campaigns. The boundary starts with a resident uint8 NumPy image and ends with
the complete dense NumPy field. Preparation/compilation is reported separately.
Acquisition/decoding are excluded. Reported table entries are medians across
campaigns, including the median of their p95/p99 values.

```bash
export PYTHONPATH=DIC/LucasKanadeICGN/python/src
python DIC/LucasKanadeICGN/benchmarks/results/roi-full-20261030/generate_masks.py \
  /tmp/live-dic-roi-masks
python DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
  --backend native-pyramid --device-type gpu --fixture-set uhd-tensile \
  --motion-model affine --outlier-policy median --pipeline-depth 2 \
  --frames 160 --warmups 12 --campaigns 3 \
  --roi-mask /tmp/live-dic-roi-masks/large.npy
```

`run_roi_cases.py` alternates baseline/candidate controls for the translation
timing fixtures before measuring large/small/fragmented ROI. `run_tensile_cases.py`
uses only the original UHD reference and frame 40, alternating identity and
known affine deformation. Both accept the mask directory and baseline Python
source directory. The two data sets are distinguished in the JSON: tiled timing
fixtures qualify execution cost, while the original tensile images qualify
affine metrology. Frames 100/199 are not required to converge from zero seeds.

## Original UHD tensile images: affine, 16/16 B-spline, median

| Configuration | Selected | Minimum accepted | FPS | Age median ms | p95 ms | p99 ms | Maximum displacement RMSE px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Starting revision, no ROI | 32,400 | 31,648 | 97.85 | 20.07 | 21.76 | 23.36 | 0.02435 |
| Candidate, no ROI | 32,400 | 31,648 | 93.31 | 21.53 | 24.13 | 25.45 | 0.02435 |
| Candidate, all true | 32,400 | 31,648 | 86.59 | 22.84 | 26.69 | 31.62 | 0.02435 |
| Large compact ROI | 25,300 | 25,295 | 96.89 | 20.53 | 23.40 | 25.77 | 0.02392 |
| Small ROI | 1,400 | 1,400 | 125.62 | 16.11 | 18.71 | 19.26 | 0.01857 |
| ROI with slits and a hole | 24,410 | 24,404 | 95.22 | 20.13 | 25.83 | 27.40 | 0.02408 |

These acceptance counts include both images; errors are evaluated on accepted
selected windows. The formal UHD regressions additionally use the existing
interior population and retain the displacement RMSE <0.05 px, p95 <0.10 px,
coverage >=90%, and median-gradient component RMSE <0.002 criteria. Raw outputs
are not held to the median-gradient bound. Pure translation is tested separately
using translations of the original UHD reference, as well as a continuously
sampled fractional-translation fixture.

## Performance decision: real-time non-regression remains open

`predeclared-margins.json` and `tensile-predeclared-margins.json` were written from
reference/reference repetition **before** evaluating the corresponding candidate.
The rule uses the observed relative range of three reference campaigns, rather
than an invented universal millisecond threshold.

The experiments are too variable to establish non-regression within those
margins. In the alternating translation-fixture controls, the old revision itself
drifts from 93.85 to 88.68 FPS; affine controls drift from 76.39 to 67.80 FPS.
Earlier old-revision campaigns reached 102.66 and 92.12 FPS respectively.
This demonstrates drift in the measurement environment; its physical cause was
not established. In particular, do not attribute every difference to ROI or
assert that it is thermal throttling without additional instrumentation.

All-true and absent-mask fields are equal in the regression tests, using the
same active solver/median pipelines and no ROI transfer resources. Nevertheless,
the all-true wall-clock campaign is slower than the absent-mask campaign above.
That discrepancy is retained as evidence of an unresolved performance question,
not discarded or used to claim a causal speedup/slowdown. More stable paired
hardware campaigns are required to close real-time reception.

All configurations in the tensile table exceed 60 FPS in these measured streams.
Their result ages are a separate quantity and generally exceed 16.67 ms with
two images in flight. This does not establish an application-wide latency
guarantee. No image-to-GPU-only timings are substituted for the Python boundary.

## Optional subgroup

`benchmark_roi_subgroup.py` uses the same changing timing images, 12 actual
stream warmups, three campaigns of 160 frames, two in flight, affine 16/16,
B-spline, uniform weighting and median filtering. It forces subgroup selection
and records its numerical comparison with the portable solver. Reports cover
the old revision, candidate without ROI, candidate all true, and fragmented ROI.
This remains a single-level path with a smaller capture range than the full
pyramid; its acceptance must not be compared as though the algorithms matched.

The dense dispatch is retained. Excluded workgroups perform no IC-GN iterations;
full-image target preprocessing still has a cost even for a small ROI. No
unmeasured compaction benefit or zero-overhead partial-window mode is claimed.

## Final validation

`validation.json` records the final source hashes and installed-wheel check.
The starting revision passes 125 tests on the Intel device. The candidate passes
142 tests on Intel and 142 on Linux/Mesa lavapipe, with GPU tests explicitly
enabled and no unittest skips. Subgroup execution is qualified on Intel;
lavapipe exercises the incompatible-specialization/portable branches.
Ruff check/format and the Sphinx build with warnings as errors pass.

The `py3-none-any` wheel was installed with pip into a fresh Python 3.12
environment. From `/tmp`, with `PYTHONPATH` unset, the installed native engine
produced all 35 selected measurements of a small ROI and exported/reloaded its
NPZ with `allow_pickle=False`, including the exact mask, selection and hash.
The shared browser WGSL also executed successfully on the Intel adapter,
accepting 900 subsets on each of eight changing 512-square inputs.

The CI workflow now includes ROI regressions on lavapipe, but these local
changes have not been pushed or executed by remote CI. No Windows/macOS
hardware GPU execution is claimed.

```bash
ICGN_RUN_GPU_TESTS=1 python -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -v
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  LIBGL_ALWAYS_SOFTWARE=1 ICGN_RUN_GPU_TESTS=1 \
  python -m unittest discover -s DIC/LucasKanadeICGN/python/tests -v
```

Use the actual software-driver manifest path supplied by the target Linux
distribution. CPU/software execution validates correctness, not live GPU rates.
