# Lightweight live presentation qualification

## Initial renderer (`b68e488`)

The predeclared `criteria.json` compares frozen scalar-display `e781957` with
`b68e488`, order A/B/B/A. Presentation is 10 Hz maximum / 1920 px in both,
with bounded two-field recording overlap explicitly enabled. Numerical profile
is common-zero: affine 16/16, B-spline, median, two levels 8/16 iterations,
0.005 tolerance, uniform, fixed first reference, no temporal features or ROI.
Input is the original UHD reference as a static simulated 60Hz camera.
This is not physical-camera or moving-deformation FPS qualification.

Intel Lunar Lake, Mesa 25.2.8, Chromium 155, Linux 7.0.0-38, performance power
profile. Each recording-off/on observation has 4s warmup and 20s measurement.
Tests/builds were completed first; none ran during timing. Cold setup and final
drain/quality readback are outside observations. `run_campaign.py` and raw
commands retain exact local paths, which must be recreated to rerun.

| Renderer | Calculated fields/s, no recording | Committed fields/s, recording |
|---|---:|---:|
| Scalar controls (2) | 43.250–44.361 | 29.667–29.855 |
| Initial vectors (2) | 43.614–43.645 | 29.423–29.576 |

There is **no measured renderer speedup**. Calculation is within the observed
reference spread. Recording and admission-p95 are inconclusive under the
reference-only envelope: one candidate falls outside, one inside. Do not call
that a passed nonregression gate or hide the small slowdown. Candidate stored
field admission p50/p95/p99 is 38.8–39.2 / 53.7–55.14 / 59.71–60.70 ms;
controls are 38.1–38.5 / 52.43–53.32 / 59.52–62.00 ms. These include cache and
IndexedDB admission; they are neither GPU time nor exposure latency.
Actual preview cadence is 9.39–9.79 Hz, a maximum rather than a deadline.

All four final stored fields have finite float32 parameters, 31,654 accepted
of 32,400 points, identity displacement RMSE 5.8743e-7 px and maximum 2.7439e-5 px.
No field is unsaved after draining. This final-field quality check does not
assert accuracy on every possible moving camera frame. The moving analytical
plate and original frame040 qualifications are separate.

## Functional evidence

The GPU pixel oracle checks the current image exactly, norm palette, fixed
current tips under magnification, tails/length, rejected crosses, no ROI glyphs,
unchanged field buffers, image-only reference preview and reduced preview size.
It executes on hardware and SwiftShader. The actual original UHD plate has
100% far-field coverage and displacement RMSE about 0.00907 px.

Both hardware/software Results checks preserve eight UHD fields and match an
independent NumPy strain operator (maximum difference below 6.4e-17). Palette
and range changes do not increase analysis runs/component computations and
retain identical statistics. The actual recording pipeline preserves six UHD
fields through hold, pause, disk failure, recovery, source change and export.
Offline server-stopped UHD reload and CSV/JSON export also pass.
`tests.log` records 283 passing native/browser-enabled tests, without skips.

The later readability refinement changes only automatic glyph spacing/weight:
it needs its own source revision and timing record. The initial results above
must not be relabelled as measurements of that refinement.

## Final automatic overview (`0275a01`)

The readability refinement is qualified separately under
`overview-criteria.json`, with the same A/B/B/A numerical and timing settings.
The final renderer uses roughly 20 columns by 12 rows, selecting near block
centres, thicker shafts and manual-spacing overrides. It still reads actual
measured vectors, never interpolated field values. `overview-vectors.png` shows
the actual final view. Hardware and SwiftShader pixel oracles pass again, including an actual
centred-block sampling check; all
283 native/browser-enabled tests pass again (`overview-tests.log`).

| Renderer | Calculated fields/s, no recording | Committed fields/s, recording |
|---|---:|---:|
| Scalar controls (2) | 30.953–34.907 | 23.683–24.164 |
| Final overview (2) | 30.536–30.558 | 22.526–24.354 |

This matrix is globally slower than the first, including its unchanged control.
Do not compare the best value of one matrix with the worst of the other or
attribute the global drift to the final renderer. The cause was not isolated.
Calculation lies inside the wider reference spread. Recording remains
**inconclusive**: one candidate is slower than the reference-only envelope and
one is faster. Candidate admission p50/p95/p99 spans 52.90–53.90 /
71.86–87.45 / 80.32–172.75 ms; retain the isolated long tail. Controls span
53.30–54.20 / 74.10–74.17 / 80.82–86.52 ms. Neither matrix qualifies 4K60
or proves a renderer speedup/nonregression for recording.

The final stored identity fields are finite and retain the same 31,654/32,400
acceptance and 5.8743e-7 px RMSE. Every timed sample uses the persistent prepared
reference (zero per-pair preparation), and all runs drain with zero unsaved
fields. This validates ownership and the first-reference contract; the long
storage/admission tail still matters to real-time integration.

`overview-*.json.gz` and `run_overview_campaign.py` preserve all final raw data
and commands. Run `python3 summarize.py` to verify gzip hashes, numerical and
ownership gates, and recompute `summary.json` for both matrices. Presentation
sampling never changes full-resolution computed/saved field geometry.

## Files

Raw reports are losslessly gzipped. `raw-manifest.json` records their original
SHA-256 and byte sizes; build manifests distinguish dirty functional-test sites
from the clean timing site. `live-vectors.png` and `zoomed-vectors.png` show
actual initial-renderer output, not the later overview refinement. No source
image or canonical shader was regenerated. The packaged numerical source tree
is unchanged. Remote CI is configured but was not run/observed for this work.

The final device reports explicitly identify Intel xe-2lpg and Google
SwiftShader. Review without WebGPU and without a Worker also passes on a
31,474-value UHD import with zero correlations. `verification.json` retains
source hashes and commands for these post-timing checks.
