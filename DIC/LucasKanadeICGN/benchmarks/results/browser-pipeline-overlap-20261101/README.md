# Bounded recording overlap and independent preview

This campaign qualifies an **opt-in standalone PWA supervisor**, not a new
numerical solver or native Python default. The existing GPU Worker still owns
upload, reference preparation, solving and field rendering. The main thread
owns presentation and asynchronous IndexedDB admission. Two reserved CPU slots
allow computation and recording to overlap without an unbounded backlog.

## Timing protocol

`criteria.json` was committed before timing. All twelve timed runs used the
clean site assembled at `60f055d`, GPU reference preparation, consecutive pairs,
affine 16/16, B-spline, two fixed levels, median and unchanged solver schedules.
The static fake 60 Hz camera uses the original UHD tensile reference, or an
unresized 1080p crop. Source hashes/crop, negotiated capture settings, numerical
configuration, adapter limits/features, browser version and raw samples are in
each report. No physical camera was opened.

For each resolution the order was A/B/C/C/B/A: synchronous, queue only, queue
plus 30 Hz / 1920 px preview, then the reverse. Each fresh browser used 4 s
warmup and 20 s observation. `commands.txt` contains the exact commands in order.
No tests/builds ran during these windows. Power profile was `performance`;
`environment.json` records the Linux/Mesa/Intel configuration. Other ordinary
desktop processes were not terminated; this is not a thermally isolated lab.

The primary rate counts **committed disk fields** between main-thread monotonic
snapshots. Computed replies are reported separately. One commit can straddle a
counting window; the predeclared two-field allowance accounts for this. The
final pause drains pending work outside the timed window before the last
persisted field is read and scored. The report's pipeline/stop snapshots precede
that drain; they can correctly show work in flight. They are not final states.

Latency starts at Worker arrival and ends at results-library/disk admission.
It excludes camera exposure and upstream delivery. Admission samples inside
the counting interval produce each repeat's median/p95/p99. The counts are
roughly 336–401 at 1080p and 90–133 at UHD; p99 from such short observations is
not a hard latency guarantee. GPU-call timing excludes later recording and is
not a GPU hardware timestamp. Do not mix these boundaries.

`summary.json` is reconstructed by `python analyze.py` (NumPy required). The
script verifies all compressed payload hashes, checks the predeclared
quality/ownership gates and prints every repeat. Both candidates exceed the
faster baseline plus the count-boundary allowance at both sizes. Two repeats
are evidence for this control, not confidence intervals or all-device proof.
The human-readable tables and interpretation are in
[`docs/reference/performance.rst`](../../../docs/reference/performance.rst#browser-recording-overlap).
Queue-only throughput increases admission age; the smaller UHD preview lowers
both rendering/bitmap cost and admission age here. No 4K60 claim is made.

Every last persisted field has full input dimensions and the original grid,
finite Float32 parameters, the same accepted population and sub-`8e-7` px
identity displacement RMSE. Every measured pair has adjacent **received-input**
ordinals. This does not imply that a busy camera delivered every exposure, nor
qualify precision on arbitrary moving scenes. The original tensile numerical
qualification and the native/browser tests remain separate evidence.

## Ownership, failure and interchange controls

`hardware-clean.json.gz` and `software-clean.json.gz` are actual UHD file-input
checks from the clean `a5ed52b` site. Seven local PNGs alternate unchanged original
reference/frame-40 bytes. The test holds real IndexedDB writes, computes two
fields, requires Pause to wait, injects quota failure, retains both completed
fields in RAM, retries without duplicates, drains a source change and exports
then reimports six fields. Parameters, status and iterations match the captured
computed arrays exactly; the saved fields remain 3840 x 2160 / 240 x 135 while
the preview is 1920 x 1080. SwiftShader times are correctness evidence only.

`late-decode-negative.json.gz` preserves a deliberate failing control. Apply
`late-decode-negative.patch` inside a copy of the assembled site to remove only
the late-playback pause guard. The qualifier holds decoding file 5 until the
disk failure has paused recording, then releases it. This control advances the
cursor to 5 with only files 1–4 recorded; the fixed application stays at 4 and
eventually records files 1–6 in order. The failing report is retained, not
reclassified as a successful qualification.

`software-first-correction.json.gz` is the earlier successful software run that
verified the naturally slow-decode case before the deliberate decoder hold was
added. Its build label is `d1e9604` plus dirty fixes; `manifest.json` records
matching application hashes and the later clean reports remove source ambiguity.
The final `tools/qualify_browser_pipeline.py` includes the deliberate hold, saves
partial reports on assertion failure and is configured in its own CI job. No
new remote CI execution is asserted.

Reproduce from the checkout and a fresh site:

```bash
python tools/build_pages.py --output /tmp/overlap-site
python tools/qualify_browser_pipeline.py --site /tmp/overlap-site \
  --output /tmp/overlap-hardware.json
python tools/qualify_browser_pipeline.py --site /tmp/overlap-site --software \
  --output /tmp/overlap-software.json
```

`full-tests.log.gz` records 278 successful native/browser-enabled tests, no skips
(`ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python -m unittest discover -s
DIC/LucasKanadeICGN/python/tests`). `default-uhd-workbench.json.gz` additionally
checks the unchanged default UI on eight saved UHD fields, interchange and an
independent NumPy strain operator. The full-suite and GUI runs precede timing;
they do not contribute timing samples. Ruff, Node syntax, repository links and
strict Sphinx checks also pass locally. Default settings and canonical numerical
shaders were not changed by this work. Original images were not regenerated.
