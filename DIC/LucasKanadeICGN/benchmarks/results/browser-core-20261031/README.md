# Portable browser core qualification

This campaign accompanies the extraction of the dependency-free `web/core`
engine, its temporal/full-ROI port and the standalone worker PWA. The native
layout change is qualified separately in `../binding-budget-20261031/`.
Pierre Margerit's foundational work remains the starting point of this project.

## Boundaries and admission criteria

`criteria.json` was committed before executing the temporal browser graph.
Each of sixty chronological frames is retained for both translation and affine,
at one and two independent slots: 240 comparisons per implementation pair.
Black frames 20–22, a history reset at frame 40, age expiry, a one-pixel crack
and an internal hole are included. These are correctness comparisons, not FPS
measurements. No frames or rejected results are removed from the reports.

`final-hardware-temporal-roi.json` compares native Vulkan with Chromium WebGPU
on the Intel GPU. `final-software-temporal-roi.json` compares native lavapipe
with Chromium SwiftShader. Both contain the exact configurations, native
adapter, versions, input hashes, source hashes and all per-frame differences.
Neither has an exceedance: zero status differences; respectively ten and
sixteen iteration differences across the complete sequence. The largest
common-accepted parameter differences are 0.001481 and 0.001451. These are
cross-runtime tolerances, not a claim of bit identity.

`temporal-roi.json`, `slots-temporal-roi.json` and
`software-temporal-roi.json` are earlier immutable implementation checks.
They are retained for provenance, not competing current speed tables.

`uhd-roi.json` checks the actual archived 3840 × 2160 tensile frame 40 and
plate-hole frame 95 against the native engine. Geometric exclusions are exact;
all statuses agree. Tensile truth is evaluated only on the explicitly defined
selected interior population, with unchanged scientific thresholds. The
unmasked extraction control is `extraction-uhd.json`.

## Demonstration and its negative control

The PWA uses a 1080p compressed demonstration derived by the checked-in
`tools/prepare_browser_demo.py`. Original UHD fixtures were not regenerated.
Its manifest identifies the inputs, image formation, encoding command and
file/pixel hashes. A 30 Hz source cannot demonstrate 60 Hz camera acquisition.

The HUD truth RMSE uses **every accepted subset**, with no interior crop.
Around frames 146–148 it can spike to about 0.687 px. This is retained and
highlighted in the UI, not corrected by changing the algorithm or population.
`demo-native-errors.json` is the original diagnostic. The reproducible
`../../qualify_browser_demo.py` replays the same decoded video in the native latency
profile and records decoded-frame hashes and the worst point per frame.

The worst point is at (55.5, 567.5). Its true x displacement at frame 146 is
−60.638 px, putting its true target outside the image; the solver accepts
−1.732 px instead. The same wrong basin occurs natively. This limits the demo's
scientific interpretation and is separate from the lossless UHD interior gate.

`display-initial-full-poll.json` is the initial 70-second display/offline
control. That harness retrieved its growing history every half second. The
final harness polls only completion/error counters while timing; full logs
are retrieved after the measurement. Reports identify those different
observer conditions rather than attributing their difference to LK changes.

## Reproduction

From the checkout root, with the package installed and Chromium available:

```bash
python3 DIC/LucasKanadeICGN/benchmarks/qualify_browser_core.py \
  --output /tmp/browser-hardware.json
python3 DIC/LucasKanadeICGN/benchmarks/qualify_browser_core.py \
  --native-device cpu --software-browser --output /tmp/browser-software.json
python3 DIC/LucasKanadeICGN/benchmarks/qualify_browser_uhd.py \
  --output /tmp/browser-uhd.json
python3 tools/build_pages.py --output _site
python3 tools/smoke_browser_app.py --site _site --frames 1800 --duration 70 \
  --offline --export --report /tmp/display.json --screenshot /tmp/display.png
python3 DIC/LucasKanadeICGN/benchmarks/qualify_browser_demo.py \
  --output /tmp/demo-native.json
```

The last optional developer diagnostic needs FFmpeg. FFmpeg is not installed
by the package, used by the application or needed to assemble existing assets.
Do not run other GPU work, builds or tests during a timed display campaign.

## Coverage limits

Executed locally: Linux Chromium 155, Intel Lunar Lake Vulkan, a dedicated
worker, software Chromium SwiftShader and native lavapipe. Hardware camera
permissions/capture/locks, Android Chrome, iOS/iPadOS Safari, desktop Safari,
Firefox and the main-thread rendering fallback remain unqualified. Portable
OS CI does not prove those browser/device combinations. The new software
browser CI and Pages workflows are prepared locally; no remote run or
deployment is claimed. No Pages setting was enabled.

## Final display and installation evidence

`display-final-70s-offline.json` contains 2,092 worker records spanning 70.507 s.
The lightweight observer measured 29.66 completed fields/s from the 30 Hz
source. Worker-available to observable GPU field plus HUD: median 9.60 ms,
p95 12.70 ms, p99 14.10 ms, maximum 27.10 ms. Minimum acceptance is 90%.
`display-summary.json` and `display-over-time.png` reproduce the aggregate and
300-frame quantiles with `analyze_display.py`; the full uncropped truth spikes
remain visible. Age drifts during the run; its cause was not isolated.

`display-build.json` records clean source `a545b8a` and the canonical shader
hashes. `display-screenshot.png` shows the actual rendered app after offline
reload; its short-window HUD after reload is not the sustained-run aggregate.
The timed observer was the pre-export-check version of
`tools/smoke_browser_app.py` at that source revision. Ordinary desktop apps
remained open; no tests/builds ran during the measurement.

`display-export.json` separately validates all 8,040 CSV rows and metadata JSON;
one worker full-field map/decode took 6.90 ms. This is one explicit cold export,
not a throughput or quantile result. CSV formatting/download and main-worker
request delivery are outside that number. `display-software.json` validates
the same finite-output and download contract on Chromium SwiftShader; the
replay is deliberately slowed. No application/console/GPU error was captured.
`demo-native-final.json` is the reproducible 200-frame negative-control report,
including the worst coordinate and decoded pixel hash for each image.

`installed-wheel.json`, `validation.json` and the archived logs verify the
pure-Python wheel, six WGSL/ten JS core resources, outside-checkout UHD ROI and
no-pickle NPZ, editable/native-extra installation and CLI. The installed Python
3.12 suite executes 202 tests with both GPU and browser flags: all pass, no
skips. This is local evidence, not a remote CI result. The installation used
NumPy 2.5.4 whereas native timing/parity used NumPy 2.4.6; neither package nor
runtime dependencies were pinned/upgraded in the project by this check.

`capability-profile.json` records the measured resolution recommendation with
all five retained samples per candidate (two warmups). This later advisory run
chose 1080p: the maximum of five samples was 6.80 ms there and 17.50 ms at UHD.
These short zero-motion R8-upload measurements exclude decoding, colour input
and display. They neither overrule the changing-trajectory gates nor establish
UHD camera performance. The initial camera resolution now follows that measured
recommendation; the shipped analytic demonstration remains explicitly 1080p.

`display-final-ui-smoke.json` checks the final camera-resolution-selection and
pre-capture control acknowledgement changes at `4b792c1`; it again verifies
rendering, CSV/JSON and server-stopped offline reload. These UI-only changes do
not alter the numerical graph measured in the 70-second report.
