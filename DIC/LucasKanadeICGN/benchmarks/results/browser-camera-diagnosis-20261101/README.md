# 1080p camera pipeline diagnosis — 2026-11-01

This is a **simulated camera control**, not a test of the user's physical camera.
Chromium delivers two identical Y4M frames at 60 Hz: an unresized 1920 × 1080
crop at (900, 500) of the checked-in native-UHD tensile reference, original luma
and neutral chroma. No physical camera or camera settings were touched. The input
hash, crop, browser identity, adapter and complete controller settings are in
the reports. No GN formula, iteration default or tolerance was changed.

Detailed numerical tables belong in the
[performance reference](../../../docs/reference/performance.rst#camera-pipeline-diagnosis).

## Records and boundaries

- `baseline.json.gz`: clean `2851e7b`, fixed/off, fixed/recording and consecutive/off.
  Roughly 338 ms CPU pair preparation and 9 ms observable solver calls explain
  the few-FPS consecutive case. This establishes a plausible mechanism, not the
  settings of a separate camera session. The user subsequently confirmed that
  their session selected consecutive images.
- `timing-reset.json.gz`: diagnostic UI iteration before HUD throttling. The new
  mode correctly reports its own age instead of a median dominated by previous
  fixed-reference samples. Its cadence differs from the baseline; this was not
  a controlled A/B performance campaign and must not be called a causal gain or
  regression. Builds/checks overlapped parts of the exploratory run.
- `sampled-hud.json.gz`: clean `b9ee40e`, stage HUD sampled at about 1 Hz, with
  the maintained diagnostic command. No other test/build was launched during
  this final timing control. The user desktop/workload was not isolated. The
  result remains a diagnostic observation, not a sustained-rate qualification.

Completed-field rate uses application completion counters over the observation
span. It includes capture admission, GPU processing, display and recording when
selected. Short-window counts and scheduling need not equal the negotiated
60 Hz input setting exactly. Reference-preparation and observable-solver wall
samples have distinct boundaries and are not physical GPU timestamps. Source
exposure-to-display latency is not measured. The static texture/motion and short
three-second warmup/ten-second observation do not qualify arbitrary camera scenes.

The controller now clears rolling ages/completion times on comparison changes.
A deterministic-clock test verifies that a 55 ms new-mode sample cannot inherit
the previous 100 ms median/FPS window. The stage line reports the latest sampled
reference preparation, solver call, recorded full-readback and main-thread
storage/admission. The latter includes cache/index/review work, not isolated disk
I/O. Neither this reporting fix nor the label change accelerates preparation.

## Reproduce

```bash
python3 tools/build_pages.py --output /tmp/live-dic-camera-site
DIC/LucasKanadeICGN/python/.venv/bin/python tools/diagnose_camera_pipeline.py \
  --site /tmp/live-dic-camera-site --output /tmp/camera-diagnosis.json
```

The script always uses a fake camera. `--software` selects SwiftShader;
`--resolution 3840x2160` requests a separate UHD control. Neither was used for
the timing table above. Derived Y4M inputs are temporary; no image fixture was
regenerated or changed. Gzipped reports are lossless and `files.json` records
uncompressed SHA-256/length. Local fixes are not evidence of a remote CI run or
of an already-updated public Pages application.

The final local native/browser-enabled suite passed 263 tests with no skips
(`suite.log.gz`). The actual UHD workbench passed eight saved fields, mobile
layout and independent NumPy operator comparison (`workbench.json.gz`). Ruff,
JavaScript syntax and strict Sphinx checks passed locally. No new remote CI
execution is claimed.
