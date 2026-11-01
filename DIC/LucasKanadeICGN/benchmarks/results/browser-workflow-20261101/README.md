# Explicit prepare / solve / replay workflow

The starting revision is `06f646a`. The implementation is split across
`bd01d36`, `78c7c1d`, `fa5bb54` and `54756f6` plus the final keyboard guard.
The packaged numerical source tree remains
`c691f0be12a0a0e4d93720628d486321c4f0bfb0`: no Python/native/Three.js core,
canonical WGSL equations, interpolation, solver defaults or fixtures changed.

## Behaviour

Input seeks preview decoded images only. Reference/mask preparation remains a
cold operation; mask editing presents the fixed reference, not the current
previewed target. Solve starts/stops calculation and selects one target or
its remaining sequence. The GUI initially selects saving so ordinary analyses
produce replayable fields. Source changes preserve that preference but require
an explicit new start. Recording and custom session defaults remain unchanged.

Results places quantity, Play/Pause, rate and slider beside the image, with its
colourbar visible above the field. Acquisition statistics are hidden in review.
Playback/slider navigation never requests correlation. Quantity and appearance
persist, while saved per-field recipes and measurement annotations are restored.
Apply a batch recipe explicitly for uniform processing. Fast seeks coalesce,
including when a playback load is already in progress. Source operations cancel
pending review seeks. The Space key controls saved-result playback; reference
capture is unavailable from the review keyboard shortcut.

## UHD evidence

`hardware.json.gz` and `software.json.gz` exercise the actual UI at 3840x2160
on Intel Vulkan and Chromium SwiftShader. Unmodified plate reference/frame095/
frame225 PNG bytes are staged in natural file order. The test proves:

- preparation and input seeking produce zero target fields;
- mask edit/cancel switches to the reference and returns to the input preview;
- three target calculations save exactly three fields and open Results;
- replay keeps the chosen quantity despite different saved per-field views;
- changing quantity during playback remains effective on following fields;
- user seeks [first, last, middle] issue only the first and final requested load;
- seeking during a held playback load waits for it, then loads the latest seek;
- every review operation leaves the correlation count at three;
- parameter/status/iteration array fingerprints match before and after review.

`results.png` is an actual final UI screenshot. These are correctness tests,
not acquisition-FPS or all-OS/all-GPU qualification.

Other retained hardware checks:

- `pipeline.json.gz`: six full UHD fields survive held writes, pause, quota
  failure, retained RAM recovery, late decoding, source replacement and project
  export. Exact numerical arrays remain checked.
- `workbench.json.gz`: eight fields, NPZ/project interchange, saved recipes,
  measurements, replay and the independent NumPy strain operator (maximum
  difference about 1.94e-17).
- `editor.json.gz`: original and edited plate masks and both original load
  states remain within existing scientific gates; displacement RMSE is below
  0.00936 px with 100% coverage on the defined far-field population.
- `local-sources.json.gz`: explicit fixed/consecutive single-target calculation,
  full ROI after flat-image failure, image-size rejection and local video seek/
  play/pause. Native/browser status mismatches are zero; compared parameter
  RMSE is below 7.83e-6 and maximum below 0.001395 in this run.
- `review-no-webgpu.json.gz`: an actual 31,474-value UHD field is processed with
  no WebGPU and no analysis Worker, producing zero correlations.
- `camera-functional.json.gz`: a **two-second static fake-camera smoke check**
  of explicit Start/Pause, first-reference reuse and saved-field admission.
  Its short timing samples are not a performance campaign or physical-camera test.

The complete local native/browser-enabled unit suite passes 283 tests without
skips (`tests.log`). Ruff, Node syntax, strict Sphinx and repository-link checks
also pass. CI gains the manual-workflow check; no new remote CI result is claimed.
Each raw report retains its actual build record. Earlier reports use dirty
intermediate sites; the final playback checks use clean `54756f6`. Supplemental `keyboard-hardware.json.gz` and `keyboard-software.json.gz`
repeat the complete workflow after the final shortcut guard, including an R key
that cannot capture a reference while reviewing. Those builds are marked dirty
and their final source hashes are in `verification.json`. Images and solver settings were
not weakened to pass a test.

## Reproduce

From the repository, with its development environment:

```bash
python3 tools/build_pages.py --output /tmp/live-dic-workflow-app
DIC/LucasKanadeICGN/python/.venv/bin/python tools/qualify_browser_workflow.py \
  --site /tmp/live-dic-workflow-app --output /tmp/workflow-hardware
DIC/LucasKanadeICGN/python/.venv/bin/python tools/qualify_browser_workflow.py \
  --site /tmp/live-dic-workflow-app --software --output /tmp/workflow-software
```

This uses the ordinary workflow with `?smoke=1&case=manual`; canned diagnostic
URLs separately request an unrecorded preview calculation. Gzip records are
lossless; `raw-manifest.json` retains uncompressed SHA-256 and byte counts.
Input pixel/PNG hashes are retained in the workflow reports. No original camera
images are stored by result recording; keep source images separately.
