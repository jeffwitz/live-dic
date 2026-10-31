# Browser mask editor and scientific display qualification

Implementation starts from `b285dec` and adds no changes to the Python package
or canonical WGSL numerical sources. Application code is committed in `bb5beb6`,
with preview-memory cleanup in `019e351` and exact conic boundary correction in
`8480694`. The reports retain their assembled revision and shader/input hashes.

## What is checked

- Seven CPU contracts cover multiple rectangles, ellipses, circles and polygons; complete pixel-support
  containment, nested holes/islands, overlap and duplicate handling, concave
  geometry, exact circle boundary pixels and all original UHD preset pixels.
- Original 3840 x 2160 plate reference, frame 95 (250/0 MPa) and frame 225
  (250/80 MPa), without resized or newly generated scientific images.
- Real DevTools mouse events create a rectangular hole, an elliptic island,
  another circle and a polygon. A circle is resized numerically and moved by
  dragging; deletion/undo and applying all six shapes are checked.
- Independent NumPy analytical displacement/strain comparison and the archived
  plane-strain VM convention. Truth never enters registration.
- A constant synthetic field produces the expected GPU VM colour. Arrows
  visibly change rendered pixels; source field values remain bit-identical;
  invalid origins do not draw arrows. The zoomed actual viewport retains the
  grey excluded hole at its expected centre.
- Explicit finite CSV/JSON export retains vector geometry and the full binary
  mask. The compressed `plate-export-offline.json.gz` is a lossless complete
  smoke/export report; it also records server-stopped offline reload. That
  check uses `019e351`, before the conic boundary correction, whose original
  preset pixels remain exactly unchanged.

`hardware.json` and `software.json` identify the actual Chromium adapter and
source build. `editor.png` shows the reference-space shapes; `von-mises-arrows.png`
shows frame 225 with 4x presentation zoom and magnified vectors. Full-page
DevTools screenshots may mishandle nested scrolling, so the scientific figure
uses the actual viewport, not the complete-page capture.

Quality values and their populations are centralized in the performance
reference. The selected far-field population changes with edited masks. The
rows are correctness gates, not a claim that masking improves precision, and
not a live FPS campaign. Near-hole gradient accuracy and phone/touch hardware
remain outside this qualification. A static pair is labelled as such in the UI.
The ordinary camera/video path has no new complete-field readback.

## Reproduce

With the Python package, Node and Chromium installed, from the repository root:

```bash
python3 tools/build_pages.py --output _site
python3 -m unittest discover -s DIC/LucasKanadeICGN/python/tests -p test_browser_roi_editor.py -v
python3 tools/qualify_browser_editor.py --site _site \
  --report /tmp/editor-hardware.json --screenshots /tmp/editor-hardware
python3 tools/qualify_browser_editor.py --site _site --software \
  --report /tmp/editor-software.json --screenshots /tmp/editor-software
python3 tools/smoke_browser_app.py --site _site --plate --frames 1 \
  --export --offline --report /tmp/plate-offline.json
ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -v
```

The software-browser workflow now prepares the pointer/render qualification
in CI. Only local results are claimed here; no push, PR, Pages activation or
remote CI run is performed by this task.

The final hardware/software interaction reports use the same `8480694` build.
The dirty flag covers documentation edits only; numerical/PWA source matches
that revision. The later HUD tooltip correction only labels the plate's
far-field population correctly. It does not change the values scored or drawn.
The archived full export/offline check precedes the endpoint correction; the
exact full-image preset hash test proves its mask unchanged.

## Final validation

`tests-209.log` executes 209 tests with native GPU and browser flags enabled:
all pass, no skips. The complete suite and final software-editor replay ran
concurrently; their elapsed times are not a performance qualification.
`validation.json` records the commands, zero native-package changes, and style /
packaged-resource checks. `final-sphinx.log` is the strict completed docs build.
The ordinary Python wheel resources remain byte-identical (44 files); this
follow-up changes the standalone app, its development tests and assembly only.
