# Browser results workbench correctness

Hardware Chromium 155 (Linux/Vulkan, Intel Lunar Lake) and Chromium SwiftShader
both execute the actual original UHD plate pair through the PWA, freeze the
field, apply Gaussian displacement smoothing + VSG Green-Lagrange strain,
place line/gauge endpoints through pointer events, export NPZ, record the second
plate load and reimport the saved field. No console errors occur.

Independent NumPy Gaussian convolution/least-squares/Green-strain calculation
agrees at the checked interior point within 7e-17 on both adapters. The fixed-pair result
population remains 31,474 accepted of 32,220 selected. These are correctness
checks, not a full-field metrological error bound or recording-FPS qualification.
The original plate images and canonical solver shaders are unchanged.

```bash
python3 tools/build_pages.py --output _site
python3 tools/qualify_browser_workbench.py --site _site --output /tmp/workbench
python3 tools/qualify_browser_workbench.py --site _site --software --output /tmp/workbench-cpu
```

Use the project Python environment. The output contains a no-pickle NPZ and
viewport screenshots as well as the report. The checked-in reports identify the
build's source revision and dirty content version. Screenshots show the first
qualified implementation before subsequent interface refinements.

`history-hardware.json` and `history-software.json` extend the UI check to a
point/gauge extraction through three saved fields and a native CLI-style
compressed UHD sequence import. The latter explicitly asks for missing image
extent and selects field 2/2. Computed arrays, invalid samples and provenance
remain separate from processing recipes. The independent strain oracle retains
its pre-existing tolerance; no threshold was relaxed for the UI work.

`project-hardware.json` adds saving four UHD fields as one results project and
importing the validated batch, yielding eight library entries. The report stores
archive size/hash; no original fixture is regenerated. Masks shared by a
recording are saved once. Subsequent support-span labels distinguish smoothing
from the larger composed smoothing-plus-VSG strain support.

## Region, project-state and responsive follow-up

`regions-hardware.json` / `regions-software.json` add a real two-corner UHD
region measurement, checked against NumPy statistics of the exported processed
field. They also verify restoration of that region after project import and
check the unload guard before/after a full project download request. The
Space key activates a focused button without triggering acquisition shortcuts.
`regions-mobile-analysis.png` captures the same 4K data at a 390 CSS-pixel
emulated viewport with no horizontal page overflow. Desktop Chromium emulation
is not physical mobile hardware qualification.

The complete local suite passed 242 tests with native GPU and browser tests
enabled, no skips. These tests exercise correctness, not isolated live FPS.

## Navigation and review-only follow-up

`navigation-hardware.json` / `navigation-software.json` additionally exercise
compatible-analysis batch recipes, bounded two-hertz result replay and the
saved-field slider. `mobile-chart.png` shows the responsive region histogram.
These are UI correctness runs, not acquisition-rate measurements.

`review-no-webgpu.json` simulates a missing navigator.gpu and OffscreenCanvas
before app startup, then opens the actual archived-output UHD NPZ. It recovers
31,474 valid equivalent-strain values with statistics checked against NumPy.
`review-no-worker.json` also removes Worker, exercising pure CPU main-thread
analysis and an ordinary HTML canvas. The harness initially bootstraps Chromium
using SwiftShader; no correlation is run in the review-only application page.
These checks do not certify an untested browser or provide CPU IC-GN fallback.

```bash
python tools/qualify_browser_review.py --site _site --software \
  --field /tmp/workbench/processed-field.npz --output /tmp/review
# Add --no-worker to exercise main-thread analysis.
```

## Computed coefficients and installed distribution

`coefficients-hardware.json` materializes a native 3840 x 2160 quadratic field
from the archived tensile reference/frame 40 with B-spline, 16/16, two levels
and spatial policy disabled. Its 31,593 accepted samples are then imported into
the UI. Computed d²u/dx² statistics at 0.01 mm/px match NumPy in 1/mm and remain
identical after Gaussian/VSG displacement processing. Re-export preserves every
original parameter, status and iteration. `computed-curvature.png` deliberately
shows those unsmoothed coefficients, including their visible noise. This is
coefficient interchange/display qualification, not a new accuracy or speed
claim for the quadratic estimator.

`coefficients-software.json` repeats the UI coefficient checks on SwiftShader
using the same native output. The native field hash connects the two runs.

```bash
python tools/qualify_browser_coefficients.py --site _site --output /tmp/coefficients
python tools/qualify_browser_coefficients.py --site _site --software \
  --field /tmp/coefficients/native-quadratic.npz --output /tmp/coefficients-software
```

`installed-wheel.json` and `installed-wheel-tests.txt` record a fresh pure-Python
wheel installation outside the checkout, the six canonical WGSL resources,
unchanged runtime dependency versions and 247 passing tests with native GPU and
browser flags (no skips). Tests still read their source/fixture/standalone-app
files from the checkout; native and Python browser-engine imports/resources use
the installed wheel. The standalone PWA remains outside the wheel by design.
The wheel contains no first-party OS/architecture binary or SPIR-V.

## Applied-recipe export barrier

`project-barrier-hardware.json` / `project-barrier-software.json` repeat the
actual UHD workbench test with a project export invoked immediately after an
analysis-window edit. The script independently reads manifest.json and requires
that selected field's new window (9) rather than the previous recipe. Export
pauses acquisition and awaits the pending derived analysis outside the live loop.
The existing NumPy field oracle and raw-array preservation checks still pass.
