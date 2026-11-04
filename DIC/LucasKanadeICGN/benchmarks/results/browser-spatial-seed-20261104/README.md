# Browser Solve GPU seed propagation — 2026-11-04

Functional qualification of source `449fb09ab48a31a1d3e801d2c0f3308b7c35d442`,
built with a clean working tree. This is not a throughput, camera or recording
latency campaign. Native timing reports in `../gpu-spatial-seed-20261104/`
describe a different supervisor and must not be presented as browser FPS.

`uhd-gui.json` records actual PNG input through the PWA controls, dedicated
Worker, session, disk storage, saved-field retrieval and NPZ export. Inputs are
the original 3840 × 2160 tensile reference and frame 40, with archived SHA-256
values. No image was resized or regenerated. Chromium 155 ran on Intel
xe-2lpg hardware; capabilities and source/shader hashes are retained in JSON.

The settings were local translation 32/16, B-spline, two levels (8/16
iterations), tolerance 0.005, ZNCC minimum 0.8, explicit median smoothing and
global affine initialization at six levels. The test first selected disabled
propagation, then two passes per level using the real Solve menu. Changing the
option marked preparation pending until Start. The first saved field retained
its original configuration after the second calculation.

The evaluated population contains 30,916 subset centres at least 32 pixels
from all image edges. Both settings accepted all those points, with vector
displacement RMSE 0.00944863 px and p95 0.01562031 px. All translation-model
gradients remained exactly zero. The script's existing gates were unchanged:
coverage >=99.5%, RMSE <0.01518230373115895 px, p95 <0.02722860762111737 px.
This smooth case verifies integration and preserved accuracy; it does not
demonstrate additional recovery on a difficult scene.

The exported NPZ was reopened with `allow_pickle=False`; its `config_json`
retained `spatialPropagationPasses=2` and parameters were finite. Its SHA-256 is
in the report. The 4.1 MiB numerical export remains a local test artifact rather
than another binary archive in Git.

Additional checks at this revision:

- Full native/browser-enabled suite: **352 tests passed, no skips**.
- Four targeted propagation tests also passed on Chromium SwiftShader. These
  include an injected rejected point actually refined from a neighbour seed,
  preservation of all unqueued states, ROI permissions, bitmap input and warmed
  resource reuse. Software execution does not qualify hardware performance.
- A fresh outside-checkout wheel installation passed those four tests using
  its packaged core/WGSL resources. The pure-Python wheel contained all 49
  package files; no first-party compiled binary was added.
- Ruff, ESLint, changed-module syntax checks, strict Sphinx and repository links
  passed locally. No new remote CI result is claimed.

Reproduce from the repository root with the project environment:

```bash
python3 tools/build_pages.py --output /tmp/live-dic-spatial-ui
python3 tools/qualify_browser_spatial.py \
  --site /tmp/live-dic-spatial-ui --output /tmp/live-dic-spatial-ui-qualified
ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests
ICGN_RUN_BROWSER_TESTS=1 ICGN_BROWSER_SOFTWARE=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p test_browser_spatial.py
```

Use a new qualification output directory on each run. The setting is
experimental, disabled by default, Analysis-only and restricted to local
translation/B-spline/uniform full windows, CPU reference preparation and one
slot. It shares numerical kernels with the native experiment; its prior is
the initial seed at each level. Native public constructors remain unchanged.
