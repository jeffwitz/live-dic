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
