# UHD field-figure export qualification — 31 October 2026

These are actual standalone-workbench PNG downloads from the original checked-in
UHD plate-with-hole pair. The interface keeps its computed fields and geometry;
this change adds self-contained presentation legends and captions only. Pierre
Margerit's numerical foundation remains unchanged.

`hardware.json` and `software.json` retain the complete workbench qualification,
build identifiers and source fingerprints, archive hashes, independent NumPy
operator checks and PNG observations. The commands below completed with exit code
zero. Each build reports the preceding commit plus dirty source fingerprint;
the candidate code is archived by the commit containing this directory.

```bash
python3 tools/build_pages.py --output /tmp/live-dic-figure-qualified-site
python3 tools/qualify_browser_workbench.py --site /tmp/live-dic-figure-qualified-site --output /tmp/live-dic-figure-qualified-hardware
python3 tools/qualify_browser_workbench.py --site /tmp/live-dic-figure-qualified-site --output /tmp/live-dic-figure-qualified-software --software
```

Both runs execute the real original UHD correlation, reversible processing,
measurements, native NPZ/project interchange, analysis during resize, history and
replay. Each additionally exports strain, displacement and categorical status:

- Decode the actual PNG and compare every full-resolution field/annotation
  pixel with an independent pre-export canvas composition (SHA-256 equality).
- Check all footer text bounds, reference geometry and foundational credit.
- Check the sequential palette endpoints, strain units, all six exact status
  colours and the same six colours in the on-screen status legend.
- Preserve fields, statuses and iteration arrays; analytical operators still
  agree with the independent NumPy calculation to the existing tolerances.

The hardware PNGs are retained for visual inspection; both reports contain the
hashes of every PNG. Pixel equality here validates export, not the mathematical
accuracy of a measured displacement. These correctness runs are not FPS
campaigns. They qualify Chromium on the local Intel hardware and SwiftShader;
no physical camera, phone, Safari or Firefox qualification is inferred. PNG is a
presentation artifact; retain NPZ/projects for complete data and provenance.
