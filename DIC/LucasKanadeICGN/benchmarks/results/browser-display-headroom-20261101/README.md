# Display headroom controls, not a new FPS qualification

The owner asked whether separating video/upload and display into threads could
recover the remaining throughput. The application already has a GPU Worker
(upload, reference preparation, solve, field rendering) and main-thread bitmap
presentation. Durable recorded-field admission is awaited before acknowledging
the frame. Extra Workers alone do not remove shared GPU work or that dependency.

These **short diagnostic controls** use the original native-UHD reference as a
static fake 60 Hz camera, GPU reference preparation, consecutive pairs, no
recording, 3 s warmup and 12 s observation. Power profile was `performance`;
the selected browser adapter is Intel `xe-2lpg`. No physical camera was opened.
The order is baseline, size guard, then no presentation, once each. This is not
a repeated/alternated speedup qualification. It also skips the earlier full
campaign's two fixed-reference cases, so its warmup history/protocol differs.
Do not use its lower absolute FPS to replace or directly compare the previous
campaign's results.

- Baseline: 4.739 completed fields/s, 184.3 ms median combined reference/solve call.
- Size guard: 4.884 fields/s, 182.2 ms combined call.
- No presentation: 5.079 fields/s, 178.6 ms combined call.

The last control removes **both** GPU field drawing and bitmap creation/display,
without modifying images, subsets, solver kernels, iterations or reference
preparation. It is an unusable-UI headroom control, not an optimized product.
Most pair time remains. The small observed differences do not establish a
repeatable gain for a size guard or another display Worker. They cannot justify
promising 60 FPS by moving presentation alone.

The retained product correction only avoids assigning identical canvas width
and height on every bitmap: those assignments reset the presentation surface.
The canvas and numerical image remain at their original dimensions. No new
thread, early storage acknowledgement, unbounded queue or scientific option was
introduced. A future compute/storage overlap must bound pending fields, keep
unsaved fields on failures, and drain on pause/source change/export. A separate
display clock can present the latest field, but must not silently drop recorded
measurements or change consecutive input-pair semantics.

## Evidence and reproduction

The base application is `aef6c0a`. Diagnostic tool additions were uncommitted
and are retained as `diagnostic-tool.patch`; defaults remain the full four-case
campaign. `variants.json` and each report's build record identify the temporary
site edits and hashes. `manifest.json` includes uncompressed SHA256 hashes for
the losslessly gzip-compressed JSON reports. No original fixtures were changed.

Use an isolated checkout of `aef6c0a`, apply `diagnostic-tool.patch` to that
checkout, then assemble `/tmp/control-normal` with `tools/build_pages.py`.
Copy that site into fresh `/tmp/control-guard` and `/tmp/control-none`, applying
`resize-guard.patch` and `no-presentation.patch` respectively with `patch -p1`
inside each site. Do not patch the numerical checkout or use these sites as the
normal application. For each site, run:

```bash
python tools/diagnose_camera_pipeline.py --site /tmp/control-normal \
  --output /tmp/control-normal.json --gpu-reference --resolution 3840x2160 \
  --comparison consecutive --recording off --duration 12 --warmup 3
```

The product size guard was checked with the real UHD workbench: eight saved
fields, NumPy operator comparison, figure/NPZ interchange and zero reported
errors. This is correctness evidence, not a performance result. The complete native-GPU/browser-enabled local suite passes 270 tests without
skips (`full-tests.log`). Ruff, Node syntax and strict Sphinx checks pass.
No new remote CI or deployment is claimed.
