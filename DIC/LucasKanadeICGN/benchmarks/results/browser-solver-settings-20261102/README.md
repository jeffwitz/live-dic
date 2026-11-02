# Apply edited solver settings before calculation

Code revision: `6c249ae` (clean site assembly). The native engine, shared core
and canonical WGSL are unchanged. This campaign qualifies cold UI orchestration,
storage ownership and replay, not GPU performance or all operating systems.

## Reproduction

```bash
python3 tools/build_pages.py --output /tmp/live-dic-settings-site
python3 tools/qualify_solver_settings.py --site /tmp/live-dic-settings-site \
  --output /tmp/settings-hardware
python3 tools/qualify_solver_settings.py --site /tmp/live-dic-settings-site \
  --software --output /tmp/settings-software
```

The tool uses the unchanged original native 3840 × 2160 tensile reference,
creates a temporary zero-filled `(64,-32)` integer translation, and writes
both as transient TIFFs. Neither images nor their scientific provenance are
regenerated in the repository. Natural ordering places Def before Ref; this
also exercises the input-order issue without inferring filename semantics.

## Checked workflow

1. Save a first failed field with the existing temporal default. On this sparse
   pair only 209/32,400 grid positions are accepted; common-interior RMSE is
   about 71 px. This intentionally failing control remains recorded.
2. Enable Fourier and 32/16 affine settings, then press Start without recapture.
   The retained reference is re-prepared once before solving; all 23,108 common
   interior points are accepted and meet the existing <0.05 px RMSE gate.
3. Replay the old failed field. Its arrays/hashes, configuration and epoch stay
   unchanged and its Results summary says temporal. Replaying calculates nothing.
4. Change only preview cadence and solve the same target again. Parameters are
   bit-identical and the preparation epoch is unchanged; no rebuild occurs.
5. Apply a solver policy change explicitly: preparation occurs, no target field
   is added. Selecting the last image as reference suggests the other image as
   target; deliberate identity seeking remains possible.
6. Apply a rectangle/hole ROI, edit solver settings, then Start. The retained mask
   hash and selected count remain exact. Coverage/error gates evaluate only the
   selected common interior, excluding geometric omissions.
7. Invalid GN settings stop before preparing/solving and add no field.

`hardware.json` and `software.json` retain source/input hashes, actual options,
per-field array hashes, epochs, pair metadata, accepted counts and known-truth
quality populations. Both runs have no application errors. Integer-shift error
here is not subpixel metrology qualification; the complete enabled local suite
also includes the existing native/browser subpixel and UHD regressions.

## Validation and scope

- 301 complete local native/browser tests pass without skips (`tests.log`).
- The workflow passes on Intel WebGPU hardware and Chromium SwiftShader.
- Ruff, formatting, changed JavaScript syntax, strict Sphinx and repository
  link checks pass locally. The software-browser CI step is added; no new remote
  CI outcome is claimed.
- Optional `--reference` and `--target` TIFF paths were used locally to diagnose
  an experimental pair at 4112 × 3008. Those images, their hashes and derived
  private measurements are deliberately not archived here.
- Default options and unchanged calculation paths are preserved. No timings or
  FPS/nonregression claims are inferred from these functional controls. Applying
  changed settings is cold preparation before Start; it adds no per-frame work.
