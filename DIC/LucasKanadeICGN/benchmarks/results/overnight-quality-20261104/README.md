# Overnight consolidation — 2026-11-04

The retained changes improve propagation without changing numerical defaults:

- Exact empty-work termination skips later searches only after a pass queues
  no new solves. The first search of every new target/level remains active.
- Analysis exposes opt-in 32-invocation solves. The native extended paired case
  improves from 56.02 to 61.62 fields/s; p95 remains 18.01 ms. The browser's
  decoded-ImageBitmap/full-field control improves from 26.05 to 30.09 fields/s.
  These are different input/return boundaries, not interchangeable camera FPS.
- Live vectors have optional contrast outlines and sequential palettes sharing
  the colourbar tables, with original colours and geometry preserved by default.
  GPU pixel oracles cover hardware/software rendering, ROI exclusions, closure,
  saturated colours and no lazy palette allocation. Desktop/narrow layouts are
  inspected; emulation does not qualify a real phone.
- Documentation explains propagation versus median smoothing, GPU terminology,
  local-model mismatch despite a correct global seed, and Python/browser scope.
  The Python sequence example exposes existing workgroup-width options.
- Native research preparation now rejects replacing bound reference resources.
  Its existing synchronous-only restriction remains explicit.

Final local validation: **362 tests, zero skipped**, native and browser flags
both enabled. ESLint, Ruff check/format and strict Sphinx pass. The full package,
benchmarks, tools and documentation are included in Python style checks.
Separate native CPU and browser software runs are retained, alongside an
installed pure-Python wheel outside the checkout. All 49 package files match
final source. The installed example processes original UHD inputs with two
slots and exports NumPy fields without pickle; the wheel ROI case also passes.

Three original UHD default-output fingerprints remain exactly identical before
and after the optional browser width: affine traction, translation traction,
and plate-with-hole ROI. Width changes intentionally permit tiny floating
reduction differences under fixed agreement and analytical gates. Status and
quality evidence, iteration differences, raw times and negative controls are
in the linked campaigns. No gate or iteration budget was relaxed.

See [the hour-long memory/timing soak](../propagation-long-soak-20261104/),
[native width controls](../propagation-workgroup-widths-20261104/),
[extended width stability](../propagation-width-stability-20261104/), and
[browser input/width controls](../browser-propagation-workgroup-20261104/).
Other dated propagation pilots preserve rejected sampler, subgroup, frontier,
fusion and score-cache experiments; unproven caches were not enabled in the
browser. Reference-versus-reference variability is retained before candidates.

`validation.json` records boundaries and actual coverage. Logs are compressed
without rewriting. Renderer reports and the final UHD UI report are retained;
large derived fields and wheel binaries remain outside the repository. The
screenshots show the new display choices; they are not metrology or speed proof.
The final code still requires separate physical-camera/storage qualification
and validation on other devices. All work is committed locally; no push,
remote CI or GitHub Pages update is claimed.
