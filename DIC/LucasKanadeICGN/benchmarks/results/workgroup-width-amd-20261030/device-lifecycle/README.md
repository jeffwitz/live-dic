# AMD device-loss investigation and lifecycle correction

## Cause

The failure was reproducible in the inherited temporal test
`CorrectedTemporalTests.test_zero_default_and_first_temporal_frame_match_bitwise`.
This test creates 27 engines (three initialization modes, three pyramid
depths, three motion models) and retains their bound `close` methods through
`unittest.addCleanup`, even after explicitly closing each group of engines.

The old native `close()` destroyed dimension-dependent buffers and textures,
but left device, pipeline, bind-group and cache references in closed engines.
On this Radeon 780M / RADV Phoenix host, the trace recorded roughly 270 MB
of additional GTT usage per newly prepared engine. The driver's reported GTT
budget was exactly **4,294,967,296 bytes (4 GiB)**. In the failing trace,
GTT usage reached **4,292,915,200 bytes** before an affine/three-level
registration. RADV rejected that command submission with `-28`, and wgpu
aborted with `Parent device is lost`, exit code 134.

Available host RAM remained about 25 GiB. The constrained resource was the
driver's GTT budget, rather than the image's NumPy allocation or a failed
numerical assertion. Kernel journal access showed non-zero GPU VM accounting
at process teardown; it did not report a shader fault or a GPU reset at the
submission failure. The cause is supported by controlled resource-lifetime
replays, rather than assigning blame to the driver based on its error string.

## Controlled experiments

- Unmodified code: the isolated test reproduced the abort, with GTT usage
  accumulating across closed engines (`before-trace.log`).
- Adding only `device.destroy()` at close, while retaining pipelines and
  other wrappers: still aborted (`destroy-only-trace.log`).
- Dropping all references in closed engine objects and collecting Python
  cycles: the same 27-engine test completed, with final GTT usage around
  47 MB (`drop-all-references-trace.log`). This was a diagnostic intervention,
  not the production implementation.
- The production correction releases GPU wrapper/cache references, destroys
  only an owning device, and collects cyclic wrappers after those references
  have gone out of scope. It preserves configuration/diagnostic attributes
  and keeps shared-level close from destroying the parent's device.

`GPUDevice` and its queue can participate in Python reference cycles. GPU
objects can retain their device as well. Resource-release loops run in a
separate helper frame so their last local loop value cannot keep a pipeline
alive during the owner's collection. No private wgpu release API is used.

Close is now explicitly permanent and idempotent; `set_reference` on a closed
engine raises `ICGNError`. Existing test/benchmark calls that manually
destroyed the device after `close()` were updated to use `close()` alone.
Solver shaders, numerical defaults and registration scheduling are unchanged.
The cleanup/collection work happens at close, outside steady-state timing.

## Reproduction

From the repository root, using the installed project environment:

```bash
ICGN_RUN_GPU_TESTS=1 DIC/LucasKanadeICGN/python/.venv/bin/python \
  -m unittest discover -s DIC/LucasKanadeICGN/python/tests \
  -p test_fused_pyramid.py \
  -k test_zero_default_and_first_temporal_frame_match_bitwise -v
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/results/workgroup-width-amd-20261030/device-lifecycle/trace_lifecycle.py
```

The trace adds prints around construction, registration and close; it leaves
the original numerical assertions intact. On this host it reads the
`mem_info_*_used` counters under `/sys/class/drm/card1/device`; these are
system-wide counters, so desktop activity also contributes. Other hosts may
use a different DRM card number or expose no counters.

The pre-fix runtime source is the campaign commit
`c0c072c24c7bf52a93aaee58b4e754a954ef7d36`, whose source hashes are retained
in the original reports. Those reports and measurements are unchanged.
The correction is a local working-tree change; this is not a remote CI claim.

New regression tests verify that retained closed native/full-pyramid engines
no longer retain their devices, repeated close is harmless, closed-reference
reuse raises a clear error, and closing a shared level leaves the parent
producing bit-identical fields.

## Validation of the correction

- Formerly aborting isolated temporal test: passed (2.632 seconds).
- Full GPU-enabled suite: **193 tests passed**, 83.766 seconds, on this AMD
  host (`gpu-suite-after.log`). This includes the new lifetime/ownership tests
  and the native-UHD numerical tests. No tests were disabled for this run.
- Portable suite: 193 discovered, 121 passed, 72 GPU tests explicitly skipped
  (`portable-after.log`).
- Ruff over `src`, `tests` and `../benchmarks`: passed.
- Formatting checks for the changed runtime, regression and trace files:
  passed.
- Sphinx HTML with warnings treated as errors: passed (`sphinx-after.log`).
- UHD replay at each workgroup width: the same 200 images were processed,
  with 180 timed frames at depth 1. All 180 per-frame quality records match
  the original report exactly, and final parameters/status/iterations are
  **bit-identical** for both widths (`parity-summary.json`). These two replays
  check numerical preservation; they are not a new three-repeat performance
  qualification.
- Sixty successive two-level full-pyramid engines were constructed, prepared,
  used and closed while all sixty Python objects remained retained. GTT
  usage was **189,267,968 bytes at every ten-engine checkpoint**, with no
  growing memory use (`retained-engine-stress.log`). Every identity solve
  accepted subsets on the deterministic random 128 x 128 reference.

The corrected mixed-model/depth 27-engine trace also completes
(`after-trace.log`). Its final GTT counter remains around 2.65 GB; the fix
does not promise immediate return of all native allocator memory to a zero
counter. The repeated two-level lifetime stress above checks absence of
continued accumulation in the normal repeated-engine scenario. GTT counters
are system-wide, and driver/native allocation lifetimes also contribute.

These are observed local checks. They establish resolution of this reproduced
failure on this GPU/driver configuration, rather than a universal driver or
hardware guarantee. The original failure logs remain in the parent archive's
`validation/` directory.
