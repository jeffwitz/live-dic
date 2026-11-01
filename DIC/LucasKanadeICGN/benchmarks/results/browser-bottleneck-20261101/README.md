# Locating the native-engine to browser-camera 4K throughput gap

The owner asked to identify where the roughly 60 FPS engine becomes a roughly
7 FPS application, before choosing another optimization. The measurements
localize the main difference to **rebuilding the reference on every pair**,
followed by external-image transfer and synchronization/recording. Display
suspension alone does not restore the native engine's throughput.

The historical native image-to-field tables exclude reference preparation,
capture, decoding, display and saving. The consecutive application does more
work. The controlled native run here uses the original UHD tensile reference
as both input images, affine 16/16, B-spline, two fixed levels, median, 8/16
iterations and 0.005 tolerances. The native temporal profile also matches the
application's prediction/fine-correction/divergence settings. It measures
15.183 ms median and 60.456 calls/s over thirty synchronous calls. Its p95 is
22.270 ms: this short repeat does not renew a 16.67 ms deadline claim. Preparation
is about 1.76 s once, excluded from the steady-state rate. Historical 11 ms
values belong to their archived inputs/conditions, not this new repeat.

The resident-browser controls use the same grayscale bytes, no camera, display
or database. After five warmups, each configuration measures 15 ordinary calls,
10 diagnostic calls, then 15 ordinary calls. Only the **30 ordinary calls**
contribute the reported wall medians. A prepared reference/full-field call is
21.3 ms. Rebuilding that reference each pair raises it to 103.65 ms. The exact
PWA-style summary followed by a separate full-field export is 111.6 ms;
preloaded external ImageBitmap inputs raise it to 141.15 ms. These are wall
times, not GPU timestamps. ImageBitmap and actual camera VideoFrame costs must
not be equated; actual camera-source stages are measured separately below.

## Reference preparation is the main additional graph

The unchanged preparation graph alone takes 86.8 ms for grayscale bytes or
101.35 ms for a preloaded RGBA ImageBitmap, including input copy/completion.
Its shader prepares Gaussian-filtered spline derivatives, reference intensities,
subset means and normalization products, Hessians and their inverses at both
levels. These operations are absent from each steady-state fixed-reference
call; the LK solver's inverse Hessians are already available in that case.

`reference-steps.json.gz` replays the **same original passes in order**, with
one completion fence after each pass to locate expensive stages. This is an
intrusive wall-time diagnosis, not a replacement graph or a GPU timing claim.
The largest individual pass is `reference level 0 prepare_subsets`, about
28–31 ms including its fence. It computes all subset statistics and Hessian
inverses, not just matrix inversion. Fine-level Gaussian passes, spline passes
and gradients each cost roughly 5–11 ms including fences. Coarse preparation
adds more. An empty submission/fence itself costs about 2.6–2.75 ms, so small
pass values cannot be interpreted as pure kernel costs or subtracted blindly.
Do not sum medians from these isolated passes to reproduce the unsplit graph.

## Actual application, zero live rendering

`camera-stages.json.gz` uses the actual PWA, original native-UHD grayscale image
delivered as a static fake 60 Hz camera, a dedicated GPU Worker, unchanged
numerical settings, and a two-slot CPU recording queue. All four cases have
**zero live draws and zero live bitmap transfers**. Reference views and final
pause flushes occur outside the observation window. Both camera inputs remain
3840 x 2160; saved fields remain 240 x 135. No physical camera was opened.

An extra completion fence after input upload, and another after GPU reference
preparation, separate the stages. These fences perturb throughput. The script
labels its output as intrusive diagnosis, never the unmodified application's
FPS benchmark. Four seconds warmup precede twelve seconds observation per case.
Cases run sequentially; timing variation is retained rather than attributed
to the options. The scene is an identity-motion control, not general motion
qualification.

The resulting stage breakdown is in `summary.json` and the central performance
reference. The decisive observations are:

- Fixed reference: target preprocessing/solving/statistics remain around
  15–18 ms; camera input upload/completion adds about 9–10 ms.
- Consecutive pairs: both input uploads/completion take about 17 ms, reference
  preparation adds 81–93 ms, and target preprocessing/solving/statistics take
  about 15–19 ms. The LK stage has not become ten times slower.
- Recorded-field export is about 4–5 ms; Worker/main handoff is about 1 ms.
  Storage admission has its own variable 19–52 ms duration, including an awaited
  IndexedDB transaction. This is elapsed asynchronous time, not main-thread CPU
  occupancy, and it overlaps subsequent computation.
- Between accepted inputs there remains about 9–14 ms median idle/admission
  gap in this diagnostic. Camera cadence, main-thread acknowledgement and
  admission are outside the resident engine call.

Period, field-ready latency and disk-admission age differ. In recorded
consecutive mode the field is CPU-ready about 136 ms after Worker arrival,
whereas its disk/library admission is about 195 ms. The next calculation runs
during much of that write. Adding storage time to all calculation stages would
double-count overlap and predict the wrong throughput.

The unmodified low-cadence controls additionally preserve two complete
consecutive-mode runs with **no live rendering**, at 6.889 and 6.945 committed
fields/s. Their numerical solver/preparation calls remain about 124 ms. The
other controls vary with system load; the complete twenty-run cadence matrix
was stopped at the owner's request. It is not presented as complete.

## Evidence, instrumentation and limits

- Base source/site: clean `1baf913`. Original image hashes and configuration are
  in the raw reports. Same Intel Lunar Lake integrated hardware, Linux/Vulkan,
  Mesa 25.2.8, Chromium 155 and performance power profile as the preceding
  overlap campaign. Desktop background load was not eliminated.
- `core-cause.json.gz` contains native and browser call samples, plus an
  attempted GPU timestamp capture. **All 1,080 timestamp durations were zero**
  despite nonzero wall times. They were rejected and are never used for cost
  attribution. The ordinary wall samples and finite numerical outputs are
  independent of those invalid timestamp values.
- `reference-steps.json.gz` contains twelve repeats per input type and a finite
  post-profile field with 31,654 accepted subsets and identity displacement
  RMSE below 0.005 px. All ordinary core samples also satisfy that gate.
- `camera-stages.json.gz` retains Worker stage events, main-thread field
  ownership/readback metadata, actual IndexedDB transaction timings, queue
  admissions and input ordinals. Both recorded last fields are finite and pass
  the same existing identity gate; no unsaved fields or storage errors occur.
- `manifest.json` gives SHA-256 hashes of uncompressed raw payloads. Reports are
  losslessly gzip-compressed. The complete original diagnostic scripts and the
  two site patches are included; they modify temporary sites, not shipped
  shaders or public engine defaults. New resource allocations/fences belong to
  explicit diagnostic runs only.
- `full-tests.log.gz` records 279 passing native/browser-enabled local tests,
  no skips, before timing. Numerical kernels were not edited for this diagnosis.
  No new remote CI, Pages deployment or all-device claim is made.

Reproduce in an isolated checkout at the base revision, with the project's
ordinary Python development dependencies, from the repository root:

```bash
diagnostics=DIC/LucasKanadeICGN/benchmarks/results/browser-bottleneck-20261101
python tools/build_pages.py --output /tmp/live-dic-cadence-site
cp "$diagnostics/profile_dic_gpu.js" /tmp/profile_dic_gpu.js
python "$diagnostics/profile_dic_core.py"
python "$diagnostics/profile_reference_steps.py"
python "$diagnostics/profile_camera_stages_setup.py"
PYTHONPATH=tools python /tmp/profile_camera_stages.py \
  --site /tmp/live-dic-camera-stage-site --output /tmp/live-dic-camera-stages.json \
  --resolution 3840x2160 --duration 12 --warmup 4 --gpu-reference \
  --overlap-pipeline --suspend-preview --preview-width 1920
```

The scripts write new reports under `/tmp`; do not overwrite these archived
records or regenerate original fixtures. The first diagnostic script also
creates `/tmp/live-dic-cause-site`, which the second reuses.

The next optimization decision should target the new per-pair preparation graph
first, then external-image upload and avoidable readback/synchronization. A
fixed reference is a different scientific comparison mode, not a transparent
way to claim faster consecutive DIC. No such policy change is made here.
