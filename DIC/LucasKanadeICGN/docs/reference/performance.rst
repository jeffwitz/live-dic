Performance on the validation machine
=====================================

This page contains only measurements for the current kernels. Earlier campaign
snapshots were removed because their filter layout, subgroup occupancy, or
execution graph no longer describes the implementation. Re-run the documented
commands whenever a kernel, driver, browser, or timing boundary changes.

The results are machine-specific regression references, not portable
performance guarantees.

Current backend policy
----------------------

* DIC-RT is evaluated first as a Python-supervised NumPy pipeline. See
  :doc:`../explanation/supervision`.
* ``ICGNWGPUFullPyramid()`` with two levels is the recommended product
  configuration for that boundary.
* ``ICGNWebGL(backend="webgpu-wgsl")`` executes the same two-level
  full-pyramid WGSL graph in Chromium.
* ``ICGNWebGL()`` is the Python-controlled browser compatibility entry point
  and selects the optimized Three.js/TSL single-scale solver with automatic
  WebGL2 fallback. It is not the low-latency NumPy recommendation.
* Both native engines expose bounded sequence execution. ``ICGNWGPU()`` is
  the single-scale engine; its optional selective recovery is not the normal
  pyramid path.
* GPU-IO is deprecated and has no current performance claim.

See :doc:`../explanation/backend-selection` for the complete decision table.

Validation machine
------------------

.. list-table:: Hardware and software environment
   :header-rows: 1

   * - Component
     - Value
   * - Measurement date
     - 2026-10-29
   * - CPU
     - Intel Core Ultra 5 228V, 8 cores
   * - GPU
     - Intel Lunar Lake integrated graphics ``8086:64a0``
   * - Kernel driver
     - ``xe``
   * - Operating system
     - Ubuntu 24.04.5 LTS
   * - Kernel
     - Linux 7.0.0-38-generic
   * - Browser
     - Chromium 155.0.8059.79 snap
   * - Browser WebGPU
     - Three.js r186, Intel Xe2-LPG Vulkan adapter
   * - Native WebGPU
     - wgpu-py 0.32.0, wgpu-native Vulkan, same physical GPU

Current workload and protocol
-----------------------------

The offline validation images are tiled to 3840 x 2160. The resulting grid is
240 x 135, or 32,400 subsets with ``subset_size=16`` and the default
``subset_stride=16``. No published timing on this page uses overlapping
subsets. The affine and translation rows in the primary UHD campaign use
Gaussian/inverse-spline target filtering, 4 x 4 cubic B-spline interpolation,
a ``5e-3`` fine-scale tolerance, minimum ZNCC ``0.8``, and the default
displacement UOD pass. No camera or H.264 stage is included.

Primary backend performance measurements use
``outlier_policy="replace"``. It tests only displacement and replaces the
complete local state after a trigger. ``flag`` and ``disabled`` are scientific
control modes. ``median`` is an explicit smoothing mode and is not represented
by the default performance rows. See
:doc:`../explanation/displacement-outliers` for the exact rule and the
false-positive, injected-outlier, and localized-feature regressions.
Quadratic performance remains reported separately because its state and solver
cost differ from affine mode.

Primary image-to-field stream measurements
------------------------------------------

The primary deployable boundary starts with an already acquired, contiguous
``uint8`` NumPy grayscale image and stops only after the complete NumPy
parameter field, status, iterations, and statistics are available to Python.
Reference preparation, camera acquisition, and decoding are excluded.

The persistent engines process a changing sequence which cycles the four
distinct qualified ``frame_001``, ``frame_005``, ``frame_010``, and
``frame_020`` fixtures. Three warm-ups precede 12 measured frames. Calls are
synchronous, so latency, result age, and stream period describe the same
one-frame-at-a-time execution. Browser rows report the median of three
campaigns after the binary transport change.

.. list-table:: UHD resident-image to materialized-field performance
   :header-rows: 1

   * - Backend and model
     - Median / p95 latency
     - Sustained throughput
   * - Native full pyramid, 2 levels, affine
     - **11.571 / 12.185 ms**
     - **85.37 FPS**
   * - Native full pyramid, 2 levels, translation
     - **10.992 / 11.397 ms**
     - **90.72 FPS**
   * - Raw browser WGSL, 2 levels, affine
     - **41.094 / 44.423 ms**
     - **24.27 FPS**
   * - Raw browser WGSL, 2 levels, translation
     - **39.996 / 43.995 ms**
     - **24.82 FPS**
   * - Three.js/TSL, 1 scale, affine
     - **43.203 / 56.336 ms**
     - **21.91 FPS**
   * - Three.js/TSL, 1 scale, translation
     - **39.666 / 44.002 ms**
     - **25.14 FPS**

.. list-table:: Native displacement-UOD policy control
   :header-rows: 1

   * - Model and policy
     - Median / p95 latency
     - Throughput
     - Median / maximum RMSE
   * - Affine, ``replace``
     - 11.571 / 12.185 ms
     - 85.37 FPS
     - 0.04161 / 0.06274 px
   * - Affine, ``disabled``
     - **10.851 / 11.436 ms**
     - **92.24 FPS**
     - 0.04161 / 0.07810 px
   * - Translation, ``replace``
     - 10.992 / 11.397 ms
     - 90.72 FPS
     - **0.03590 / 0.04063 px**
   * - Translation, ``disabled``
     - **10.095 / 10.539 ms**
     - **98.63 FPS**
     - 0.04487 / 0.06548 px

Omitting the post-pass saved about 0.72 ms affine and 0.90 ms translation in
these campaign medians. ``disabled`` is not numerically equivalent to the
default: no wrong-basin replacement occurs, and the maximum error increased on
this sequence. Current production tables therefore use ``replace``. The raw
mode is intended for solver studies, explicit downstream outlier handling, and
isolated timing.

The native and raw-browser WGSL rows execute the same two-level algorithm.
Their host paths remain different. Native wgpu uploads NumPy bytes in-process
and reads one packed state buffer. The browser API now transfers each
contiguous grayscale frame unchanged over a persistent same-origin binary long
poll and returns one packed binary state field. It no longer performs
per-frame PNG encoding, data-URL transport, image decoding, or JSON field
serialization. Three.js also keeps the upload packed at four 8-bit pixels per
word and expands it on the GPU.

Quadratic-model qualification measurement
-----------------------------------------

The quadratic model has a distinct cost and is not included in the affine and
translation table above. On the same validation machine, a dedicated
three-campaign run used a two-level native full pyramid, 32 x 32 subsets,
direct Keys-cubic interpolation, one target in flight, three warm-ups, and 12
changing UHD frames per campaign. The resident NumPy image to materialized
12-component NumPy field boundary measured:

* campaign median latencies of 61.940, 56.017, and 71.522 ms;
* campaign p95 latencies of 85.279, 78.054, and 107.940 ms;
* sustained throughput of 16.98, 17.31, and 14.64 FPS;
* 7,580 to 7,788 accepted subsets on the 120 x 67 grid;
* approximately 3.17 s of excluded one-time reference preparation.

This is a correctness-oriented first implementation, not a 4K60 quadratic
result. Its exact second-order composition currently inverts a lifted 6 by 6
increment operator for every active subset and iteration. The measurement
therefore establishes the current engineering baseline and keeps it separate
from the much faster affine and translation models.

Reproduce it from the repository root with::

   python3 DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
     --backend native-pyramid --motion-model quadratic \
     --subset-size 32 --interpolation cubic --pyramid-levels 2 \
     --pipeline-depth 1 --warmups 3 --frames 12 --campaigns 3

Across the four changing fixtures, affine native and raw WGSL accepted 31,258
to 31,654 subsets with 0.06466 px maximum truth RMSE. Translation native
accepted 31,312 to 31,654 with 0.04063 px maximum RMSE; raw WGSL differed by at
most one accepted subset and less than ``2e-6`` px in maximum RMSE. Three.js is
single-scale, so its difficult-frame acceptance is not algorithmically
comparable to the two-level paths.

The binary path removes the former hundreds of milliseconds of serialization,
but an UHD frame still crosses the Python/Chromium process boundary, is copied
into a GPU resource, and the field crosses back. In the current affine runs,
the browser body copy took about 9.8 ms median and the synchronized GPU graph
about 17--20 ms. Complete calls sustain about 20--22 FPS, so the browser
path is still not a 60 Hz NumPy stream. The current native timing contract
separates input normalization, direct texture
upload, command encoding, queue submission, blocking completion wait,
mapped-buffer copy, and NumPy result construction. For a synchronous
translation call in a four-run sequence campaign, representative medians were
0.019, 0.694, 0.449, 0.096, 8.703, 0.145, and 0.339 ms,
respectively. The completion wait is host blocking time, not an isolated GPU
timestamp.

Reproduce this campaign with:

.. code-block:: bash

   python benchmarks/benchmark_image_to_field.py \
     --width 3840 --height 2160 --frames 12 --warmups 3 --campaigns 3 \
     --backend three-tsl --backend webgpu-wgsl \
     --motion-model translation

Native full-pyramid bounded stream
----------------------------------

``ICGNWGPUFullPyramid.register_stream`` allocates independent target textures,
pyramid reductions, solver states, transferred seeds, and final readbacks for
each in-flight slot. Immutable reference data and pipelines remain shared.
The input is uploaded directly from the contiguous NumPy buffer; no PNG,
browser, DevTools, or JSON conversion is involved.

The following campaign used four runs, each with eight warm-ups and 50 changing
UHD frames. The table reports the median campaign. Result age starts when the
iterator accepts a NumPy frame and ends after its complete NumPy field and
statistics have been constructed. Stream period is total wall-clock duration
divided by 50; it is the correct sustained-throughput metric even when
completed results arrive in short bursts.

.. list-table:: Native two-level full-pyramid stream trade-off
   :header-rows: 1

   * - Model and targets in flight
     - Median / p95 result age
     - Stream period
     - Throughput
   * - Affine, depth 1
     - **11.354 / 11.899 ms**
     - 11.368 ms
     - 88.00 FPS
   * - Affine, depth 2
     - 21.008 / 21.683 ms
     - **10.471 ms**
     - **95.51 FPS**
   * - Affine, depth 3
     - 31.761 / 35.991 ms
     - 10.523 ms
     - 95.03 FPS
   * - Translation, depth 1
     - **10.496 / 11.229 ms**
     - 10.708 ms
     - 93.43 FPS
   * - Translation, depth 2
     - 19.131 / 19.665 ms
     - **9.607 ms**
     - **104.09 FPS**
   * - Translation, depth 3
     - 29.854 / 36.126 ms
     - 9.655 ms
     - 103.57 FPS

Depth one remains the low-age choice. Depth two is the recommended
throughput/latency compromise: it gains 8.5% affine throughput and 11.4%
translation throughput over depth one, while approximately doubling result
age. A third slot does not improve throughput and increases median age above
29 ms.

Reproduce one row with:

.. code-block:: bash

   python benchmarks/benchmark_image_to_field.py \
     --backend native-pyramid --width 3840 --height 2160 \
     --campaigns 4 --frames 50 --warmups 8 --pipeline-depth 2 \
     --motion-model affine

Kernel configuration represented by the results
------------------------------------------------

* shared WGSL FIR: 32 x 8 horizontal and 8 x 32 vertical output tiles in
  workgroup memory, with the central 25 coefficients of the compatible
  31-value record;
* Three.js FIR: a 16 x 16 output tile and all 31 coefficients;
* Three.js solver: the measured 32-lane hardware subgroup, eight subset pixels
  per lane on the validation GPU;
* portable shared-WGSL solver: one 16-lane workgroup per subset;
* full pyramid: GPU-resident five-tap reduction, per-level FIR coefficients,
  affine seed transfer, seeded solves, final UOD pass, and one packed
  readback.

The opt-in GPU tests validate partial FIR tiles, exact integer pyramid
reduction, fresh seeded state, selective-retry neighbour preservation, and the
translation-only fractional-displacement contract. They also verify complete-
state replacement and flag-only preservation for an injected displacement
outlier, at most 1% false positives on a healthy noisy field, preservation of
a one-subset-wide affine-gradient band, and exact agreement of the explicit
3 by 3 median with an independent translation/affine/quadratic oracle. The
portable suite and the opt-in GPU suite are both executed in CI; the latter
must not convert shader, adapter, or execution failures into skipped tests.

Timing boundaries
-----------------

.. list-table:: Reported boundaries
   :header-rows: 1

   * - Boundary
     - Includes
     - Excludes
   * - Browser GPU timestamp
     - FIR, reductions when configured, IC-GN, seed transfer, optional UOD pass
     - upload, Python transport, readback
   * - Browser synchronized wall time
     - the same GPU graph plus browser synchronization
     - Python transport and result conversion
   * - Native complete Python call
     - validation, upload, GPU graph, packed readback, NumPy result statistics
     - acquisition and downstream consumers
   * - Sequence period
     - total bounded-stream wall time divided by returned frames
     - result age when more than one frame is in flight
   * - Acquisition-to-result
     - the deployed measurement pipeline
     - nothing relevant to the application deadline

Reproduce the current campaign
------------------------------

Native full pyramid:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_full_pyramid.py \
       --width 3840 --height 2160 --levels 2 3 --warmups 8 --repeats 50

Raw browser WGSL:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend webgpu-wgsl --pyramid-levels 2 \
       --translation 4 -2 --warmups 8 --repeats 50

Three.js nominal and difficult pairs:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend three-tsl --translation 2 -1 --warmups 8 --repeats 50

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend three-tsl --translation 4 -2 --warmups 8 --repeats 50

Native sequence pipeline:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_pipeline.py \
       --repeats 30 --warmups 5 --campaigns 4 --depths 1 2 3

Run the hardware regressions before publishing a new campaign:

.. code-block:: console

   $ ICGN_RUN_GPU_TESTS=1 PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 -m unittest discover \
       -s DIC/LucasKanadeICGN/python/tests -p 'test_*.py' -v

External SYSCL comparison
-------------------------

The related SYSCL project remains part of the algorithmic provenance, but
there is currently no qualified apples-to-apples timing row. Add one only
after both projects are run on the same machine, input, iteration schedule,
and explicitly equivalent timing boundary.

Interpretation
--------------

These measurements demonstrate offline 4K60 processing margin on one GPU and
one controlled dataset. They do not include camera acquisition, frame-arrival
jitter, a codec, downstream processing, or operating-system deadline
guarantees. Median and p95, accepted subsets, RMSE, queue depth, and result
latency must be reported together when qualifying a deployment.

Full-window ROI qualification
-----------------------------

The native ``full`` ROI implementation has a separate
`qualification report with raw stream campaigns
<https://github.com/jeffwitz/live-dic/tree/main/DIC/LucasKanadeICGN/benchmarks/results/roi-full-20261030>`_.
It measures translation and affine execution, the original UHD tensile images,
all-true/absent masks, compact/fragmented regions and the optional subgroup path.
The observed hardware drift prevents a conclusive real-time non-regression
decision within the predeclared margins. These ROI campaigns do not replace
the established unmasked measurements above. See :doc:`../how-to/use-roi`.

Partial-support ROI qualification
---------------------------------

The explicitly enabled ``partial`` policy has its own
`UHD report and raw campaigns
<https://github.com/jeffwitz/live-dic/tree/main/DIC/LucasKanadeICGN/benchmarks/results/roi-partial-20261030>`_.
It separates complete and partial-window accuracy, preparation cost and actual
stream latency/throughput, including optional visibility truncation. Translation
uses pure shifts of the original UHD reference; affine uses its archived tensile
frame 40. Hardware drift keeps temporal non-regression reception open. These
measurements do not establish a universal zero-overhead partial mode.
See :doc:`../how-to/use-partial-roi` for support definitions and limits.

The `native-UHD fractional-motion and crack addendum
<https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/benchmarks/results/roi-partial-20261030/uhd-fractional.md>`_
qualifies 8/8 and 16/16 on a known fractional translation using both the original
UHD speckle and a continuous analytic UHD texture. It distinguishes raw and
median fields, single-level cubic and two-level B-spline, and includes the
full-UHD visibility case. Small-image oracle tests do not substitute for these
qualification runs.

Temporal initialization: cumulative native-UHD motion
------------------------------------------------------------

The latest matched scheduling/correction campaign is summarized below.
The earlier measurements following it explain the original slowdown report;
their absolute timings should not be used to compute the new code's speedup.

Latest matched native-UHD control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three alternating process runs per variant replay the identical 200-frame
trajectory. Twenty frames establish history and warm the actual slots; frames
20--199 are timed. The machine is Intel LNL, Vulkan/Mesa 25.2.8, wgpu-py
0.32.0, performance profile. All cases use affine B-spline, size/stride 16/16,
uniform weights, median policy, two levels, temporal initialization, no ROI
and unchanged iteration ceilings (8,16). The boundary is resident NumPy image
to materialized NumPy field and statistics, without truth scoring.

.. list-table:: Medians of three campaign statistics; p95 range retains all runs
   :header-rows: 1

   * - Variant
     - Slots
     - FPS
     - Age median
     - Age p95 range
   * - Previous code, 93ceb37
     - 1
     - 71.00
     - 14.06 ms
     - 16.14--16.48 ms
   * - Batched dispatches and fused reduction
     - 1
     - 75.05
     - 13.48 ms
     - 15.26--17.95 ms
   * - Plus optional fine correction
     - 1
     - 76.17
     - 13.17 ms
     - 14.98--17.84 ms
   * - Previous code, 93ceb37
     - 2
     - 77.29
     - 25.69 ms
     - 29.14--29.50 ms
   * - Batched dispatches and fused reduction
     - 2
     - 80.94
     - 24.55 ms
     - 27.84--37.17 ms
   * - Plus optional fine correction
     - 2
     - 82.42
     - 24.04 ms
     - 27.10--37.81 ms

Every new-variant run exceeds 60 FPS, but the p95 latency target is not
consistently met. Pooling all 540 ages gives p95 16.38 ms before changes,
17.32 ms with batching/fusion, and 17.01 ms with optional correction at one
slot. Thus the campaign supports a higher median throughput, not a
tail-latency improvement or a hard real-time guarantee. Paired throughput
changes vary in sign, particularly at two slots. The old code also runs
faster than in the preceding campaign: a general regression or a gain caused
only by changing the power profile has not been established.

Batching/fusion preserve all per-frame quality statistics and final arrays
bit for bit in this control. Fine correction reduces accepted fine iterations
from 2.649 to 2.290, with final fixed-support coverage 99.996%, displacement
RMSE 0.02858 px, p95 error 0.05305 px and strain-component RMSE 0.001446.
This population contains 26,924 fixed interior positions; all-grid accepted
error remains about 2.58 px near the deformed image boundaries. Those errors,
rejections and nonconverged windows are retained and still cost runtime.

The `complete report, commands, source hashes and raw samples
<https://github.com/jeffwitz/live-dic/tree/main/DIC/LucasKanadeICGN/benchmarks/results/realtime-optimization-20261030>`_
include negative prototypes and the individual p95/p99 values. The optional
correction is disabled by default and does not change the fixed reference,
ROI population, iteration ceilings or the two solves per image.

Earlier capture-range qualification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The opt-in native full pyramid projects accepted fine history onto the next
coarsest grid, then solves every scale. A 200-frame 3840 x 2160 tensile
sequence reaches stretch 1.1 about (1920,1080). On a fixed 26,924-position
interior support, the final temporal/two-level result retains 99.967% coverage,
0.02838 px displacement RMSE and 0.001431 strain-component RMSE. Zero/two
levels retains only 1.645%, with 63.37 px error among accepted estimates.
Median smoothing is explicit; these gradient errors do not qualify raw fields.
A black frame produces zero updates and recovers on the next normal frame.

The following numbers are medians of three campaign statistics, on Intel
Graphics LNL, Vulkan/Mesa 25.2.8, wgpu-py 0.32.0. Boundary: resident uint8 NumPy
image to materialized NumPy field and statistics. Rendering, acquisition and
truth scoring are excluded. B-spline, uniform weights, size/stride 16/16,
explicit median policy and unchanged iteration ceilings are used throughout.

.. list-table:: Temporal initialization, one streaming slot
   :header-rows: 1
   :widths: 30 10 15 15 15 15

   * - Motion
     - Levels
     - FPS
     - Age median
     - Age p95
     - Age p99
   * - Tensile affine
     - 1
     - 66.99
     - 15.13 ms
     - 19.70 ms
     - 22.12 ms
   * - Tensile affine
     - 2
     - 59.92
     - 16.62 ms
     - 20.15 ms
     - 21.65 ms
   * - Pure translation
     - 1
     - 86.67
     - 11.37 ms
     - 14.14 ms
     - 14.60 ms
   * - Pure translation
     - 2
     - 86.76
     - 11.37 ms
     - 13.77 ms
     - 14.13 ms

Affine does not meet age p95 <16.67 ms. Translation/one level/one slot does
in all three campaigns (11.01--14.81 ms); translation/two levels/one slot
does not consistently do so (12.48--17.12 ms). No two-slot configuration
meets this age criterion. These cumulative-motion campaigns do not replace
the earlier small-motion results or establish performance non-regression.
No defaults were changed.

A subsequent same-workload regression control on the original native UHD
frame 40 measures 12.55--13.10 ms before temporal changes, 12.42--12.44 ms
in current zero mode, and 12.53--12.74 ms with temporal initialization
(two repetitions, same median policy and two levels, no ROI). These controls
show no general slowdown. The longer sequence also exhibits substantial
timing drift, including on an unchanged image; sampled controls observe GPU
power limiting. A single cause has not been isolated. See the
`controlled diagnostic and raw evidence
<https://github.com/jeffwitz/live-dic/blob/main/DIC/LucasKanadeICGN/benchmarks/results/temporal-init-20261030/performance-diagnostic.md>`_.
Repeated-image controls do not replace ordered-sequence quality or latency
qualification, and the 16.62 ms sequence median is not a universal engine speed.

Reapplying the system performance profile and repeating all three native-UHD
campaigns measures 16.90 ms median / 20.36 ms p95 and 59.23 FPS for temporal
affine/two levels/one slot. Translation/two levels/one slot measures 12.10 ms
median / 15.05 ms p95 and 80.83 FPS. Accepted counts and final numerical errors
match the preceding campaign. The system already reported performance before
the command, so this is not a confirmed balanced-versus-performance comparison
and demonstrates no affine speedup. The
`profile log, full table and raw repeat
<https://github.com/jeffwitz/live-dic/tree/main/DIC/LucasKanadeICGN/benchmarks/results/temporal-init-20261030/performance-retest>`_
retain the unchanged configuration and all timing samples.

:download:`Raw report <../../benchmarks/results/temporal-init-20261030/report.json>`
and :download:`per-frame metrics <../../benchmarks/results/temporal-init-20261030/metrics.csv>`
include both fixed and own populations, rejected counts, all timing samples,
preparation times, hardware and input hashes. The campaign README in the same
directory provides the command, the complete two-slot table, image-formation
error convention and limitations. See :doc:`../how-to/use-temporal-initialization`.
