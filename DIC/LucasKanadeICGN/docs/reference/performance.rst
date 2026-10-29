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
a ``5e-3`` fine-scale tolerance, minimum ZNCC ``0.8``, and spatial median
consistency. No camera or H.264 stage is included.

All current affine and quadratic measurements include component-wise
normalized median replacement. Its isolated analytic A/B qualification shows
a 29.6--36.3% reduction in curvature RMSE, including the noisy case, with a
12.9--20.5% gradient-RMSE trade-off. See
:doc:`../explanation/component-wise-median` for the exact protocol, equations,
and legacy/current commit comparison. Quadratic performance remains reported
separately because its state and solver cost differ from affine mode.

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
     - **12.069 / 13.163 ms**
     - **82.80 FPS**
   * - Native full pyramid, 2 levels, translation
     - **10.526 / 11.170 ms**
     - **94.03 FPS**
   * - Raw browser WGSL, 2 levels, affine
     - **45.369 / 52.165 ms**
     - **21.98 FPS**
   * - Raw browser WGSL, 2 levels, translation
     - **40.213 / 45.024 ms**
     - **24.21 FPS**
   * - Three.js/TSL, 1 scale, affine
     - **46.763 / 62.198 ms**
     - **20.21 FPS**
   * - Three.js/TSL, 1 scale, translation
     - **39.966 / 45.706 ms**
     - **24.54 FPS**

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
     - **11.955 / 13.620 ms**
     - 12.168 ms
     - 82.18 FPS
   * - Affine, depth 2
     - 22.616 / 25.208 ms
     - **11.433 ms**
     - **87.50 FPS**
   * - Affine, depth 3
     - 35.322 / 42.598 ms
     - 11.524 ms
     - 86.78 FPS
   * - Translation, depth 1
     - **10.331 / 11.164 ms**
     - 10.373 ms
     - 96.42 FPS
   * - Translation, depth 2
     - 18.888 / 19.827 ms
     - **9.474 ms**
     - **105.58 FPS**
   * - Translation, depth 3
     - 29.705 / 35.718 ms
     - 9.420 ms
     - 106.16 FPS

Depth one remains the low-age choice. Depth two is the recommended
throughput/latency compromise: it gains 6.5% affine throughput and 9.5%
translation throughput over depth one, while approximately doubling result
age. A third slot adds 0.6 FPS in translation mode, does not improve affine
throughput in the current rerun, and increases median age beyond 35 ms.

Reproduce one row with:

.. code-block:: bash

   python benchmarks/benchmark_image_to_field.py \
     --backend native-pyramid --width 3840 --height 2160 \
     --campaigns 4 --frames 50 --warmups 8 --pipeline-depth 2 \
     --motion-model affine

Diagnostic repeated-target and kernel results
---------------------------------------------

The following measurements remain useful for kernel optimization, but the
browser rows are GPU timestamps with a target uploaded before the timed loop.
They must not be compared directly with the complete native call above.

.. list-table:: October 29 current kernels
   :header-rows: 1
   :widths: 36 25 39

   * - Boundary and translation
     - Current median / p95
     - Current numerical result
   * - Native affine full pyramid, 2 levels, ``(4, -2)``
     - **10.684 / 13.238 ms**
     - 31,724 accepted; 0.001873 px RMSE
   * - Native affine full pyramid, 3 levels, ``(4, -2)``
     - **10.891 / 13.161 ms**
     - 31,734 accepted; 0.001937 px RMSE
   * - Raw browser WGSL affine, 2 levels, ``(4, -2)``
     - **8.681 / 11.698 ms**
     - 31,724 accepted; 0.001804 px RMSE
   * - Three.js TSL affine subgroup, ``(4, -2)``
     - **26.657 / 28.441 ms**
     - 22,401 accepted; 0.096354 px RMSE
   * - Three.js TSL affine subgroup, ``(2, -1)``
     - **12.568 / 14.552 ms**
     - 31,654 accepted; 0.0000305 px RMSE

The recommended native two-level result corresponds to 93.6 calls/s from the
median and 75.5 calls/s from p95. The complete-call boundary includes target
upload, GPU pyramid construction, all coarse-to-fine solves, packed readback,
and NumPy result construction. Camera acquisition and downstream consumers are
excluded.

The raw-browser two-level GPU boundary corresponds to 115.2 frames/s from the
median. Its synchronized browser wall time was 10.8 ms median and 14.2 ms p95.
The Three.js nominal browser wall time was 14.3 / 16.0 ms; the difficult
single-scale pair measured 28.7 / 30.7 ms.

Current translation-only A/B comparison
---------------------------------------

The same protocol was repeated with ``motion_model="translation"``. The
translation solver uses an analytic 2 by 2 inverse and omits all four affine
Jacobian components and update terms. Result arrays retain the six-parameter
layout, with the deformation components set to zero.

.. list-table:: Affine versus translation-only current kernels
   :header-rows: 1

   * - Execution boundary
     - Affine median / p95
     - Translation median / p95
     - Median change
   * - Native one scale, ``(4, -2)``, complete call
     - 34.683 / 35.766 ms
     - **14.948 / 17.696 ms**
     - **-56.9%**
   * - Native full pyramid, 2 levels, ``(4, -2)``, complete call
     - 10.684 / 13.238 ms
     - **10.121 / 12.352 ms**
     - **-5.3%**
   * - Raw browser WGSL, 2 levels, ``(4, -2)``, GPU timestamp
     - **8.681 / 11.698 ms**
     - 8.777 / 11.099 ms
     - +1.1%
   * - Three.js subgroup, one scale, ``(2, -1)``, GPU timestamp
     - 12.568 / 14.552 ms
     - **9.087 / 10.560 ms**
     - **-27.7%**

The gain is largest when IC-GN iterations dominate. Both two-level paths
average about one iteration per level on this pair; image reduction, B-spline
prefiltering, upload, readback, and result construction therefore dominate.
The raw-WGSL median difference is within run-to-run variation and does not show
a speed-up in this campaign, although its p95 decreased by 5.1%.

Translation mode accepted 30,185 subsets at 0.0000642 px RMSE in the native
one-scale difficult case, 31,781 at 0.002110 px in the native pyramid, 31,783
at 0.002141 px in raw WGSL, and the same 31,654 subsets at 0.0000654 px in
Three.js. These results validate displacement only; the model cannot measure
strain. See :doc:`../explanation/motion-models`.

Current fractional-interpolation comparison
-------------------------------------------

The interpolation campaign uses a continuous analytic texture sampled before
and after the fractional translation ``(0.5, -0.375)``. It therefore does not
use ``np.roll`` or preselect one reconstruction method while generating the
target. Native figures use eight warm-ups and 50 complete ``register()``
calls. Raw-browser figures use three warm-ups and seven GPU timestamp samples;
Three.js figures use five warm-ups and ten timestamp samples. Binary host
transport and readback are excluded from browser rows.

.. list-table:: UHD interpolation modes on the current kernels
   :header-rows: 1

   * - Execution boundary
     - ``bspline`` median / p95
     - ``linear`` median / p95
     - ``cubic`` median / p95
   * - Native full pyramid, 2 levels, complete call
     - **11.962 / 13.724 ms**
     - **7.844 / 9.149 ms**
     - **9.162 / 10.830 ms**
   * - Raw browser WGSL, 2 levels, GPU timestamp
     - 20.935 / 22.427 ms
     - **10.586 / 13.528 ms**
     - 15.633 / 17.616 ms
   * - Three.js/TSL, 1 scale, GPU timestamp
     - 45.115 / 51.933 ms
     - **29.357 / 32.276 ms**
     - 37.381 / 42.067 ms

The matching native RMSE values were 0.00866 px for ``bspline``, 0.01934 px
for ``linear``, and 0.01024 px for ``cubic``. All three native p95 values are
below the 16.667 ms frame period on this workload. The tuned schedules reduce
the average fine solve from 6.43 to 2.30 iterations for B-spline, 11.35 to 4.17
for linear, and 7.56 to 2.60 for direct cubic. This is not a portable 4K60
guarantee. See
:doc:`../explanation/interpolation` for the implementation and selection
guidance.

Reproduce the native row with:

.. code-block:: bash

   python benchmarks/benchmark_interpolation.py \
     --width 3840 --height 2160 --levels 2 \
     --translation 0.5 -0.375 --warmups 8 --repeats 50

Current native sequence throughput
----------------------------------

The bounded single-scale sequence API was measured in four campaigns, each
using five warm-ups and 30 UHD targets with an exact ``(2, -1)`` translation.

.. list-table:: Current bounded sequence measurements
   :header-rows: 1

   * - Execution
     - Median period
     - Throughput
     - Median result latency
   * - synchronous ``register()``
     - 11.539 ms
     - 86.7 FPS
     - 11.398 ms
   * - bounded pipeline, depth 1
     - 11.805 ms
     - 84.7 FPS
     - not retained
   * - **bounded pipeline, depth 2**
     - **11.026 ms**
     - **90.7 FPS**
     - **21.963 ms**
   * - bounded pipeline, depth 3
     - 10.780 ms
     - 92.8 FPS
     - 32.093 ms

All configurations accepted 31,654 subsets and reported ``3.066e-5`` px RMSE.
Two slots remain the recommended throughput setting: a third slot adds only
2.1 FPS while adding another processed frame of result latency. Pipeline depth
is unrelated to image-pyramid depth.

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
  affine seed transfer, seeded solves, final median pass, and one packed
  readback.

The opt-in GPU tests validate partial FIR tiles, exact integer pyramid
reduction, fresh seeded state, selective-retry neighbour preservation, and the
translation-only fractional-displacement contract. They also inject isolated
affine-gradient and quadratic-curvature outliers to verify component-wise
median replacement. All 99 tests pass when GPU execution is enabled.

Timing boundaries
-----------------

.. list-table:: Reported boundaries
   :header-rows: 1

   * - Boundary
     - Includes
     - Excludes
   * - Browser GPU timestamp
     - FIR, reductions when configured, IC-GN, seed transfer, median pass
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
