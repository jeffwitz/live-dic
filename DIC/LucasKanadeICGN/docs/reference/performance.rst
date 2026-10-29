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

* ``ICGNWGPUFullPyramid()`` with two levels is the recommended native
  coarse-to-fine configuration.
* ``ICGNWebGL(backend="webgpu-wgsl")`` executes the same two-level
  full-pyramid WGSL graph in Chromium.
* ``ICGNWebGL()`` remains the browser entry-point default and selects the
  optimized Three.js/TSL single-scale solver with automatic WebGL2 fallback.
* ``ICGNWGPU()`` is the native single-scale and bounded-sequence engine. Its
  optional selective recovery is not the normal pyramid path.
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

The offline validation texture is tiled to 3840 x 2160. The resulting grid is
240 x 135, or 32,400 subsets of 16 x 16 pixels. The solver uses six affine
parameters, Gaussian/inverse-spline target filtering, 4 x 4 cubic B-spline
interpolation, a ``1e-3`` fine-scale tolerance, minimum ZNCC ``0.8``, and
spatial median consistency. No camera or H.264 stage is included.

Every row uses eight warm-ups and 50 measured registrations. Native rows are
complete Python ``register()`` wall times. Browser rows are GPU timestamp
durations and exclude PNG/DevTools transport and result readback.

Current UHD results
-------------------

.. list-table:: October 29 current kernels
   :header-rows: 1
   :widths: 36 25 39

   * - Boundary and translation
     - Current median / p95
     - Current numerical result
   * - Native full pyramid, 2 levels, ``(4, -2)``
     - **10.194 / 11.073 ms**
     - 31,724 accepted; 0.001874 px RMSE
   * - Native full pyramid, 3 levels, ``(4, -2)``
     - **10.280 / 11.151 ms**
     - 31,734 accepted; 0.001939 px RMSE
   * - Raw browser WGSL, 2 levels, ``(4, -2)``
     - **8.789 / 9.375 ms**
     - 31,724 accepted; 0.001804 px RMSE
   * - Three.js TSL subgroup, ``(4, -2)``
     - **24.300 / 26.627 ms**
     - 22,401 accepted; 0.096354 px RMSE
   * - Three.js TSL subgroup, ``(2, -1)``
     - **11.893 / 12.798 ms**
     - 31,654 accepted; 0.0000305 px RMSE

The recommended native two-level result corresponds to 98.1 calls/s from the
median and 90.3 calls/s from p95. The complete-call boundary includes target
upload, GPU pyramid construction, all coarse-to-fine solves, packed readback,
and NumPy result construction. Camera acquisition and downstream consumers are
excluded.

The raw-browser two-level GPU boundary corresponds to 113.8 frames/s from the
median. Its synchronized browser wall time was 10.4 ms median and 11.5 ms p95.
The Three.js nominal browser wall time was 13.5 / 14.7 ms; the difficult
single-scale pair measured 26.2 / 28.7 ms.

Current fractional-interpolation comparison
-------------------------------------------

The interpolation campaign uses a continuous analytic texture sampled before
and after the fractional translation ``(0.5, -0.375)``. It therefore does not
use ``np.roll`` or preselect one reconstruction method while generating the
target. Native figures use three warm-ups and ten complete ``register()``
calls. Browser figures use three warm-ups and seven GPU timestamp samples;
PNG/DevTools transport and readback are excluded from those rows.

.. list-table:: UHD interpolation modes on the current kernels
   :header-rows: 1

   * - Execution boundary
     - ``bspline`` median / p95
     - ``linear`` median / p95
     - ``cubic`` median / p95
   * - Native full pyramid, 2 levels, complete call
     - 19.433 / 21.199 ms
     - **15.118 / 16.224 ms**
     - 19.677 / 21.717 ms
   * - Raw browser WGSL, 2 levels, GPU timestamp
     - 33.126 / 36.084 ms
     - **23.371 / 26.789 ms**
     - 34.996 / 36.136 ms
   * - Three.js/TSL, 1 scale, GPU timestamp
     - 38.843 / 48.645 ms
     - **29.310 / 32.192 ms**
     - 42.357 / 42.785 ms

The matching native RMSE values were 0.00925 px for ``bspline``, 0.01990 px
for ``linear``, and 0.01089 px for ``cubic``. The linear native p95 is below
the 16.667 ms frame period on this workload, but the margin is only 0.44 ms;
this is not a portable 4K60 guarantee. Direct cubic preserved most B-spline
accuracy but did not improve native time here because it still performs 16
target reads and required more iterations. See
:doc:`../explanation/interpolation` for the implementation and selection
guidance.

Reproduce the native row with:

.. code-block:: bash

   python benchmarks/benchmark_interpolation.py \
     --width 3840 --height 2160 --levels 2 \
     --translation 0.5 -0.375 --warmups 3 --repeats 10

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
reduction, fresh seeded state, and selective-retry neighbour preservation. All
67 tests pass when GPU execution is enabled.

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
