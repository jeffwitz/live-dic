Real-time performance and latency
=================================

"Real time" is a property of the complete deployed pipeline, not a GPU kernel
in isolation. A DIC system must define its deadline, sustain it over time, and
report late or dropped frames.

Latency boundaries
------------------

.. list-table:: Timing boundaries
   :header-rows: 1

   * - Boundary
     - Includes
     - Excludes
   * - browser GPU timestamp
     - target FIR, configured pyramid passes, IC-GN, median pass
     - upload, Python transport, readback
   * - browser synchronized wall time
     - the same GPU graph plus browser synchronization
     - Python transport, decode, result conversion
   * - native complete Python call
     - validation, upload, GPU graph, packed readback, NumPy statistics
     - acquisition and downstream consumers
   * - bounded sequence period
     - total sequence wall time divided by returned frames
     - result age when multiple frames are in flight
   * - acquisition-to-result
     - the complete deployed measurement pipeline
     - nothing relevant to the application deadline

Current validated result
------------------------

The authoritative machine-specific measurements are maintained on
:doc:`../reference/performance`. The current October 29 audit uses UHD frames,
eight warm-ups, and 50 measured registrations.

The recommended two-level native full pyramid measured 10.684 ms median and
13.238 ms p95 for the complete Python call on the ``(4, -2)`` pair.
It accepted 31,724 of 32,400 subsets with 0.001873 px truth RMSE. Three levels
measured 10.891 / 13.161 ms. Both p95 values are below the 16.667 ms 60 Hz
period on the validation GPU, but acquisition and downstream work remain
excluded.

The raw-browser two-level WGSL graph measured 8.681 / 11.698 ms on the GPU and
10.7 / 12.6 ms at the synchronized browser boundary. The nominal ``(2, -1)``
Three.js path measured 12.568 / 14.552 ms GPU and 14.3 / 16.0 ms browser wall.
These browser numbers do not include the complete Python transport boundary.

Throughput is not latency
-------------------------

The current bounded native single-scale campaign measured:

.. list-table:: Sequence period and result age
   :header-rows: 1

   * - Execution
     - Median period
     - Throughput
     - Median result latency
   * - synchronous ``register()``
     - 11.539 ms
     - 86.7 FPS
     - 11.398 ms
   * - **two frames in flight**
     - **11.026 ms**
     - **90.7 FPS**
     - **21.963 ms**
   * - three frames in flight
     - 10.780 ms
     - 92.8 FPS
     - 32.093 ms

Pipelining improves throughput by keeping the queue occupied, but it allows
additional frames to remain in flight. A two-slot period of 11.026 ms does not
mean that an individual result is 11.026 ms old. Pipeline depth is also
unrelated to image-pyramid depth.

Frame-rate budgets
------------------

.. list-table:: Common periods
   :header-rows: 1

   * - Nominal rate
     - Period
   * - 30 Hz
     - 33.3 ms
   * - 60 Hz
     - 16.7 ms
   * - 100 Hz
     - 10.0 ms

Use p95 or a stricter percentile rather than the median to qualify a sustained
rate. Also report queue depth, missed deadlines, accepted subsets, numerical
error, and the age of each frame when its result becomes available.

Qualification checklist
-----------------------

Before calling an application real time:

#. timestamp frames at acquisition;
#. reuse one engine and one prepared reference;
#. run representative texture, noise, displacement, and deformation cases;
#. record complete-call and acquisition-to-result distributions;
#. report accepted subsets and iterations with timing;
#. bound queues and count dropped or late frames;
#. run long enough to observe thermal and scheduling effects;
#. validate the target GPU, driver, browser version, and power mode.
