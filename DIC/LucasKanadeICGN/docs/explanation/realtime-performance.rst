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
:doc:`../reference/performance`. The primary October 29 audit uses a changing
UHD sequence, three warm-ups, and 12 measured resident-image to field calls.

On a changing four-image sequence, the recommended two-level native full
pyramid measured 11.920 ms median and 13.072 ms p95 for the affine
resident-image to materialized-field call. Translation mode measured 10.526 ms
median and 11.170 ms p95. Across the four fixtures, the affine path accepted
31,252 to 31,654 subsets and the translation path 31,312 to 31,654. Their
maximum truth RMSE values were 0.0627 px and 0.0406 px respectively. Both
native p95 values are below the 16.667 ms 60 Hz period, but acquisition and
downstream work remain excluded.

At the same complete Python boundary, the raw-browser two-level WGSL path
measured 585.594 / 612.275 ms in affine mode and 470.765 / 494.262 ms in
translation mode. Three.js measured 596.772 / 613.626 ms and 482.453 / 496.790
ms respectively. The browser kernels themselves take tens rather than hundreds
of milliseconds; PNG encoding, DevTools transport, browser decoding, JSON
readback, and conversion dominate. GPU timestamps remain useful diagnostics,
but they are not stream performance claims.

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
