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
measured 40.873 / 44.782 ms in affine mode and 40.213 / 45.024 ms in
translation mode. Three.js measured 43.297 / 59.107 ms and 39.966 / 45.706 ms
respectively. These figures use the persistent binary target and packed-result
transport: no per-frame PNG, image decoding, DevTools frame payload, or JSON
field remains. The browser process boundary and the required GPU upload still
keep these paths above the 16.667 ms target. GPU timestamps remain useful
diagnostics, but they are not stream performance claims.

Throughput is not latency
-------------------------

The current bounded two-level full-pyramid campaign used four runs of 50
changing UHD frames after eight warm-ups per run. The table reports the median
campaign:

.. list-table:: Sequence period and result age
   :header-rows: 1

   * - Model and targets in flight
     - Stream period
     - Throughput
     - Median result age
   * - Affine, depth 1
     - 11.561 ms
     - 86.53 FPS
     - **11.478 ms**
   * - Affine, depth 2
     - **10.735 ms**
     - **93.16 FPS**
     - 21.408 ms
   * - Translation, depth 1
     - 10.373 ms
     - 96.42 FPS
     - **10.331 ms**
   * - Translation, depth 2
     - **9.474 ms**
     - **105.58 FPS**
     - 18.888 ms

Pipelining improves throughput by keeping the queue occupied, but it allows
additional frames to remain in flight. A two-slot affine period of 10.735 ms
does not mean that an individual result is 10.735 ms old. A third slot reached
94.97 FPS affine and 106.16 FPS translation, but median result age was 31.894
and 29.705 ms. Pipeline depth is unrelated to image-pyramid depth.

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
