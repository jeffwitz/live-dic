Real-time performance and latency
=================================

"Real time" is a property of the complete deployed pipeline, not a GPU kernel
in isolation. A DIC system must define its deadline, sustain it over time, and
report late or dropped frames.

Latency boundaries
------------------

This implementation exposes several useful but different boundaries:

.. list-table:: Timing boundaries
   :header-rows: 1

   * - Boundary
     - Includes
     - Excludes
   * - GPU timer query
     - target FIR, IC-GN, median pass
     - upload, Python transport, readback
   * - browser benchmark wall time
     - FIR, solver, synchronization
     - Python transport, decode, result readback
   * - browser ``register`` total
     - upload, FIR, solver, readback after decode
     - Python conversion, transport, image decode
   * - complete Python call
     - all work inside ``engine.register``
     - acquisition before the call and downstream consumers
   * - acquisition-to-result
     - the complete measurement pipeline
     - nothing relevant to the application deadline

Validated reference result
--------------------------

For the ``t=0 -> 10`` analytical translation on Intel Lunar Lake, the current
30-run campaigns report:

.. list-table:: Current validated measurements
   :header-rows: 1

   * - Metric
     - GPU-IO
     - Three.js/TSL
   * - median GPU prefilter + solver
     - 3.74-3.90 ms
     - 3.40-3.56 ms
   * - p95 GPU prefilter + solver
     - 4.13-4.33 ms
     - 3.71-4.56 ms
   * - median browser benchmark wall time
     - 16.5-17.0 ms
     - 5.1-5.8 ms
   * - accepted subsets
     - 900 / 1,024
     - 900 / 1,024
   * - mean iterations over accepted subsets
     - 5.957
     - 5.952
   * - vector RMSE
     - 3.819e-5 pixel
     - 4.076e-5 pixel

The GPU work is well below a 16.7 ms budget. GPU-IO's browser benchmark wall
time is near the nominal 60 Hz period, while the Three.js/TSL boundary is below
it on this machine. Both benchmarks exclude Python frame encoding, DevTools
transport, decoding, and result transfer. Neither therefore proves a sustained
60 Hz Python application.

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
rate. Also report queue depth, missed deadlines, and the age of the frame when
the result becomes available.

Throughput is not latency
-------------------------

Pipeline overlap can improve throughput without reducing the age of an
individual result. Conversely, always waiting for readback produces clear
latency measurements but can limit overlap. State which behavior the
measurement uses.

Qualification checklist
-----------------------

Before calling an application real time:

#. timestamp frames at acquisition;
#. reuse one engine and one prepared reference;
#. run representative texture, noise, and displacement cases;
#. record complete-call and acquisition-to-result distributions;
#. report accepted subsets and iterations with timing;
#. bound queues and count dropped or late frames;
#. run long enough to observe thermal and scheduling effects;
#. validate the target GPU, driver, Chromium version, and power mode.
