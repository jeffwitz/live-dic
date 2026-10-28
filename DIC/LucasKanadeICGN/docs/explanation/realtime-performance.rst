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
   * - native execute plus readback
     - WGSL FIR, IC-GN, median pass, synchronization, packed readback
     - input upload and Python result statistics
   * - acquisition-to-result
     - the complete measurement pipeline
     - nothing relevant to the application deadline

Validated reference result
--------------------------

The authoritative machine-specific results, including the external SYCL
implementation, are maintained on the :doc:`../reference/performance` page.

For the ``t=0 -> 10`` analytical translation on Intel Lunar Lake, the current
30-run campaigns report:

.. list-table:: Current validated measurements
   :header-rows: 1

   * - Metric
     - GPU-IO
     - Three.js/TSL
     - native wgpu-py
   * - median GPU prefilter + solver
     - 3.301 ms
     - 0.675 ms
     - not timestamped
   * - p95 GPU prefilter + solver
     - 4.210 ms
     - 1.227 ms
     - not timestamped
   * - median browser benchmark wall time
     - 10.3 ms
     - 2.7 ms
     - not applicable
   * - median complete Python call
     - 61.877 ms
     - 56.203 ms
     - **1.882 ms**
   * - p95 complete Python call
     - 65.655 ms
     - 59.117 ms
     - **2.382 ms**
   * - accepted subsets
     - 900 / 1,024
     - 900 / 1,024
     - 900 / 1,024
   * - mean iterations over accepted subsets
     - 5.957
     - 5.959
     - 5.952
   * - vector RMSE
     - 3.873e-5 pixel
     - 2.749e-5 pixel
     - 2.693e-5 pixel

The 512 x 512 GPU work is below a 16.7 ms budget for both backends. The
Three.js/TSL boundary is substantially lower on this machine. The Three.js
runtime measured a 32-lane subgroup and
selected its fused 16-lane solver for these campaigns. Both benchmarks exclude
Python frame encoding, DevTools transport, decoding, and result transfer.
Neither therefore proves a sustained 60 Hz Python application.

The native complete-call values include direct input upload and packed result
readback but exclude acquisition. At UHD, one 30-run campaign measured 16.160
ms median and 16.569 ms p95. That is inside a 60 Hz period with very little
margin; it is a promising integration result, not a sustained system claim.

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
