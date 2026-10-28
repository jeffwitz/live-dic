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

Published reference result
--------------------------

The authoritative machine-specific results, including the external SYCL
implementation, are maintained on the :doc:`../reference/performance` page.

For the ``t=0 -> 10`` analytical translation on Intel Lunar Lake, the
published 30-run campaign snapshots report:

.. list-table:: Current validated measurements
   :header-rows: 1

   * - Metric
     - GPU-IO
     - Three.js/TSL
     - shared WGSL
     - native wgpu-py
   * - median GPU prefilter + solver
     - 3.301 ms
     - 0.675 ms
     - 0.817 ms
     - not timestamped
   * - p95 GPU prefilter + solver
     - 4.210 ms
     - 1.227 ms
     - 1.298 ms
     - not timestamped
   * - median browser benchmark wall time
     - 10.3 ms
     - 2.7 ms
     - 2.6 ms
     - not applicable
   * - median complete Python call
     - 61.877 ms
     - 56.203 ms
     - 55.916 ms
     - **1.812 ms**
   * - p95 complete Python call
     - 65.655 ms
     - 59.117 ms
     - 59.555 ms
     - **2.187 ms**
   * - accepted subsets
     - 900 / 1,024
     - 900 / 1,024
     - 900 / 1,024
     - 900 / 1,024
   * - mean iterations over accepted subsets
     - 5.957
     - 5.959
     - 5.954
     - 5.953
   * - vector RMSE
     - 3.873e-5 pixel
     - 2.749e-5 pixel
     - 2.700e-5 pixel
     - 2.696e-5 pixel

The 512 x 512 GPU work is below a 16.7 ms budget for all browser backends. The
Three.js/TSL boundary is substantially lower on this machine. The Three.js
runtime measured a 32-lane subgroup and
selected its fused 16-lane solver for these campaigns. The browser GPU and
browser-wall benchmarks exclude Python frame encoding, DevTools transport,
decoding, and result transfer. They therefore do not prove a sustained 60 Hz
Python application.

The native complete-call values include direct input upload and packed result
readback but exclude acquisition. At UHD, three 30-run campaigns measured a
median campaign of 14.621 ms median and 15.748 ms p95. Both are inside a 60 Hz
period, but acquisition and downstream consumers are still excluded; this is
a promising integration result, not a sustained system claim.

A later audit measured 14.605 ms median and 15.018 ms p95 for the same native
one-scale boundary. For the full-pyramid path, only the recommended two-level
configuration retained a repeated p95 margin below 16.667 ms; three levels
remained above 60 FPS by median but not by p95. See
:doc:`../reference/performance` for the audit table.

For a saturated sequence, the bounded two-slot native path measured a 13.380
ms median campaign period, equivalent to 74.7 FPS, with 26.617 ms median result
latency. This distinction is important: pipelining increases throughput by
keeping the GPU queue occupied, but it deliberately allows one additional
frame to remain in flight. It does not make an individual result arrive in
13.380 ms.

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
