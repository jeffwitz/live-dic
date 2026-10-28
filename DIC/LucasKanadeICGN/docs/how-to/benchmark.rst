Benchmark the DIC pipeline
==========================

Use the built-in benchmark to characterize the GPU prefilter and solver under a
stable workload:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWebGL


   reference = "DIC/LucasKanadeICGN/assets/frame_000.png"
   target = "DIC/LucasKanadeICGN/assets/frame_010.png"

   with ICGNWebGL() as engine:
       benchmark = engine.benchmark(reference, target, repeats=30, warmups=3)

   print(f"GPU median: {benchmark.gpu_median_ms:.3f} ms")
   print(f"GPU p95: {benchmark.gpu_p95_ms:.3f} ms")
   print(f"Browser wall median: {benchmark.wall_median_ms:.3f} ms")
   print(f"Browser wall p95: {benchmark.wall_p95_ms:.3f} ms")

The command-line equivalent is:

.. code-block:: console

   $ icgn-webgl reference.png target.png \
       --benchmark --repeats 30 --warmups 3

Interpret the timers
--------------------

``gpu_times_ms``
   WebGL timer-query or WebGPU timestamp duration for target prefiltering plus
   IC-GN and median replacement. It excludes Python transport and result
   readback.

``wall_times_ms``
   Browser-side elapsed time around the same prefilter-and-solve operation. It
   includes synchronization while waiting for the GPU query, but excludes
   Python image conversion, DevTools transport, target decoding, and result
   readback.

Complete Python call
   Measure ``engine.register(frame)`` with ``time.perf_counter``. This includes
   the Python-to-browser and browser-to-Python path and is the correct boundary
   for a Python application.

Acquisition-to-result latency
   Timestamp frames at acquisition and results after DIC processing. This is the
   relevant metric for the complete real-time system.

Use representative data
-----------------------

Iteration count depends on displacement, texture, noise, and rejected subsets.
Benchmark the expected motion and image quality, not only the analytical test
pair. Report at least median and p95 latency, accepted-subset count, and mean
iteration count.

For the measured Three.js, GPU-IO, and SYCL results on the validation machine,
see :doc:`../reference/performance`. For the interpretation of timing
boundaries, see :doc:`../explanation/realtime-performance`.
