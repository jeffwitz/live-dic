Benchmark the DIC pipeline
==========================

Use the built-in benchmark to characterize the GPU prefilter and solver under a
stable workload:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWebGL


   reference = "DIC/LucasKanadeICGN/assets/frame_000.png"
   target = "DIC/LucasKanadeICGN/assets/frame_010.png"

   with ICGNWebGL() as engine:
       preparation = engine.set_reference(reference)
       benchmark = engine.benchmark_target(target, repeats=30, warmups=3)

   print(f"Reference construction: {preparation.reference_build_wall_ms:.3f} ms")
   print(f"Complete preparation call: {preparation.total_wall_ms:.3f} ms")
   print(f"GPU median: {benchmark.gpu_median_ms:.3f} ms")
   print(f"GPU p95: {benchmark.gpu_p95_ms:.3f} ms")
   print(f"Browser wall median: {benchmark.wall_median_ms:.3f} ms")
   print(f"Browser wall p95: {benchmark.wall_p95_ms:.3f} ms")

This form deliberately keeps the one-time reference construction outside the
warm-ups and measured optical-flow runs. ``engine.benchmark(reference, target)``
remains available as a convenience wrapper, but does not return its
``ReferenceInfo`` object.

WebGPU buffer upload and pipeline compilation are lazy. The first warm-up may
therefore be much slower than steady registration even after
``set_reference``. Keep at least one warm-up, report its count, and do not fold
that first-use cost into either the reference-construction number or the
steady optical-flow distribution.

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

``ReferenceInfo``
   ``runtime_build_wall_ms`` measures resolution-dependent host storage and TSL
   graph construction. ``reference_build_wall_ms`` measures gradients,
   Hessians, and metadata derived from the reference. ``preparation_wall_ms``
   covers both in the browser, while ``total_wall_ms`` is the complete Python
   preparation call. GPU-IO allocates textures while its private page starts,
   so it reports zero runtime-build time; if dimensions change, the page restart
   remains visible in ``total_wall_ms``.

Complete Python call
   Measure ``engine.register(frame)`` with ``time.perf_counter``. This includes
   the Python-to-browser and browser-to-Python path and is the correct boundary
   for a Python application.

Native timing fields
   ``ICGNWGPU.register`` reports ``uploadWallMs``,
   ``executeReadbackWallMs``, and ``totalWallMs`` in
   ``RegistrationResult.timing``. The execute/readback field is synchronized;
   the total includes validation, upload, GPU work, packed readback, and NumPy
   result statistics.
   With the default automatic policy, ``set_reference`` also records median
   candidate times in ``engine.solver_tuning_ms`` and assigns the selected
   pipeline name to ``engine.solver``. That tuning cost remains in reference
   preparation rather than the per-frame loop.

Acquisition-to-result latency
   Timestamp frames at acquisition and results after DIC processing. This is the
   relevant metric for the complete real-time system.

Use representative data
-----------------------

Iteration count depends on displacement, texture, noise, and rejected subsets.
Benchmark the expected motion and image quality, not only the analytical test
pair. Report at least median and p95 latency, accepted-subset count, and mean
iteration count.

For the measured native wgpu-py, Three.js, GPU-IO, PyO3, and SYCL results on
the validation machine, see :doc:`../reference/performance`. For the
interpretation of timing boundaries, see
:doc:`../explanation/realtime-performance`.

From a repository checkout, reproduce the UHD campaign for either browser
backend with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend three-tsl --repeats 30 --warmups 5

Use ``--backend webgpu-wgsl`` for the canonical portable shader in Chromium,
or ``--backend gpuio`` for the WebGL2 path.
Use ``--backend wgpu-native`` for the browser-free complete Python-call path.
Add ``--native-solver portable`` or ``--native-solver subgroup`` to bypass
automatic selection during a diagnostic campaign.

To distinguish native sequence throughput from per-result latency, compare
bounded pipeline depths explicitly:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_pipeline.py \
       --repeats 30 --warmups 5 --campaigns 4 --depths 1 2 3

The reported period is total sequence wall time divided by the number of
frames. It must not be presented as result latency when more than one slot is
in flight.
