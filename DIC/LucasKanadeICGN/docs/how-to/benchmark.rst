Benchmark the DIC pipeline
==========================

Original UHD images for numerical regression
---------------------------------------------

The repository includes four original 3840 x 2160 virtual tensile-test PNGs
under ``DIC/LucasKanadeICGN/assets/uhd_tensile``. They were copied byte for
byte from the source experiment in ``DIC_FROM_H264``. They are native UHD
images, not tiled 512 x 512 textures. Their source generator, SHA256 hashes,
rendering conventions, and complete 200-state machine truth are archived
beside them. Tests run offline without that other repository or a video decoder.

The source generator stretches about pixel coordinate ``(1920, 1080)``. Its
targets use bilinear inverse remapping and uint8 quantization. These images
therefore have a known affine geometric truth with discrete rendering error;
they must not be described as exact continuous-texture samples.

Run the fixture checks and full-resolution numerical regressions with:

.. code-block:: console

   $ ICGN_RUN_GPU_TESTS=1 PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 -m unittest discover -s DIC/LucasKanadeICGN/python/tests \
       -p test_uhd_fixtures.py -v

Every image is checked for identity recovery. The reference-to-frame-40
case checks affine displacement with raw and median policies, and gradients
with the explicit median policy. The raw gradient estimate is not qualified
to the same bound. Coverage and error are checked separately, excluding a
32-pixel border. Later frames 100 and 199 remain available as large-deformation
inputs, but their identity tests do not qualify zero-seed recovery of the
complete deformation. No timing threshold is imposed by these correctness
tests. The Linux software Vulkan CI job executes them on lavapipe.

These inputs are separate from the tiled fixtures in the performance commands
below and the on-demand continuous analytical generators. See the fixture
README for all bounds and :download:`the archived manifest
<../../assets/uhd_tensile/manifest.json>` for provenance.

Measure processing time
-----------------------

For the Python-supervised product boundary, the primary measurement is the
complete call from a resident NumPy image to a materialized NumPy field. Time
``engine.register(frame)`` or use ``benchmark_image_to_field.py``. The
browser-only benchmark below is a diagnostic for the GPU graph; it must not be
used as the Python stream result. See :doc:`../explanation/supervision`.

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
   IC-GN and the optional UOD pass. It excludes Python transport and result
   readback.

``wall_times_ms``
   Browser-side elapsed time around the same prefilter-and-solve operation. It
   includes synchronization while waiting for the GPU query, but excludes
   Python input preparation, binary process transport, and result readback.

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
   for the current Python supervisor. A future JavaScript-supervised Three.js
   stream would need a separate browser-native end-to-end benchmark.

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

For the current native wgpu-py, raw-WGSL, Three.js, and bounded-sequence
results on the validation machine, see :doc:`../reference/performance`. For the
interpretation of timing boundaries, see
:doc:`../explanation/realtime-performance`.

For the comparable deployable boundary, use changing frames and time the public
call from a resident NumPy image until its complete NumPy field is returned:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
       --width 3840 --height 2160 --frames 12 --warmups 3 \
       --motion-model affine

This persistent-engine benchmark excludes one-time reference preparation and
fixture generation. It includes every per-frame transfer, synchronization,
readback, and Python result conversion. Its synchronous stream period is also
the result age. Use this measurement for backend selection; use GPU timestamps
only to diagnose kernels.

From a repository checkout, reproduce the UHD campaign for either browser
backend with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend three-tsl --repeats 50 --warmups 8

Use ``--backend webgpu-wgsl`` for the canonical portable shaders and default
two-level pyramid in Chromium. Add
``--pyramid-levels 1`` for a comparable single-scale campaign, or
``--pyramid-levels 3`` to measure the larger-capture-range graph.
Use ``--backend three-tsl --force-three-webgl`` to measure the Three.js WebGL2
fallback separately from its normal WebGPU path.
Use ``--backend wgpu-native`` for the browser-free complete Python-call path.
Add ``--native-solver portable`` or ``--native-solver subgroup`` to bypass
automatic selection during a diagnostic campaign.
Add ``--motion-model translation`` to compare the displacement-only analytic
solver with the default affine model. Keep the interpolation, pyramid depth,
motion, warm-up count, and repetition count identical for an A/B comparison.

Archive a backend-comparison campaign with its provenance and raw timing
samples:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/compare_native_wgpu.py \
       --native-only --native-solver auto \
       --repeats 50 --warmups 8 \
       --output campaign-native-auto.json

The JSON records the Git commit and dirty-worktree flag, complete command,
Python, NumPy, wgpu and DIC-RT versions, platform, input paths and SHA-256
hashes, selected solver, device, numerical error, accepted count, and
individual timing samples. Keep this file with any table derived from the
campaign.

To distinguish native sequence throughput from per-result latency, compare
bounded pipeline depths explicitly:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_pipeline.py \
       --repeats 30 --warmups 5 --campaigns 4 --depths 1 2 3

The reported period is total sequence wall time divided by the number of
frames. It must not be presented as result latency when more than one slot is
in flight.

Measure the recommended GPU-resident native full-pyramid path with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_full_pyramid.py \
       --width 3840 --height 2160 --levels 2 3 --warmups 8 --repeats 50

This command reports each requested full-pyramid depth on the same exact
``(4, -2)`` target. It also records the coarse-to-fine
iteration schedule, accepted subset count, truth RMSE, and number of queue
submissions.

Repeat it with ``--motion-model translation`` when the application can assume
rigid subset motion. Do not compare its zero deformation terms with affine
strain output; compare displacement, accepted subsets, iterations, and timing.
