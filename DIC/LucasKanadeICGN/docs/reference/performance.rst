Performance on the validation machine
=====================================

This page records measurements obtained on the development machine. They are
reference results for regression detection, not portable performance claims.
Driver versions, browser versions, power policy, and thermal state can change
the result on otherwise identical hardware.

.. note::

   The current pyramid campaign was run after the opt-in hardware regressions
   passed on the validation GPU. It includes axis-oriented FIR tiling, the
   corrected reduction arithmetic, seeded-solve state handling, and the final
   packed readback.

``gpuio`` results are retained as historical baselines. GPU-IO is deprecated
for application use; current WebGL2 compatibility measurements should use
``three-tsl`` with ``force_three_webgl=True``.

Validation machine
------------------

.. list-table:: Hardware and software environment
   :header-rows: 1

   * - Component
     - Value
   * - Measurement date
     - 2026-10-29 (latest optimization audit)
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
   * - WebGL path
     - ANGLE, Mesa Intel OpenGL 4.6
   * - WebGPU path
     - Three.js r186, Intel Xe2-LPG Vulkan adapter
   * - Native WebGPU path
     - wgpu-py 0.32.0, wgpu-native Vulkan, same physical GPU
   * - SYCL path
     - DPC++ ``-O2``, Level Zero, same physical GPU

Common workload
---------------

All compared implementations use the raw 512 x 512 validation images, without
H.264 decoding. The measured pair is ``t=0 -> 10``, whose analytical
translation is ``(2, -1)`` pixels. The common configuration is:

* 32 x 32 grid, or 1,024 subsets;
* 16 x 16 pixels per subset, with 900 usable interior subsets;
* six affine parameters per subset;
* sigma-one Gaussian and cubic B-spline target coefficients;
* cubic B-spline interpolation over 4 x 4 coefficients;
* convergence tolerance ``1e-3`` and minimum ZNCC ``0.8``;
* at most 30 inverse-compositional iterations;
* spatial median consistency test enabled.

Revalidation audit
------------------

The latest UHD optimization audit directly compared starting commit
``d18d10c`` with the optimized kernels using eight warm-ups and 50 measured
registrations. Native rows are complete Python-call wall times; browser rows
are GPU timestamps.

.. list-table:: October 29 UHD kernel optimization audit
   :header-rows: 1
   :widths: 31 19 19 14 17

   * - Boundary and translation
     - Starting median / p95
     - Optimized median / p95
     - Median reduction
     - Optimized quality
   * - Native full pyramid, 2 levels, ``(4, -2)``
     - 14.820 / 15.639 ms
     - **10.194 / 11.073 ms**
     - **31.2%**
     - 31,724 accepted; 0.001874 px RMSE
   * - Native full pyramid, 3 levels, ``(4, -2)``
     - 15.946 / 16.937 ms
     - **10.280 / 11.151 ms**
     - **35.5%**
     - 31,734 accepted; 0.001939 px RMSE
   * - Raw browser WGSL, 2 levels, ``(4, -2)``
     - 14.874 / 15.476 ms
     - **8.789 / 9.375 ms**
     - **40.9%**
     - 31,724 accepted; 0.001804 px RMSE
   * - Three.js TSL subgroup, ``(4, -2)``
     - 29.906 / 32.261 ms
     - **24.300 / 26.627 ms**
     - **18.7%**
     - 22,401 accepted; 0.096354 px RMSE
   * - Three.js TSL subgroup, ``(2, -1)``
     - 13.469 / 13.997 ms
     - **11.893 / 12.798 ms**
     - **11.7%**
     - 31,654 accepted; 0.0000305 px RMSE

The native two- and three-level p95 values are now below the 16.667 ms period
on this machine. This supersedes the older three-level 60 Hz warning in the
historical revalidation table below. Browser wall time remains higher than its
GPU timestamp because it also includes synchronization and result transport.

The same optimized shader was exercised through the bounded single-scale
sequence API in four campaigns of 30 UHD frames. The synchronous median period
was 11.539 ms (86.7 FPS). Two in-flight slots reduced it to 11.026 ms (90.7
FPS) with 21.963 ms median result latency; three slots reached 10.780 ms (92.8
FPS) with 32.093 ms latency. All configurations accepted 31,654 subsets and
reported ``3.066e-5`` px RMSE.

The current policy derived from this audit is:

* ``ICGNWGPUFullPyramid()`` with two levels is the recommended native
  coarse-to-fine configuration;
* ``ICGNWebGL(backend="webgpu-wgsl")`` also defaults to the two-level shared
  full-pyramid graph in Chromium;
* ``ICGNWGPU()`` remains the single-scale/streaming engine; its optional
  selective recovery is historical compatibility, not the normal pyramid;
* Three.js/TSL remains the browser entry-point default and uses the
  hardware-width subgroup path when available.

See :doc:`../explanation/backend-selection` for the complete decision table.

Historical campaign revalidation (October 28)
---------------------------------------------

The published tables below are campaign snapshots, not constants of the
hardware. On 2026-10-28, after the Three.js/WebGL2 fallback and browser pyramid
changes, the documented commands were rerun on the same machine. All accepted
counts, status fields, mean translations, and truth RMSE values reproduced at
the displayed precision. The main timing checks were:

.. list-table:: Revalidation of the published performance claims
   :header-rows: 1
   :widths: 26 19 22 33

   * - Boundary
     - Published campaign
     - Revalidation
     - Interpretation
   * - 512 x 512 Three.js WebGPU, GPU median
     - 0.675 ms
     - 0.668 ms
     - Three paired campaigns; 900 accepted
   * - 512 x 512 GPU-IO, GPU median
     - 3.301 ms
     - 3.293 ms
     - Three paired campaigns; 900 accepted
   * - 512 x 512 native complete call
     - 1.812 / 2.187 ms median / p95
     - 1.588 / 1.853 ms
     - 900 accepted; same ``2.696e-5`` px RMSE
   * - 512 x 512 Three.js complete Python call
     - 56.203 / 59.117 ms median / p95
     - 59.360 / 65.430 ms
     - Browser transport boundary; 900 accepted
   * - 512 x 512 raw WGSL one-scale complete call
     - 55.916 / 59.555 ms median / p95
     - 58.156 / 61.775 ms
     - Explicitly forced to one scale; 900 accepted
   * - 512 x 512 GPU-IO complete Python call
     - 61.877 / 65.655 ms median / p95
     - 69.636 / 73.527 ms
     - Browser transport boundary; 900 accepted
   * - UHD native one-scale complete call
     - 14.621 / 15.748 ms median / p95
     - 14.605 / 15.018 ms
     - 31,654 accepted; same ``3.075e-5`` px RMSE
   * - UHD native full pyramid, two levels
     - 14.811 / 15.163 ms median / p95
     - 14.580 / 15.478 ms
     - Corrected-reduction campaign; 31,724 accepted
   * - UHD native full pyramid, three levels
     - 15.458 / 16.210 ms median / p95
     - 16.012 / 16.813 ms
     - October 28 ``r32float`` campaign; superseded above
   * - UHD raw browser pyramid, two levels, ``(4, -2)``
     - 15.931 / 16.622 ms GPU median / p95
     - 14.961 / 15.587 ms
     - Corrected-reduction campaign; 31,720 accepted
   * - UHD Three.js/WebGL2 compatibility path
     - 27.246 / 27.730 ms GPU median / p95
     - 28.511 / 29.328 ms
     - 31,654 accepted; same ``4.116e-5`` px RMSE
   * - UHD GPU-IO historical path
     - 80.738 / 82.522 ms GPU median / p95
     - 79.665 / 80.304 ms
     - 31,654 accepted; same ``4.182e-5`` px RMSE

The reproduced GPU medians support the relative conclusions below. Complete
browser-call timings vary more because they include PNG encoding, DevTools
transport, decoding, and result conversion. Short-campaign p95 values also
vary more than medians and are not hard real-time guarantees. In particular,
this October 28 snapshot did not support a p95 4K60 claim for the three-level
native pyramid. The newer October 29 optimization audit above supersedes that
conclusion for the current kernels. The two-level native pyramid remains the
recommendation because it is the smaller sufficient graph. All percentages,
FPS conversions, timing margins, and throughput ratios are scoped to the
table from which they are computed.

Backend comparison
------------------

The browser values are the median of three campaigns. Each campaign uses five
warm-ups followed by 30 measured runs. The SYCL values come from 30 runs after
three warm-ups.

.. list-table:: Pipeline measurements on this machine
   :header-rows: 1
   :widths: 18 20 25 10 10 9 8

   * - Implementation
     - Runtime
     - Timed boundary
     - Median
     - p95
     - Accepted
     - Vector RMSE
   * - Three.js/TSL
     - WebGPU, Vulkan, subgroup solver
     - GPU prefilter, IC-GN, median pass
     - **0.675 ms**
     - **1.227 ms**
     - 900 / 1,024
     - 2.749e-5 px
   * - shared WGSL browser, one scale
     - raw WebGPU, Vulkan, portable workgroup
     - GPU prefilter, IC-GN, median pass
     - 0.817 ms
     - 1.298 ms
     - 900 / 1,024
     - 2.700e-5 px
   * - GPU-IO
     - WebGL2, OpenGL
     - GPU prefilter, IC-GN, median pass
     - 3.301 ms
     - 4.210 ms
     - 900 / 1,024
     - 3.873e-5 px
   * - DIC_FROM_H264 SYCL
     - DPC++, Level Zero
     - upload, prefilter, IC-GN, readback
     - 3.651 ms
     - 4.280 ms
     - 900 / 1,024
     - 1.12e-5 px

The Three.js/TSL median GPU time is 79.6% lower than GPU-IO for this case, or
4.89 times the throughput at this timing boundary. GPU-IO now fuses every
IC-GN iteration into one fragment draw, keeps the affine warp and reference
atlases in registers, and writes the subset state only once. Relative to the
previous 3.768 ms measurement, that reduced its median GPU time by 12.4%. The
runtime measured a 32-lane hardware subgroup but this historical campaign used
the older fused 16-active-lane Three.js solver. The current implementation
fills the measured subgroup, while a portable 16-lane workgroup-memory solver
remains the fallback. See :doc:`../explanation/gpu-optimization` for the
current mappings.

The raw WebGPU adapter executes exactly the portable WGSL file used by
wgpu-py. This historical one-scale 512 x 512 GPU median is 21.0% above the specialized Three.js
subgroup path, while its truth RMSE is within ``5e-7`` pixel. This is the
measured cost of choosing one directly shared portable kernel on this adapter.

The SYCL row has a stricter timing boundary: its ``total_ms`` includes upload
and readback, while browser GPU queries exclude both. The raw figures are useful
on the same page, but they must not be interpreted as a strictly equivalent
backend ratio. The separately observed SYCL registration time was 2.573 ms;
the current fused browser solvers do not isolate registration from their
prefilter and median passes.

Browser synchronization
-----------------------

Browser wall time measures the same GPU work plus backend synchronization. It
still excludes Python image conversion, DevTools transport, image decoding, and
result readback.

.. list-table:: Browser benchmark wall time
   :header-rows: 1

   * - Backend
     - Median
     - p95
   * - Three.js/TSL
     - **2.7 ms**
     - **3.4 ms**
   * - shared WGSL browser
     - 2.6 ms
     - 3.1 ms
   * - GPU-IO
     - 10.3 ms
     - 11.9 ms
   * - SYCL
     - not applicable
     - not applicable

The large browser-wall difference is primarily a synchronization-path
difference. It is not an acquisition-to-result latency comparison. Use
:doc:`../explanation/realtime-performance` for the timing boundaries required
to qualify a complete application.

Three.js WebGL2 fallback
------------------------

The Three.js backend now selects a serial TSL transform-feedback solver when
WebGPU is unavailable. A forced-WebGL2 campaign on the same 512 x 512 pair used
five warm-ups and 30 measured runs. GPU-IO was measured immediately before it
with the same process boundary and image pair.

.. list-table:: WebGL2 compatibility backends at 512 x 512
   :header-rows: 1

   * - Backend
     - GPU median
     - GPU p95
     - Browser wall median
     - Browser wall p95
     - Accepted
     - Truth RMSE
   * - Three.js/TSL WebGL2
     - **1.466 ms**
     - **1.565 ms**
     - **8.4 ms**
     - **8.7 ms**
     - 900 / 1,024
     - 3.905e-5 px
   * - GPU-IO
     - 3.869 ms
     - 5.228 ms
     - 10.8 ms
     - 12.6 ms
     - 900 / 1,024
     - 3.873e-5 px

The status fields were identical. Across the 900 jointly accepted subsets, the
maximum absolute difference between any affine parameter was
``1.4913e-4`` and the mean absolute difference was ``4.5489e-6``. Iteration
counts were not bit-identical because the two GLSL programs accumulate in a
different order.

At 3840 x 2160, the forced Three.js/WebGL2 path measured 27.246 ms GPU median,
27.730 ms GPU p95, 33.4 ms browser-wall median, and 37.4 ms browser-wall p95
over eight runs after three warm-ups. It accepted 31,654 subsets with a
``4.116e-5`` px truth RMSE. The historical GPU-IO campaign measured 80.738 ms
GPU median on the same nominal translation. Three.js/WebGL2 is therefore a
substantially better compatibility backend, but neither WebGL2 implementation
meets the 16.667 ms 4K60 budget.

Force the fallback on a WebGPU-capable machine with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend three-tsl --force-three-webgl \
       --repeats 8 --warmups 3

Native wgpu-py comparison
-------------------------

The experimental ``ICGNWGPU`` backend was measured on the same 512 x 512
``t=0 -> 10`` pair in three campaigns with five warm-ups and 30 runs. Unlike the browser rows, its
complete-call boundary includes input validation, direct texture upload, FIR,
IC-GN, median replacement, packed readback, and NumPy result construction.
There is no PNG encoding, Chromium process, DevTools transfer, or JavaScript
array conversion.

.. list-table:: Complete Python-call comparison at 512 x 512
   :header-rows: 1

   * - Backend
     - Median
     - p95
     - Accepted
     - Truth RMSE
   * - native wgpu-py
     - **1.812 ms**
     - **2.187 ms**
     - 900 / 1,024
     - 2.696e-5 px
   * - Three.js/TSL through Chromium
     - 56.203 ms
     - 59.117 ms
     - 900 / 1,024
     - 2.749e-5 px
   * - shared WGSL through Chromium, one scale
     - 55.916 ms
     - 59.555 ms
     - 900 / 1,024
     - 2.700e-5 px
   * - GPU-IO through Chromium
     - 61.877 ms
     - 65.655 ms
     - 900 / 1,024
     - 3.873e-5 px

The browser complete-call values are dominated by PNG and DevTools transport;
they are not the browser GPU values in the earlier table. Within the native
call, target upload measured 0.044 ms median and compute plus packed readback
measured 1.647 ms median. Native solver preparation measured 1.993 ms for the
portable pipeline and 3.109 ms for the subgroup pipeline, then selected the
portable path.

The separate Rust/PyO3 upload milestone compiles the same WGSL and borrows the
NumPy frame directly. Its initial synchronized UHD texture-upload probe
measured 1.783 ms median and 6.029 ms p95. Because the complete solver is not
yet hosted by Rust, this result is reported as an integration baseline, not as
a faster backend. See :doc:`../explanation/native-backend`.

Provenance and reproduction
---------------------------

The Three.js and GPU-IO comparison can be rerun from the repository root:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/compare_backends.py \
       --repeats 30 --warmups 5

Run the native and browser complete-call comparison with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/compare_native_wgpu.py \
       --repeats 30 --warmups 5 --webgpu-wgsl-pyramid-levels 1

The comparison script defaults to one raw-browser scale so that the historical
table remains like for like even though normal ``webgpu-wgsl`` applications
now default to two pyramid levels.

Detailed browser campaign results and the four-target numerical comparison are
kept in ``DIC/LucasKanadeICGN/BENCHMARK.md``.

The SYCL implementation is maintained in the
`DIC_FROM_H264 project <https://github.com/jeffwitz/DIC_FROM_H264>`_. The
referenced checkout is commit
`77267f9 <https://github.com/jeffwitz/DIC_FROM_H264/commit/77267f92920ea8984533371cc6a62fef149ff6e2>`_,
and the standalone comparison source is
`icgn_sycl.cpp <https://github.com/jeffwitz/DIC_FROM_H264/blob/77267f92920ea8984533371cc6a62fef149ff6e2/experiments/codec_observation_closure/sycl_probe/icgn_sycl.cpp>`_.

UHD validation
--------------

The native backend was also measured on the same reproducible 3840 x 2160
pair. Three campaigns used five warm-ups and 30 measured complete Python calls;
the table reports the median campaign summary.

.. list-table:: Native UHD complete-call breakdown
   :header-rows: 1

   * - Measurement
     - Median
     - p95
   * - Complete ``register`` call
     - **14.621 ms**
     - **15.748 ms**
   * - Direct target upload
     - 0.403 ms
     - not recorded
   * - Compute plus packed readback
     - 13.876 ms
     - not recorded

Reference preparation measured 1,036.5 ms and remained outside the per-frame
loop. The portable solver was forced for this source-sharing campaign; the
automatic policy still compares it with the optional subgroup pipeline during
normal preparation. The final registration accepted 31,654 subsets and
produced a ``3.075e-5 px`` truth RMSE. The median fits inside the 16.667 ms
period, with 0.919 ms of p95 margin. It excludes acquisition
and downstream work and is therefore evidence for the transport design, not a
qualified 4K60 application.

The corresponding median Three.js campaign measured 14.462 ms GPU and 15.9 ms
synchronized browser wall. The native complete-call median is 8.0% lower than
that browser-wall value while including per-sample target upload, packed
readback, and NumPy statistics. These boundaries are deliberately shown
separately rather than presented as an exact compute-only comparison.

Reproduce this native campaign with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend wgpu-native --repeats 30 --warmups 5

Native sequence pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

``register`` minimizes the latency of one isolated frame but synchronizes its
packed readback before Python can submit the next target. The bounded
``register_stream`` API assigns independent target, intermediate, state, and
readback resources to each in-flight slot. Reference buffers remain shared.

Four balanced campaigns used five warm-ups and 30 UHD targets per method. The
campaign order was rotated to limit warm-up and frequency bias. Period
percentiles below are computed across the four campaign-average periods;
latency percentiles are first computed across frames within each campaign.

.. list-table:: UHD native sequence throughput and result latency
   :header-rows: 1

   * - Method
     - Median period
     - Period p95
     - Throughput
     - Median result latency
   * - synchronous ``register``
     - 14.792 ms
     - 15.149 ms
     - 67.6 FPS
     - 14.671 ms
   * - one mapped slot
     - 14.493 ms
     - 14.802 ms
     - 69.0 FPS
     - 14.447 ms
   * - **two slots**
     - **13.380 ms**
     - **14.203 ms**
     - **74.7 FPS**
     - 26.617 ms
   * - three slots
     - 13.077 ms
     - 14.008 ms
     - 76.5 FPS
     - 38.921 ms

Every method accepted 31,654 subsets, produced the same ``3.075e-5 px`` truth
RMSE, and returned bit-for-bit identical parameter and status fields. Two
slots are the recommended throughput setting: the third slot adds only 2.3%
throughput while adding another 12.3 ms of median result latency.

This demonstrates offline 4K60 *processing throughput*. It does not include a
camera, frame arrival jitter, or downstream consumers. The period is total
sequence wall time divided by frame count; result latency is measured from
target upload until its packed field has been converted to NumPy.

Reproduce the comparison with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_pipeline.py \
       --repeats 30 --warmups 5 --campaigns 4 --depths 1 2 3

Historical optional selective native recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   This section documents the older ``ICGNWGPU(pyramid_levels=2 or 3)``
   exception path. It is not the recommended native pyramid. Use
   ``ICGNWGPUFullPyramid`` for normal coarse-to-fine processing.

The optional native pyramid was measured with an exact ``(4, -2)`` pixel
translation, which deliberately exceeds the robust basin of many fine subsets.
``pyramid_levels`` counts the fine image. One scale is the default fast path;
two scales add a half-resolution selective recovery.

.. list-table:: Selective pyramid recovery on the validation machine
   :header-rows: 1

   * - Resolution and scales
     - Median call
     - p95
     - Accepted
     - Recovered failures
     - Truth RMSE
   * - 512 x 512, one scale
     - 3.189 ms
     - 3.682 ms
     - 634 / 1,024
     - 0
     - 0.0906 px
   * - 512 x 512, two scales
     - 5.834 ms
     - 6.396 ms
     - 900 / 1,024
     - 266
     - 0.0760 px
   * - 3840 x 2160, one scale
     - 36.247 ms
     - 38.285 ms
     - 22,401 / 32,400
     - 0
     - 0.0964 px
   * - 3840 x 2160, two scales
     - 90.035 ms
     - 91.359 ms
     - 31,654 / 32,400
     - 9,253
     - 0.0811 px

These are recovery measurements, not the nominal ``(2, -1)`` throughput case.
The two-scale path preserves every result accepted by the first fine solve, so
its RMSE intentionally retains any fine-scale false convergence. On the
512 x 512 validation, all six parameters, status, and iteration count of all
634 initially accepted subsets remained bit-for-bit identical. Fixed border
subsets whose interpolation support cannot lie inside the image are excluded
from recovery triggering and scoring.

The one-scale WGSL output also remained bit-for-bit identical to the wheel
before the pyramid implementation. Its 512 x 512 packed-field SHA-256 for the
shipped ``t=20`` pair was
``74ef8ba550fba255a18d9bf904dfb591892d0a71358e37e070646ad01b544b1f``
before and after. A repeated nominal UHD pipeline campaign measured 13.469 ms
per target versus the earlier 13.380 ms, a 0.7% difference within the observed
campaign spread; the numerical result was unchanged.

Reproduce the recovery campaign with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_pyramid.py \
       --warmups 3 --repeats 10 --levels 1 2

Add ``--width 3840 --height 2160`` for the UHD campaign. The current CPU
five-tap reduction makes active 4K recovery an exception path rather than a
4K60 mode. When no recovery is configured, none of these resources or
dispatches exists.

Pre-tiling GPU-resident full-pyramid campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The separate ``ICGNWGPUFullPyramid`` backend was measured on the same Intel
GPU with the same exact ``(4, -2)`` translation. It uploads only the finest
target, performs reduction and coarse-to-fine seed transfer on the GPU, submits
one command buffer, and reads back only the finest packed field. The two-level
configuration is the class default and the recommended native path.

.. list-table:: Full-pyramid validation on the Intel Lunar Lake GPU
   :header-rows: 1

   * - Resolution and backend
     - Iterations coarse to fine
     - Median call
     - p95
     - Accepted
     - Truth RMSE
   * - 512 x 512, one scale
     - 30
     - 2.641 ms
     - 3.548 ms
     - 634 / 1,024
     - 0.0906 px
   * - 512 x 512, full pyramid 2
     - 12, 30
     - 1.740 ms
     - 2.242 ms
     - 900 / 1,024
     - 0.00000955 px
   * - 512 x 512, full pyramid 3
     - 6, 8, 20
     - 2.073 ms
     - 2.625 ms
     - 900 / 1,024
     - 0.0000102 px
   * - 3840 x 2160, one scale
     - 30
     - 33.238 ms
     - 35.125 ms
     - 22,401 / 32,400
     - 0.0964 px
   * - 3840 x 2160, full pyramid 2
     - 12, 30
     - **14.580 ms (68.6 FPS)**
     - 15.478 ms
     - 31,724 / 32,400
     - 0.00182 px
   * - 3840 x 2160, full pyramid 3
     - 6, 8, 20
     - **16.012 ms (62.5 FPS)**
     - 16.813 ms
     - 31,734 / 32,400
     - 0.00194 px

This campaign used five warm-ups and 30 measured runs after the hardware
regressions qualified the corrected integer-scale reduction. Two levels
provide the repeated 4K60 compute margin for this
controlled translated texture on this machine. Three levels remained above 60
FPS by median, but should be reserved for a larger expected
capture range and are not claimed to meet the 60 Hz p95 boundary. None of
these measurements is an acquisition-to-result guarantee. The three-level
iteration ceilings were reduced from ``(8, 12, 30)`` to ``(6, 8, 20)`` after
an earlier alternating A/B campaign preserved its accepted field.
The fine level converged in 1.39 iterations on average in the current campaign, so reducing the
ceilings recovered only a fraction of the extra-scale cost; target reduction,
prefiltering, seed transfer, and additional dispatches remain fixed costs. A
one-level ``ICGNWGPUFullPyramid`` result matched the portable ``ICGNWGPU``
packed field bit for bit at 512 x 512.

A more aggressive descending schedule, ``(12, 8, 6)``, reduced the alternating
campaign median from 15.895 to 15.345 ms. It also reduced accepted subsets from
31,681 to 31,260. It is therefore available through
``iterations_per_level`` for latency-oriented applications, but is not the
robust default: the small time gain did not justify losing 421 measurements on
this validation pair.

Reproduce the campaign with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_wgpu_full_pyramid.py \
       --width 3840 --height 2160 --levels 2 3 --warmups 5 --repeats 30

Pre-tiling raw-browser full-pyramid campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same ``icgn.wgsl`` and ``full_pyramid.wgsl`` graph now runs through raw
WebGPU in Chromium. The browser uploads only the finest target, retains every
reduced image and coefficient field on the GPU, and transfers affine seeds
coarse to fine. Two levels are the browser default; one level keeps the
original single-scale dispatch path.

The current focused campaigns used five warm-ups and 20 measured runs per
depth on the same Intel GPU. GPU timestamps include reductions,
per-level prefilters and solves, seed transfers, and the final median pass, but
exclude PNG transport and result readback.

.. list-table:: Raw browser WebGPU full-pyramid UHD validation
   :header-rows: 1

   * - Translation and levels
     - Iterations coarse to fine
     - GPU median
     - GPU p95
     - Accepted
     - Truth RMSE
   * - ``(2, -1)``, one scale
     - 30
     - 17.247 ms
     - 17.762 ms
     - 31,654 / 32,400
     - 0.0000308 px
   * - ``(2, -1)``, two levels
     - 12, 30
     - **16.299 ms (61.4 FPS)**
     - 16.981 ms
     - 31,654 / 32,400
     - 0.0000301 px
   * - ``(2, -1)``, three levels
     - 6, 8, 20
     - **16.540 ms (60.5 FPS)**
     - 17.608 ms
     - 31,654 / 32,400
     - 0.0000298 px
   * - ``(4, -2)``, one scale
     - 30
     - 61.393 ms
     - 62.151 ms
     - 22,401 / 32,400
     - 0.0964 px
   * - ``(4, -2)``, two levels
     - 12, 30
     - **14.961 ms (66.8 FPS)**
     - 15.587 ms
     - 31,720 / 32,400
     - 0.00184 px
   * - ``(4, -2)``, three levels
     - 6, 8, 20
     - **15.269 ms (65.5 FPS)**
     - 15.568 ms
     - 31,734 / 32,400
     - 0.00194 px

The difficult pair is the deciding result: two levels recover 9,319 additional
subsets while reducing the median by 75% because the single-scale failures run
to their 30-iteration ceiling. Two levels are therefore the robust browser
default. Three remain useful for a larger expected capture range. The nominal
two-level p95 is slightly above 16.667 ms in this pre-tiling campaign, as is the
three-level p95. The October 29 audit at the top of this page supersedes these
timings for the current kernels. Neither campaign is a hard end-to-end
real-time qualification.

Reproduce either pair offline with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend webgpu-wgsl --pyramid-levels 2 \
       --translation 4 -2 --warmups 5 --repeats 20

Browser validation
~~~~~~~~~~~~~~~~~~

All three dynamic-resolution browser paths were also validated at 3840 x 2160
on the same machine before the pyramid became the raw-WGSL default. This
historical single-scale comparison uses ``pyramid_levels=1`` for shared WGSL.
The offline reference tiles the shipped speckle texture and
the target applies an exact ``(2, -1)`` pixel translation. This produces a 240
x 135 grid, or 32,400 subsets. Each of three campaigns used five warm-ups and
30 measured optical-flow runs. The table reports the median campaign summary.

.. list-table:: UHD preparation and optical-flow measurements
   :header-rows: 1

   * - Phase
     - Boundary
     - Three.js/TSL
     - shared WGSL
     - GPU-IO
   * - Runtime construction
     - storage allocation and shader/graph construction
     - 4.3 ms
     - 37.9 ms
     - included in page startup
   * - Reference construction
     - grayscale filtering, gradients, Hessians, and metadata
     - 1,144.3 ms
     - 1,094.7 ms
     - 1,261.1 ms
   * - Browser preparation
     - runtime plus reference construction
     - 1,148.8 ms
     - 1,132.6 ms
     - 1,261.1 ms
   * - Complete Python preparation
     - conversion, PNG transport, and browser preparation
     - 1,456.9 ms
     - 1,451.9 ms
     - 2,583.3 ms
   * - Optical flow median
     - GPU prefilter, IC-GN, and median pass
     - **14.462 ms GPU; 15.9 ms browser wall**
     - 16.490 ms GPU; 17.9 ms browser wall
     - 80.738 ms GPU; 85.9 ms browser wall
   * - Optical flow p95
     - same boundary
     - **17.097 ms GPU; 18.1 ms browser wall**
     - 17.068 ms GPU; 18.8 ms browser wall
     - 82.522 ms GPU; 90.2 ms browser wall

All final registrations accepted 31,654 subsets. Three.js recovered
``(1.999991, -0.999997)`` pixels with a ``3.076e-5`` pixel vector RMSE; shared
WGSL recovered the same rounded translation with a ``3.078e-5`` pixel RMSE;
GPU-IO recovered ``(1.999991, -0.999996)`` pixels with a ``4.182e-5`` pixel
RMSE. The shared portable kernel uses 14.0% more GPU time than the Three.js
subgroup kernel at the median boundary. Three.js provides 5.58 times the
GPU-IO throughput at that boundary.
GPU-IO is functionally valid at UHD, but its one-fragment-per-subset mapping is
not a 4K60 performance path on this machine.

The equivalent current SYCL configuration was rebuilt and measured on the same
offline pair. It uses mixed precision, column layout, the FIR prefilter, a
constant inverse Hessian, no acceleration or tolerance guard, and median
replacement.

.. list-table:: UHD browser and SYCL comparison
   :header-rows: 1

   * - Measurement
     - Three.js/TSL
     - shared WGSL
     - GPU-IO
     - SYCL mixed/double reduction
     - SYCL float reduction
   * - Prefilter plus registration median
     - **14.462 ms**
     - 16.490 ms
     - 80.738 ms
     - 19.099 ms
     - 18.111 ms
   * - Prefilter plus registration p95
     - **17.097 ms**
     - 17.068 ms
     - 82.522 ms
     - 19.863 ms
     - 19.062 ms
   * - Upload plus compute plus download median
     - not included
     - not included
     - not included
     - 19.573 ms
     - 18.509 ms
   * - Accepted subsets
     - 31,654
     - 31,654
     - 31,654
     - 31,654
     - 31,654
   * - Mean iterations
     - 5.943
     - 5.942
     - 5.947
     - 5.944
     - 5.940
   * - Truth RMSE
     - 3.076e-5 px
     - 3.078e-5 px
     - 4.182e-5 px
     - 2.555e-5 px
     - 2.712e-5 px

At the comparable compute boundary, optimized Three.js uses 75.7% of the
mixed/double-reduction SYCL time: a 24.3% reduction, or 1.32 times the
throughput, on this machine. It uses 79.9% of the numerically closer
``--float-reduce`` time, a 20.1% reduction or 1.25 times the throughput. The
Three.js timer excludes input upload and output readback, so the separate SYCL
totals are shown rather than folded into those ratios. These values demonstrate
functional UHD support and backend compute performance, not acquisition-to-
result 4K60 qualification. WebGPU upload and pipeline compilation are lazy;
the five unreported warm-ups absorb first-use work.

Reproduce the campaign offline with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --backend three-tsl --repeats 30 --warmups 5

Replace ``three-tsl`` with ``webgpu-wgsl --pyramid-levels 1`` for the canonical
portable browser shader at the same single-scale boundary, or with ``gpuio``
for the optional WebGL2 measurement.

