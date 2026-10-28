Performance on the validation machine
=====================================

This page records measurements obtained on the development machine. They are
reference results for regression detection, not portable performance claims.
Driver versions, browser versions, power policy, and thermal state can change
the result on otherwise identical hardware.

Validation machine
------------------

.. list-table:: Hardware and software environment
   :header-rows: 1

   * - Component
     - Value
   * - Measurement date
     - 2026-10-28
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
   * - shared WGSL browser
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
runtime measured a 32-lane hardware subgroup and selected the fused 16-lane
Three.js solver; a portable workgroup-memory solver is selected automatically
when that condition is not met. See :doc:`../explanation/gpu-optimization` for
the two tiling strategies.

The raw WebGPU adapter executes exactly the portable WGSL file used by
wgpu-py. Its 512 x 512 GPU median is 21.0% above the specialized Three.js
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
   * - shared WGSL through Chromium
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
       --repeats 30 --warmups 5

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

Browser validation
~~~~~~~~~~~~~~~~~~

All three dynamic-resolution browser paths were also validated at 3840 x 2160 on
the same machine. The offline reference tiles the shipped speckle texture and
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
RMSE. The shared portable kernel is 14.0% slower than the Three.js subgroup
kernel at the median GPU boundary. Three.js is 5.58 times faster than GPU-IO.
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

Replace ``three-tsl`` with ``webgpu-wgsl`` for the canonical portable browser
shader, or with ``gpuio`` for the optional WebGL2 measurement.

