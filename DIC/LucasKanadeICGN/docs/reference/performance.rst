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
   * - SYCL path
     - DPC++ ``-O2``, Level Zero, same physical GPU

Common workload
---------------

All three implementations use the raw 512 x 512 validation images, without
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
     - **0.672 ms**
     - **1.225 ms**
     - 900 / 1,024
     - 2.749e-5 px
   * - GPU-IO
     - WebGL2, OpenGL
     - GPU prefilter, IC-GN, median pass
     - 3.768 ms
     - 4.362 ms
     - 900 / 1,024
     - 3.819e-5 px
   * - DIC_FROM_H264 SYCL
     - DPC++, Level Zero
     - upload, prefilter, IC-GN, readback
     - 3.651 ms
     - 4.280 ms
     - 900 / 1,024
     - 1.12e-5 px

The Three.js/TSL median GPU time is 82.2% lower than GPU-IO for this case. The
three campaign ratios were 23.1%, 17.4%, and 17.2%. The Three.js p95 ranged
from 1.174 to 1.227 ms, while GPU-IO ranged from 4.253 to 4.613 ms. The runtime
measured a 32-lane hardware subgroup and selected the fused 16-lane solver; a
portable workgroup-memory solver is selected automatically when that condition
is not met.

The SYCL row has a stricter timing boundary: its ``total_ms`` includes upload
and readback, while browser GPU queries exclude both. The raw figures are useful
on the same page, but they must not be interpreted as a strictly equivalent
backend ratio. The separately observed SYCL registration time was 2.573 ms;
the corresponding GPU-IO registration time was estimated at 3.55 ms. The
Three.js registration stage has not been isolated from its prefilter and median
passes.

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
     - **3.5 ms**
   * - GPU-IO
     - 16.6 ms
     - 18.4 ms
   * - SYCL
     - not applicable
     - not applicable

The large browser-wall difference is primarily a synchronization-path
difference. It is not an acquisition-to-result latency comparison. Use
:doc:`../explanation/realtime-performance` for the timing boundaries required
to qualify a complete application.

Provenance and reproduction
---------------------------

The Three.js and GPU-IO comparison can be rerun from the repository root:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/compare_backends.py \
       --repeats 30 --warmups 5

Detailed browser campaign results and the four-target numerical comparison are
kept in ``DIC/LucasKanadeICGN/BENCHMARK.md``.

The SYCL implementation is maintained in the
`DIC_FROM_H264 project <https://github.com/jeffwitz/DIC_FROM_H264>`_. The
referenced checkout is commit
`77267f9 <https://github.com/jeffwitz/DIC_FROM_H264/commit/77267f92920ea8984533371cc6a62fef149ff6e2>`_,
and the standalone comparison source is
`icgn_sycl.cpp <https://github.com/jeffwitz/DIC_FROM_H264/blob/77267f92920ea8984533371cc6a62fef149ff6e2/experiments/codec_observation_closure/sycl_probe/icgn_sycl.cpp>`_.

UHD Three.js validation
-----------------------

The dynamic-resolution Three.js path was also validated at 3840 x 2160 on the
same machine. The offline reference tiles the shipped speckle texture and the
target applies an exact ``(2, -1)`` pixel translation. This produces a 240 x
135 grid, or 32,400 subsets. Each of three campaigns used five warm-ups and 30
measured optical-flow runs. The table reports the median campaign summary.

.. list-table:: UHD preparation and optical-flow measurements
   :header-rows: 1

   * - Phase
     - Boundary
     - Result
   * - Runtime construction
     - storage allocation and TSL graph construction
     - 5.1 ms
   * - Reference construction
     - grayscale filtering, gradients, Hessians, and metadata
     - 996.3 ms
   * - Browser preparation
     - runtime plus reference construction
     - 1,003.1 ms
   * - Complete Python preparation
     - conversion, PNG transport, and browser preparation
     - 1,291.3 ms
   * - Optical flow median
     - WebGPU prefilter, IC-GN, and median pass
     - **13.341 ms GPU; 14.7 ms browser wall**
   * - Optical flow p95
     - same boundary
     - **14.405 ms GPU; 16.0 ms browser wall**

The final registration accepted 31,654 subsets, recovered a mean translation
of ``(1.999991, -0.999997)`` pixels, required 5.943 iterations on average, and
produced a vector RMSE of ``3.076e-5`` pixel. The optimized subgroup solver is
6.13 times faster than the earlier 81.817 ms TSL baseline.

The equivalent current SYCL configuration was rebuilt and measured on the same
offline pair. It uses mixed precision, column layout, the FIR prefilter, a
constant inverse Hessian, no acceleration or tolerance guard, and median
replacement.

.. list-table:: UHD Three.js and SYCL comparison
   :header-rows: 1

   * - Measurement
     - Three.js/TSL
     - SYCL mixed/double reduction
     - SYCL float reduction
   * - Prefilter plus registration median
     - **13.341 ms**
     - 19.099 ms
     - 18.111 ms
   * - Prefilter plus registration p95
     - **14.405 ms**
     - 19.863 ms
     - 19.062 ms
   * - Upload plus compute plus download median
     - not included
     - 19.573 ms
     - 18.509 ms
   * - Accepted subsets
     - 31,654
     - 31,654
     - 31,654
   * - Mean iterations
     - 5.943
     - 5.944
     - 5.940
   * - Truth RMSE
     - 3.076e-5 px
     - 2.555e-5 px
     - 2.712e-5 px

At the comparable compute boundary, optimized Three.js uses 69.9% of the
mixed/double-reduction SYCL time: a 30.1% reduction, or 1.43 times the
throughput, on this machine. It uses 73.7% of the numerically closer
``--float-reduce`` time, a 26.3% reduction or 1.36 times the throughput. The
Three.js timer excludes input upload and output readback, so the separate SYCL
totals are shown rather than folded into those ratios. These values demonstrate
functional UHD support and backend compute performance, not acquisition-to-
result 4K60 qualification. WebGPU upload and pipeline compilation are lazy;
the five unreported warm-ups absorb first-use work.

Reproduce the campaign offline with:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src \
       python3 DIC/LucasKanadeICGN/benchmarks/benchmark_4k.py \
       --repeats 30 --warmups 5

