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
     - WebGPU, Vulkan
     - GPU prefilter, IC-GN, median pass
     - **3.461 ms**
     - **3.713 ms**
     - 900 / 1,024
     - 4.076e-5 px
   * - GPU-IO
     - WebGL2, OpenGL
     - GPU prefilter, IC-GN, median pass
     - 3.794 ms
     - 4.160 ms
     - 900 / 1,024
     - 3.819e-5 px
   * - DIC_FROM_H264 SYCL
     - DPC++, Level Zero
     - upload, prefilter, IC-GN, readback
     - 3.651 ms
     - 4.280 ms
     - 900 / 1,024
     - 1.12e-5 px

The Three.js/TSL median GPU time is 8.8% lower than GPU-IO for this case. The
three campaign ratios were 91.0%, 91.3%, and 91.2%. Tail latency did not always
improve: the Three.js p95 ranged from 3.707 to 4.562 ms, while GPU-IO ranged
from 4.128 to 4.326 ms.

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
     - **5.6 ms**
     - **6.2 ms**
   * - GPU-IO
     - 16.6 ms
     - 19.2 ms
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

