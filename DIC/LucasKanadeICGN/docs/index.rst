Real-Time DIC in Python toward 4K60
===================================

.. important:: Foundational work by Pierre Margerit

   This work would not exist without `Pierre Margerit's original tests_gpuio
   project <https://github.com/MargeritPierre/tests_gpuio>`_. His GPU-IO
   experiments, repository structure, and WebGL groundwork are the foundation
   on which this Python real-time DIC implementation is built. The work
   documented here is a continuation of that foundation.

.. admonition:: Long-term target: qualified 4K at 60 FPS

   The engineering objective is sustained DIC on 3840 x 2160 frames at 60
   frames per second over the complete acquisition-to-result pipeline. The
   native wgpu-py complete Python-call boundary is 14.621 ms median and 15.748
   ms p95 on the validation machine. Both are below the 16.667 ms frame period,
   with 0.919 ms of p95 margin. A later documentation audit reproduced this at
   14.605 ms median and 15.018 ms p95.
   Direct upload, GPU computation, packed readback, and NumPy result
   construction are included; camera acquisition and downstream consumers are
   not. A bounded two-slot sequence reaches a 13.380 ms median period (74.7
   FPS) but 26.617 ms median result latency. 4K60 processing throughput is
   therefore demonstrated offline, while end-to-end acquisition remains to be
   qualified.

.. important:: Browser backend policy

   ``three-tsl`` is the supported browser backend for new applications. It
   selects WebGPU first and automatically falls back to Three.js/WebGL2.
   ``gpuio`` is deprecated and remains available temporarily only for
   historical reproducibility and replacement qualification. See
   :doc:`how-to/migrate-from-gpuio`.

This documentation describes a Python workflow for GPU-accelerated digital
image correlation (DIC). The current implementation performs affine
inverse-compositional Gauss-Newton registration on grayscale frames through
Three.js/TSL with WebGPU and WebGL2 backends, raw browser WebGPU with shared
WGSL kernels, and native wgpu-py backends. The deprecated GPU-IO/WebGL2 path is
retained as a historical comparison. Three.js is the default browser backend.
All paths have been validated at 3840 x 2160 pixels.

The native backend also offers opt-in selective pyramid recovery for subsets
that fail the initial fine solve. One scale remains the default fast path. A
separate ``ICGNWGPUFullPyramid`` backend now keeps the normal coarse-to-fine
path entirely GPU-resident and defaults to two levels. On the controlled
``(+4, -2)`` validation pair it
measured 14.811 ms at 4K with two levels and 15.458 ms with three levels after
reducing the three-level iteration ceilings. A later audit retained a 4K60 p95
margin for two levels, while the three-level median remained above 60 FPS but
its p95 did not. These figures qualify offline compute on that pair, not
camera-to-consumer latency.

The engineering objective is sustained, measurable DIC processing from Python.
That means keeping the GPU runtime alive, preparing the reference only once,
separating compute time from transport and readback, and reporting the complete
latency of the deployed acquisition-to-result path.

Choose the section that matches your task:

* :doc:`tutorials/first-registration` if you are using the package for the first
  time;
* :doc:`tutorials/realtime-sequence` to build a persistent sequence loop;
* :doc:`how-to/process-live-frames` for NumPy frames from an acquisition source;
* :doc:`how-to/benchmark` to measure throughput and latency correctly;
* :doc:`reference/api` for the exact Python interface;
* :doc:`reference/performance` for measured native wgpu-py, Three.js, GPU-IO,
  PyO3, and SYCL results on the validation machine;
* :doc:`explanation/gpu-optimization` for the backend tiling, lane grouping,
  and reduction strategies;
* :doc:`explanation/shared-kernel` for the common WGSL source and its exact
  host/buffer contract;
* :doc:`explanation/native-backend` for the direct NumPy-to-WebGPU path and
  the Rust/PyO3 milestone;
* :doc:`explanation/realtime-performance` to understand what "real time" means
  for this pipeline.

Current scope
-------------

The implementation currently provides a resolution-dependent grid of 16 x 16
subsets, six affine parameters per subset, cubic B-spline target interpolation,
ZNCC validation, and median outlier replacement. The test dataset is offline;
camera acquisition and codec integration are deliberately outside the current
path.

.. note::

   This is a soft real-time numerical pipeline, not a hard real-time system.
   GPU timing alone is not sufficient to claim a frame rate. Always measure the
   complete Python call and the acquisition-to-result latency on the target
   machine.

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/first-registration
   tutorials/realtime-sequence

.. toctree::
   :hidden:
   :caption: How-to guides

   how-to/install
   how-to/migrate-from-gpuio
   how-to/process-live-frames
   how-to/benchmark
   how-to/export-results

.. toctree::
   :hidden:
   :caption: Reference

   reference/api
   reference/result-layout
   reference/command-line
   reference/performance

.. toctree::
   :hidden:
   :caption: Explanation

   explanation/pipeline
   explanation/icgn
   explanation/gpu-optimization
   explanation/shared-kernel
   explanation/native-backend
   explanation/realtime-performance
   explanation/limitations
   explanation/acknowledgements
