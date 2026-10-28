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
   with 0.919 ms of p95 margin.
   Direct upload, GPU computation, packed readback, and NumPy result
   construction are included; camera acquisition and downstream consumers are
   not. 4K60 is therefore still a target rather than an end-to-end
   qualification.

This documentation describes a Python workflow for GPU-accelerated digital
image correlation (DIC). The current implementation performs affine
inverse-compositional Gauss-Newton registration on grayscale frames through
GPU-IO/WebGL2, Three.js/TSL WebGPU, raw browser WebGPU with a shared WGSL
kernel, and native wgpu-py backends. The default remains Three.js. All four
paths have been validated at 3840 x 2160 pixels.

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
