Python-supervised real-time DIC toward 4K60
============================================

.. important:: Foundational work by Pierre Margerit

   This work would not exist without `Pierre Margerit's original tests_gpuio
   project <https://github.com/MargeritPierre/tests_gpuio>`_. His GPU-IO
   experiments, repository structure, and WebGL groundwork are the foundation
   on which this Python real-time DIC implementation is built. The work
   documented here is a continuation of that foundation.

New to GPU computing? Start with :doc:`explanation/webgpu-fundamentals`.
It assumes Python and numerical arrays, explains the technologies and memory
model, and leads to a runnable example, a glossary, and an optimization guide.

.. important:: Primary product boundary: Python supervision

   DIC-RT is designed to run inside a long-lived Python measurement
   application. Python owns reference changes, configuration, acquisition
   integration, bounded queues, timing, failure handling, and consumption of
   the NumPy displacement and parameter fields. For resident NumPy streams,
   ``ICGNWGPUFullPyramid`` is the recommended backend because WebGPU remains
   in-process.

   The browser adapters are maintained for portability, browser-native use,
   and numerical qualification. Their binary transport removes PNG and JSON,
   but it cannot share NumPy memory with Chromium. A competitive Three.js
   streaming deployment would require a persistent JavaScript supervisor to
   own browser-native frame ingestion, GPU scheduling, backpressure, and local
   result consumption. Python could configure and monitor that supervisor
   without carrying every UHD image and complete field. This JavaScript
   supervisor is not implemented here today. See
   :doc:`explanation/supervision`.

.. admonition:: Long-term target: qualified 4K at 60 FPS

   The engineering objective is sustained DIC on 3840 x 2160 frames at 60
   frames per second over the complete acquisition-to-result pipeline. On a
   changing four-image sequence, the recommended two-level native full
   pyramid measured 11.571 ms median / 12.185 ms p95 in affine mode and
   10.992 / 11.397 ms in translation mode. These resident-NumPy-image to
   materialized-NumPy-field boundaries are below the 16.667 ms frame period.
   Camera acquisition and downstream consumers are not included. The bounded
   two-slot full-pyramid stream sustains 95.51 FPS affine and 104.09 FPS
   translation, with 21.008 and 19.131 ms median result ages. 4K60 processing
   throughput is therefore demonstrated offline, while end-to-end acquisition
   remains to be qualified.

.. important:: Browser backend policy

   ``three-tsl`` is the supported backend for browser applications. It
   selects WebGPU first and automatically falls back to Three.js/WebGL2.
   ``ICGNWebGL`` is a Python compatibility and qualification adapter, not the
   recommended low-latency NumPy data plane.
   ``gpuio`` is deprecated and remains available temporarily only for
   historical reproducibility and replacement qualification. See
   :doc:`how-to/migrate-from-gpuio`.

.. important:: Displacement outlier policy

   The default ``outlier_policy="replace"`` applies a normalized UOD test only
   to recentered ``tx`` and ``ty``. A detected wrong-basin candidate receives
   status 2 and its complete motion state is replaced by neighbour medians.
   ``"flag"`` preserves the raw state while setting status 2; ``"disabled"``
   omits the post-pass. ``"median"`` is an explicit smoothing mode: it applies
   a centred 3 by 3 component median to every valid subset without assigning
   outlier status. Earlier coefficient-triggered modes were removed because
   they modified 66--81% of a healthy noisy field while pretending to detect
   outliers. The rule, validation contracts, and limitations are in
   :doc:`explanation/displacement-outliers`.

This documentation describes a Python-supervised workflow for GPU-accelerated digital
image correlation (DIC). The current implementation performs translation,
affine, or quadratic inverse-compositional Gauss-Newton registration on grayscale frames through
Three.js/TSL with WebGPU and WebGL2 backends, raw browser WebGPU with shared
WGSL kernels, and native wgpu-py backends. The deprecated GPU-IO/WebGL2 path is
retained as a historical comparison. Three.js is the default browser backend.
All paths have been validated at 3840 x 2160 pixels.

The recommended native coarse-to-fine engine is
``ICGNWGPUFullPyramid``, whose two-level default keeps reduction, prefiltering,
seed transfer, solving, and final readback on the GPU.  The older
``ICGNWGPU(pyramid_levels=2 or 3)`` selective-recovery mode remains available
only as an optional exception path: it first runs a fine solve and performs
CPU-orchestrated recovery for failures. The current changing-frame UHD
campaign measures the complete resident-image to materialized-field boundary;
kernel-only browser timings are retained only as diagnostics. These figures
qualify offline processing from a NumPy frame, not camera-to-consumer latency.

The engineering objective is sustained, measurable DIC processing from Python.
That means keeping the GPU runtime alive, preparing the reference only once,
separating compute time from transport and readback, and reporting the complete
latency of the deployed acquisition-to-result path.

Choose the section that matches your task:

* :doc:`explanation/webgpu-fundamentals` if GPU programming is new to you;
* :doc:`tutorials/first-gpu-kernel` to run a small WGSL calculation from Python;
* :doc:`how-to/optimize-webgpu` to develop and measure efficient kernels;
* :doc:`how-to/run-on-cpu` to select a software CPU adapter explicitly;
* :doc:`reference/gpu-vocabulary` for GPU terminology and source responsibilities;
* :doc:`tutorials/first-registration` if you are using the package for the first
  time;
* :doc:`tutorials/realtime-sequence` to build a persistent sequence loop;
* :doc:`how-to/process-live-frames` for NumPy frames from an acquisition source;
* :doc:`how-to/benchmark` to measure throughput and latency correctly;
* :doc:`reference/api` for the exact Python interface;
* :doc:`reference/performance` for the current native wgpu-py, raw-WGSL,
  Three.js, and bounded-sequence results on the validation machine;
* :doc:`explanation/backend-selection` for the exact defaults and recommended
  execution path for each public engine;
* :doc:`explanation/supervision` for the Python product boundary and the
  JavaScript supervisor required by a competitive browser-native stream;
* :doc:`explanation/gpu-optimization` for the backend tiling, lane grouping,
  and reduction strategies;
* :doc:`explanation/interpolation` for the selectable B-spline, bilinear, and
  Keys-cubic target reconstruction paths;
* :doc:`explanation/motion-models` for translation, affine, and quadratic solving;
* :doc:`explanation/displacement-outliers` for UOD replacement, flag-only,
  and disabled policies and their scientific validation contracts;
* :doc:`explanation/subset-size` for selectable subset size and overlapping stride;
* :doc:`explanation/shared-kernel` for the common WGSL source and its exact
  host/buffer contract;
* :doc:`explanation/native-backend` for the direct NumPy-to-WebGPU path and
  native WGSL subgroup qualification;
* :doc:`explanation/realtime-performance` to understand what "real time" means
  for this pipeline.

Current scope
-------------

The implementation provides a resolution-dependent grid of configurable 8 x 8,
16 x 16, or 32 x 32 subsets, with translation, six-parameter affine, or
twelve-parameter quadratic motion,
selectable B-spline, bilinear, or
Keys-cubic target interpolation, ZNCC validation, and optional displacement
outlier detection. The test dataset is offline;
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
   tutorials/first-gpu-kernel

.. toctree::
   :hidden:
   :caption: How-to guides

   how-to/install
   how-to/migrate-from-gpuio
   how-to/process-live-frames
   how-to/benchmark
   how-to/export-results
   how-to/optimize-webgpu
   how-to/run-on-cpu
   how-to/use-roi
   how-to/use-partial-roi
   how-to/compare-plate-hole-masks

.. toctree::
   :hidden:
   :caption: Reference

   reference/api
   reference/result-layout
   reference/command-line
   reference/performance
   reference/gpu-vocabulary

.. toctree::
   :hidden:
   :caption: Explanation

   explanation/pipeline
   explanation/webgpu-fundamentals
   explanation/backend-selection
   explanation/supervision
   explanation/icgn
   explanation/interpolation
   explanation/motion-models
   explanation/displacement-outliers
   explanation/subset-size
   explanation/gpu-optimization
   explanation/shared-kernel
   explanation/native-backend
   explanation/realtime-performance
   explanation/limitations
   explanation/acknowledgements
