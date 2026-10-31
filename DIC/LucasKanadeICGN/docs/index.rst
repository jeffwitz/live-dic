live-dic: real-time DIC from Python
===================================

.. important:: Foundational work by Pierre Margerit

   live-dic would not exist without `Pierre Margerit's original tests_gpuio
   project <https://github.com/MargeritPierre/tests_gpuio>`_, its GPU-IO
   experiments and WebGL groundwork. This project continues that foundation.

live-dic is a numerical engine for Python measurement applications. A
persistent Python supervisor provides NumPy images, manages references and
bounded work, and consumes displacement, deformation and curvature fields.
The recommended engine is **ICGNWGPUFullPyramid**: native wgpu-py executes
WGSL on the GPU without Chromium or per-frame image encoding.

.. admonition:: Engineering goal: qualified 4K at 60 FPS

   The objective is sustained processing of 3840 x 2160 images at 60 FPS,
   with both throughput and result age measured. Native benchmarks have
   executed on tested AMD, Intel and NVIDIA GPUs under Linux/Vulkan.
   Coverage differs by workload and adapter; see :ref:`gpu-hardware-coverage`.
   The numerical-engine boundary starts with a resident NumPy image and ends
   with the materialized NumPy field. Camera acquisition and downstream
   processing must also be measured before claiming a deployed 4K60 system.

Start here
----------

* :doc:`how-to/install` to install native dependencies and optional browser support.
* :doc:`tutorials/first-registration` for the first native Python registration.
* :doc:`tutorials/realtime-sequence` and :doc:`how-to/process-live-frames` for
  persistent NumPy streams.
* :doc:`how-to/use-roi` for a boolean reference mask; white/True retains pixels.
  The default full-window policy rejects any subset touching a forbidden pixel.
  :doc:`how-to/use-partial-roi` covers explicit boundary supports.
* :doc:`reference/api` for exact defaults and public interfaces.

New to GPU programming? Read :doc:`explanation/webgpu-fundamentals`, then
:doc:`tutorials/first-gpu-kernel`, :doc:`reference/gpu-vocabulary`, and
:doc:`how-to/optimize-webgpu`. These assume engineering and Python knowledge,
not graphics-programming experience.

Choose an execution path
-------------------------

``ICGNWGPUFullPyramid()`` is the native coarse-to-fine path. Two fixed levels
are the default; every configured level runs for every target. Native
``ICGNWGPU()`` is single-scale by default, with optional selective recovery
retained as a separate compatibility strategy. Both support bounded streaming.
One slot minimizes result age; the API default of two prioritizes throughput.

``ICGNWebGL()`` retains Three.js/TSL as its browser default, with WebGPU and
WebGL2 fallback. Raw browser ``webgpu-wgsl`` loads the common WGSL full
pyramid. GPU-IO is deprecated. Browser ROI requests are rejected explicitly.
The existing CLI and one-shot helper remain browser interfaces; their defaults
have not changed. See :doc:`explanation/backend-selection`.

Browser binary transport removes per-frame PNG and JSON encoding, but still
copies images and fields between Python and Chromium. A competitive
browser-native Three.js stream would need its own JavaScript supervisor,
which is not implemented here. See :doc:`explanation/supervision`.

Current numerical scope
-----------------------

Native engines estimate translation, six-parameter affine motion (default),
or twelve-parameter quadratic motion. Native subset widths are 8, 15, 16,
17, 31, 32 and 33, with independently configurable stride and uniform or
Gaussian integration. B-spline is the default interpolator; bilinear and
Keys cubic are explicit alternatives. ROI supports translation and affine.

The default spatial policy detects displacement outliers and replaces the
complete local state. Flag-only and raw modes preserve different measurement
contracts. Explicit ``outlier_policy="median"`` smooths valid states; it is
not outlier detection. See :doc:`explanation/displacement-outliers`.
Temporal initialization, velocity prediction and finest-level stopping are
opt-in; see :doc:`how-to/use-temporal-initialization`.

Use native engines as context managers or close them explicitly. Native
close is permanent and idempotent and releases the owning GPU resources;
shared pyramid levels preserve their parent's device. Result arrays remain
usable after close. See :doc:`reference/api`.

Evidence and limits
--------------------

:doc:`reference/performance` is the canonical location for timing tables,
scientific populations and hardware coverage. It distinguishes default
backend comparisons, analytic temporal trajectories, fixed-pair experiments
and rejected prototypes. :doc:`how-to/benchmark` explains measurement.

These are soft real-time pipelines. Portable CI, software Vulkan execution,
hardware correctness and hardware performance are different qualifications.
No current result certifies every GPU, OS or architecture. Consult
:doc:`explanation/limitations` and retain the configuration and validity masks
when exporting a measurement with :doc:`how-to/export-results`.

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
   how-to/use-temporal-initialization
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
   explanation/browser-portability
   explanation/temporal-initialization
   explanation/realtime-performance
   explanation/limitations
   explanation/acknowledgements
