Real-Time DIC in Python
=======================

.. important:: Foundational work by Pierre Margerit

   This work would not exist without `Pierre Margerit's original tests_gpuio
   project <https://github.com/MargeritPierre/tests_gpuio>`_. His GPU-IO
   experiments, repository structure, and WebGL groundwork are the foundation
   on which this Python real-time DIC implementation is built. The work
   documented here is a continuation of that foundation.

This documentation describes a Python workflow for GPU-accelerated digital
image correlation (DIC). The current implementation performs affine
inverse-compositional Gauss-Newton registration on 512 x 512 grayscale frames
through a headless WebGL2 backend.

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
* :doc:`explanation/realtime-performance` to understand what "real time" means
  for this pipeline.

Current scope
-------------

The implementation currently provides a fixed 32 x 32 grid of 16 x 16 subsets,
six affine parameters per subset, cubic B-spline target interpolation, ZNCC
validation, and median outlier replacement. The test dataset is offline; camera
acquisition and codec integration are deliberately outside the current path.

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

.. toctree::
   :hidden:
   :caption: Explanation

   explanation/pipeline
   explanation/icgn
   explanation/realtime-performance
   explanation/limitations
   explanation/acknowledgements
