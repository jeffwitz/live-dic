Python supervision and browser-native execution
================================================

Product boundary
----------------

DIC-RT is a headless numerical engine intended to be supervised by Python.
The supervisor is not a graphical interface. It is the application layer that
owns:

* acquisition integration and frame timestamps;
* reference creation and replacement;
* interpolation, motion-model, pyramid, and convergence configuration;
* bounded queues, backpressure, and dropped-frame policy;
* engine lifetime, error reporting, and recovery;
* complete image-to-field and acquisition-to-result measurements;
* consumption, storage, or publication of NumPy result fields.

For a source that produces NumPy arrays and a consumer that needs complete
NumPy fields, the recommended deployment is
``ICGNWGPUFullPyramid``. It keeps the wgpu runtime in the Python process,
uploads the contiguous image directly to the GPU resource, and materializes
one packed result in Python. This is the architecture used by the current
4K60 compute objective.

Why the browser adapter remains slower
--------------------------------------

``ICGNWebGL`` keeps Chromium persistent and now transfers targets and results
as binary data. It no longer encodes each target as PNG, sends an image
through DevTools, decodes it in JavaScript, or returns the field as large JSON
arrays. Those changes reduce the complete UHD call from hundreds of
milliseconds to roughly 40--43 ms median on the validation machine.

Binary transport is not shared memory. A Python-supervised browser call still
performs this sequence:

.. code-block:: text

   NumPy uint8 frame
          |
          | inter-process binary copy
          v
   Chromium Uint8Array
          |
          | GPU upload
          v
   Three.js or raw WebGPU graph
          |
          | GPU readback + inter-process binary copy
          v
   NumPy parameter and status fields

The Three.js GPU graph can therefore be much faster than the complete
``engine.register(frame)`` call observed by Python. Kernel timestamps and
complete-call latency answer different questions.

What a JavaScript supervisor would change
-----------------------------------------

A competitive browser-native stream needs a persistent JavaScript supervisor
in the same process as Three.js. It should:

* receive browser-native video or image frames without a Python round trip;
* prepare the reference once and retain all GPU resources;
* submit frames through a bounded queue and define backpressure explicitly;
* keep intermediate and downstream processing on the GPU when practical;
* consume, reduce, visualize, or store results in JavaScript;
* send only configuration, health information, selected statistics, or
  deliberately sampled fields to Python.

Python can remain the experiment-level control plane: it may configure the
run, monitor health and timing, and collect reduced outputs. It should not be
the per-frame transport for every UHD image and complete field if browser
throughput is the objective.

This repository does not currently provide that JavaScript supervisor. The
browser implementations expose the numerical kernels and a headless Python
adapter, which are the prerequisites for it. Adding a supervisor would be a
separate integration layer and must be qualified at the same complete
acquisition-to-consumer boundary as the Python path.

Decision rule
-------------

.. list-table:: Choose the supervisor from the real data boundary
   :header-rows: 1
   :widths: 34 28 38

   * - Data ownership
     - Recommended execution
     - Reason
   * - NumPy input and complete NumPy output
     - Python + ``ICGNWGPUFullPyramid``
     - Avoids the Chromium process boundary and matches the intended product
       model.
   * - Browser-native input and browser-local output
     - JavaScript supervisor + Three.js or raw WebGPU
     - Keeps the stream and result consumers in the browser process; this
       supervisor remains future work.
   * - NumPy input with browser compatibility required
     - Python + ``ICGNWebGL``
     - Functional and useful for qualification, but the binary process copies
       remain in the complete latency.
   * - Historical GPU-IO reproduction
     - Python + explicit ``gpuio`` backend
     - Retained only for reproducibility and migration checks.

No choice removes the required CPU-to-GPU resource transfer unless acquisition
already produces a GPU resource that the selected runtime can consume. The
important optimization is to avoid unnecessary representation and process
boundaries, and to measure the boundary that the deployed application actually
uses.
