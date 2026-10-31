Process live NumPy frames
=========================

Use one processing thread and one persistent engine instance. The engines are
intentionally not thread-safe. For NumPy camera frames and a normal
coarse-to-fine solve, prefer
:class:`~lucas_kanade_icgn.ICGNWGPUFullPyramid`; use
:class:`~lucas_kanade_icgn.ICGNWebGL` when browser or WebGL2 compatibility is
the deployment requirement.

This loop is the Python supervisor: it owns frame timestamps, backpressure,
reference lifetime, engine errors, and result consumption. If frames originate
and remain in a browser, do not route every frame through this NumPy loop merely
to call Three.js. The standalone PWA supplies a JavaScript supervisor for the shared raw-WebGPU
core to keep that
data plane browser-native; see :doc:`../explanation/supervision`.

Frame contract
--------------

With the default Three.js backend, frame dimensions are established by the
reference image and every target must match them. Accepted inputs are:

* a two-dimensional grayscale array;
* an RGB array with shape ``(height, width, 3)``;
* an RGBA array with shape ``(height, width, 4)``;
* ``uint8`` or another numeric dtype convertible to ``uint8``.

Floating-point arrays with maximum value at or below 1 are treated as normalized
images and scaled to 8-bit grayscale.

The optional GPU-IO backend also derives its allocation from the reference;
its width and height must be multiples of 16. Changing dimensions rebuilds its
private runtime page. Maximum dimensions depend on the WebGPU storage-buffer
or WebGL2 texture limits of the selected adapter.

Use a frame iterator
--------------------

Connect the acquisition layer to the DIC engine through an iterator or a bounded
queue. The DIC layer does not need to know which camera library produced the
array:

.. code-block:: python

   from collections.abc import Iterable

   import numpy as np

   from lucas_kanade_icgn import ICGNWGPUFullPyramid


   def process_frames(reference: np.ndarray, frames: Iterable[np.ndarray]) -> None:
       with ICGNWGPUFullPyramid() as engine:
           engine.set_reference(reference)

           for frame_index, result in enumerate(engine.register_stream(frames)):
               if result.mean_translation is None:
                   print(f"frame {frame_index}: no accepted subsets")
                   continue

               dx, dy = result.mean_translation
               print(
                   f"frame {frame_index}: dx={dx:.4f}, dy={dy:.4f}, "
                   f"accepted={result.accepted}"
               )

Keep acquisition bounded
------------------------

If acquisition can outrun DIC processing, use a bounded queue. Choose the
overflow policy from the experiment requirements:

* keep every frame when temporal sampling must be preserved, accepting growing
  latency or back-pressure;
* keep only the most recent frame when bounded latency matters more than a
  contiguous frame history;
* stop with an explicit error when dropping either data or latency would make
  the measurement invalid.

Registration compares every target with a fixed reference. With default zero
initialization, targets have no motion-history dependency. With optional
``ICGNWGPUFullPyramid(initialization="temporal")``, skipping frames increases
the motion change from the last processed image and can exceed the local
capture range. Keep processed targets chronological; record physical sampling
times and dropped frames. See :doc:`use-temporal-initialization`.

Refresh the reference deliberately
----------------------------------

Call ``set_reference(new_reference)`` again when the measurement protocol
requires a new reference state. This recomputes all reference-dependent
gradients, Hessians, and solver metadata. Do not place it in the per-frame loop
unless that cost and change of reference are intentional.

Avoid hidden latency
--------------------

For the lowest current latency:

* reuse one engine;
* provide contiguous arrays;
* avoid image display and file output in the processing loop;
* measure the outer Python call with ``time.perf_counter``;
* move logging and persistence to another consumer when possible.

The native backend reads a contiguous grayscale ``uint8`` NumPy buffer
directly during ``queue.write_texture``. ``np.ascontiguousarray`` returns the
original object when the input already satisfies that contract, so there is no
PNG or grayscale conversion in the normal camera path. One CPU-to-GPU upload
is still required because ordinary NumPy memory is not GPU texture memory.

The Three.js and raw WGSL browser transports send each contiguous grayscale
NumPy frame as unchanged binary bytes over the private loopback server and
return one packed binary field. They do not encode targets as PNG or return
large JSON arrays. Three.js/WebGPU additionally unpacks four 8-bit pixels per
word on the GPU instead of building a UHD float array in JavaScript.

This is still not shared memory: Chromium receives its own byte buffer, uploads
it to the GPU, and sends the result back to Python. Current UHD complete-call
measurements are about 45--47 ms median, so native wgpu remains the 60 Hz NumPy
stream path. Include this inter-process cost whenever a browser backend is
qualified.
