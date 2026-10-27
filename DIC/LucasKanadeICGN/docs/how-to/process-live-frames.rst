Process live NumPy frames
=========================

Use one processing thread and one persistent :class:`~lucas_kanade_icgn.ICGNWebGL`
instance. The engine is intentionally not thread-safe.

Frame contract
--------------

Each frame must be exactly 512 x 512 pixels. Accepted inputs are:

* a two-dimensional grayscale array;
* an RGB array with shape ``(512, 512, 3)``;
* an RGBA array with shape ``(512, 512, 4)``;
* ``uint8`` or another numeric dtype convertible to ``uint8``.

Floating-point arrays with maximum value at or below 1 are treated as normalized
images and scaled to 8-bit grayscale.

Use a frame iterator
--------------------

Connect the acquisition layer to the DIC engine through an iterator or a bounded
queue. The DIC layer does not need to know which camera library produced the
array:

.. code-block:: python

   from collections.abc import Iterable

   import numpy as np

   from lucas_kanade_icgn import ICGNWebGL


   def process_frames(reference: np.ndarray, frames: Iterable[np.ndarray]) -> None:
       with ICGNWebGL() as engine:
           engine.set_reference(reference)

           for frame_index, frame in enumerate(frames):
               result = engine.register(frame)
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

The registration currently compares every target with a fixed reference, so
dropping a target does not corrupt a temporal solver state. It does change the
physical sampling times and must therefore be recorded.

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

The current transport still converts each NumPy frame to PNG before sending it
to Chromium. This is a known integration cost and must be included in real-time
qualification.
