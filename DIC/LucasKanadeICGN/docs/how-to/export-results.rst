Export registration results
===========================

Export from the command line
----------------------------

Use ``--output`` to write a compressed NumPy archive:

.. code-block:: console

   $ icgn-webgl \
       DIC/LucasKanadeICGN/assets/frame_000.png \
       DIC/LucasKanadeICGN/assets/frame_001.png \
       DIC/LucasKanadeICGN/assets/frame_005.png \
       DIC/LucasKanadeICGN/assets/frame_010.png \
       --output results.npz

Load it with NumPy:

.. code-block:: python

   import numpy as np


   archive = np.load("results.npz")
   print(archive.files)
   print(archive["parameters"].shape)    # (3, 32, 32, 6)
   print(archive["displacement"].shape)  # (3, 32, 32, 2)

Export from Python
------------------

Keep the scientific fields and the validity information together:

.. code-block:: python

   np.savez_compressed(
       "one_result.npz",
       parameters=result.parameters,
       displacement=result.displacement,
       status=result.status,
       iterations=result.iterations,
       accepted=result.accepted,
       flagged=result.flagged,
   )

Preserve measurement metadata
-----------------------------

For traceable experiments, store the reference identifier, target timestamp,
solver parameters, GPU name, package version, and timing boundary alongside the
arrays. A GPU timer value without its included stages is ambiguous and should
not be labeled as end-to-end latency.

See :doc:`../reference/result-layout` for field definitions.
