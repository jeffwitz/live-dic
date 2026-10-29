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

   import json

   import numpy as np


   archive = np.load("results.npz")
   print(archive.files)
   print(archive["parameters"].shape)    # (3, 32, 32, 6)
   print(archive["displacement"].shape)  # (3, 32, 32, 2)
   config = json.loads(archive["config_json"].item())
   print(config["outlierPolicy"])

``config_json`` is a scalar JSON string rather than an object array, so the
archive remains readable with NumPy's default ``allow_pickle=False``. It
records whether the exported field was raw, flagged, replaced, or smoothed.

Export from Python
------------------

Keep the scientific fields and the validity information together:

.. code-block:: python

   import json


   np.savez_compressed(
       "one_result.npz",
       parameters=result.parameters,
       displacement=result.displacement,
       status=result.status,
       iterations=result.iterations,
       accepted=result.accepted,
       flagged=result.flagged,
       config_json=json.dumps(dict(result.config), sort_keys=True),
   )

Preserve measurement metadata
-----------------------------

For traceable experiments, store the reference identifier, target timestamp,
solver parameters, GPU name, package version, and timing boundary alongside the
arrays. A GPU timer value without its included stages is ambiguous and should
not be labeled as end-to-end latency.

See :doc:`../reference/result-layout` for field definitions.
