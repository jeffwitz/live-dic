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


   with np.load("results.npz", allow_pickle=False) as archive:
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

   result.save_npz("one_result.npz")

Use this method to retain parameters, status, iterations, parameter ordering,
configuration and population counts, plus the reference mask and grid selection
when a ROI is present. The JSON fields use strict serialization; the archive
opens with ``allow_pickle=False``. Exporting only displacement loses the local
model, validity and ROI provenance.

.. code-block:: python

   with np.load("one_result.npz", allow_pickle=False) as archive:
       parameters = archive["parameters"]
       accepted = archive["status"] > 0.5
       config = json.loads(archive["config_json"].item())
       print(config["motionModel"], config["outlierPolicy"])
       print(parameters[accepted].shape)

Status 2 is accepted by the engine's mask, but its interpretation depends on
``outlierPolicy``: replaced with ``replace``, retained raw with ``flag``.
``median`` repairs rejected parameter slots from valid neighbours but preserves
their solver statuses; at coarse pyramid levels a private marker makes the
repair seed-eligible. It does not smooth accepted measurements. Keep that
policy with every field. See :doc:`../explanation/displacement-outliers`.

The complete :download:`sequence example <../examples/register_sequence.py>`
writes one self-contained NPZ per target, a source/configuration manifest and a
CSV index without accumulating the sequence in RAM.

Preserve measurement metadata
-----------------------------

For traceable experiments, store the reference identifier, target timestamp,
solver parameters, GPU name, package version, and timing boundary alongside the
arrays. A GPU timer value without its included stages is ambiguous and should
not be labeled as end-to-end latency.

See :doc:`../reference/result-layout` for field definitions.
