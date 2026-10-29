Result layout and status values
===============================

Grid geometry
-------------

The frame is divided into ``floor(width / 16)`` by ``floor(height / 16)``
subsets. Each grid point owns a 16 x 16 subset centered at coordinates
``(8 + 16i, 8 + 16j)``. Border subsets that cannot support cubic interpolation
are rejected. A 512 x 512 frame produces a 32 x 32 grid; a 3840 x 2160 frame
produces a 240 x 135 grid.

Affine parameters
-----------------

``RegistrationResult.parameters`` has shape ``(grid_height, grid_width, 6)``.
The final axis is:

.. list-table:: Affine parameter order
   :header-rows: 1

   * - Index
     - Name
     - Meaning
   * - 0
     - ``tx``
     - horizontal translation
   * - 1
     - ``du_dx``
     - horizontal displacement gradient in x
   * - 2
     - ``du_dy``
     - horizontal displacement gradient in y
   * - 3
     - ``ty``
     - vertical translation
   * - 4
     - ``dv_dx``
     - vertical displacement gradient in x
   * - 5
     - ``dv_dy``
     - vertical displacement gradient in y

When an engine is constructed with ``motion_model="translation"``, this public
layout is unchanged. Indices 1, 2, 4, and 5 are exact zeros by construction;
only ``tx`` and ``ty`` are estimated. This preserves consumers that read
``displacement`` or export the full parameter array, but the zero gradients
must not be interpreted as measured zero strain.

``RegistrationResult.displacement`` selects indices 0 and 3 and has shape
``(grid_height, grid_width, 2)``.

Status field
------------

``RegistrationResult.status`` has shape ``(grid_height, grid_width)``.

.. list-table:: Status values
   :header-rows: 1

   * - Value
     - Meaning
     - Included by ``accepted_mask``
   * - ``2``
     - accepted after spatial median replacement
     - yes
   * - ``1``
     - converged and accepted by ZNCC
     - yes
   * - ``0``
     - iteration limit reached without convergence
     - no
   * - ``-1``
     - invalid geometry, border, or singular update
     - no
   * - ``-2``
     - insufficient intensity norm or failed ZNCC
     - no

Iteration field
---------------

``RegistrationResult.iterations`` stores the number of executed IC-GN
iterations for each subset. ``mean_iterations`` averages this field over
accepted subsets only.

Timing dictionary
-----------------

A normal Three.js ``register`` call reports these timing keys:

.. list-table:: Registration timing keys
   :header-rows: 1

   * - Key
     - Included work
   * - ``gpuMs``
     - WebGPU timestamp for target prefiltering, IC-GN, and median replacement
   * - ``wallMs``
     - browser-side synchronization around the same GPU pipeline
   * - ``encodingWallMs``
     - Python grayscale conversion and PNG encoding
   * - ``pythonCallWallMs``
     - complete Python ``register`` call, including transport and readback

GPU-IO returns its more detailed legacy stage keys. In all browser backends,
inspect the reported boundary before comparing values.

The native ``ICGNWGPU`` engine reports:

.. list-table:: Native registration timing keys
   :header-rows: 1

   * - Key
     - Included work
   * - ``uploadWallMs``
     - direct NumPy buffer to ``r8unorm`` texture submission
   * - ``executeReadbackWallMs``
     - FIR, IC-GN, median replacement, synchronization, and packed readback
   * - ``totalWallMs``
     - synchronous native call through result statistics and optional pyramid recovery

These are host wall-clock boundaries, not WebGPU timestamp-query durations.
For synchronous registration, ``totalWallMs`` starts before target validation
and conversion. Streaming uses a different boundary: its per-frame total runs
from slot submission through result construction and optional recovery. It
excludes the earlier target conversion, input-producer work, and time spent by
the consumer after the result is yielded. ``pipelineLatencyWallMs`` ends at
mapped readback and therefore excludes result statistics and recovery. Use an
external acquisition timestamp to measure end-to-end camera latency.
