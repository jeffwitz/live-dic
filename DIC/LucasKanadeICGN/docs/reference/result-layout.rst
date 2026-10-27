Result layout and status values
===============================

Grid geometry
-------------

The fixed 512 x 512 frame is divided into a 32 x 32 grid with 16-pixel spacing.
Each grid point owns a 16 x 16 subset centered at coordinates
``(8 + 16i, 8 + 16j)``. Border subsets that cannot support cubic interpolation
are rejected.

Affine parameters
-----------------

``RegistrationResult.parameters`` has shape ``(32, 32, 6)``. The final axis is:

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

``RegistrationResult.displacement`` selects indices 0 and 3 and has shape
``(32, 32, 2)``.

Status field
------------

``RegistrationResult.status`` has shape ``(32, 32)``.

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

A normal ``register`` call reports browser-side stage timings:

.. list-table:: Registration timing keys
   :header-rows: 1

   * - Key
     - Included work
   * - ``prefilterWallMs``
     - target upload synchronization and cubic-coefficient prefilter
   * - ``solveWallMs``
     - IC-GN solve and median replacement, including GPU wait
   * - ``solveGpuMs``
     - GPU timer query for the solve
   * - ``readbackMs``
     - result texture readback inside JavaScript
   * - ``totalWallMs``
     - browser-side upload, prefilter, solve, and readback after image decoding

These values exclude Python preprocessing, PNG encoding, DevTools transport,
and JavaScript image decoding. Measure the outer Python call for application
latency.
