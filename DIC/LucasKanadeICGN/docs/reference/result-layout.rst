Result layout and status values
===============================

Grid geometry
-------------

For ``subset_size=S`` and ``subset_stride=D``, the complete-subset grid is
``1 + floor((width - S) / D)`` by ``1 + floor((height - S) / D)``. Each point
owns an ``S x S`` subset centered at ``(S/2 + D*i, S/2 + D*j)`` in
pixel-edge coordinates. In zero-based pixel-index coordinates (integer pixel
centres), the same point is ``((S-1)/2 + D*i, (S-1)/2 + D*j)``; the browser
CSV and analytical fixture scoring use that convention. Canvas rendering adds
the half-pixel offset when placing these coordinates. The stride
defaults to the size. Border subsets that cannot support the selected
interpolation are rejected. With the default ``S=D=16``, a 512 x 512 frame
produces a 32 x 32 grid and a 3840 x 2160 frame produces a 240 x 135 grid.
``ReferenceInfo`` and ``RegistrationResult`` expose the effective
``subset_size`` and ``subset_stride``. See :doc:`../explanation/subset-size`.

Motion parameters
-----------------

``RegistrationResult.parameters`` has shape
``(grid_height, grid_width, parameter_count)``. Translation and affine
engines return six components; quadratic engines return twelve. The first six
components always have the following order:

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

``motion_model="quadratic"`` appends:

.. list-table:: Quadratic parameter extension
   :header-rows: 1

   * - Index
     - Name
     - Meaning at the subset centre
   * - 6
     - ``d2u_dx2``
     - second x derivative of horizontal displacement
   * - 7
     - ``d2u_dxdy``
     - mixed derivative of horizontal displacement
   * - 8
     - ``d2u_dy2``
     - second y derivative of horizontal displacement
   * - 9
     - ``d2v_dx2``
     - second x derivative of vertical displacement
   * - 10
     - ``d2v_dxdy``
     - mixed derivative of vertical displacement
   * - 11
     - ``d2v_dy2``
     - second y derivative of vertical displacement

The quadratic basis uses ``x^2/2``, ``xy``, and ``y^2/2``, so the
stored pure quadratic coefficients are second derivatives rather than doubled
polynomial coefficients. ``result.parameter_order`` gives the exact names for
the active result.

``RegistrationResult.displacement`` selects indices 0 and 3 and has shape
``(grid_height, grid_width, 2)``.

Status field
------------

``RegistrationResult.status`` has shape ``(grid_height, grid_width)``.

Native ROI results additionally use ``-3`` for geometrically excluded windows.
This is not a failed solve. ``roi_selected`` and ``reference_valid`` distinguish
the static geometric and numerical reference masks; ``populations`` labels the
denominators used for success rates. Undefined RMSE is ``None``. See
:doc:`../how-to/use-roi` for finite dense fields and self-contained NPZ exports.

.. list-table:: Status values
   :header-rows: 1

   * - Value
     - Meaning
     - Included by ``accepted_mask``
   * - ``2``
     - displacement outlier detected; state replaced or preserved according
       to ``outlierPolicy``
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

``outlier_policy="median"`` preserves these solver statuses while replacing
the active coefficients with a centred 3 by 3 component median. Status 2 is
therefore never introduced merely because explicit smoothing was requested.

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
     - WebGPU timestamp for target prefiltering, IC-GN, and optional UOD
   * - ``wallMs``
     - browser-side synchronization around the same GPU pipeline
   * - ``binaryInputWallMs``
     - browser-side materialization of the binary target body
   * - ``binaryPackingWallMs``
     - browser-side packing of the returned subset states (eight floats for
       translation/affine, sixteen for quadratic)
   * - ``inputPreparationWallMs``
     - Python validation and contiguous grayscale conversion before publication
   * - ``encodingWallMs``
     - compatibility key; zero for Three.js and raw WGSL binary targets
   * - ``resultBuildWallMs``
     - construction of writable NumPy component views and summary statistics
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
     - FIR, IC-GN, optional UOD, synchronization, and packed readback
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
