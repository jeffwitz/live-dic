Shared numerical kernel and adapter contract
============================================

This is an implementation reference. For the meaning of shaders, pipelines,
bind groups, and dispatches, first read :doc:`webgpu-fundamentals` and
:doc:`../reference/gpu-vocabulary`. To change a kernel, follow
:doc:`../how-to/optimize-webgpu`.

The portable WebGPU path follows one rule: the IC-GN equations are defined
once, while each host owns only resource allocation, transfers,
dispatch, and result conversion. The canonical sources are
``lucas_kanade_icgn/shaders/icgn.wgsl`` and ``full_pyramid.wgsl``. Both the raw
browser WebGPU adapter and wgpu-py load those files without generating or
translating shader code.

``icgn.wgsl`` also defines the ``INTERPOLATION_MODE`` pipeline override. Both
hosts compile the same solver body into B-spline, bilinear, and Keys-cubic
pipelines. Only the sampling function is specialized; increment solving,
inverse composition, convergence, status, and UOD rules remain single-source.
The ``MOTION_MODEL`` override similarly specializes the same source for the
two-parameter translation, six-parameter affine, or twelve-parameter
quadratic model. Pipeline specialization removes unused branches from the
per-subset loop.
``OUTLIER_POLICY`` specializes the displacement-only post-pass. ``replace``
returns complete-state neighbour medians after a ``tx`` or ``ty`` trigger;
``flag`` preserves raw coefficients and changes only the status. With
``disabled``, native and raw-browser hosts omit the dispatch and read the raw
solver state directly.

Three.js remains the default *browser entry point* because its
subgroup-specialized TSL solver is optimized for browser execution and it supplies
the automatic WebGL2 fallback. The recommended native coarse-to-fine entry
point is instead ``ICGNWGPUFullPyramid``. Three.js shares browser reference
preparation and the same result contract, but keeps an optimized TSL
implementation. TSL now also
compiles a serial solver to GLSL transform feedback when Three.js falls back to
WebGL2. The deprecated GPU-IO backend remains a separate historical
implementation because WebGL cannot execute the canonical WGSL compute
shaders directly. It is not a target for further kernel development.

This source-sharing policy is independent of supervision. The current product
supervisor is Python, so native wgpu is the low-latency NumPy path. Exploiting
Three.js kernel performance without the Python/Chromium copies requires a
persistent JavaScript supervisor and browser-local frame/result ownership; see
:doc:`supervision`.

Architecture
------------

.. code-block:: text

                    grayscale reference image
                              |
                              v
             +----------------------------------+
             | shared browser reference builder |
             | F, gradients, H^-1, metadata     |
             +----------------------------------+
                       |                 |
                       v                 v
             +----------------+  +------------------+
             | Three.js / TSL |  | raw browser     |
             | subgroup path  |  | WebGPU adapter  |
             +----------------+  +------------------+
                       |                 |
                       |                 | loads
                       |                 v
                       |       +--------------------+
                       |       | canonical WGSL     |
                       |       | reduce, sample, LK,|
                       |       | seed, UOD          |
                       |       +--------------------+
                       |                 ^
                       |                 | loads
                       |       +--------------------+
                       |       | wgpu-py adapter    |
                       |       | direct NumPy I/O   |
                       |       +--------------------+
                       |                 |
                       +--------+--------+
                                v
                      common NumPy result schema

The shared browser preparation module is
``web/common/ReferencePreparation.js``. It is parameterized by intensity
scale: Three.js uses normalized intensities, while the canonical WGSL and
wgpu-py use ``[0, 255]``. The raw adapter builds reduced references once with
the same mirrored five-tap filter as the WGSL target reductions. Normalized
residuals make both intensity conventions mathematically equivalent, and
cross-backend tests check the resulting fields.

Numerical contract
------------------

Geometry
~~~~~~~~

* shared-WGSL subset width ``S``: 8, 15, 16, 17, 31, 32, or 33 pixels,
  default 16;
* shared-WGSL subset stride ``D``: integer from 1 through ``S``, default ``S``;
* grid dimensions: ``1 + floor((width - S) / D)`` by
  ``1 + floor((height - S) / D)``;
* subset centre: ``(S/2 + D gx, S/2 + D gy)``;
* local coordinates: ``[-(S-1)/2, +(S-1)/2]`` on each axis;
* integration weights: uniform by default, or a centred Gaussian in native
  wgpu with an explicit sigma in pixels;
* affine parameter order: ``[tx, du/dx, du/dy, ty, dv/dx, dv/dy]``.
* quadratic extension: ``[d2u/dx2, d2u/dxdy, d2u/dy2, d2v/dx2,
  d2v/dxdy, d2v/dy2]``.

The translation model retains this layout but only indices 0 and 3 are active;
the other four entries are zero. Its convergence value is the Euclidean norm
of the translation increment. See :doc:`motion-models`.

Quadratic mode uses the same first six entries and appends the second
derivatives. Its geometry and convergence checks sample a 3 by 3 set over the
subset so interior curvature is represented, not only corner motion.

The convergence value is the maximum Euclidean displacement caused by the
increment at the four subset corners. Single-scale engines default to a
``1e-3`` pixel threshold and a hard limit of 30 iterations. Full-pyramid hosts
provide their qualified per-level schedules, currently ``5e-3`` at the fine
level with a limit of 16 iterations. The default minimum ZNCC is ``0.8``.

GPU buffers
~~~~~~~~~~~

.. list-table:: Canonical WGSL buffer layout
   :header-rows: 1

   * - Binding
     - Data
     - Layout
   * - 0
     - target image
     - ``texture_2d<f32>``, ``r8unorm``
   * - 1
     - configuration
     - six ``u32`` followed by two ``f32`` values, 32 bytes
   * - 2
     - horizontal FIR temporary
     - one ``f32`` per pixel
   * - 3
     - cubic target coefficients
     - one ``f32`` per pixel
   * - 4
     - reference data
     - ``vec4(F, gx, gy, padding)`` per pixel
   * - 5
     - inverse Hessian
     - 1, 9, or 36 ``vec4`` values per subset for translation, affine, or quadratic
   * - 6
     - reference metadata
     - two, four, or seven ``vec4`` values per subset
   * - 7
     - solver state
     - two ``vec4`` values, or four in quadratic mode
   * - 8
     - radius-15 FIR weights and optional native partial supports
     - ``u32`` words: 31 float bit patterns, then optional packed geometry
   * - 9
     - UOD output state
     - same layout as solver state
   * - 10, optional
     - coarse-to-fine seed state
     - two or four read-only ``vec4`` values per subset
   * - 11, optional
     - active/retry mask for seeded solves
     - one read-only ``u32`` per subset

Translation and affine states use ``[tx, du/dx, du/dy, ty]`` followed by
``[dv/dx, dv/dy, status, iterations]``. Quadratic states use three parameter
vectors in documented public order and a fourth
``[status, iterations, padding, padding]``. Hosts must not reinterpret this order.
The full 31-value FIR record preserves the common buffer contract. The current
WGSL prefilter reads its central 25 entries; the six omitted tails have a total
absolute weight below ``7e-8``.

The first 31 words at binding 8 retain exactly the previous float bytes; only
these words are bitcast to float by the shader. Native partial ROI preparation
can append the partial count, dense subset indices and packed binary supports.
There is no additional binding. Complete-window pipelines compile without
per-pixel mask tests. Native ROI metadata appends neighbourhood permissions and,
for partial windows, support bounding boxes after the ordinary reference records.
See :doc:`../how-to/use-partial-roi` for this explicitly enabled variant.

Status contract
~~~~~~~~~~~~~~~

``1`` means accepted, ``2`` means a displacement outlier was detected,
``0`` means the iteration limit was reached, and negative values mean rejected.
Status 2 may contain a neighbour-median replacement or the raw solver state,
depending on ``outlierPolicy``.
The explicit ``median`` policy preserves the incoming solver status even though
it replaces every active coefficient with a centred 3 by 3 component median.
``-1`` is used for invalid geometry or a singular update; ``-2`` is used for
insufficient normalization energy or ZNCC.

Shared functions
----------------

The WGSL file separates the algorithm into ordinary functions:

* ``sample_cubic`` reconstructs the target intensity;
* ``solve_increment`` applies the inverse Hessian;
* ``compose_inverse_update`` composes translation or affine inverse updates;
* ``compose_quadratic_inverse_update`` composes the exact lifted second-order update;
* ``corner_change`` evaluates convergence;
* ``convergence_status`` applies tolerance and ZNCC rules.

The portable solve entry point retains the GPU-specific organization: one
16- or 32-lane workgroup owns one subset and reduces three packed partial-sum
vectors. Sixteen is the default; native callers select the width with
``solver_workgroup_size``. Quadratic mode then reuses two of those vectors for
a second reduction of its six curvature products. It deliberately does not
declare a fourth reduction buffer: doing so reduced affine occupancy on the
validation GPU even when the pipeline override selected affine mode. With the
default 16 lanes, each lane processes 4, 16, or 64 pixels for ``S=8``, ``16``,
or ``32``; 32 lanes halve those per-lane counts. A
``workgroupUniformLoad`` makes the convergence decision uniform, so completed
subsets leave the iteration loop without violating barrier uniformity.

The shared ``solve_seeded`` entry point calls the same ``solve_subset`` body
with bindings 10 and 11. The ordinary ``solve`` entry point does not reference
those bindings, so one-scale browser and native pipelines retain their original
ten-entry bind-group layout. Optional selective recovery allocates seed
resources lazily; the recommended full-pyramid browser and native adapters
allocate them once for every finer level. The ``retry_seeded`` entry point uses the same equations
and seed bindings, but preserves the existing state of disabled subsets.
Selective recovery needs those already accepted neighbours during the UOD
pass. Fresh ``solve_seeded`` dispatches instead write an explicit invalid state
for a disabled subset, so a cleared buffer cannot be mistaken for a computed
partial estimate.

``full_pyramid.wgsl`` orchestrates the pyramid without duplicating IC-GN
equations. The browser uses ``reduce_x`` and ``reduce_y``; native uses the
pixel-equivalent tiled ``reduce_fused`` entry point without an intermediate
texture. ``transfer_seed`` interpolates, recentres and rescales the coarser
polynomial field; its ROI variant applies prepared link permissions.
Native temporal entry points project and merge accepted history. The optional
``refine_temporal_seed`` entry point applies the measured coarse change to
fine history when complete age-matched stencils exist. All numerical solve
operations remain in ``icgn.wgsl``. Native records ordered dispatches in one
compute pass; the browser retains its separate compute passes. Both submit
one command buffer and read back only the final field.

Change policy
-------------

A formula change that preserves the buffer contract belongs in the canonical
WGSL and, while Three.js remains specialized, in its corresponding named TSL
helper. The Three.js WebGPU and WebGL2 solvers call those same TSL helpers but
keep different accumulation organizations. The native subgroup solver is also
a specialized implementation: its ``icgn_subgroup.wgsl`` source must be
updated and qualified whenever an LK formula changes. A buffer-layout change
must update both WebGPU hosts and the contract tests.
Transfer, allocation, browser, and NumPy conversion changes remain local to
their adapters. GPU-IO changes remain independently qualified against the same
image pairs until its retirement.

The comparison scripts report status equality, jointly accepted subsets,
translation RMSE, and maximum absolute disagreement. Matching output types is
not considered sufficient: numerical agreement is part of the backend
contract.

The native autotuner applies that rule before considering timings. It executes
the portable and subgroup solvers on the same shifted reference, rejects
non-finite results, incompatible accepted-subset masks, or affine fields beyond
the documented tolerance, and times only the remaining candidates. This check
is performed during reference preparation and adds no work to registration.

The current guard permits at most ``max(1, ceil(0.005 * grid_count))`` differing
accepted-subset flags. On jointly accepted subsets, the aggregate RMSE over
the six affine parameters must not exceed ``2e-3`` and the maximum absolute
difference must not exceed ``2e-2``. Translation and gradient components have
different units: these are implementation-compatibility thresholds, not a
displacement accuracy specification. If the portable path accepts no subsets,
the candidate must also accept none. Independent known-motion validation is
still required.
