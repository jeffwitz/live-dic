Shared numerical kernel and adapter contract
============================================

The portable WebGPU path follows one rule: the affine IC-GN equations are
defined once, while each host owns only resource allocation, transfers,
dispatch, and result conversion. The canonical source is
``lucas_kanade_icgn/shaders/icgn.wgsl``. Both the raw browser WebGPU adapter
and wgpu-py load that file without generating or translating shader code.

Three.js remains the default because its subgroup-specialized TSL solver is
faster on the validation GPU. It shares browser reference preparation and the
same result contract, but keeps an optimized TSL implementation. GPU-IO is a
separate WebGL2 compatibility implementation because WebGL cannot execute
WGSL compute shaders.

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
                       |       | FIR, LK, median    |
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
wgpu-py use ``[0, 255]``. Normalized residuals make both conventions
mathematically equivalent, and cross-backend tests check the resulting fields.

Numerical contract
------------------

Geometry
~~~~~~~~

* subset size: 16 x 16 pixels;
* grid dimensions: ``floor(width / 16)`` by ``floor(height / 16)``;
* subset centre: ``(8 + 16 gx, 8 + 16 gy)``;
* local coordinates: ``[-7.5, 7.5]`` on each axis;
* affine parameter order: ``[tx, du/dx, du/dy, ty, dv/dx, dv/dy]``.

The convergence value is the maximum Euclidean displacement caused by the
increment at the four subset corners. The default threshold is ``1e-3`` pixel,
the default minimum ZNCC is ``0.8``, and the hard iteration limit is 30.

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
     - nine ``vec4`` values per subset
   * - 6
     - reference metadata
     - four ``vec4`` values per subset
   * - 7
     - solver state
     - two ``vec4`` values per subset
   * - 8
     - radius-15 FIR weights
     - 31 ``f32`` values
   * - 9
     - median-filtered state
     - two ``vec4`` values per subset

The two packed state vectors are ``[tx, du/dx, du/dy, ty]`` and
``[dv/dx, dv/dy, status, iterations]``. Hosts must not reinterpret this order.

Status contract
~~~~~~~~~~~~~~~

``1`` means accepted, ``2`` means accepted after spatial median replacement,
``0`` means the iteration limit was reached, and negative values mean rejected.
``-1`` is used for invalid geometry or a singular update; ``-2`` is used for
insufficient normalization energy or ZNCC.

Shared functions
----------------

The WGSL file separates the algorithm into ordinary functions:

* ``sample_cubic`` reconstructs the target intensity;
* ``solve_increment`` applies the inverse Hessian;
* ``compose_inverse_update`` composes the affine inverse update;
* ``corner_change`` evaluates convergence;
* ``convergence_status`` applies tolerance and ZNCC rules.

The solve entry point retains the GPU-specific organization: one 16-lane
workgroup owns one subset and reduces three packed partial-sum vectors. A
``workgroupUniformLoad`` makes the convergence decision uniform, so completed
subsets leave the iteration loop without violating barrier uniformity.

Change policy
-------------

A formula change that preserves the buffer contract belongs in the canonical
WGSL and, while Three.js remains specialized, in its corresponding named TSL
helper. A buffer-layout change must update both WebGPU hosts and the contract
tests. Transfer, allocation, browser, and NumPy conversion changes remain
local to their adapters. GPU-IO changes remain independently qualified against
the same image pairs.

The comparison scripts report status equality, jointly accepted subsets,
translation RMSE, and maximum absolute disagreement. Matching output types is
not considered sufficient: numerical agreement is part of the backend
contract.
