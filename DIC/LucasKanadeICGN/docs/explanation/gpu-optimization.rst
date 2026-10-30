GPU tiling and grouping strategies
==================================

If these terms are new to you, start with :doc:`webgpu-fundamentals` and
:doc:`../tutorials/first-gpu-kernel`. This page specifies the current live-dic
mapping. :doc:`../how-to/optimize-webgpu` explains how to choose and measure
such strategies, and :doc:`../reference/gpu-vocabulary` defines the terms.

All active backends evaluate the same selected IC-GN motion model, but WebGPU and WebGL2
offer different synchronization models. This page distinguishes three notions
that older descriptions sometimes conflated: the configurable *numerical subset*,
the rectangular *FIR output tile*, and the one-dimensional *solver workgroup*.
They are independent scheduling choices.

With ``motion_model="translation"``, the subset and workgroup mappings stay
unchanged. The specialized kernels omit four Jacobian products, the general
6 by 6 increment solve, affine composition, and four-corner convergence test.
Keeping the scheduling identical preserves pyramid and result-buffer
compatibility while reducing arithmetic inside each iteration.

.. list-table:: Current kernel mapping
   :header-rows: 1
   :widths: 27 22 20 31

   * - Kernel/path
     - Logical output
     - Threads
     - Memory/reduction strategy
   * - Shared WGSL FIR, horizontal
     - 32 x 8 pixels
     - 32 x 8
     - 56 x 8 shared source tile, 25 taps
   * - Shared WGSL FIR, vertical
     - 8 x 32 pixels
     - 8 x 32
     - 8 x 56 shared source tile, 25 taps
   * - Three.js WebGPU FIR
     - 16 x 16 pixels
     - 256 (one-dimensional)
     - 46 x 16 or 16 x 46 shared source tile, 31 taps
   * - Portable shared-WGSL IC-GN
     - one 8 x 8, 16 x 16, or 32 x 32 subset
     - 16
     - 4, 16, or 64 pixels per lane, then workgroup-memory reduction
   * - Three.js specialized IC-GN on the validation GPU
     - one selected square subset
     - 32
     - one hardware subgroup, size-specialized pixel distribution
   * - Pyramid reduction, each separable pass
     - 16 x 16 destination pixels
     - 16 x 16
     - five direct texture reads per output
   * - Seed transfer and displacement UOD
     - one subset per invocation
     - groups of 256
     - independent one-dimensional records

Common numerical tile
---------------------

One output grid point owns a square reference subset and a local model:
two translation, six affine, or twelve quadratic parameters. The following
example uses affine motion and size 16. The target value at each of the 256 pixels is
reconstructed from a 4 x 4 neighbourhood of cubic B-spline coefficients. The
direct ``cubic`` mode also uses 4 x 4 samples, while ``linear`` uses four. The
reference image, gradients, inverse Hessian, ``J^T F``, and ``sum(J)`` are
prepared once and reused for every target.

.. code-block:: text

   output grid point (gx, gy)
              |
              v
   reference tile for default S=16: 16 columns x 16 rows

        x=0   1   2                         15
      +----+----+----+--------------------+----+
   y=0| p  | p  | p  |        ...         | p  |
      +----+----+----+--------------------+----+
   y=1| p  | p  | p  |        ...         | p  |
      +----+----+----+--------------------+----+
       ...|                 256 pixels          |
   y=15   |                                     |
      +----+----+----+--------------------+----+

   each p -> affine warp -> 4 or 16 target values
          -> F, g, g^2, F*g, and six J*g contributions

The target FIR prefilter is separable. The coefficient vector contains the same
31 radius-15 weights as the optimized SYCL configuration. A horizontal pass
writes an intermediate image and a vertical pass produces the cubic
coefficient image, avoiding a 31 x 31 two-dimensional convolution.

The shared WGSL evaluates the central 25 taps. The six omitted outer weights
have a combined absolute magnitude below ``7e-8``; even for a 255-valued input,
their worst-case contribution is below ``1.8e-5`` grayscale unit per pass. The
opt-in GPU regression compares the resulting coefficients with an independent
CPU convolution. Three.js and its WebGL2 fallback retain all 31 taps.

The shared WGSL also stages each pass in workgroup memory. The horizontal pass
produces a 32 x 8 output tile from a 56 x 8 halo, while the vertical pass uses
an 8 x 32 output tile with an 8 x 56 halo. Thus 256 outputs require 448 source
reads instead of 6,400 uncached reads for a 25-tap pass.

.. code-block:: text

   horizontal FIR                         vertical FIR

   56 source columns x 8 rows             8 columns x 56 source rows
   +--------------------------+            +--------+
   | halo | 32 outputs | halo |            |  halo  |
   +--------------------------+            +--------+
                                               |
                                           8 x 32 outputs
                                               |
                                           +--------+
                                           |  halo  |
                                           +--------+

The TSL path uses a conservative 16 x 16 output tile: 46 x 16 source values
for the horizontal pass and 16 x 46 for the vertical pass. WebGL2 keeps its
serial prefilter because workgroup memory and barriers are not available
through that fallback.

Partial edge workgroups still execute every halo load before their barrier,
including a few cells that no valid output consumes. After applying the mirror
border rule, both tiled implementations clamp those speculative coordinates to
the image extent. Coordinates required by valid convolution outputs are already
in range after reflection, so this guard changes no coefficient used by the
solver. It only prevents an unused edge invocation from forming an out-of-range
buffer address on small images or pyramid levels.

This robustness guard is specialized when the pipeline is compiled rather
than evaluated for every UHD load. The shared WGSL hosts use the original fast
pipeline for axes of at least 44 pixels and compile a clamped variant only
when a configured pyramid contains smaller levels. Three.js makes the same decision while
building TSL because its image dimensions are host constants. The normal UHD
critical path therefore keeps the pre-existing filter instructions.

These FIR tiling rules apply only to ``interpolation="bspline"``. The direct
``linear`` and ``cubic`` graphs skip both passes. Their trade-off moves into
the solver: bilinear sampling performs four direct image reads per warped
pixel, while Keys cubic convolution performs 16. See :doc:`interpolation`.

WebGPU solver mappings
----------------------

The portable shared-WGSL solver assigns one workgroup of 16 active invocations to one subset.
Lane ``x`` owns column ``x`` and loops over the 16 rows. The 256 pixels are
therefore split evenly: every lane evaluates 16 warped pixels per iteration.
Different subsets are independent and are scheduled as separate workgroups.

.. code-block:: text

   one subset / one workgroup

          lane 0   lane 1   lane 2                 lane 15
             |        |        |                       |
             v        v        v                       v
   row  0   (0,0)    (1,0)    (2,0)       ...        (15,0)
   row  1   (0,1)    (1,1)    (2,1)       ...        (15,1)
     ...      |        |        |                       |
   row 15   (0,15)   (1,15)   (2,15)      ...        (15,15)
             |        |        |                       |
             +--------+--------+---------- ... --------+
                                  |
                           nine scalar sums
                                  |
                            lane 0 updates
                           the affine warp

Each lane accumulates nine quantities: ``sum(g)``, ``sum(g^2)``, ``sum(F*g)``,
and the six components of ``J^T g``. They are packed into three four-component
vectors so the reduction performs three vector operations rather than nine
independent scalar operations.

Three.js specialized subgroup reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At startup the runtime probes the hardware subgroup width. When a subgroup can
contain the subset, that result nominates a TSL kernel that fills the measured
subgroup rather than leaving lanes idle. During reference preparation, the
actual production solver writes its own ``subgroupSize`` to a diagnostic
buffer. The subgroup kernel is accepted only when that value matches the
candidate workgroup width; otherwise the runtime replaces it with the portable
workgroup-memory solver. This matters because a lightweight probe and a more
complex solver need not receive the same compiler-selected subgroup width. On
the validation adapter, both report 32 lanes, so each lane processes eight of
the 256 pixels. Subgroup collectives then reduce the partial vectors without
workgroup-memory round trips.

.. code-block:: text

   one hardware subgroup of lane-local partial vectors

   [00][01][02][03][04][05][06][07] ... [24][25][26][27][28][29][30][31]
     \_________________________________________________________________/
                     subgroupAdd(vec4) x 3
                                   |
                                   v
                     complete sums available to lane 0

The normalized residual, inverse-Hessian multiplication, inverse composition,
and corner test use the reduced values. Lane 0 writes the final state. All
IC-GN iterations remain inside one compute dispatch, so converged subsets stop
expensive image sampling without a new Python, JavaScript, or GPU submission.

Portable shared-WGSL workgroup reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native portable solver supports 16-, 32-, and experimental 128-lane mappings; 16 remains the
default. Three shared ``vec4`` arrays hold the partial sums and a binary
reduction tree uses workgroup barriers between its stages:

.. code-block:: text

   16 values  -> 8 pairs -> 4 pairs -> 2 pairs -> 1 sum
      offset      8          4          2          1

   32 values  -> 16 pairs -> 8 -> 4 -> 2 -> 1 sum
      offset       16        8    4    2

Use ``solver_workgroup_size=32`` to split each subset's pixel samples across
twice as many invocations. The browser shared-WGSL host keeps the default
16-invocation pipeline unless it supplies another pipeline override.
The experimental value ``128`` assigns two pixels per invocation for a 16 by
16 subset, but increases the reduction tree to seven stages instead of four.
It is an opt-in measurement candidate, not an automatic performance upgrade.

The canonical WGSL uses ``workgroupUniformLoad`` to make the shared status a
provably uniform loop condition. A converged subset can therefore leave the
iteration loop early while every barrier that is reached remains in uniform
control flow. This path does not depend on a particular subgroup width and is
executed by raw browser WebGPU with 16 invocations and by native wgpu-py with
the selected 16-, 32-, or 128-invocation mapping.

Three.js selects the subgroup path only when the width probe permits it and the
production-pipeline validation agrees. ``EngineInfo.subgroup_probe_width``,
``subgroup_solver_width``, and ``subgroup_validation`` expose that decision;
``EngineInfo.solver`` reports ``portable-workgroup-subgroup-mismatch`` when the
check triggers the fallback. The raw browser adapter deliberately selects the
portable shared WGSL. Native
``ICGNWGPU`` goes one step further: during reference preparation it measures
the complete portable and subgroup WGSL pipelines on a
representative shifted frame, then retains the faster one. This matters
because subgroup availability is a capability result, not a performance
guarantee. The native policy uses its own preparation-time measurements;
separate campaigns can select different winners when the medians are close.
The Three.js optimization is independently qualified and does not establish
the performance of the native specialization. See :doc:`native-backend`.

Shared WebGPU full pyramid: ordered GPU passes
-----------------------------------------------

The fixed pyramid described here remains the recommendation. The optional
native adaptive policy is a separate latency experiment; it does not change
the default graph and fails the current tensile accuracy check.

The native ``ICGNWGPUFullPyramid`` and raw browser ``webgpu-wgsl`` paths follow
the established
coarse-to-fine pyramidal LK organization exposed by `NVIDIA VPI
<https://docs.nvidia.com/vpi/algo_optflow_lk.html>`_ and `OpenCV CUDA
<https://docs.opencv.org/4.x/d7/d05/classcv_1_1cuda_1_1SparsePyrLKOpticalFlow.html>`_.
They do not use one monolithic megakernel. Instead, dependent compute
dispatches are recorded into one command encoder and submitted once for the
calculation:

.. code-block:: text

   finest uint8 target upload
             |
             +--> 5-tap image reduction --> next target level --> ...
             v
   per-level spline prefilters retained on the GPU
             |
             v
   coarsest solve + optional UOD
             |
             v
   polynomial seed interpolation, recentering, and scale conversion
             |
             v
   next solve + optional UOD --> ... --> finest solve + optional UOD
             |
             v
   one packed finest-level readback

Native batches the ordered dispatches into one compute pass, with buffer
clears before it and packed readback copies after it. The browser retains
separate compute passes. Each compute dispatch has its own WebGPU usage scope;
dependent dispatches do not require CPU synchronization. This is different
from putting the whole pyramid in one kernel: workgroup barriers cannot
synchronize separate workgroups inside a dispatch. Reference pixels,
gradients, inverse Hessians, metadata, target spline
coefficients, affine states, and transferred seeds are allocated once and stay
on the same device. The two-level default uses ``(8, 16)`` maximum iterations
from coarse to fine. Three levels use the ``(6, 8, 16)`` schedule.
Setting one level creates only the original fine-level resources and
dispatches; no reduction or seed transfer is encoded.

The browser's separable five-tap reduction stores its horizontal intermediate in a
single-channel ``r32float`` texture, using four bytes per intermediate pixel.
The input UNORM sample is first converted back to its integer gray level with
``round(sample * 255)``. Both convolution passes operate in units from 0 to 255,
like the CPU reference builder, and normalization is deferred until the final
8-bit texture write. The five-tap weights are multiples of ``1/16``; the two-pass
result is a multiple of ``1/256``, exactly representable in float32 in this
range. This avoids truncating a normalized sum that has rounded just below an
integer. The GPU regression tests compare the resulting pixels to the CPU
reduction. The browser binds the intermediate as unfilterable and reads it
with ``textureLoad``.

Native evaluates the identical filter in one ``reduce_fused`` dispatch per
level. A 16 x 16 invocation group cooperatively loads a reflected source tile
into shared memory, joins one barrier, then computes 256 reduced pixels.
The required footprint is 35 x 35 pixels, stored with padding in a 36 x 36
array of unsigned integer gray levels. The 5 x 5 product weights are integers
from the vector ``[1, 4, 6, 4, 1]``. Their weighted sum is at most 65,280;
integer division by 256 gives exactly the historical uint8 truncation. The
kernel therefore removes an intermediate texture and a dispatch without
changing reduced pixels. All lanes participate in loading and the barrier,
including partial image-edge tiles; every reflected load is bounded before
access. Regression tests compare constants, ramps, impulses, random textures
and odd dimensions against the independent CPU reduction.

A fresh solve writes an explicit state for every dispatched subset, including
disabled workgroups, so an untouched zero-filled buffer cannot masquerade as
a computed status-0 estimate. Selective retries use a separate entry point
sharing the same equations and preserve disabled subsets' existing states;
the UOD pass can therefore still use their accepted neighbours. Seed
interpolation ignores unavailable coarse subsets. If none of the four coarse
neighbours provides an estimate, the finer subset is still executed from a
zero motion-model seed instead of being suppressed.

Keep temporal counters local to a workgroup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The opt-in velocity path uses diagnostics kernels with 256 invocations per
workgroup, one per subset. This differs from the configurable IC-GN workgroup
(16 lanes by default, optionally 32) that
cooperatively processes the pixels of one subset. Prediction and history
update need integer counts, not a new floating-point reduction.

Incrementing one global counter for every accepted subset creates many writes
to the same address. The velocity path first counts in workgroup memory, then
one lane adds that group's total to global memory. For 32,400 fine subsets,
this reduces global increments to at most 127 per counter. It preserves exact
integer counts and does not change the order of the GN floating-point sums.

.. code-block:: text

   initialize local counter -> barrier
   each valid invocation copies/predicts its subset and increments local count
   barrier -> one lane adds the local total to the global counter

Unused invocations in the last group skip data accesses but still join both
barriers. The per-subset helper can return early; the compute entry point must
continue to the final barrier. Counts share the existing packed readback, so
there is no extra CPU/GPU synchronization. Fewer global operations do not
guarantee a faster kernel on every device: the isolated control preserves all
arrays and counts but shows substantial timing variation. The complete final
profile is qualified separately in :ref:`latency-reduction-profile`.

Iteration limits are ceilings rather than fixed costs: each subset exits as
soon as its corner-change and ZNCC criteria converge. The qualified defaults
are ``(8, 16)`` for two levels and ``(6, 8, 16)`` for three, with fine-scale
tolerance ``5e-3``. The former ``1e-3`` fine tolerance over-resolved the
quantized fractional validation images: it increased fine iterations without
improving displacement RMSE. Lower custom ceilings or tighter tolerances must
still be qualified against accepted subsets and displacement error. Reduction,
prefilter, seed-transfer, and dispatch work remains fixed even when iteration
ceilings are reduced.

Three.js/WebGL2: serial TSL through transform feedback
------------------------------------------------------

When WebGPU is unavailable, Three.js compiles a separate TSL solver to a
WebGL2 vertex program and captures its outputs with transform feedback. One
invocation owns one subset and performs the complete selected-square accumulation and
iteration loop serially. It calls the same TSL interpolation, Hessian solve,
inverse composition, convergence, and UOD helpers as the WebGPU variants,
but contains no workgroup arrays, barriers, or subgroup operations.

.. code-block:: text

   one transform-feedback invocation / subset
              |
              +--> serial S x S accumulation
              +--> affine inverse-compositional update
              +--> repeat until convergence or iteration ceiling
              |
              v
   translation + linear terms + status buffers

Three.js PBO-backed storage nodes provide indexed reads for the target
coefficients, reference fields, Hessians, and neighbouring subset states. The
horizontal prefilter, vertical prefilter, solver, and UOD kernels are grouped
in one ordered ``computeAsync`` call so buffer ping-pong remains visible between
passes without a JavaScript synchronization after each kernel.

This path exists for compatibility, not for 4K60. It avoids maintaining another
set of equations and is materially faster than the historical GPU-IO backend
on the validation machine, but it still exposes parallelism only between
subsets.

GPU-IO/WebGL2: one fused fragment per subset
--------------------------------------------

WebGL2 fragment shaders do not expose WebGPU workgroups, shared workgroup
memory, subgroup collectives, or a barrier between neighbouring fragments.
GPU-IO therefore maps one output fragment to one subset. That fragment scans
all selected columns and rows serially, while the GPU still processes many
independent subsets in parallel.

.. code-block:: text

   fragment (gx, gy)
          |
          +--> scan row 0:  pixels 0 ... S-1
          +--> scan row 1:  pixels 0 ... S-1
          |          ...
          +--> scan row S-1: pixels 0 ... S-1
          |
          +--> update six affine parameters
          +--> repeat in the same fragment until convergence
          |
          +--> one final MRT write: translation, linear part, status

The current implementation applies four WebGL-specific optimizations:

* all IC-GN iterations are fused into one draw call, so the warp stays in
  registers and intermediate state is not written to textures every iteration;
* the nine inverse-Hessian blocks and four metadata blocks are fetched once per
  subset and retained across iterations;
* cubic interpolation is evaluated as four horizontal dot products followed by
  one vertical combination;
* image statistics and all six ``J^T g`` terms are accumulated during the same
  square scan.

The solver writes three multiple-render-target attachments only after the
subset converges, is rejected, or reaches the iteration limit. The following
UOD pass remains separate because it needs the completed neighbouring
subset field.

Why GPU-IO does not emulate WebGPU lanes
----------------------------------------

A WebGL2 emulation could render 16 fragments per subset, store nine partial
sums in several textures, then launch another draw to reduce and update the
warp. It would require at least two draws per IC-GN iteration and repeated
partial-sum/state traffic. There is no portable cross-fragment synchronization
that can keep the reduction and the next iteration inside one draw.

For these small square subsets the extra render-target bandwidth and roughly 60 draw
submissions at the 30-iteration limit are a worse fit than the fused serial
fragment. The implementation therefore chooses fewer submissions and less
global memory traffic over artificial lane-level parallelism. This is the main
architectural reason the WebGL2 backend scales less effectively than WebGPU at
UHD.

Scaling consequences
--------------------

At 512 x 512 there are 1,024 subsets; at 3840 x 2160 there are 32,400. The
WebGPU backend exposes parallelism both *between* subsets and *within* each
subset. Both WebGL2 paths expose only the first level. The measured results on
the validation machine reflect that distinction: Three.js/WebGL2 and GPU-IO
remain compatibility paths at UHD, while cooperative Three.js/WebGPU is the
default performance path. See :doc:`../reference/performance` for the exact
timing boundaries and campaign results.

Neither strategy removes image upload, output readback, Python transport, or
camera latency. Those stages must be measured separately before claiming the
project's end-to-end 4K60 objective.
