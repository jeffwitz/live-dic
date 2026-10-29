GPU tiling and grouping strategies
==================================

All backends evaluate the same affine IC-GN equations, but WebGPU and WebGL2
offer different synchronization models. This page distinguishes three notions
that older descriptions sometimes conflated: the 16 x 16 *numerical subset*,
the rectangular *FIR output tile*, and the one-dimensional *solver workgroup*.
They are independent scheduling choices.

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
     - one 16 x 16 subset
     - 16
     - workgroup-memory reduction
   * - Three.js specialized IC-GN on the validation GPU
     - one 16 x 16 subset
     - 32
     - one hardware subgroup, eight pixels per lane
   * - Pyramid reduction, each separable pass
     - 16 x 16 destination pixels
     - 16 x 16
     - five direct texture reads per output
   * - Seed transfer and median consistency
     - one subset per invocation
     - groups of 256
     - independent one-dimensional records

Common numerical tile
---------------------

One output grid point owns one 16 x 16 reference subset and one six-parameter
affine warp. In the default mode, the target value at each of the 256 pixels is
reconstructed from a 4 x 4 neighbourhood of cubic B-spline coefficients. The
direct ``cubic`` mode also uses 4 x 4 samples, while ``linear`` uses four. The
reference image, gradients, inverse Hessian, ``J^T F``, and ``sum(J)`` are
prepared once and reused for every target.

.. code-block:: text

   output grid point (gx, gy)
              |
              v
   reference tile: 16 columns x 16 rows

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

The canonical portable solver always uses the 16-lane mapping. Three shared
``vec4`` arrays hold the partial sums and a fixed reduction tree uses
workgroup barriers between its four stages:

.. code-block:: text

   16 values  -> 8 pairs -> 4 pairs -> 2 pairs -> 1 sum
      offset      8          4          2          1

   [0]+[8]   [0]+[4]    [0]+[2]    [0]+[1]    [0]
   [1]+[9]   [1]+[5]    [1]+[3]       ...       |
      ...       ...         ...                   v
                                               lane 0

The canonical WGSL uses ``workgroupUniformLoad`` to make the shared status a
provably uniform loop condition. A converged subset can therefore leave the
iteration loop early while every barrier that is reached remains in uniform
control flow. This path does not depend on a particular subgroup width and is
executed unchanged by raw browser WebGPU and wgpu-py.

Three.js selects the subgroup path only when the width probe permits it and the
production-pipeline validation agrees. ``EngineInfo.subgroup_probe_width``,
``subgroup_solver_width``, and ``subgroup_validation`` expose that decision;
``EngineInfo.solver`` reports ``portable-workgroup-subgroup-mismatch`` when the
check triggers the fallback. The raw browser adapter deliberately selects the
portable shared WGSL. Native
``ICGNWGPU`` goes one step further: during reference preparation it measures
the complete portable shared-WGSL and subgroup SPIR-V pipelines on a
representative shifted frame, then retains the faster one. This matters
because subgroup availability is a capability result, not a performance
guarantee. On the validation Intel driver the native SPIR-V subgroup pipeline
was slower, while the Three.js WGSL subgroup pipeline remained the browser's
faster implementation.

Shared WebGPU full pyramid: ordered GPU passes
-----------------------------------------------

The native ``ICGNWGPUFullPyramid`` and raw browser ``webgpu-wgsl`` paths follow
the established
coarse-to-fine pyramidal LK organization exposed by `NVIDIA VPI
<https://docs.nvidia.com/vpi/algo_optflow_lk.html>`_ and `OpenCV CUDA
<https://docs.opencv.org/4.x/d7/d05/classcv_1_1cuda_1_1SparsePyrLKOpticalFlow.html>`_.
They do not use one monolithic megakernel. Instead, all dependent compute
passes are recorded into one command encoder and submitted once for the
calculation:

.. code-block:: text

   finest uint8 target upload
             |
             +--> horizontal 5-tap reduction --> vertical 5-tap reduction
             |                                      |
             |                                      +--> next target level
             |                                              ...
             v
   per-level spline prefilters retained on the GPU
             |
             v
   coarsest solve + median consistency
             |
             v
   affine seed interpolation and scale conversion
             |
             v
   next solve + median --> ... --> finest solve + median
             |
             v
   one packed finest-level readback

Pass boundaries provide device-wide ordering without a host synchronization
point. Reference pixels, gradients, inverse Hessians, metadata, target spline
coefficients, affine states, and transferred seeds are allocated once and stay
on the same device. The two-level default uses ``(8, 16)`` maximum iterations
from coarse to fine. Three levels use the ``(6, 8, 16)`` schedule.
Setting one level creates only the original fine-level resources and
dispatches; no reduction or seed transfer is encoded.

The separable five-tap reduction stores its horizontal intermediate in a
single-channel ``r32float`` texture, using four bytes per intermediate pixel.
The input UNORM sample is first converted back to its integer gray level with
``round(sample * 255)``. Both convolution passes operate in units from 0 to 255,
like the CPU reference builder, and normalization is deferred until the final
8-bit texture write. The five-tap weights are multiples of ``1/16``; the two-pass
result is a multiple of ``1/256``, exactly representable in float32 in this
range. This avoids truncating a normalized sum that has rounded just below an
integer. The GPU regression tests compare the resulting pixels to the CPU
reduction. Both adapters bind the intermediate as unfilterable and read it
with ``textureLoad``.

A fresh solve writes an explicit state for every dispatched subset, including
disabled workgroups, so an untouched zero-filled buffer cannot masquerade as
a computed status-0 estimate. Selective retries use a separate entry point
sharing the same equations and preserve disabled subsets' existing states;
the median pass can therefore still use their accepted neighbours. Seed
interpolation ignores unavailable coarse subsets. If none of the four coarse
neighbours provides an estimate, the finer subset is still executed from a
zero affine seed instead of being suppressed.

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
invocation owns one subset and performs the complete 16 x 16 accumulation and
iteration loop serially. It calls the same TSL interpolation, Hessian solve,
inverse composition, convergence, and median helpers as the WebGPU variants,
but contains no workgroup arrays, barriers, or subgroup operations.

.. code-block:: text

   one transform-feedback invocation / subset
              |
              +--> serial 16 x 16 accumulation
              +--> affine inverse-compositional update
              +--> repeat until convergence or iteration ceiling
              |
              v
   translation + linear terms + status buffers

Three.js PBO-backed storage nodes provide indexed reads for the target
coefficients, reference fields, Hessians, and neighbouring subset states. The
horizontal prefilter, vertical prefilter, solver, and median kernels are grouped
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
all 16 columns and 16 rows serially, while the GPU still processes many
independent subsets in parallel.

.. code-block:: text

   fragment (gx, gy)
          |
          +--> scan row 0:  pixels 0 ... 15
          +--> scan row 1:  pixels 0 ... 15
          |          ...
          +--> scan row 15: pixels 0 ... 15
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
  16 x 16 scan.

The solver writes three multiple-render-target attachments only after the
subset converges, is rejected, or reaches the iteration limit. The following
median pass remains separate because it needs the completed neighbouring
subset field.

Why GPU-IO does not emulate WebGPU lanes
----------------------------------------

A WebGL2 emulation could render 16 fragments per subset, store nine partial
sums in several textures, then launch another draw to reduce and update the
warp. It would require at least two draws per IC-GN iteration and repeated
partial-sum/state traffic. There is no portable cross-fragment synchronization
that can keep the reduction and the next iteration inside one draw.

For this 16 x 16 problem the extra render-target bandwidth and roughly 60 draw
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
