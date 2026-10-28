GPU tiling and grouping strategies
==================================

Both backends evaluate the same affine IC-GN equations, but WebGPU and WebGL2
offer different synchronization models. This page explains how the 16 x 16
image subsets are tiled, how work is grouped, and why the two implementations
have different scaling at UHD resolution.

Common numerical tile
---------------------

One output grid point owns one 16 x 16 reference subset and one six-parameter
affine warp. The target value at each of the 256 pixels is reconstructed from a
4 x 4 neighbourhood of cubic B-spline coefficients. The reference image,
gradients, inverse Hessian, ``J^T F``, and ``sum(J)`` are prepared once and
reused for every target.

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

   each p -> affine warp -> 4 x 4 target-coefficient tile
          -> F, g, g^2, F*g, and six J*g contributions

The target FIR prefilter is separable. A horizontal radius-15 pass writes an
intermediate image and a vertical radius-15 pass produces the cubic coefficient
image. This avoids a 31 x 31 two-dimensional convolution and gives both
backends the same interpolation input.

Three.js/WebGPU: one cooperative group per subset
-------------------------------------------------

The WebGPU solver assigns one workgroup of 16 active invocations to one subset.
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

Fast subgroup reduction
~~~~~~~~~~~~~~~~~~~~~~~

At startup the runtime probes the hardware subgroup width. When a subgroup can
contain all 16 active lanes, subgroup collectives reduce the partial vectors
without workgroup-memory round trips.

.. code-block:: text

   16 lane-local partial vectors

   [00][01][02][03][04][05][06][07][08][09][10][11][12][13][14][15]
     \_____________________________________________________________/
                     subgroupAdd(vec4) x 3
                                   |
                                   v
                     complete sums available to lane 0

Lane 0 computes the normalized residual, multiplies by the constant inverse
Hessian, composes the inverse warp, and tests the four subset corners. The new
warp is broadcast to the group before the next iteration. All IC-GN iterations
remain inside one compute dispatch, so converged subsets stop expensive image
sampling without a new Python, JavaScript, or GPU submission.

Portable workgroup reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the subgroup probe fails, the same 16-lane tiling is retained but three
shared ``vec4`` arrays hold the partial sums. A fixed reduction tree uses
workgroup barriers between its four stages:

.. code-block:: text

   16 values  -> 8 pairs -> 4 pairs -> 2 pairs -> 1 sum
      offset      8          4          2          1

   [0]+[8]   [0]+[4]    [0]+[2]    [0]+[1]    [0]
   [1]+[9]   [1]+[5]    [1]+[3]       ...       |
      ...       ...         ...                   v
                                               lane 0

The barriers are executed uniformly even when a subset has converged. An
active flag prevents further interpolation and update work while preserving
legal workgroup synchronization. This path is slower than hardware subgroup
collectives but does not depend on a particular subgroup width.

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
subset. GPU-IO exposes only the first level. The measured results on the
validation machine reflect that distinction: GPU-IO remains a correct offline
and regression backend at UHD, while the cooperative Three.js solver is the
default performance path. See :doc:`../reference/performance` for the exact
timing boundaries and campaign results.

Neither strategy removes image upload, output readback, Python transport, or
camera latency. Those stages must be measured separately before claiming the
project's end-to-end 4K60 objective.
