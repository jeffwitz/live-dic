Prepare a changing reference on GPU
===================================

In inverse compositional Gauss–Newton, gradients and the inverse Hessian belong
to the reference image. They can be reused for a fixed reference. Comparing
consecutive images changes that reference on every pair: copying the old
Hessian would change the numerical problem. The first browser implementation
rebuilt it on CPU, including readback, filtering, inversion and allocation.
This dominated the observed 1080p consecutive-camera cost.

The optional ``referencePreparation: 'gpu'`` graph moves that preparation to
portable WGSL. The IC-GN solver is unchanged. CPU preparation remains the
default; the new mode is experimental. See :doc:`../how-to/run-in-browser` to
activate it and :doc:`../reference/js-api` for ``registerPair``.

What is prepared and when
--------------------------

Cold ``setReference`` computes geometry, full-ROI selections, conservative
neighbour/seed permissions and buffer offsets. It creates every pipeline,
texture, uniform and scratch buffer. These static objects persist across pairs.
An explicit reference/ROI reconfiguration or applying changed Solve settings
rebuilds them. Starting with unchanged settings reuses them.

A hot pair uses this ordered graph in one command encoder/submission::

   Previous frame -> independent reference texture
                         |
                 build reference pyramid
                         |
            for each level: prepare intensity
            -> B-spline coefficients, if selected
            -> gradients -> mean, products, H and inverse H
                         |
   Current frame -> target pyramid -> unchanged coarse-to-fine IC-GN
                         -> configured spatial policy
                         -> final field/count readback -> display/recording

The reference and target textures are distinct. Otherwise uploading the target
before submission could overwrite the reference before its preparation runs.
Coarse reference textures may be reused for the target only after preparation
of all reference levels has finished. Pass ordering provides that dependency;
no CPU synchronization is inserted between them.

An empty ROI submits no pair computation. Otherwise the numeric admissibility
count joins the final readback. The changing dense validity mask remains on GPU;
exposing a cached CPU mask would misdescribe the new reference. Static ROI
selection remains separate from reference validity and current convergence.

Arithmetic and GPU organization
--------------------------------

* For B-spline interpolation, the Gaussian has the existing seven sigma-one
  coefficients. Separable horizontal and vertical passes retain reflected
  boundary coordinates.
* The cubic coefficient prefilter retains the CPU causal/anti-causal recursion,
  pole, gain and initial boundary sum. One invocation owns a row or a column;
  independent lines run in parallel. Replacing this by a convenient symmetric
  short filter would change the boundary convention.
* Linear and Keys cubic preparation use the existing unfiltered intensity and
  central-gradient conventions. The target interpolator remains unchanged.
* A 64-invocation workgroup owns one subset. Its anchored mean sums differences
  from a shared reference intensity, keeping constant-image variance zero.
  Lanes accumulate Jacobian products over different pixels. Shared reductions
  form the Hessian and metadata. Translation uses the analytical 2 × 2 inverse;
  affine uses pivoted Gauss–Jordan elimination distributed across matrix entries.
* Keeping the augmented matrix in workgroup memory avoids a private 6 × 12
  array spilling registers in every invocation. This was the main observed
  preparation bottleneck in the first GPU prototype. Shared reductions use
  less than 16 KiB per workgroup and require no subgroup feature or width.
* Overlapping subsets keep raw reference intensity and their own mean in
  metadata; non-overlapping subsets retain the historical centred-pixel layout.
  Inverse-Hessian offsets and the six public affine parameter slots are unchanged.

The preparation layout uses five storage buffers, one uniform and one sampled
texture. Three float32 scratch planes occupy about 95 MiB at UHD per finest
level, within a 128 MiB storage binding. Additional independent input textures,
coarse levels and normal solver resources also consume memory; the resource
report counts them. It does not report physical/free VRAM.

Float32 preparation is compared to the existing float64 CPU oracle, with
explicit tolerances. Healthy textures, constants, odd/even subsets, overlap,
interpolators, both models, ROI and fractional/affine motions are tested on the
actual browser GPU path. A software adapter repeats those executions. This is
not a proof of identical decisions on arbitrarily ill-conditioned textures.
No solver accuracy gate is relaxed for the new path.

History, recording and timing
-----------------------------

Changing the reference changes the coordinate system of the motion field.
``registerPair`` discards previous temporal histories on GPU and uses the fixed
coarse-to-fine solve for the new pair. It does not reuse velocities from another
reference or silently accumulate incremental strain. Later fixed-reference
calls can start a new history from those empty buffers.

The reported pair wall time includes GPU preparation and observable solving.
It is not an isolated GPU timestamp. The API returns no invented separate
preparation duration. Explicit reference diagnostics allocate/read back only
when requested while idle; they do not belong in the live path.

A completed-camera FPS observation additionally includes frame delivery,
worker scheduling, rendering and optional recording. Recording still reads the
full field and waits for durable browser storage admission. A static simulated
camera and a physical sensor are different experiments. Numerical UHD controls
use the original lossless analytical fixtures rather than the encoded demo.
