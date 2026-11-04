Displacement outliers and explicit field median
================================================

live-dic provides one deliberately narrow spatial robustness check: a universal
outlier detection (UOD) test on the two displacement components.  It is a
guard against subsets that converge into a wrong correlation basin.  It is
not a denoising filter for affine gradients or quadratic curvatures. A
separate ``median`` policy can fill rejected subsets with a neighbour-based
estimate so a coarse pyramid level can offer it as a seed to the next finer
solve. It does not smooth accepted measurements.

For diagrams comparing measurement smoothing, coarse median seeds and
photometrically tested propagation, see :doc:`../reference/propagation`.

The policy is selected when the engine is created:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   engine = ICGNWGPUFullPyramid(outlier_policy="replace")

``replace`` is the default. ``flag`` preserves the raw solver coefficients,
``disabled`` omits the UOD post-pass, and ``median`` repairs rejected subsets
from valid neighbours. The median width defaults to a 3 by 3 subset-grid
stencil; shared-WGSL engines also accept 5 by 5, 7 by 7, and 9 by 9.

Detection rule
--------------

For each accepted subset, the eight direct neighbours are expressed at the
centre of the current subset.  This recentering is important for affine and
quadratic fields: a neighbour's translation is not compared at the wrong
spatial position.  The detector then evaluates ``tx`` and ``ty`` independently:

.. math::

   s_k = \frac{|p_k - \operatorname{median}(n_k)|}
                 {\operatorname{median}(|n_k - \operatorname{median}(n_k)|) + 0.1}.

The subset is flagged when either score is greater than 2.  At least four
accepted neighbours are required.  Status 2 records a detected outlier.

The 0.1-pixel floor and threshold 2 make this a coarse wrong-basin guard.  In
a low-noise DIC field they correspond roughly to a neighbour disagreement of
more than 0.2 pixel when the local MAD is small.  They must not be interpreted
as a calibrated coefficient uncertainty.

The four policies
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 32 20

   * - Policy
     - Detection
     - Returned state
     - GPU post-pass
   * - ``replace``
     - ``tx`` and ``ty``
     - all active coefficients are replaced by their recentered neighbour
       medians when either displacement component is flagged
     - executed
   * - ``flag``
     - ``tx`` and ``ty``
     - raw IC-GN coefficients are preserved; only status becomes 2
     - executed
   * - ``disabled``
     - none
     - raw IC-GN state and status
     - omitted
   * - ``median``
     - solver status is not accepted
     - rejected target coefficients are replaced by the recentered component
       medians of accepted, ROI-permitted neighbours; the target's rejected
       status is preserved
     - executed; coarse repaired states are seed-eligible

Replacing the complete state is intentional.  A displacement outlier normally
means the local optimizer converged into the wrong basin, so retaining its
gradients or curvatures would create a hybrid polynomial with no clear
physical or optimization meaning.  The ``flag`` policy is preferable when a
metrology workflow must publish raw estimates and handle outliers downstream.
The ``disabled`` policy is useful for raw numerical studies and for measuring
the solver without robustness-pass cost.

Median-seeded recovery
-----------------------

The ``median`` policy bypasses the UOD score and changes only subsets whose
solver status is not accepted. The target reference window must itself be
valid, and at least four accepted neighbours must be available. Invalid
reference windows and ROI-excluded subsets are never filled. The failed target
is excluded from the median; each accepted neighbour is expressed at the
target centre before the component-wise median is computed. For a quadratic
state, the displacement, first derivatives, and curvatures are recentered
together. ROI neighbour permissions prevent a stencil from crossing excluded
regions.

At each intermediate pyramid level, the repaired coefficients are written to
the filtered state buffer with a private seed-available marker. Coarse-to-fine
transfer can then use the estimate even though that coarse solve failed. The
next finer level runs the normal IC-GN solve and assigns its own status. The
raw coarse solver state and its failure status remain unchanged. At the finest
level, repaired coefficients can be inspected, but the original non-accepted
status is retained: this repair does not claim that the point converged or
re-evaluate ZNCC.

Accepted subsets are copied through unchanged. This is not median smoothing of
the field. The Results workbench has a separate, explicit post-processing
recipe for spatially smoothing saved displacements.

The configured stencil is measured in subset-grid cells. A 9 by 9 stencil
contains at most 80 neighbours for a failed centre. Stencils at image or ROI
boundaries can have fewer samples; fewer than four leaves the failed estimate
untouched. The method can still seed a wrong basin if its valid neighbours
belong to different motion regions, so it is an initialization aid rather
than an acceptance rule or a convergence guarantee.

Validation contracts
--------------------

The opt-in GPU regression suite separates three questions that a smoothing
benchmark would confound:

* a healthy fractional-translation field with one-gray-level Gaussian noise
  must have at most 1% false positives;
* an injected half-pixel displacement outlier must be detected, with
  ``replace`` changing the complete state and ``flag`` preserving it;
* a one-subset-wide band with true ``du/dx = 0.004`` must remain present and
  must not be flagged merely because it is spatially localized.
* the ``median`` policy must repair an injected failed subset from valid
  neighbours for translation, affine, and quadratic states, leave accepted
  neighbours unchanged, preserve final failure status, avoid ROI-excluded
  targets, and make a repaired coarse estimate eligible for seed transfer.

These are behavioural guards, not a full metrological qualification.  Real
camera noise, texture variation, discontinuities, and multiple GPU vendors
still require application-specific evaluation.

Why coefficient-wise median filtering was removed
--------------------------------------------------

An earlier experimental API applied the same normalized median rule to every
affine and quadratic coefficient.  On a healthy noisy field it modified
66--81% of subsets.  Its apparent 30--56% curvature-RMSE improvement on smooth
analytic fields was therefore ordinary median smoothing, not demonstrated
outlier robustness.  It also erased a localized one-subset-wide deformation
band and could degrade gradients through quadratic recentering.

That coefficient-wise smoothing behavior is no longer the solver-time
``median`` policy. The current policy uses a median only to initialize a
rejected point at the next pyramid scale; it never changes accepted states.
For derivative smoothing, use the Results workbench's explicit saved-field
recipe or a NumPy post-processing operation with a documented kernel and
transfer function.

A future all-coefficient detector would need coefficient-specific uncertainty
floors, for example

.. math::

   \epsilon_k = c\,\hat{\sigma}\sqrt{(H^{-1})_{kk}},

and qualification of false-positive rate, injected-outlier detection, and
localized-feature bias. This future detector could use the neighbourhood
estimate with ``retry_seeded`` and accept the rerun according to its
correlation score. The current median seed repair is limited to subsets already
rejected by the solver and relies on the next pyramid level to retry them.
