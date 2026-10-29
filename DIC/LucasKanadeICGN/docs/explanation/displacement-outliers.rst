Displacement outliers and explicit field median
================================================

DIC-RT provides one deliberately narrow spatial robustness check: a universal
outlier detection (UOD) test on the two displacement components.  It is a
guard against subsets that converge into a wrong correlation basin.  It is
not a denoising filter for affine gradients or quadratic curvatures.  A
separate ``median`` policy is available when that smoothing is explicitly
wanted.

The policy is selected when the engine is created:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   engine = ICGNWGPUFullPyramid(outlier_policy="replace")

``replace`` is the default.  ``flag`` preserves the raw solver coefficients,
``disabled`` omits the complete post-pass, and ``median`` applies a centred
3 by 3 component median to every valid subset.

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
     - none
     - every active coefficient is replaced by its centred 3 by 3 component
       median; solver status is preserved
     - executed

Replacing the complete state is intentional.  A displacement outlier normally
means the local optimizer converged into the wrong basin, so retaining its
gradients or curvatures would create a hybrid polynomial with no clear
physical or optimization meaning.  The ``flag`` policy is preferable when a
metrology workflow must publish raw estimates and handle outliers downstream.
The ``disabled`` policy is useful for raw numerical studies and for measuring
the solver without robustness-pass cost.

Explicit median smoothing
-------------------------

The ``median`` policy deliberately bypasses the UOD score.  For every valid
subset, the subset itself and every valid direct neighbour are expressed at
the current subset centre.  Each active translation, affine, or quadratic
coefficient is then replaced independently by the median of those available
samples.  The centre is included, so a fully populated interior stencil has
nine samples.  Boundary stencils use the valid samples that exist.

Recentring preserves the meaning of a spatial polynomial: neighbour
translations and first derivatives are transported to the current centre
before their component medians are calculated.  The operation is applied at
every pyramid level and the filtered coarse state seeds the next finer level.
It leaves accepted/rejected status and iteration count unchanged; in
particular, status 2 remains reserved for UOD detections from ``replace`` or
``flag``.

This policy is a nonlinear spatial low-pass filter.  It can reduce noise, but
it can also attenuate a narrow deformation band or a real discontinuity.  Its
output is not the raw minimizer of the local IC-GN objective, and correlation
is not recomputed after filtering.  Accuracy and timing claims must therefore
name the policy explicitly.

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
* the explicit ``median`` policy must match an independent centred 3 by 3
  component-median oracle for translation, affine, and quadratic states,
  including boundary stencils, without changing solver status.

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

Those coefficient-triggered modes are no longer public options.  If derivative
smoothing is wanted, ``outlier_policy="median"`` now exposes it explicitly as
a centred GPU post-pass, separate from the solver's outlier status.  A NumPy
post-processing operation remains preferable when a different kernel or a
fully user-controlled transfer function is required.

A future all-coefficient detector would need coefficient-specific uncertainty
floors, for example

.. math::

   \epsilon_k = c\,\hat{\sigma}\sqrt{(H^{-1})_{kk}},

and qualification of false-positive rate, injected-outlier detection, and
localized-feature bias.  A robust recovery could then use the neighbourhood
estimate as a seed for ``retry_seeded`` and accept the rerun according to its
correlation score instead of overwriting a converged state.
