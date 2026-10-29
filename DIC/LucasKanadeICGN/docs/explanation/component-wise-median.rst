Median validation and replacement policies
==========================================

The component-wise normalized median replacement is an experimental DIC-RT
contribution. It extends the normalized median test from displacement vectors
to the coefficients of a local affine or quadratic displacement polynomial.
Detection and replacement are two independent choices: the filter can inspect
only centre displacement or every active coefficient, then replace either only
the flagged coefficients or the complete local motion state.

This distinction is important. A subset can contain a reliable translation
and first derivative but one unstable curvature coefficient. Replacing its
complete polynomial discards useful information. Testing only the translation,
on the other hand, leaves the unstable curvature untouched. The DIC-RT rule
keeps the coherent coefficients and replaces only the local outliers.

.. important:: Literature position

   A targeted search performed on 29 October 2026 found established normalized
   median tests for PIV displacement vectors, second-order DIC shape functions,
   and robust post-processing of DIC/DVC displacement fields. It did not find a
   published method combining all four properties used here: recentering local
   polynomial shape functions, testing every affine and quadratic coefficient
   independently, using coefficient-specific physical floors, and replacing
   only the flagged coefficients.

   This bounded search is evidence that the formulation is unusual; it is not
   a formal proof of novelty. A broader systematic literature review would be
   required before making a priority claim in a publication.

Why translation-only validation is insufficient
------------------------------------------------

The former implementation computed neighbour medians for all coefficients but
used only the two subset-centre translations to decide whether a result was an
outlier. When either translation was flagged, the complete affine or quadratic
state was replaced. Consequently:

* an isolated bad curvature could survive when translation was plausible;
* one bad translation could erase otherwise plausible gradients and
  curvatures;
* the decision threshold had pixel units and could not be applied directly to
  dimensionless gradients or inverse-pixel curvatures.

The current implementation exposes both the historical whole-state policy and
the DIC-RT component-wise policy, so their metrological effect can be measured
without changing kernels.

Algorithm
---------

For every accepted subset :math:`s`, the direct 3 by 3 neighbourhood provides
up to eight accepted neighbours :math:`n`. Each neighbour polynomial is first
recentered from its own subset centre to the centre of :math:`s`. This produces
comparable coefficients :math:`\widetilde p_{n\rightarrow s,k}` for component
:math:`k`.

.. code-block:: text

   accepted direct neighbours
              |
              v
   recenter every neighbour polynomial
   at the current subset centre
              |
              v
   for each coefficient selected by median_scope
      median + median absolute deviation
              |
              v
   normalized residual > 2 ?
        |                 |
       yes                no
        |                 |
   mark k as outlier   retain k
        \_________________/
                  |
      median_replacement
       |             |
    flagged          all
       |             |
   replace k     replace complete state
       \_____________/
              |
       status 2 if changed

For one component, the neighbour median and robust local spread are

.. math::

   m_{s,k} &= \operatorname{median}_{n}
              \left(\widetilde p_{n\rightarrow s,k}\right),\\
   d_{s,k} &= \operatorname{median}_{n}
              \left|\widetilde p_{n\rightarrow s,k}-m_{s,k}\right|.

The normalized residual is

.. math::

   r_{s,k} = \frac{|p_{s,k}-m_{s,k}|}{d_{s,k}+\epsilon_k}.

Component :math:`k` is flagged when :math:`r_{s,k}>2`. At least four accepted
neighbours are required. Otherwise the subset is left unchanged. Status 2
means that the selected replacement policy changed the result; consult the
configuration to know whether individual coefficients or the complete state
were replaced.

Selecting the filtered coefficients
-----------------------------------

The filter is fixed when the engine is constructed. Two orthogonal options are
available on every backend:

.. code-block:: python

   # Default: test every coefficient, replace only those classified as outliers.
   engine = ICGNWGPUFullPyramid(
       median_scope="all",
       median_replacement="flagged",
   )

   # Test every coefficient; one outlier replaces the complete local state.
   engine = ICGNWGPUFullPyramid(
       median_scope="all",
       median_replacement="all",
   )

``median_scope`` selects the coefficients allowed to trigger the filter.
``median_replacement`` selects what is changed after any trigger. This gives
four explicit strategies:

.. list-table:: Median-filter strategies
   :header-rows: 1

   * - ``median_scope``
     - ``median_replacement``
     - Behaviour
   * - ``displacement``
     - ``flagged``
     - Test :math:`t_x,t_y`; replace only the flagged displacement component.
   * - ``all``
     - ``flagged``
     - Test every active coefficient; replace only flagged coefficients. **Default.**
   * - ``displacement``
     - ``all``
     - Test :math:`t_x,t_y`; either displacement outlier replaces the complete state.
   * - ``all``
     - ``all``
     - Test every coefficient; any outlier replaces the complete state.

All strategies recenter neighbouring affine or quadratic polynomials before
computing centre-displacement medians. ``replacement="all"`` also computes
every active neighbour median so that a trigger replaces a coherent complete
polynomial, not a mixture of old and filtered values.

For ``motion_model="translation"`` the two scopes are numerically equivalent
because :math:`t_x` and :math:`t_y` are the only active coefficients. The
selected values are reported as ``medianScope`` and ``medianReplacement`` in
reference and registration configuration dictionaries. Changing either
specialization requires preparing the reference for the intended engine.

Unit-aware robustness floors
----------------------------

A single additive floor cannot serve translations, gradients, and curvatures
because their units differ. DIC-RT expresses every floor as its displacement
effect at the subset edge. Let :math:`h=(S-1)/2` be the half extent of a subset
of width :math:`S`:

.. list-table:: Robustness floors
   :header-rows: 1

   * - Components
     - Coefficient floor :math:`\epsilon_k`
     - Edge-displacement interpretation
   * - :math:`t_x,t_y`
     - :math:`0.1` px
     - historical normalized-median noise floor
   * - affine gradients
     - :math:`0.001/h`
     - 0.001 px at the subset edge
   * - pure curvatures :math:`u_{xx},u_{yy},v_{xx},v_{yy}`
     - :math:`0.001/(h^2/2)`
     - 0.001 px from the corresponding half-square basis term
   * - cross curvatures :math:`u_{xy},v_{xy}`
     - :math:`0.001/h^2`
     - 0.001 px from the cross term

The 0.1-pixel translation floor and threshold 2 follow the normalized median
test convention. The 0.001-pixel derivative effect is a DIC-RT engineering
choice and must be qualified for the intended images and metrology.

Measured curvature result
-------------------------

The comparison below uses exactly the same analytic image generator, GPU, and
current implementation. Only the two median-policy options change:

* 768 x 768 continuous analytic texture rasterized to 8-bit grayscale;
* centred 31 x 31 Gaussian subsets (sigma 12), stride 16, B-spline interpolation;
* eight measured registrations after three warm-ups;
* exact second-order displacement truth evaluated at every subset centre.

.. image:: component-median-curvature.svg
   :alt: Relative curvature RMSE before and after component-wise median replacement
   :align: center
   :width: 90%

.. list-table:: Analytic four-policy result for the 31 x 31 Gaussian subset
   :header-rows: 1

   * - Policy ``scope/replacement``
     - Moderate: displacement / gradient / curvature RMSE
     - Strong: displacement / gradient / curvature RMSE
     - Noise :math:`\sigma=1`: displacement / gradient / curvature RMSE
   * - ``displacement/flagged``
     - 0.006894 / 6.787e-4 / 6.849e-5
     - 0.007570 / 7.761e-4 / 7.787e-5
     - 0.024683 / 2.419e-3 / 2.364e-4
   * - ``displacement/all``
     - 0.006894 / 6.787e-4 / 6.849e-5
     - 0.007570 / 7.761e-4 / 7.787e-5
     - 0.024683 / 2.419e-3 / 2.364e-4
   * - ``all/flagged`` **default**
     - 0.006898 / 7.822e-4 / **4.635e-5**
     - 0.007565 / 9.146e-4 / **5.358e-5**
     - 0.024676 / 2.728e-3 / **1.496e-4**
   * - ``all/all``
     - 0.008531 / 1.042e-3 / **3.203e-5**
     - 0.009133 / 1.253e-3 / **3.587e-5**
     - 0.032474 / 3.588e-3 / **1.034e-4**

The default component-wise policy reduces curvature RMSE by 31--37% relative
to displacement-only detection. Whole-state replacement after any coefficient
outlier reduces it further, by 53--56% relative to displacement-only, but its
displacement RMSE is 21--32% worse and its gradient RMSE 32--37% worse than the
default. The new ``all/all`` strategy is therefore useful when curvature is
the dominant observable, but it is not a better general default.

Runtime cost
------------

``displacement`` detection needs two robust tests; ``all`` needs six affine or
twelve quadratic tests. Whole-state replacement also needs all active medians
even when only displacement can trigger. The 768 x 768 analytic timings varied
between runs and did not establish a stable ordering between policies. Select
the policy for its metrological behaviour, not for a claimed frame-rate gain.

A separate UHD quadratic check (31 x 31 Gaussian subsets, stride 16, two
pyramid levels, 20 frames in each of three campaigns) measured 86.4--92.9 ms
median depending on policy. Those values qualify the correctness-first
quadratic path and must not be confused with the approximately 10--12 ms UHD
affine configuration used for the 4K60 target.

After adding the replacement option, the unchanged default affine policy was
revalidated on five campaigns of 20 changing UHD frames after five warm-ups.
The median campaign measured 11.182 ms median, 11.972 ms p95, and 89.09 FPS
from the resident NumPy image to the complete NumPy field. It therefore keeps
the required margin below the 16.667 ms frame period on this machine.

Browser-supervised calls have additional transport and synchronization noise.
See :doc:`../reference/performance` for the qualified default ``all``
measurements.

Backend coverage
----------------

All four policies are in the canonical WGSL used by native wgpu-py and raw
browser WebGPU. The six-component affine policies are also implemented in
Three.js/TSL and the deprecated GPU-IO compatibility backend.
Translation-only kernels test only their two active coefficients and do not
pay for inactive affine comparisons.

Limits and interpretation
-------------------------

This is a local consistency rule, not a constitutive model. A real crack,
interface, strain localization, or other physical discontinuity can be
different from its direct neighbourhood and may therefore be classified as an
outlier. The component-wise rule is less destructive than replacing the full
polynomial, but it can still suppress genuine localized derivatives.

For that reason:

* retain status 2 in exported results;
* qualify the rule on representative discontinuities, not only smooth analytic
  fields;
* report gradient and curvature errors separately;
* do not describe the output as an unfiltered measurement.

Literature boundary
-------------------

The normalized residual, neighbourhood size, threshold 2, and 0.1-pixel
translation floor are derived from Westerweel and Scarano, `Universal outlier
detection for PIV data <https://doi.org/10.1007/s00348-005-0016-6>`_. That
method validates displacement vectors rather than independently recentered
shape-function coefficients.

Second-order local DIC shape functions are established by work such as Bai et
al., `A novel 2nd-order shape function based digital image correlation method
for large deformation measurements
<https://doi.org/10.1016/j.optlaseng.2016.09.010>`_. Robust DIC/DVC field
post-processing also exists, for example Becker and Marrow, `A robust finite
element-based filter for digital image and volume correlation displacement
data <https://doi.org/10.1007/s11340-021-00718-5>`_. Neither source identified
in the targeted search describes the coefficient-wise rule implemented here.

Reproduce the current half of the comparison from the repository root with::

   PYTHONPATH=DIC/LucasKanadeICGN/python/src \
   python3 DIC/LucasKanadeICGN/benchmarks/benchmark_quadratic_weighting.py \
     --warmups 3 --repeats 8

