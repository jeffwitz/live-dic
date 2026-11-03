Spatial initialization experiments
==================================

A converged local IC-GN fit and a robust search strategy are different parts of
a DIC pipeline. An affine initialization can place the field close to the
solution while leaving individual subsets in the wrong attraction basin.
Adding scales also enlarges the physical support of a fixed-size coarse patch;
it does not guarantee recovery. This page describes a research harness, not a
new native or browser option. Existing solver defaults remain unchanged.

What DIS contributes
--------------------

Kroeger et al., `Fast Optical Flow using Dense Inverse Search
<https://arxiv.org/html/1603.03590v1>`_, section 2.2, combine independent inverse
search with aggregation of overlapping patches. Their aggregation weights each
pixel contribution by its photometric residual. Excessively large patch updates
are reset to their initial estimate. The resulting dense field initializes the
next scale; variational refinement is a separate step.

The `OpenCV implementation
<https://github.com/opencv/opencv/blob/5.0.0/modules/video/src/dis_flow.cpp>`_
additionally compares spatial candidates before local optimization. It tests a
neighbour displacement on the current patch and selects it by the current
patch error. This differs from replacing a measured field by neighbour medians.
The paper and OpenCV are distinct references; the benchmark below is not an
implementation of either complete algorithm.

Controlled hypothesis selection
-------------------------------

``benchmarks/spatial_seed_reference.py`` is a CPU oracle for translation-only
candidate selection. It evaluates the current estimate, an optional original
prior and up to eight permitted neighbouring displacements on each destination
patch, using normalized correlation. Native IC-GN then recomputes the state
from the selected seed with its existing B-spline interpolation, Hessian,
convergence tolerance and ZNCC acceptance threshold. A neighbour coefficient
is a hypothesis, never an accepted measurement by itself.

The oracle reads an immutable snapshot on each pass, so its parallel update
schedule differs from OpenCV point-by-point propagation. ROI neighbour bits
prevent forbidden links. Reference selection, reference numerical validity and
final solver acceptance retain their existing meanings. A rejected but finite
state can supply a candidate: its displacement is independently scored again
at the destination, then optimized and tested by IC-GN.

The experiment uses OpenCV bilinear sampling for candidate scoring; final
optimization remains native B-spline. It supports neither affine/quadratic
candidate transport nor partial supports. It performs CPU readbacks, scoring
and uploads between solves, so its elapsed times cannot qualify live operation.
OpenCV is optional research tooling and is not added to the package requirements.

Reproduce on the public UHD fixture
-----------------------------------

From the repository root, with the native dependencies and optional
``opencv-python-headless`` installed in the intended environment:

.. code-block:: console

   python3 DIC/LucasKanadeICGN/benchmarks/benchmark_spatial_seed_candidates.py \
     DIC/LucasKanadeICGN/assets/uhd_tensile/reference.png \
     DIC/LucasKanadeICGN/assets/uhd_tensile/frame_040.png \
     --mode spatial --output /tmp/seed-spatial

   python3 DIC/LucasKanadeICGN/benchmarks/benchmark_spatial_seed_candidates.py \
     DIC/LucasKanadeICGN/assets/uhd_tensile/reference.png \
     DIC/LucasKanadeICGN/assets/uhd_tensile/frame_040.png \
     --mode continue --output /tmp/seed-continue

Output directories must be new. Both runs use two levels, 32-pixel subsets,
stride 16, no spatial post-filter, iteration caps 8/16, tolerance 0.005 and
ZNCC threshold 0.8. Pass indices C/F count extra coarse/fine re-solves. The
``continue`` control uses the same number of re-solves and iteration caps,
without neighbour selection; it separates propagation from additional local
iterations. It does not equalize the extra candidate-scoring work.

``--mask`` accepts a boolean NPY. ``--initial-affine`` accepts a JSON object
with ``dx``, ``dy`` and four displacement gradients in ``linear``; its origin
is the image centre ``((W-1)/2, (H-1)/2)``. The harness exports original-resolution
NPZ fields, pass counts, configuration and source/input hashes. It relies on
private native seed methods and is development tooling, not a supported engine
API. Compare coverage and analytic displacement error together. This 32/16
translation experiment does not inherit the qualification of affine 16/16.

Public UHD correctness evidence
-------------------------------

The archived `UHD control report
<https://github.com/jeffwitz/live-dic/tree/main/DIC/LucasKanadeICGN/benchmarks/results/spatial-seed-candidates-20261104>`_
retains both modes and all pass counts. At 32/16, the interior coverage stays
100 percent, with vector RMSE 0.0101823 px before extra passes and 0.0101954 px
after scored propagation; equal-schedule continuation gives 0.0101881 px.
These are correctness measurements on the archived analytic pair, not timing
or general metrology qualification.

Before production integration
-----------------------------

The :doc:`../reference/performance` timing campaign measures the current
two-pass CPU oracle at about 6.54 seconds per UHD field, compared with 14.4 ms
for the production GPU-resident pyramid in the same campaign. Candidate scoring
on CPU dominates. This prototype is unsuitable for live acquisition; its
robustness result does not establish an affordable production implementation.

A production stage must retain data on the GPU, share its WGSL between hosts,
keep ROI exclusions and geometry, and qualify translations, spatially varying
analytic fields, discontinuities, noise and false candidates. Re-estimation
must preserve the final scientific rejection criteria. Photometric coverage on
private images without ground truth does not prove displacement accuracy.
See :doc:`../reference/performance` for qualified performance boundaries and
:doc:`displacement-outliers` for the distinct post-filter policies.

GPU research prototype
----------------------

``benchmarks/gpu_spatial_seed.py`` now provides an experimental, synchronous,
single-slot host around the native full pyramid. It is not a public engine
option and is not connected to the browser interface. Numerical defaults remain
unchanged. It supports translation, uniform full subsets and B-spline only.
The fixed reference, reference ROI, Hessians and all pipelines are prepared once.

The shared ``spatial_seed.wgsl`` scores the current displacement, original prior
and permitted direct neighbours on the destination reference window. Each
candidate is a hypothesis, never a copied final measurement. Workgroups read
an immutable snapshot, so a parallel pass is Jacobi propagation; it does not
reproduce OpenCV's directional sequential sweeps. Scoring uses bilinear raw
intensities and normalized correlation; final refinement uses the existing
B-spline IC-GN solver with its unchanged acceptance criteria.

An improvement of more than 0.0001 in score and 0.001 px in displacement is
required to enqueue a new seed. ROI permissions and numerical reference validity
restrict both destinations and sources. An atomic counter appends work indices;
no global sort is required because each re-solve owns one output cell. A small
finalization dispatch writes indirect dispatch dimensions. The opt-in
``solve_compact_translation`` entry point runs the existing IC-GN equations on
that list, preserving the dense output grid. A uniform guard before all barriers
handles unused workgroups, including the final row of a two-dimensional dispatch.
If an accepted original point's trial fails, its original state is retained.

All reduction, scoring, list construction and re-solves are encoded before one
queue submission. There is no intermediate host readback or per-image compilation.
The normal readback remains the finest field. Explicit cold diagnostics can read
the work counts separately; they are outside the timing loop. The research host
rejects streaming because only its first slot has been instrumented.

The benchmark distinguishes scoring every admissible window from scoring only
current rejections. Compaction saves IC-GN work but does not save the cost of
scoring every window. Restricting candidate search to rejected windows is a
different opt-in policy: it cannot escape a wrong basin already marked accepted.
A sampled-score control also exists; it changes only candidate evaluation, not
the final IC-GN window or equations, and must be qualified separately.

Median measurements and median seeds
------------------------------------

The original CPU propagation experiment explicitly disabled all median passes.
It did not establish that median filtering is ineffective in a pyramid.
The GPU controls now distinguish three operations:

* ``median`` applies the existing centred 3 x 3 component median to accepted
  coarse measurements only. Statuses and iteration counts are retained; an
  invalid own point is not filled. The filtered coarse buffer is actually used
  for coarse-to-fine transfer. The final fine field remains raw.
* ``seed-median`` is a separate initialization-only control. Numerically
  admissible status-0 coarse estimates also contribute. With at least four
  permitted sources, an unavailable coarse seed can receive a median seed.
  Availability is recorded only in the private transfer buffer, never in the
  raw coarse measurement or the final field. Fine IC-GN still has to converge.
* ``median-candidate`` tests a neighbourhood median photometrically before
  selectively running another local solve. This differs from unconditionally
  overwriting an initialization or smoothing the final measurements.

Independent GPU tests compare the first two medians with NumPy, verify that the
filtered buffer is transferred rather than ignored, preserve raw measurements,
and check selective work and indirect dispatch counts. These tests establish
implementation correctness, not a universal robustness improvement. A median
suppresses isolated errors; it cannot generally resolve a whole neighbourhood
that has converged to the same incorrect basin.

Reproduce the original-UHD numerical/timing controls with an installed native
environment, from the repository root:

.. code-block:: console

   python3 DIC/LucasKanadeICGN/benchmarks/benchmark_gpu_spatial_seed.py \
     --output /tmp/gpu-seed-campaign

The output directory must be new. See :doc:`../reference/performance` for the
measured costs, evaluated populations and remaining real-time limitations.
