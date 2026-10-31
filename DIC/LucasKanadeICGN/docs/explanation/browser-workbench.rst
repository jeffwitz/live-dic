Browser workbench design and scientific contract
================================================

The browser companion complements the native Python supervisor. Its interface
is organized around a short path: choose input, define the reference and
ROI, run correlation, then inspect and process the resulting fields. Pierre
Margerit's foundational numerical work remains the basis of the engine.

Interface research
------------------

The review used official product documentation on 31 October 2026. It is a
representative comparison, not a claim that every DIC package was tested or
that commercial solvers were executed locally.

* `Ncorr manual 1.2.2 <https://www.ncorr.com/download/ncorrmanual_v1_2_2.pdf>`_:
  staged reference/current/ROI/analysis workflow, a visible strain-fit scale,
  point inspection and separate displacement/strain plots.
* `VIC-2D manual 7 <https://correlated.kayako.com/api/v1/articles/86/attachments/5544/url>`_:
  image/data/extraction organization, timeline navigation, line profiles,
  extensometers and explicit strain/post-processing settings.
* `MatchID software <https://www.matchid.eu/software>`_:
  datasets, result views, extraction/chart tools and reusable analysis recipes.
* `ZEISS CORRELATE <https://www.zeiss.com/metrology/en/software/zeiss-correlate.html>`_:
  persistent evaluation steps, reference systems and virtual extensometers.
* `DICe VSG processor <https://dicengine.github.io/dice/class_d_i_ce_1_1_v_s_g___strain___post___processor.html>`_:
  displacement-neighbourhood strain calculation as an explicit postprocessor.

The adopted principles are progressive disclosure, a large central field view,
explicit acquisition/measurement state, reversible processing and named physical
quantities. No proprietary source, icon set or screen layout is copied. Stereo,
material identification, stress reconstruction, instrument synchronization and
commercial metrology certification are outside this application.

Numerical analysis module
-------------------------

``web-app/ui/postprocess.js`` operates on a materialized field. It never changes
the engine parameters, statuses or iterations. "Computed" means exactly the
solver output, including any explicitly selected solver median policy; it does
not mean unsmoothed if that policy was enabled.

Displacement median/Gaussian smoothing is a separate, optional operation. Its
odd 3/5/7/9 window is in grid points. Gaussian sigma is half the window radius,
with a minimum of half a grid step. Invalid points are not filled. Neighbours
must have a clear conservative rectangle in both the valid grid and image ROI;
this avoids averaging across a thin excluded crack. Near boundaries the support
is reduced, not extended across missing data. Smoothing changes spatial
resolution and is not outlier detection or an uncertainty estimate.

Strain is either read from the local affine/quadratic derivatives or recomputed
by a least-squares plane fitted to the displacement neighbourhood (VSG). The
fit uses double precision JavaScript arithmetic, normalized grid coordinates,
centred normal equations and a determinant rank test. At least three
non-collinear admissible points are required. Failed fits retain a separate
invalid strain mask. Translation-only fields require VSG for strain.
Smoothing requires VSG so that processed displacement and strain describe the
same operation. The subset-plus-window span is
``subsetSize + (window - 1) * subsetStride`` pixels.

Small strain uses the symmetric displacement gradient. Green-Lagrange strain
uses ``E = (F.T F - I)/2``. Principal values are the eigenvalues of that 2D
symmetric tensor. Equivalent strain uses the documented ezz=0 convention;
rotation is the 2D polar angle ``atan2(vx - uy, 2 + ux + vy)``. Length calibration
changes displacement/length units, not dimensionless strain. This is an isotropic
scale, not a lens/perspective calibration. No material stress is inferred.

A two-point extensometer snaps to actual subset centres and requires valid
endpoints. It reports change in endpoint distance divided by original distance,
which is a gauge engineering strain, distinct from a local tensor component.
Incremental comparisons remain pairwise; they are not summed into trajectories.

Archive contract
----------------

``ui/archive.js`` writes standard uncompressed ZIP/NPY NPZ archives without any
runtime dependency. Computed parameters/status/iterations and optional processed
arrays are separate. Metadata and processing recipe are scalar Unicode JSON;
image dimensions, subset geometry and the binary ROI are included. NumPy can
open these files with ``allow_pickle=False``. Float64 imported parameters are
preserved. The importer checks CRC and sizes, rejects object/pickle arrays and
limits total uncompressed input to 256 MiB. It supports C-order little-endian
numeric/Unicode arrays and a single field, not arbitrary multi-frame archives.
Archives need explicit image dimensions and subset geometry; legacy NPZ files
without these are rejected rather than guessing image extent.

Qualification
-------------

The independent CPU tests cover an analytical 3840 x 2160 affine field, exact
finite rotation/Green strain, a thin excluded crack, rank-deficient support,
Gaussian weights, unit-aware extensometry, non-mutation and NumPy archive
round trips. These validate the analysis operators, not an experimental noise
floor or the precision of an uncalibrated browser camera. Browser end-to-end
records must be reported separately from this numerical operator coverage.
