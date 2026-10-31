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
same operation. The displacement smoothing span is
``subsetSize + (window - 1) * subsetStride`` pixels. VSG on unsmoothed
displacements has the same span; applying smoothing followed by VSG composes
two neighbourhoods, giving ``subsetSize + 2 * (window - 1) * subsetStride``
for strain. Direct unsmoothed gradients retain the solver subset span. At
16/16 with a 5-point window, smoothing spans 80 px and smoothed VSG strain
spans 144 px nominally; truncated boundary support can be smaller.

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
numeric/Unicode arrays, including exact-range 64-bit integers. Native CLI
sequence archives require an explicit selected frame. Image extent comes from
stored scalar dimensions, an archived ROI shape, or explicit user confirmation;
it is never inferred from a last grid cell. Subset geometry comes from stored
scalars/configuration or confirmation. Declared ROI without its binary mask is
rejected. The input archive hash and geometry provenance survive re-export.

Qualification
-------------

The independent CPU tests cover an analytical 3840 x 2160 affine field, exact
finite rotation/Green strain, a thin excluded crack, rank-deficient support,
Gaussian weights, unit-aware extensometry, non-mutation and NumPy archive
round trips. These validate the analysis operators, not an experimental noise
floor or the precision of an uncalibrated browser camera. Browser end-to-end
records must be reported separately from this numerical operator coverage.

Presentation is separate from measurement
-----------------------------------------

Saved results use a separate reference-coordinate field canvas. Displaying them
never overlays an old field on an unrelated current camera image. Point and
gauge annotations also appear inside the viewer, while processing controls stay
beside it. The solver's dense exclusions remain visible in status mode.

Robust colour bounds use the 2nd/98th percentiles of valid measurements; signed
fields use symmetric limits. Full range and fixed bounds are explicit options.
Sequential blue/green/gold, signed blue/white/red and grayscale palettes are
presentation choices. Saturated values are counted but not removed or clipped
in the data. Histograms use the valid population, not only the unsaturated range.

Saved-field history
-------------------

The acquisition worker assigns a new analysis epoch whenever the user prepares
a reference/ROI. A consecutive pair changes its actual reference image while
retaining that preparation epoch; both input identifiers remain in comparison
metadata. The history tool requires matching epoch (or native archive hash),
image/grid geometry, model, ROI hash and comparison mode. It processes one field
at a time and sends a shared static mask once to its worker. No GPU calculation
or image decoding is repeated. Cancellation releases that worker.

A probe samples an actual grid point; a gauge computes endpoint distance strain
for each field independently. Missing values stay null and break chart lines.
The horizontal axis is saved order; input indices/timestamps are additional
export columns. The tool does not infer regular sampling, accumulate incremental
displacements or silently combine unrelated reference configurations.

Results projects
----------------

A ``.dicproject`` is a standard ZIP with ``manifest.json``, per-field NPY
parameters/status/iterations, and shared static masks. It retains every computed
field, applied recipe, grid, source-pair provenance and analysis epoch. A mask
shared by a recording is written once. Processed arrays can be regenerated from
the stored recipe; individual NPZ export also includes those arrays explicitly.
No input image, camera permission or File handle is serialized.

Import validates all fields before changing the library and checks capacity for
the complete batch. It adds fields without replacing the active acquisition
source or settings. Current library storage is 128 MiB, total uncompressed
archive input is capped at 256 MiB, and the versioned format accepts at most
512 fields / 4096 ZIP entries. CRC protects entries; ROI hashes are checked
against their canonical binary mask. Export/import does not execute code.

Mask image editing
------------------

An optional binary base is intersected with the existing exact vector mask. Undo
snapshots retain an immutable base reference, not a copied full image per edit.
Only paused edits rasterize or compose masks. PNG import/export does not change
IC-GN equations or add a pixel mask in the solver loop. Preparing a new ROI
reuses the core's full-window selection, conservative links and canonical hash.
A failed import leaves the existing geometry/calculation intact. Source and
reference operations share one admission gate, drain pending work and invalidate
stale video seeks before replacing resources.
