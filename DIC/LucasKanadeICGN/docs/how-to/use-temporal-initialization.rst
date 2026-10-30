Track ordered frames with temporal initialization
============================================================

Use this opt-in mode when motion accumulates against a fixed reference but
the change between successive frames remains within the local capture range.
Only ``ICGNWGPUFullPyramid`` implements it, with one, two, or three levels.
Its default remains ``initialization="zero"``.

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(
       initialization="temporal",
       pyramid_levels=2,
       temporal_max_age=2,
       device_type="gpu",
   ) as engine:
       engine.set_reference(reference, roi_mask=mask, roi_policy="full")
       for result in engine.register_stream(ordered_frames, pipeline_depth=2):
           print(result.accepted, result.timing.get("temporallySeededCoarseSubsets", 0))
       engine.reset_temporal_state()

Omit ``roi_mask`` when no ROI is needed. Translation, affine and quadratic
models retain their existing domains; ROI remains limited to translation and
affine. ``full`` and ``partial`` ROI supports are both respected.

Every frame runs every configured scale
------------------------------------------------------------

The previous accepted fine-resolution motion is projected onto the coarsest
grid. The coarsest solve then runs, followed by the normal coarse-to-fine
solves. Temporal initialization never skips a pyramid level and does not add
a conditional retry. With one level, projection and solving use the fine grid.
If no admissible history exists, a subset starts from zero. A first temporal
frame therefore has the same numerical result as zero initialization.

A thin ROI with no selected coarse windows still has no coarse seed route,
even if fine history exists. Temporal initialization does not bypass the ROI
or supply a second direct fine-level seed in that case.

These are fixed-reference, Lagrangian motion parameters. They are not
incremental displacements to accumulate. The reference image, ROI, gradients
and Hessians stay fixed until the next ``set_reference``.

Preserve fine detail with a coarse correction
-------------------------------------------------------------

For a two-level temporal pyramid, opt in with
``temporal_fine_correction=True``. Start with one slot when result age matters:

.. code-block:: python

   with ICGNWGPUFullPyramid(
       initialization="temporal",
       temporal_fine_correction=True,
       pyramid_levels=2,
       motion_model="affine",
       outlier_policy="median",
       device_type="gpu",
   ) as engine:
       engine.set_reference(reference)
       for result in engine.register_stream(ordered_frames, pipeline_depth=1):
           consume(result.displacement, result.accepted_mask)

The normal coarse solve still starts from projected fine history. Its change
relative to the previous accepted coarse solution then corrects the previous
fine estimate. This avoids discarding fine spatial detail when constructing
the fine seed. It requires complete admissible coarse stencils with histories
from the same accepted frame as the fine estimate; otherwise the existing
absolute coarse-to-fine seed is used. Reset clears both histories.

This is native-only and disabled by default. Translation, affine and quadratic
retain their existing configuration domains; ROI is still restricted to
translation and affine. Neither iteration ceilings nor tolerances are reduced.
Compare timing and quality on the actual sequence: lower iteration counts do
not guarantee a lower p95 result age on a shared or thermally limited machine.

Ordering, age and reset
------------------------------------------------------------

``register`` and ``register_stream`` share one history. Supply chronological
targets to both APIs; timestamps are not inferred or reordered. An engine is
not thread-safe. Finish consuming the stream before resetting history,
replacing the reference, or switching to an unrelated sequence. Frames already
submitted can update history even if their results have not yet been yielded.

``pipeline_depth`` bounds target slots, not independent histories. Queue order
ensures that the next target sees the previous submitted target's completed
history update. No intermediate CPU readback or synchronization is added.

``temporal_max_age=None`` keeps the last accepted estimate indefinitely.
A nonnegative integer permits that many consecutive rejected frames before
the seed expires: ``0`` expires it after the first rejection, ``1`` permits
one. This counts submitted frames, not seconds. A whole-ROI-excluded result
does not submit GPU work and does not age history. ``reset_temporal_state``
forgets history without rebuilding the reference; ``set_reference`` always
resets it. Reset is harmless before preparation or in zero mode.

Spatial policies and diagnostics
------------------------------------------------------------

The history uses raw accepted solver states for ``median``, ``flag`` and
``disabled``. Explicit median smoothing is therefore not fed back repeatedly
into the seeds. With ``replace``, history uses the final UOD-corrected state;
this avoids retaining the wrong-basin value that UOD replaced. Rejected raw
states are not made into raw seeds merely because smoothing filled a value.

Results retain their existing dense arrays and status meanings. In temporal
mode, ``timing`` additionally includes ``temporallySeededCoarseSubsets`` and
``temporalAcceptedUpdates``. These are counts, not durations. The first counts
coarse destinations receiving at least one admissible fine source; it is not
the number of fine estimates. Four diagnostic words share the existing final
mapped readback. ``config`` records initialization, age limit and seed policy.
Whole-ROI-excluded results bypass GPU work and omit those GPU counters.

See :doc:`../explanation/temporal-initialization` for projection and failure
modes, and :doc:`../reference/performance` for native UHD measurements.
