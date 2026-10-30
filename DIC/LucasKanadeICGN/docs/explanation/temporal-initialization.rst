Why initialize the pyramid from temporal history?
=================================================

IC-GN is a local optimizer. Image pyramids enlarge its useful capture range,
but a fixed reference can still accumulate hundreds of pixels of displacement.
Often the change between successive images is much smaller. The native full
pyramid can use its last accepted motion field as an initial estimate while
continuing to measure every image against the original reference.

The graph for every target is:

.. code-block:: text

   last accepted fine field (shared GPU history)
                     |
          project onto the coarsest grid
                     |
     coarse solve -> finer solves -> fine solve -> spatial policy
                                                    |
                                    update accepted history and ages
                                                    |
                                            final NumPy readback

Projection changes coordinates and scale
------------------------------------------------------------

Projection reuses the same polynomial transport as coarse-to-fine seed
transfer. Each source polynomial is evaluated at the destination centre,
then admissible surrounding sources are bilinearly combined. If destination
pixels represent ``s`` reference-image pixels, displacement is divided by
``s``, affine gradients are unchanged, and quadratic curvatures are multiplied
by ``s``. Existing half-pixel pyramid phase conventions are preserved. The
reverse direction is subsequently used in the normal finer-level transfers.

Only accepted historical states contribute. ROI source/destination selection
and conservative cross-level link permissions are prepared once. A missing
history never suppresses an otherwise selected destination: it starts at zero.
All configured levels are solved, irrespective of temporal seed availability.
This is neither selective recovery nor a second attempt after failure.

Optional two-level coarse correction
------------------------------------

An absolute coarse-to-fine transfer replaces fine spatial detail with the
interpolated coarse field. ``temporal_fine_correction=True`` instead retains
that detail when matched accepted history is available:

.. code-block:: text

   previous fine history -> project -> current coarse solve
            |                              |
            |          current coarse field - previous coarse history
            |                              |
            +----------- add transported change
                                           |
                                      fine solve
                                           |
                            update fine and coarse histories

All polynomial coefficients are transported using the existing recentering
and scale conventions. The difference is an initialization correction, not an
accumulation of output measurements. Both solves still compare against the
unchanged reference. No velocity extrapolation or conditional retry is added.

The GPU checks complete source stencils, ROI permissions, expiration and equal
coarse/fine history ages. It retains the normal absolute seed when those
conditions fail, including the first image. Fine and coarse histories are
shared by all slots and merged in queue order. The extra coarse field, ages
and link permissions are allocated at preparation, with no extra host round
trip. This option currently requires exactly two levels; the default remains
the absolute transfer described above.

One history, multiple slots
------------------------------------------------------------

Temporal states and unsigned frame ages belong to the engine. Images,
intermediate fields and diagnostic counters belong to individual slots.
The GPU queue orders projection, solving and history update before the next
submission can read that history. Slots therefore overlap host work and GPU
execution without creating divergent per-slot trajectories. Tests compare
depths one, two and three to synchronous registration.

Each accepted estimate resets its age to zero. A rejection preserves the
previous estimate and increments age, saturating below the never-initialized
sentinel. Expiration marks that historical source unavailable; its finite
parameter storage is retained. A later successful solve can establish new
history. Reset is an explicit preparation/control operation, not a per-frame
CPU field transfer.

Scientific scope
----------------

No reference update, displacement accumulation, velocity prediction or extra
mechanical model is introduced. Raw seeds avoid feeding repeated explicit
median smoothing back into the optimizer. UOD ``replace`` is an exception:
its corrected state replaces a suspect solution in history as well as output.
Neither policy turns convergence into a statistical uncertainty guarantee.

A corrupted frame with no accepted subsets leaves historical parameters
unchanged, but ages them. A corrupted frame that produces an accepted wrong
solution can still poison history. Sudden jumps, missing frames, occlusion,
reappearing regions, discontinuities and model mismatch remain failure modes.
History is not a guarantee that the optimizer reaches the correct basin.
No automatic retry is performed; every target follows its configured pyramid.

The option allocates one extra fine field and one age per fine subset plus
static projection permissions and small per-slot diagnostics. It adds a
projection and a merge pass, not a CPU/GPU round trip. Zero initialization
does not allocate these resources or dispatch these passes. Fewer iterations
may offset their cost, but latency improvement must be measured on the
actual sequence. See :doc:`../how-to/use-temporal-initialization`.
