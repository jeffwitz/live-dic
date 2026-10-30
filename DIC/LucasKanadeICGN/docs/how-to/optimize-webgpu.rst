Develop and optimize a WebGPU compute kernel
=================================================

Use this guide when a GPU stage is too slow or when adding a numerical
operation. First read :doc:`../explanation/webgpu-fundamentals` and run
:doc:`../tutorials/first-gpu-kernel`. The exact live-dic kernel shapes are
specified in :doc:`../explanation/gpu-optimization`.

Establish correctness and the timing boundary
--------------------------------------------------

Write down input dtype, units, array shape, border rule, and expected output.
A CPU implementation or analytic field provides an independent reference.
Include status and acceptance masks, not just the error averaged over retained
results. For DIC, evaluate displacement, gradient, and curvature when the
selected model estimates them. Include fractional motion, noise, localized
features, rejected subsets, and small/odd image dimensions. A smooth field
may reward smoothing while hiding the loss of a localized feature.

The primary Python time boundary starts with a resident NumPy image and stops
with the complete NumPy field. Record preparation separately. Warm up, retain
individual samples, and report median, p95, accepted fraction, and numerical
errors together. See :doc:`benchmark` for repository commands. Preserve input
hashes, commit, configuration, selected solver, runtime, GPU, and driver.
Interleave A/B measurements where possible; a small difference between
separate runs can reflect thermal or scheduling variation. Software adapters
can check behaviour, but their times do not qualify a hardware GPU.

Find the expensive stage
------------------------

.. code-block:: text

   normalization -> upload -> encode/submit -> GPU graph
                                  -> completion/readback -> NumPy construction

These categories are not necessarily independently additive timer values:
CPU and GPU work can overlap. A completion wait covers whatever queued work
has not finished when the host requests the result; it is not an isolated
shader duration. GPU timestamps can help isolate stages when supported, but
exclude input transport and result consumption. Timing only ``queue.submit``
misses most of the calculation.

.. list-table:: Symptom to investigation
   :header-rows: 1
   :widths: 28 36 36

   * - Observation
     - Inspect
     - Candidate change to measure
   * - First call is slower
     - Compilation, allocation, lazy initialization
     - Retain device, pipelines, and resources
   * - Good GPU time, slow Python call
     - Upload, browser transport, readback, conversions
     - Reduce representation changes and process crossings
   * - Filtering dominates
     - Repeated neighbourhood reads
     - Separable filtering and cooperative tiles
   * - Solver dominates
     - Sampling, iteration counts, reductions, local storage
     - Reuse work and examine lane mapping
   * - More threads make it slower
     - Barriers, registers, shared memory, inactive lanes
     - Try smaller groups and less temporary state
   * - Higher FPS, older results
     - In-flight frames and queue depth
     - Bound resources and compare result age

Keep stable resources out of the frame loop
------------------------------------------------

Create the device and pipelines once. Allocate textures, intermediates, bind
groups, and readback slots for the geometry. Prepare reference gradients and
inverse Hessians once per reference. Update target data and small parameters
instead of rebuilding the engine.

Pipeline overrides can specialize stable choices such as interpolation and
motion model. Cache the required variants. Many variants add preparation
time and memory; frequently changed thresholds may be better expressed as
configuration data. Compilation costs still belong in startup and reference
change latency measurements.

Choose a work decomposition
---------------------------

List independent outputs and dependencies. An elementwise operation can use
one invocation per output. LK can use one cooperating workgroup per subset.
An image filter can use a rectangular tile to reuse neighbouring input values.

Dispatch counts count workgroups. For W by H outputs with tiles Tx by Ty,
dispatch ``ceil(W/Tx)`` by ``ceil(H/Ty)`` groups. Define how local IDs map to
outputs. For subset grids, flatten a group coordinate consistently as
``gy * grid_width + gx``; forgetting the row index silently solves wrong rows.

Do not assume that groups are all resident or execute in grid order. A group
spinning until another finishes can prevent that producer from executing.
Use ordered stages for global dependencies.

Improve memory access
----------------------

Prefer adjacent invocations reading adjacent contiguous elements where
possible. This encourages efficient memory transactions. Whether coefficients
should be stored as separate arrays or per-subset records depends on which
values each lane accesses. Measure the complete pattern before replacing a
layout.

A 3840 x 2160 uint8 image occupies 8,294,400 bytes, about 8.29 MB. Expanding
it to float32 creates about 33.18 MB before intermediate images. A 240 x 135
affine state grid at eight float32 values per subset occupies 1,036,800 bytes.
Uploading bytes and reading only final states can therefore save substantial
traffic compared with moving full floating-point images between stages.

Packed inputs still need conversion into the shader's expected units.
``r8unorm`` saves input storage, but samples are floats divided by 255.
``vec4`` simplifies aligned records; it does not promise a fourfold speedup.
Larger records can increase memory traffic. Keep host dtype, byte count,
padding, and shader indexing consistent with :doc:`../explanation/shared-kernel`.

Use tiles when outputs reuse input samples
------------------------------------------

A 25-tap horizontal filter evaluated independently for 32 x 8 outputs expresses
6,400 source reads. A shared input tile with a 12-pixel halo on each side needs
only ``(32 + 24) * 8 = 448`` source values. Invocations load these together,
synchronize, then reuse them from workgroup memory.

.. code-block:: text

   Shared input tile: 56 columns x 8 rows
   +------------+--------------------------------+------------+
   | 12-column  |        32 output columns       | 12-column  |
   | left halo  |        across 8 rows           | right halo |
   +------------+--------------------------------+------------+
          load inputs -> barrier -> compute valid outputs

This reduction in expressed reads is not a measured 14x speedup. Caches,
shared-memory instructions, address calculations, barriers, and occupancy
affect the result. Compare both time and output values.

Handle the final partial tile: all invocations needed for collective loads
must reach the barrier before invalid outputs return. Validate even unused
halo reads; unused accesses must still be legal. Test small pyramid levels.
Use separable horizontal/vertical filters when their mathematics permits it;
an arbitrary two-dimensional convolution is not necessarily separable.

Reduce sums and control synchronization
---------------------------------------

Accumulate pixel contributions privately, then combine lane partial sums once
per iteration rather than contending on a shared sum for every pixel. A tree
reduction combines pairs through workgroup memory:

.. code-block:: text

   16 lane values -> 8 partial sums -> 4 -> 2 -> 1 complete sum

Stages reading another lane's writes require appropriate barriers. If lane 0
updates the motion, make the new state and exit decision visible uniformly
before the next collective operation. Follow the existing
``workgroupUniformLoad`` pattern; individual lanes must not abandon a loop
while other participants still need its barriers.

Subgroup reductions can avoid shared-memory exchanges. Request the required
feature and qualify the actual production mapping. A probe reporting width
32 does not guarantee that a different compiled solver uses width 32. Never
normalize a partial sum as if every subset pixel contributed. Keep a portable
path and compare valid variants on the target machine.

Control temporary storage and occupancy
---------------------------------------

More private variables, local matrices, and shared tile storage can reduce
the number of concurrent groups. Register spills introduce memory accesses.
Quadratic LK carries more coefficients than affine LK; larger subsets also
increase sampling work. Counting arithmetic instructions alone can miss the
actual bottleneck.

Change one resource choice at a time: tile dimensions, lane count, packing,
or reuse of scratch arrays. Compare several sizes and models. The shared
kernel reuses reduction storage for quadratic products because extra declared
storage affected the affine path; see :doc:`../explanation/shared-kernel`.
Maximum occupancy alone does not guarantee the fastest kernel.

Fuse local work and retain global stage boundaries
--------------------------------------------------

Keeping local Gauss-Newton iterations inside one dispatch avoids repeated
submissions and intermediate readbacks. Stop sampling converged subsets using
legal collective control flow.

Pyramid reduction, seed transfer, and neighbour filtering consume outputs
from other workgroups. Record these as ordered passes in one command buffer.
Workgroup barriers cannot provide the global synchronization required to
combine them into a single arbitrary shader. GPU residency is achieved by
retaining the intermediate resources between passes.

The median stage reads raw state and writes a separate filtered state.
Overwriting records while reading neighbours would make the answer depend
on execution order. Distinct input/output resources give a defined operation.

Treat precision and numerical shortcuts as experiments
------------------------------------------------------

Float32 differs from NumPy float64, and parallel sums change addition order.
Use established scientific tolerances and independent ground truth rather
than requiring bitwise identity for every reduction. Float16 arithmetic needs
feature support and analysis of range, rounding, conditioning, and error.
A half-float texture format does not automatically accelerate all arithmetic:
storage precision and calculation precision are different choices.

An intermediate rounded just below an integer can change subsequent uint8
truncation. live-dic's pyramid reduction therefore uses an integer-unit
float32 convention. Shorter filters, lower iteration limits, linear
interpolation, and smoothing also need renewed accuracy qualification.
Do not loosen acceptance thresholds to make a faster variant appear equivalent.

Finish with reproducible qualification
--------------------------------------

Run an independent oracle, edge and partial-tile cases, then numerical GPU
regressions using :doc:`benchmark`. Compare portable and specialized variants.
When canonical WGSL changes, exercise native and raw-browser consumers.
Three.js/TSL is a separate implementation and needs its own equation-parity
check.

Measure changing frames through the public API with the intended queue depth.
Retain samples and quality metrics. Keep a speed improvement only when it is
repeatable relative to observed variability. Update the shared resource
contract and the :doc:`../explanation/gpu-optimization` mapping table whenever
those change.

The `WebGPU specification <https://www.w3.org/TR/webgpu/>`_ defines API rules;
the `WGSL specification <https://www.w3.org/TR/WGSL/>`_ defines language,
memory layout, and synchronization rules. Performance choices require
experiments on the actual hardware.
