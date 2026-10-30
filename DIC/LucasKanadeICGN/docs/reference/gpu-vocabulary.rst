GPU and WebGPU vocabulary
==============================

Use this page as a lookup reference. For a first explanation, start with
:doc:`../explanation/webgpu-fundamentals`.

.. glossary::
   :sorted:

   Adapter
      An available GPU implementation and its capabilities. A software
      implementation can also appear as an adapter. Check its identity before
      interpreting a benchmark.

   Device
      The API object used to allocate GPU resources and create pipelines after
      requesting supported features and limits from an adapter.

   Host
      The CPU side of the application: Python or JavaScript orchestration,
      input preparation, scheduling, and result consumption.

   Shader
      A GPU program. A compute shader can calculate numerical fields without
      drawing anything. WGSL is a shader language; a shader module is the API
      object holding a program's source for pipeline creation.

   SPIR-V
      An intermediate representation of shader programs, used by Vulkan and
      other toolchains. live-dic distributes WGSL source rather than maintaining
      precompiled SPIR-V binaries. The runtime handles downstream translation.

   PyO3
      Rust bindings for Python extensions. The retired native prototype used
      it; the current project uses wgpu-py's distributed runtime and does not
      require contributors to compile that prototype.

   Feature
      An optional adapter capability that may need to be requested when
      creating a device. Its availability is independent of performance.

   Limit
      A resource or execution bound, such as maximum workgroup invocations
      or buffer size. Validate requirements against the selected device.

   Kernel
      In these pages, a compute shader entry point and the operation it
      implements, such as a filter pass or subset solve.

   Invocation
      One logical execution of a shader entry point. Frequently called a
      thread; it is not a Python thread.

   Lane
      Informal name for an invocation's position within a cooperating group
      or subgroup. Always specify which group is meant.

   Workgroup
      A fixed-size set of compute invocations that can share scratch memory
      and synchronize with workgroup barriers. Its size is declared in WGSL.

   Subgroup
      A device-supported execution unit with collective operations such as
      ``subgroupAdd``. It can be smaller than a workgroup. Its width must not
      be assumed from a GPU marketing name or a different shader.

   Dispatch
      A request to execute a grid of workgroups. Group counts are multiplied
      by the shader's workgroup dimensions to obtain invocation counts.

   Pipeline
      A shader entry point configured with layouts and optional compile-time
      constants. Reuse it across frames; it is not a single submitted frame.

   Override
      A WGSL constant whose value can be supplied at pipeline creation, allowing
      a specialized executable. It differs from a uniform updated as data.

   Binding
      A numbered resource slot declared in WGSL and populated by the host.

   Bind group
      A concrete set of buffers, texture views, or samplers assigned to
      bindings. Its layout specifies the permitted types and access modes.

   Storage buffer
      A typed shader view of an indexed memory region, used for substantial
      arrays and writable results. Host and shader must agree on byte layout.

   Uniform buffer
      A read-only configuration resource. Its layout rules differ from those
      of ordinary tightly packed CPU arrays.

   Texture
      An image resource with dimensions, format, and declared usage. It may
      hold input intensities, coefficients, or intermediate numerical images.

   Sampler
      A resource defining how sampling operations filter and address texture
      coordinates. Integer ``textureLoad`` operations do not use a sampler.

   UNORM
      Unsigned normalized storage: integer values are exposed as normalized
      floats when sampled. ``r8unorm`` stores one byte per pixel and exposes
      intensities divided by 255.

   Workgroup memory
      Shader scratch storage shared within one workgroup. It is not shared
      between groups and does not persist as an output between dispatches.

   Barrier
      A synchronization operation with a defined execution and memory scope.
      A workgroup barrier cannot synchronize the entire dispatch.

   Uniform control flow
      Control flow satisfying WGSL's uniformity requirements for operations
      such as barriers. A condition varying between invocations cannot safely
      decide whether only some of them reach a workgroup barrier.

   Reduction
      Combining many values into fewer values, for example summing pixel
      contributions to a Gauss-Newton increment. A tree reduction combines
      pairs in successive stages.

   Tile
      A block of outputs processed together to reuse inputs. Its halo contains
      the additional neighbouring samples needed at the block's edges.

   Coalescing
      Hardware combining adjacent lane memory accesses into efficient memory
      transactions. Layout and access patterns influence it; WGSL does not
      guarantee a universal transaction size.

   Occupancy
      How much concurrent execution a GPU can sustain for a kernel. Registers,
      workgroup storage, and hardware limits can restrict active groups.
      Maximum occupancy alone does not guarantee minimum runtime.

   Divergence
      Different invocations following different branches or loop counts.
      Hardware may execute paths with some lanes inactive. It also matters
      for the legality of collective operations.

   Command encoder
      The host object recording passes and copies. Finishing it produces a
      command buffer; recording alone performs no requested GPU calculation.

   Compute pass
      A command-recording scope containing compute dispatches. live-dic uses
      separate passes to order dependent global stages.

   Queue
      The API object accepting command buffers and scheduling resource writes.
      A completed submit call is not proof that GPU computation has finished.

   Readback
      Making a completed device result accessible to the CPU, usually through
      a copy into a mappable buffer and a completion wait.

   Mapping
      Obtaining CPU access to a compatible buffer. Mapping is governed by
      buffer usage and ownership rules; a mapped buffer is not simultaneously
      available for arbitrary GPU use.

   GPU residency
      Retaining intermediate data in device resources between stages, avoiding
      repeated CPU readback and upload. This does not mean one shader or one
      dispatch for the complete algorithm.

   Pipeline depth
      Number of target frames allowed in flight in a bounded stream. Different
      from pyramid depth, which counts image scales.

Where to find the implementation
-------------------------------------

Paths below are relative to ``DIC/LucasKanadeICGN/python/src/lucas_kanade_icgn``.
They identify source responsibilities, not additional public Python APIs.

.. list-table:: Reading map
   :header-rows: 1
   :widths: 32 34 34

   * - File
     - Responsibility
     - Start reading here
   * - ``wgpu_backend.py``
     - Native device, resources, pipeline selection, and one-level execution
     - ``ICGNWGPU.__init__``, ``_create_bind_group_layout``,
       ``_encode_registration_passes``
   * - ``full_pyramid_backend.py``
     - Persistent levels and bounded full-pyramid execution
     - ``ICGNWGPUFullPyramid``, ``set_reference``, ``register_stream``
   * - ``shaders/icgn.wgsl``
     - Portable interpolation, IC-GN equations, and spatial post-processing
     - ``prefilter_x``, ``solve_subset``, ``median_filter``
   * - ``shaders/full_pyramid.wgsl``
     - Target reduction and field transfer between scales
     - ``reduce_x``, ``reduce_y``, ``transfer_seed``
   * - ``shaders/icgn_subgroup.wgsl``
     - Optional specialized native affine solver
     - ``main``; compare with the portable equations
   * - ``shaders/subgroup_probe.wgsl``
     - Executed capability and collective-operation check
     - The probe and its Python result validation
   * - ``web/webgpu/LucasKanadeICGNWebGPU.js``
     - Browser host for canonical WGSL resources and commands
     - Resource creation and dispatch encoding
   * - ``web/three/LucasKanadeICGNThree.js``
     - Separate Three.js/TSL implementation
     - Workgroup/subgroup solver construction and WebGL2 fallback
   * - ``engine.py``
     - Python browser process and binary transport adapter
     - Engine lifecycle and frame/result transport

The shared buffers and status conventions are specified in
:doc:`../explanation/shared-kernel`. Public result fields are specified in
:doc:`result-layout`. The native feature is named ``subgroup`` in the runtime
used here; browser WebGPU uses ``subgroups``. See
:doc:`../explanation/native-backend` for the qualified loading behaviour.

Recognizing common failures
----------------------------

.. list-table:: A diagnostic is a starting point, not a substitute for validation
   :header-rows: 1
   :widths: 29 35 36

   * - Symptom
     - Likely area to inspect
     - First check
   * - No adapter available
     - Driver/runtime installation or execution environment
     - Enumerate adapters and inspect identity/backend
   * - Shader parsing or validation error
     - WGSL syntax, types, optional features
     - Read the reported shader location and device requirements
   * - Binding/layout validation error
     - Resource type, usage flags, access mode, or size
     - Compare both host declarations and shader bindings
   * - Barrier/uniformity diagnostic
     - Invocation-dependent branches or early exits
     - Ensure all required invocations reach collective operations
   * - Correct first rows, wrong later rows
     - Dispatch geometry or flattening
     - Check row stride and both group-coordinate components
   * - Finite field with poor accuracy
     - Units, layout, normalization, convergence, conditioning
     - Compare to known motion and inspect rejected/accepted masks
   * - Unexpectedly old results
     - In-flight frames or unbounded acquisition queue
     - Measure result age and enforce backpressure

Compilation success does not prove correct indexing or numerical validity.
Out-of-bounds accesses are errors, even if one driver appears to tolerate them.
A device loss is different from a candidate-local failure: returning to another
pipeline on the same lost device cannot restore execution.
