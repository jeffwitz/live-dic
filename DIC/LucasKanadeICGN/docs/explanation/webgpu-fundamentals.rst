GPU computing for engineers: from Python to WebGPU
======================================================

This chapter assumes that you can read Python and work with numerical arrays.
You do not need experience with graphics, GPU programming, or game engines.
The goal is to understand what live-dic executes, where its time goes, and
which parts you would change to improve it.

Read this chapter first, try :doc:`../tutorials/first-gpu-kernel`, then use
:doc:`../how-to/optimize-webgpu` when changing a kernel. Keep
:doc:`../reference/gpu-vocabulary` open as a vocabulary reference. The exact
live-dic mappings are in :doc:`gpu-optimization` and the resource contract is
in :doc:`shared-kernel`.

Why use a GPU for DIC?
---------------------------

A CPU is well suited to orchestration, decisions, and diverse tasks. A GPU
provides many arithmetic execution lanes and high memory bandwidth. It is
effective when a large number of data items undergo similar operations.
More lanes do not automatically make any program faster: work distribution,
memory traffic, synchronization, and transfers determine the result.

For DIC, thousands of subsets can be solved independently against the same
reference. Within a subset, many pixels contribute to a small set of sums.
This offers two levels of parallelism: solve different subsets concurrently,
and cooperate on the pixels of each subset. The final spatial filter needs
neighbouring results and therefore runs after the solver.

Python does not run once per GPU pixel. Python prepares resources and records
a batch of commands; the GPU executes the numerical loops. Moving a Python
loop into a shader is only useful if enough independent work exists and the
cost of moving data is acceptable.

Which technology does what?
---------------------------

.. list-table:: Names encountered in the repository
   :header-rows: 1
   :widths: 20 40 40

   * - Technology
     - Role
     - Meaning for live-dic
   * - WebGPU
     - An API: rules for allocating GPU resources and submitting commands
     - Supports numerical compute without displaying an image
   * - WGSL
     - A language for writing GPU shader programs
     - Contains the portable IC-GN equations and pyramid operations
   * - wgpu-py, imported as ``wgpu``
     - Python bindings exposing a WebGPU-style API
     - Lets the Python process allocate resources and launch WGSL kernels
   * - wgpu-native
     - Native runtime used by wgpu-py
     - Connects those commands to a supported system GPU API and driver
   * - Naga
     - Shader translation and validation component used by wgpu
     - Processes WGSL on the native path before driver execution
   * - Vulkan / Metal / Direct3D 12
     - Operating-system GPU APIs
     - Possible underlying execution backends; availability depends on the
       runtime, platform, and driver
   * - Three.js
     - JavaScript GPU library
     - Hosts the supported browser implementation and its WebGL2 fallback
   * - TSL, Three.js Shading Language
     - JavaScript functions and nodes describing shader operations
     - Generates GPU programs; the LK expressions remain a separate source
       from the canonical WGSL
   * - WebGL2 / GLSL
     - Graphics API and its shader language
     - Compatibility execution with different cooperation mechanisms
   * - CUDA
     - NVIDIA-specific GPU programming platform
     - A useful comparison, but not the execution API used by this project

The word *Web* does not imply that the native backend starts a browser. The
Python path uses wgpu-py directly. A canvas and a graphical interface are
unnecessary for computation.

.. code-block:: text

   Native path                         Browser paths
   Python + NumPy                      Python adapter or JavaScript application
         |                                      |
      wgpu-py                              Chromium runtime
         |                                 /              \
     wgpu-native                  raw WebGPU/WGSL      Three.js/TSL
         |                                 \              /
   system GPU API                    browser GPU implementation
         |                                      |
      driver ---------------- GPU hardware -----+

The native runtime's compiled libraries are distributed through wgpu-py.
Contributors edit Python and WGSL. The project no longer requires its own
Cargo/PyO3 build or a GLSL-to-SPIR-V compiler. This does not remove the normal
requirement for an installed GPU driver. See :doc:`native-backend`.

An adapter exposes **features** (optional operations) and **limits** (maximum
buffer sizes, workgroup dimensions, shared storage, and other capacities).
The device is created with the requirements needed by the program. A common
API does not give every GPU the same optional features, limits, or speed.
Inspect the selected adapter and qualify any specialization; use the portable
path when its requirements are satisfied. A CPU software adapter is useful
for regression checks but cannot establish a real-time hardware claim.

What is a shader if nothing is drawn?
------------------------------------------

A shader is a program executed on the GPU. Graphics pipelines have shader
stages for vertices and pixels. A **compute shader** instead reads and writes
general resources: arrays, images, intermediate sums, or displacement fields.
In this documentation, *kernel* usually means one compute entry point.

A WGSL file is source text. A shader module makes that source available to the
runtime. A compute pipeline selects an entry point, its resource layout, and
compile-time options. The runtime and driver translate it for the selected
GPU. Creating the module or pipeline can be expensive; reuse them for later
frames. Compilation need not all happen at the first API call on every driver.

The roles are different:

* the **shader** describes the operations;
* the **pipeline** is a configured executable version of those operations;
* the **bind group** supplies actual resources to the pipeline;
* the **dispatch** requests how many groups of invocations to execute.

For example, choosing ``INTERPOLATION_MODE`` when creating a live-dic pipeline
allows compilation of an interpolation-specific variant. Updating an image
in a texture changes data; it does not require recompiling the shader.

How a large problem is divided
------------------------------

An **invocation** is one logical execution of a compute entry point. A
**workgroup** contains a fixed number of invocations declared by the shader.
The CPU launches a grid of workgroups using ``dispatch_workgroups``. The
runtime schedules groups; their order is not an application guarantee.
Invocations are often called threads, and their positions are often called
lanes. Neither name means that Python creates an operating-system thread.

With ``@workgroup_size(16)``, dispatching 240 by 135 groups launches
240 x 135 x 16 = 518,400 invocations. In the portable live-dic solver, each
group owns one subset. At subset size 16, its 16 invocations share the 256
pixels, each processing 16 pixels in a loop. This is different from a
per-pixel image kernel, which might assign one output pixel to each invocation.

.. code-block:: text

   Dispatch grid: 240 columns x 135 rows of independent subsets
   +------------+------------+------------+-----+
   | group 0,0  | group 1,0  | group 2,0  | ... |
   +------------+------------+------------+-----+
   | group 0,1  | group 1,1  | group 2,1  | ... |
   +------------+------------+------------+-----+

   Inside one default 16 x 16 subset:
   16 invocations -> 16 pixels each -> partial sums -> combined sums
                                                        |
                                                   update motion

Three sizes must be kept distinct:

* **Subset size** is a numerical choice: which image samples fit a local warp.
* **Tile size** is a memory reuse choice: which nearby outputs share inputs.
* **Workgroup size** is an execution choice: how many invocations cooperate.

The subset stride sets the distance between subset centres. Reducing it adds
overlapping subsets and increases work. It does not increase the number of
threads inside a workgroup. See :doc:`subset-size`.

A **subgroup** is a smaller execution unit exposed by some devices for
collective operations such as adding lane values. It is not another name for
a workgroup. Its width can depend on the GPU and compiled shader. live-dic
qualifies specialized subgroup paths and retains a portable workgroup path;
``ICGNWGPUFullPyramid`` uses the latter. See :doc:`gpu-optimization` before
changing subgroup assumptions.

Where the data lives
-------------------------

The host is the CPU side, including Python and NumPy. Device resources are
owned by the GPU API. On an integrated GPU they may use the same physical RAM,
but ownership, format conversion, copies, and synchronization still have costs.
An arbitrary NumPy pointer is not a GPU texture.

.. list-table:: Resources and intended use
   :header-rows: 1
   :widths: 24 36 40

   * - Resource
     - What it represents
     - Example
   * - Texture
     - An image with a defined pixel format and access operations
     - The target grayscale image
   * - Storage buffer
     - An indexed array read or written by shader invocations
     - Reference samples, inverse Hessians, parameter states
   * - Uniform buffer
     - A small read-only configuration block
     - Image dimensions, grid dimensions, numerical thresholds
   * - Workgroup memory
     - Scratch storage shared only within one group
     - A filter tile or partial sums
   * - Invocation-local variables
     - Temporary state private to one invocation
     - Accumulated sums and current coordinates
   * - Readback buffer
     - A resource used to make completed GPU results accessible to the CPU
     - Packed parameters, status, and iteration count

Private variables often use registers, but large live arrays can spill into
slower memory. Workgroup storage also has a finite capacity and can limit how
many groups execute concurrently. Increasing either is not free.

``r8unorm`` means one unsigned normalized eight-bit channel: texture reads
return a float in [0, 1] representing the stored integer divided by 255.
``r32float`` means one 32-bit floating-point channel. They are not equivalent
formats. A byte value of 137 read as UNORM is approximately 137/255, not 137.
live-dic's shader converts units where needed. The interpolation and reduction
code must agree with reference preparation on both units and rounding.

A texture can be read with ``textureLoad`` at an integer pixel coordinate.
A sampler supplies rules for sampling operations, including filtering and
addressing. Hardware texture filtering is not automatically equivalent to
our cubic B-spline reconstruction or our explicit pyramid reduction. Mipmap
levels are reduced images; producing them does not perform LK or transfer
the polynomial displacement field between scales. See :doc:`interpolation`.

How the CPU connects these resources to a shader
------------------------------------------------

The shader declares numbered resource slots with ``@group`` and ``@binding``.
The host creates a matching **bind-group layout**, describing the resource
types, and a **bind group**, containing specific buffers and texture views.
Changing a binding number on one side alone breaks that contract.

A texture view specifies how the texture is exposed to a binding. A buffer's
usage flags declare permitted operations, such as shader storage, upload,
copy source, or CPU mapping. They are checked by the API; they do not make
data transfers occur automatically.

Byte layout matters. WGSL vectors and structures have alignment rules;
three floats do not always imply a twelve-byte structure or array stride.
NumPy dtype, byte count, padding, and shader indexing must agree. The
project's aligned records and exact binding table are documented in
:doc:`shared-kernel`; :doc:`../reference/result-layout` describes public fields.

Synchronization: two different problems
---------------------------------------

Inside one workgroup, invocations may cooperate through workgroup memory.
If one invocation writes a partial sum and another reads it, a barrier is
needed at the appropriate point. All participating invocations must reach
barriers through valid uniform control flow. An early return in half the
group before a barrier is an error, even if their output pixels are outside
the image. A barrier does not synchronize other workgroups.

Across workgroups, a later dependent dispatch provides a global stage
boundary when resources are used correctly. The host records the stages
in order; WebGPU handles the required resource synchronization. live-dic
uses separate compute passes for those dependencies. No Python wait is
required between a horizontal filter and its vertical consumer.

An encoder records commands; finishing it creates a command buffer. Submitting
that buffer to a queue schedules work. Submission returning does **not** mean
the result has been computed. Reading a completed result requires a completion
mechanism, such as mapping an appropriate readback buffer after its copy.

.. code-block:: text

   Python: encode passes -> submit ---------------------> map/read result
                                |                              ^
   GPU:                         +-> filter -> solve -> copy ---+

Both the GPU dependency graph and the Python result-availability boundary
matter. Timing only ``queue.submit`` measures submission overhead and misses
most of the calculation.

Follow one live-dic target
-------------------------------

``set_reference`` prepares reference samples, derivatives, inverse Hessians,
and metadata. The full pyramid retains them at every scale. This preparation
is reused for subsequent targets, until the reference or configuration changes.

For each target, the recommended native engine:

#. accepts and normalizes a grayscale NumPy image;
#. uploads the target once;
#. constructs reduced target levels on the GPU;
#. prepares B-spline coefficients when that interpolation is selected;
#. solves the coarsest field and transfers estimates to finer levels;
#. refines the field and applies the configured spatial policy;
#. copies the final packed state for readback and constructs the NumPy result.

The iterations within each subset are fused in the solver dispatch. Pyramid
levels and spatial filtering remain separate passes because they consume
results from other workgroups. GPU residency means that intermediate images
and fields stay in GPU resources, not that the entire algorithm must occupy
one enormous shader.

For streams, independent mutable resources are needed for each in-flight
target. Pipeline depth is the number of frames in flight; pyramid depth is
the number of image scales. More frames in flight can increase throughput
while increasing the age of each returned result. See
:doc:`realtime-performance` and :doc:`supervision`.

Where to read next
------------------

* Run :doc:`../tutorials/first-gpu-kernel` to see allocation, binding,
  dispatch, and readback in one small program.
* Consult :doc:`../reference/gpu-vocabulary` for terms and source locations.
* Follow :doc:`../how-to/optimize-webgpu` to investigate a slow stage.
* Read :doc:`gpu-optimization` for the actual tiling and reduction strategies.

Upstream references
-------------------

The `WebGPU specification <https://www.w3.org/TR/webgpu/>`_ defines the API,
and the `WGSL specification <https://www.w3.org/TR/WGSL/>`_ defines the shader
language, memory layout, and synchronization rules. These are detailed
references rather than first tutorials. The `wgpu-py guide
<https://wgpu-py.readthedocs.io/en/stable/guide.html>`_ explains its Python API.
For the separate browser implementation, see the `Three.js TSL documentation
<https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language>`_.
