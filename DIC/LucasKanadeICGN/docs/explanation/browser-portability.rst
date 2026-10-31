Portable browser engine and resource budget
===========================================

The browser front-end uses the canonical WGSL sources packaged with Python.
The JavaScript host is being separated from the Python binary-transport shim;
the equations must not acquire a second browser implementation.

Phase 0: binding audit (pre-change b9d23a2)
------------------------------------------

WebGPU's core guaranteed limits include eight storage buffers per shader stage,
four bind groups, 128 MiB per storage binding, a 256 MiB maximum buffer,
256 invocations per workgroup, 16 KiB workgroup storage and 8192 pixels per
texture dimension. Check actual adapter limits before constructing resources.
See the `WebGPU limits specification <https://www.w3.org/TR/webgpu/#limits>`_.
Limits count the entire explicit pipeline layout, including unused bindings,
across all groups visible to the stage, not just expressions executed by a
particular shader entry point.

.. list-table:: Existing compute pipeline layouts before packing
   :header-rows: 1

   * - Pipeline family
     - Storage buffers
     - Uniforms
     - Sampled / storage textures
     - Bind group slots
   * - IC-GN, FIR, median, native subgroup (shared base layout)
     - 8
     - 1
     - 1 / 0
     - 1
   * - Seeded IC-GN / retry / partial
     - **10**
     - 1
     - 1 / 0
     - 1
   * - Separable or fused image reduction
     - 0
     - 0
     - 1 / 1
     - 1 or 2
   * - Coarse-to-fine seed transfer
     - 3 (4 with ROI links)
     - 1
     - 0 / 0
     - 3
   * - Temporal projection / merge
     - 5
     - 1
     - 0 / 0
     - 3 / 4
   * - Velocity prediction / previous-state save
     - 6
     - 0 / 1
     - 0 / 0
     - 3 / 4
   * - Fine temporal correction
     - 7
     - 2
     - 0 / 0
     - 3

The temporal entry points use separate layouts: group 2 and group 3 are not
both declared in the same temporal pipeline layout. Empty preceding groups
consume group slots, but add no buffer bindings. Counting their module-wide
declarations together would overstate the actual pipeline requirement.

.. list-table:: Default 16/16 grid resource sizes
   :header-rows: 1

   * - Input
     - Grid
     - Reference vec4 buffer
     - Two float32 FIR planes
   * - 1280 x 720
     - 80 x 45
     - 14,745,600 bytes
     - 7,372,800 bytes
   * - 1920 x 1080
     - 120 x 67
     - 33,177,600 bytes
     - 16,588,800 bytes
   * - 3840 x 2160
     - 240 x 135
     - 132,710,400 bytes
     - 66,355,200 bytes

UHD fits a 128 MiB binding for the reference alone, but appending its inverse
Hessians and metadata would exceed that guaranteed limit. The chosen packing
therefore combines the two FIR scratch planes and combines metadata with
inverse Hessians, leaving the large reference separate. This reduces the
base layout from eight to six storage bindings and seeded solving from ten
to eight, without adding a copy, dispatch or floating-point operation.
The group indices, temporal mathematics and partial-support payload remain
unchanged. Qualification results are recorded with the implementation.

Static buffer limits do not prove that a device has enough memory to run UHD.
Capability checks must also account for all levels, reference preparation,
mutable slots and history buffers. Mobile devices are not assumed tested.
