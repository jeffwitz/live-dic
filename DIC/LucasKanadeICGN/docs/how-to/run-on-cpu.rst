Run the native solver on CPU
=============================

Use the native engines on a machine without a hardware GPU, or select CPU
execution explicitly for validation and offline processing. The same WGSL
shaders execute through a software WebGPU adapter. This is not a separate
NumPy/C++ implementation of the LK solver.

Select the execution device
---------------------------

Both ``ICGNWGPU`` and ``ICGNWGPUFullPyramid`` accept ``device_type``:

.. list-table:: Selection occurs when constructing the engine
   :header-rows: 1
   :widths: 18 48 34

   * - Value
     - Selection
     - If unavailable
   * - ``"auto"`` (default)
     - Prefer an enumerated hardware adapter; use a CPU adapter only if no
       hardware adapter is enumerated
     - Raise ``ICGNError`` when neither type is available
   * - ``"gpu"``
     - Require a hardware adapter
     - Raise ``ICGNError``; never select CPU
   * - ``"cpu"``
     - Require an adapter reported as CPU, even when hardware is available
     - Raise ``ICGNError``; never select hardware

Vulkan is preferred within the selected type. Among equally ranked backends,
``power_preference="high-performance"`` prefers discrete hardware and
``"low-power"`` prefers integrated hardware. Enumeration order breaks any
remaining ties. CPU selection does not request a particular core count.

The automatic policy does not retry every initialization failure on a CPU.
Feature, shader, allocation, and device-loss errors remain errors. In
particular, it cannot migrate an in-flight sequence from a failed GPU to CPU.
Recreate the engine and prepare the reference again to change devices.

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(
       device_type="cpu",
       motion_model="translation",
       subset_size=16,
       subset_stride=16,
       interpolation="bspline",
       outlier_policy="median",
   ) as engine:
       print(engine.execution_device_type)  # cpu
       print(engine.adapter_info)           # actual adapter/backend/driver
       preparation = engine.set_reference(reference)
       print(preparation.device)
       result = engine.register(target)
       assert result.config["executionDeviceType"] == "cpu"

The default remains the two-level full pyramid. Result shapes, motion models,
spatial policies, and bounded ``register_stream`` execution are unchanged.
``outlier_policy="median"`` is explicit smoothing; see
:doc:`../explanation/displacement-outliers` before interpreting filtered
derivatives.

Install the required runtime
----------------------------

Install the standard package using :doc:`install`. A software GPU
driver must also be provided by the system. The Python package does not
download or install drivers.

On Debian/Ubuntu, Mesa's Vulkan software driver is normally supplied by:

.. code-block:: console

   $ sudo apt install mesa-vulkan-drivers

Package names and availability depend on the distribution. No Mesa source
build is required when the distribution supplies compatible binaries. On
Windows or macOS, CPU execution likewise requires an installed implementation
exposed as a CPU adapter by wgpu; it is not guaranteed by installing the
Python wheel. This release qualifies CPU execution on Linux/Mesa only.

Inspect the implementations visible to wgpu:

.. code-block:: python

   import wgpu

   for adapter in wgpu.gpu.enumerate_adapters_sync():
       print(dict(adapter.info))

On the validation machine, the software Vulkan adapter reports
``adapter_type="CPU"``, ``backend_type="Vulkan"``, and a device name containing
``llvmpipe``. Mesa's Vulkan software implementation is commonly called
lavapipe; an OpenGL software adapter can also identify itself as llvmpipe.
Always inspect the backend as well as the device name.

There is no need to hide the hardware GPU or set environment variables for
``device_type="cpu"``. If driver discovery needs an ICD manifest override,
use the actual installed Vulkan manifest path. Filenames vary across systems;
an incorrect path can hide Vulkan and accidentally leave only another backend.
Explicit CPU selection cannot install a missing software implementation.

Diagnostics and exported results
--------------------------------

``engine.device_type`` is the normalized requested policy.
``engine.execution_device_type`` is the actual ``"cpu"`` or ``"gpu"``.
``engine.adapter_info`` describes the selected implementation. Every native
``RegistrationResult.config`` includes ``deviceType`` and
``executionDeviceType``. They therefore survive exports that preserve the
configuration. ``ReferenceInfo.device`` continues to name the adapter/backend.

These diagnostics describe where the solver executes. Reference preparation
and host orchestration already involve the CPU in hardware GPU mode.

Measure CPU execution
----------------------

The image-to-field benchmark supports the same selection:

.. code-block:: console

   $ PYTHONPATH=DIC/LucasKanadeICGN/python/src python3 \
       DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
       --backend native-pyramid --device-type cpu \
       --width 3840 --height 2160 --motion-model translation \
       --subset-size 16 --subset-stride 16 --interpolation bspline \
       --outlier-policy median --pipeline-depth 1 --warmups 8 --frames 50

The benchmark's standard inputs are changing tiled fixtures. They differ from
the analytic noise experiment below and should not be treated as the same
accuracy qualification. The browser CLI ``icgn-webgl`` remains a browser
adapter; ``device_type`` is a native Python engine option, not a Chromium flag.

A 2026-10-30 analytic UHD experiment on the Intel Core Ultra 5 228V measured
translation-only 16/16, B-spline, explicit median, two levels, independent
Gaussian noise of sigma 1 grayscale unit, 8 warm-ups and 50 calls per case:

.. list-table:: CPU software Vulkan, complete NumPy-image to NumPy-field boundary
   :header-rows: 1

   * - Analytic input field
     - Median / p95
     - Measured throughput
   * - Rigid translation
     - 218.2 / 236.7 ms
     - 4.56 FPS
   * - Affine deformation, fitted with translation-only subsets
     - 244.0 / 304.7 ms
     - 4.00 FPS
   * - Bounded curved field, fitted with translation-only subsets
     - 223.9 / 253.6 ms
     - 4.39 FPS

These figures use Mesa 25.2.8 / LLVM 20.1.2 and wgpu-py 0.32.0. They exclude
reference preparation and acquisition. Throughput is the inverse mean call
time, not the inverse median. They qualify neither affine/quadratic solver
performance on CPU nor a sustained acquisition stream. CPU execution is useful
for slower acquisition, offline analysis, and numerical checks; it is not a
4K60 claim. Use :doc:`benchmark` to measure the actual workload and thread load.
