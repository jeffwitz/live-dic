Your first GPU calculation from Python
======================================

This tutorial computes ``output[i] = 2 * input[i]`` on a GPU. Its purpose is
to make the WebGPU objects concrete before you read the DIC shaders. Knowledge
of NumPy is sufficient; read :doc:`../explanation/webgpu-fundamentals` for the
technology overview.

Install and run
---------------

Use the project's native installation, described in :doc:`../how-to/install`.
It supplies NumPy and wgpu-py. You need a working GPU driver; no browser or
canvas is required. This example uses the synchronous wgpu-py 0.32 API used
by the project.

Download :download:`vector_scale.py <../examples/vector_scale.py>` and run:

.. code-block:: console

   $ python3 vector_scale.py
   <adapter name>: 1003 correct values

From a checkout, the path is
``DIC/LucasKanadeICGN/docs/examples/vector_scale.py``. The actual adapter name
depends on the machine. A software adapter can verify the calculation, but
its execution time does not describe a hardware GPU.

The complete program
--------------------

.. literalinclude:: ../examples/vector_scale.py
   :language: python

Understand the shader
---------------------

``array<f32>`` is a runtime-sized array of 32-bit floats. NumPy supplies
``float32`` values, so each element occupies four bytes on both sides.
Binding 0 is read-only input; binding 1 is writable output. The host's bind
group assigns the two buffers to those same numbers.

``@compute`` marks the entry point. ``@workgroup_size(64)`` requests 64
invocations per group. ``global_invocation_id.x`` identifies the element
assigned to an invocation. With 1003 elements, we dispatch 16 groups,
providing 1024 invocations. The final 21 invocations return without reading
or writing data. This return is valid because the shader contains no
workgroup barrier.

This mapping is intentionally simpler than live-dic's solver: here one
invocation computes one element, whereas the portable LK solver assigns a
cooperating workgroup to one subset.

Understand the Python objects
-----------------------------

#. Request an adapter, then create a device. The adapter represents a GPU
   implementation; the device owns our resources.
#. Upload the input and allocate the output. ``STORAGE`` permits shader
   access; ``COPY_SRC`` permits copying the output for readback.
#. Create a shader module, then a pipeline selecting ``scale``. For this small
   example, ``layout="auto"`` infers a layout from the shader. live-dic uses
   explicit layouts to maintain its shared resource contract.
#. Create a bind group connecting resource numbers to actual buffers.
#. Record the pipeline, bind group, and group count into a compute pass.
#. Finish and submit the command buffer. The GPU can now execute the work.
#. Read the output. ``queue.read_buffer`` is a wgpu-py convenience operation
   which handles a readback copy and waits for completion. It is not a
   zero-cost view of device memory. live-dic manages reusable readback
   resources explicitly for streaming.

The NumPy assertion checks every element. Multiplication by two is exact for
these small values; more complex floating-point reductions require appropriate
tolerances. The example releases buffers and the device after the result has
been checked.

Experiments to build intuition
-------------------------------

Change the array length to 1024 and then 1025; the same ceiling division and
guard handle both. Change the multiplier in WGSL and the NumPy expectation
together. Change the workgroup size only if you also change the dispatch
calculation. Keep within the selected device's limits.

The example is too small to establish a GPU speed advantage. It creates a
device, uploads data, compiles a pipeline, and performs a synchronous readback
for one multiplication. A useful stream retains those objects and performs
enough work per transfer. Follow :doc:`../how-to/optimize-webgpu` for a method
of measuring and improving that kind of application.
