Install the Python runtime
==========================

System requirements
-------------------

The browser backends require:

* Python 3.10 or newer;
* Chromium or Google Chrome with WebGL2 support; WebGPU is preferred;
* a GPU and driver accepted by Chromium.

The optional native backend requires Python 3.11 or newer and a platform
supported by ``wgpu-py``. It does not require Chromium.

The package embeds GPU-IO, Three.js r186, the raw WebGPU adapter, all IC-GN
shaders, and their headless pages. It does not require a repository checkout
or a network connection at runtime.

Install the package
-------------------

From the repository root:

.. code-block:: console

   $ python3 -m venv .venv-icgn
   $ .venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python

Verify the command-line interface:

.. code-block:: console

   $ .venv-icgn/bin/icgn-webgl --help

Verify the headless GPU runtime:

.. code-block:: console

   $ .venv-icgn/bin/icgn-webgl \
       DIC/LucasKanadeICGN/assets/frame_000.png \
       DIC/LucasKanadeICGN/assets/frame_010.png \
       --truth 2 -1

The JSON output should name the GPU, report a 32 x 32 grid, and contain 900
accepted subsets for this test pair.

Select a compute backend
------------------------

Defaults are scoped to the chosen engine. See
:doc:`../explanation/backend-selection` for the complete browser, native, and
pyramid decision table.

Three.js/TSL is the default backend. A standard engine therefore selects it:

.. code-block:: python

   engine = ICGNWebGL()

Select the canonical portable WGSL
with ``ICGNWebGL(backend="webgpu-wgsl")`` or ``--backend webgpu-wgsl``. This
raw WebGPU adapter executes the same IC-GN and full-pyramid shader files as
wgpu-py and defaults to two levels; pass ``pyramid_levels=1`` for the original
single-scale graph. Three.js keeps the faster subgroup-specialized path. On
Linux, the default Python
driver supplies the native headless Vulkan flags required by Chromium.
Confirm that ``engine.info().device`` names the physical adapter rather than
``swiftshader`` before reporting performance.

If WebGPU is unavailable, Three.js automatically uses its WebGL2
transform-feedback solver. Force this path during qualification with:

.. code-block:: python

   engine = ICGNWebGL(backend="three-tsl", force_three_webgl=True)

The historical ``backend="gpuio"`` is deprecated. It remains available while
the Three.js fallback is validated across the supported browser and GPU
matrix, but new applications should not select it. Follow
:doc:`migrate-from-gpuio` when updating existing code.

Install and select native WebGPU
---------------------------------

Install the optional ``native`` dependency group:

.. code-block:: console

   $ .venv-icgn/bin/pip install -e 'DIC/LucasKanadeICGN/python[native]'

For the recommended native coarse-to-fine path, construct the two-level full
pyramid explicitly:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   engine = ICGNWGPUFullPyramid()

``ICGNWGPUFullPyramid`` sends contiguous NumPy grayscale bytes directly to a
native ``r8unorm`` texture and keeps both scales on the GPU. It returns the
same ``RegistrationResult`` layout as ``ICGNWebGL``. Use ``ICGNWGPU()``
instead when the single-scale ``register_stream`` API or the lowest isolated
frame latency is required. Native engines are selected explicitly and are not
currently used by the command-line interface.

Prepare an offline installation
-------------------------------

On a connected machine with the same operating system, architecture, and
Python version as the offline machine, build a local wheelhouse:

.. code-block:: console

   $ python3 -m pip wheel 'DIC/LucasKanadeICGN/python[native]' \
       --wheel-dir wheelhouse

This downloads wheels for NumPy, Pillow, websockets, and wgpu-py and builds the
IC-GN wheel with its HTML, JavaScript, WGSL, GLSL, and SPIR-V resources. The wgpu-py wheel
contains its wgpu-native runtime. Copy ``wheelhouse/`` to the offline machine,
where Chromium is needed only for browser backends, then run:

.. code-block:: console

   $ python3 -m pip install \
       --no-index \
       --find-links wheelhouse \
       lucas-kanade-icgn-webgl

No CDN or package index is contacted during subsequent registrations.

Build the experimental PyO3 module for offline deployment
----------------------------------------------------------

On a connected build machine with the same operating system, architecture,
and Python ABI as the offline target, install Rust and Maturin, then build the
extension wheel:

.. code-block:: console

   $ maturin build --release \
       --manifest-path DIC/LucasKanadeICGN/native/Cargo.toml \
       --out wheelhouse

Copy the resulting ``dic_rt_wgpu_native`` wheel with the wheelhouse and install
it with ``pip --no-index``. The offline target does not need Cargo, the Rust
toolchain, or source access once that wheel has been built.

Select a browser explicitly
---------------------------

The launcher searches common Chromium and Chrome executable names on Linux,
Chrome and Edge names on Windows, and the standard Chrome, Chromium, and Edge
application locations on macOS. Platform-specific GPU switches are selected
at launch: Linux keeps the qualified Vulkan path, while Windows and macOS let
Chromium select their native D3D or Metal adapter.

If automatic browser discovery fails, pass the executable path:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWebGL

   engine = ICGNWebGL(chromium="/usr/bin/chromium")

The equivalent command-line option is ``--chromium``.

Chromium sandbox
----------------

``ICGNWebGL`` passes ``--no-sandbox`` by default for container compatibility.
On a system with a working Chromium sandbox, prefer:

.. code-block:: python

   engine = ICGNWebGL(no_sandbox=False)

Use ``--enable-chromium-sandbox`` on the command line.

Run the portable unit tests
---------------------------

Install the package and its development tools, then run the standard-library
test runner from the repository root:

.. code-block:: console

   $ python -m pip install -e 'DIC/LucasKanadeICGN/python[dev]'
   $ python -m unittest discover -s DIC/LucasKanadeICGN/python/tests -v

These tests do not require Chromium or a GPU. They validate numerical helpers,
packaged offline resources, public contracts, and simulated browser policies
for Linux, Windows, and macOS. A successful unit suite does not qualify a GPU
driver, shader compiler, or performance on the host; run the registration
command above separately on every deployment platform.

Build the documentation locally
-------------------------------

.. code-block:: console

   $ .venv-icgn/bin/pip install -r DIC/LucasKanadeICGN/docs/requirements.txt
   $ .venv-icgn/bin/sphinx-build \
       -W --keep-going \
       -b html \
       DIC/LucasKanadeICGN/docs \
       DIC/LucasKanadeICGN/docs/_build/html

Open ``DIC/LucasKanadeICGN/docs/_build/html/index.html`` in a browser.
