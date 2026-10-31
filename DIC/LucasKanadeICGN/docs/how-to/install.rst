Install the Python runtime
==========================

System requirements
-------------------

Native processing requires:

* Python 3.11 or newer;
* a platform supported by the dependency wheels;
* a working GPU driver or an installed software WebGPU adapter.

The standard package includes ``wgpu-py`` and supports native processing on
platforms supported by that library. Native processing does not require
Chromium. Browser compatibility additionally requires Chromium or Chrome with
WebGL2 support; WebGPU is preferred for those paths. It is not installed by pip.
Both native engines accept ``device_type="cpu"`` when an installed software
adapter is available. See :doc:`run-on-cpu` for driver requirements and
explicit CPU/GPU selection.

The package embeds GPU-IO, Three.js r186, the raw WebGPU adapter, all IC-GN
shaders, and their headless pages. It does not require a repository checkout
or a network connection at runtime.

Install the package
-------------------

From the repository root, install the package for the intended
Python-supervised NumPy pipeline:

.. code-block:: console

   $ python3 -m venv .venv-icgn
   $ .venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python

This provides ``ICGNWGPUFullPyramid``, the recommended two-level native
backend, as well as the browser compatibility backends.

Verify the command-line interface:

.. code-block:: console

   $ .venv-icgn/bin/icgn-webgl --help

Verify a native registration using :doc:`../tutorials/first-registration`.
That tutorial uses ``ICGNWGPUFullPyramid`` and does not start Chromium.
The CLI above is still a browser interface, so the following optional check
requires Chromium:

.. code-block:: console

   $ .venv-icgn/bin/icgn-webgl \
       DIC/LucasKanadeICGN/assets/frame_000.png \
       DIC/LucasKanadeICGN/assets/frame_010.png \
       --truth 2 -1

The JSON output should identify the browser adapter and report a 32 x 32 grid.
Check acceptance and error against the fixture's known translation; do not
interpret browser availability as native runtime qualification.

Select a compute backend
------------------------

Defaults are scoped to the chosen engine. See
:doc:`../explanation/backend-selection` for the complete browser, native, and
pyramid decision table.

For Python-supervised NumPy processing, select the native full pyramid:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid


   engine = ICGNWGPUFullPyramid()

Three.js/TSL is the default only within the ``ICGNWebGL`` browser-control
class. A browser compatibility engine selects it as follows:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWebGL


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

The standard installation already includes ``wgpu-py``:

.. code-block:: console

   $ .venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python

For the recommended native coarse-to-fine path, construct the two-level full
pyramid explicitly:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   engine = ICGNWGPUFullPyramid()

``ICGNWGPUFullPyramid`` sends contiguous NumPy grayscale bytes directly to a
native ``r8unorm`` texture and keeps both scales on the GPU. It returns the
same ``RegistrationResult`` layout as ``ICGNWebGL``. Use ``ICGNWGPU()``
instead when a single-scale graph, automatic subgroup tuning, or selective
recovery is required. Both native engines expose ``register_stream``. Native
engines are selected explicitly and are not currently used by the command-line
interface.

Prepare an offline installation
-------------------------------

On a connected machine with the same operating system, architecture, and
Python version as the offline machine, build a local wheelhouse:

.. code-block:: console

   $ python3 -m pip wheel DIC/LucasKanadeICGN/python \
       --wheel-dir wheelhouse

This downloads wheels for NumPy, Pillow, websockets, and wgpu-py and builds the
IC-GN wheel with its HTML, JavaScript, and WGSL resources. The wgpu-py wheel
contains its prebuilt wgpu-native runtime. Copy ``wheelhouse/`` to the offline machine,
where Chromium is needed only for browser backends, then run:

.. code-block:: console

   $ python3 -m pip install \
       --no-index \
       --find-links wheelhouse \
       lucas-kanade-icgn-webgl

No CDN or package index is contacted during subsequent registrations.

No Rust toolchain, Cargo, Maturin, ``glslc``, shader compiler, or Vulkan SDK is
required for the live-dic package. Subgroup shaders are WGSL sources compiled by
the wgpu-native library that is already part of the platform-specific wgpu-py
wheel. As with every GPU application, an appropriate system GPU driver remains
required.

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
