Install the Python runtime
==========================

System requirements
-------------------

The browser backends require:

* Python 3.10 or newer;
* Chromium or Google Chrome with WebGL2 and WebGPU support;
* a GPU and driver accepted by Chromium.

The optional native backend requires Python 3.11 or newer and a platform
supported by ``wgpu-py``. It does not require Chromium.

The package embeds GPU-IO, Three.js r186, the IC-GN solvers, and both headless pages.
It does not require a repository checkout or a network connection at runtime.

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

Three.js/TSL is the default backend. A standard engine therefore selects it:

.. code-block:: python

   engine = ICGNWebGL()

Select the previous implementation with ``ICGNWebGL(backend="gpuio")`` or the
command-line option ``--backend gpuio``. On Linux, the default Python driver
supplies the native headless Vulkan flags required by Chromium. Confirm that
``engine.info().device`` names the physical adapter rather than ``swiftshader``
before reporting performance.

Install and select native WebGPU
---------------------------------

Install the optional ``native`` dependency group:

.. code-block:: console

   $ .venv-icgn/bin/pip install -e 'DIC/LucasKanadeICGN/python[native]'

Then construct the native engine explicitly:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPU

   engine = ICGNWGPU()

``ICGNWGPU`` sends contiguous NumPy grayscale bytes directly to a native
``r8unorm`` texture. It returns the same ``RegistrationResult`` array layout as
``ICGNWebGL``. It is experimental and is not yet selected by the command-line
interface or by the default engine.

Prepare an offline installation
-------------------------------

On a connected machine with the same operating system, architecture, and
Python version as the offline machine, build a local wheelhouse:

.. code-block:: console

   $ python3 -m pip wheel 'DIC/LucasKanadeICGN/python[native]' \
       --wheel-dir wheelhouse

This downloads wheels for NumPy, Pillow, websockets, and wgpu-py and builds the
IC-GN wheel with its HTML, JavaScript, and WGSL resources. The wgpu-py wheel
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
