# Third-party runtimes and browser assets

The browser and Python runtimes do not fetch JavaScript from a CDN. The
following files are kept in the repository so that the solver can run without
network access.

| Component | Version | Local file | SHA-256 |
| --- | --- | --- | --- |
| Tweakpane | 3.1.2 | `../../tweakpane.js` | `fe3395481ebfbc0eadb558260b6304d94c3f3c7b5b0b9282b152e408f92ef765` |
| Tweakpane Essentials | 0.1.0 | `vendor/tweakpane-plugin-essentials-0.1.0.min.js` | `1001dd26be6519682784fe34d4ef0c648ddc4da8fdf56b81fe51fe2b09dcf771` |
| GPU-IO (standalone historical demo only) | repository build | `../../gpu-io.js` | `8320a896f09375d948ed4c28ff6562b1949503d19631d5d9642c894d2bedfabb` |
| Three.js core | 0.186.1 | `three/vendor/three.core.js` | `9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6` |
| Three.js WebGPU | 0.186.1 | `three/vendor/three.webgpu.js` | `15cfce5c653541704fd9a3463c39d3e8b854bb6265ccd854d7cfe74090625cc6` |
| Three.js TSL | 0.186.1 | `three/vendor/three.tsl.js` | `6e6b92d8ce467818944f73815a96abadc6392fb34de577d479a23810b6aeeed0` |

Tweakpane and its Essentials plugin were retrieved from their official npm
packages. Their MIT licence texts are stored under `vendor/licenses/`. The
standalone historical GPU-IO demo uses the repository-root bundle; its licence
is stored at `vendor/licenses/GPU-IO-LICENSE.txt`. The Three.js modules come
from the official npm package and their MIT licence is stored at
`three/vendor/THREE-LICENSE.txt`.

The Python wheel contains the Three.js/TSL and raw-WebGPU browser adapters,
their local vendor modules, shared JavaScript core and WGSL shaders. It does
not contain GPU-IO or the standalone historical GPU-IO demonstration. These
are package resources, not Python dependencies, and let the installed package
start its supported browser backends without locating a live-dic checkout.

## Native WebGPU dependencies

The standard Python dependencies include `wgpu>=0.32,<0.33`. The published
wgpu-py wheel contains the matching wgpu-native library, so the installed
native Python backend does not download a driver library or shader at runtime.
The project's portable and subgroup-specialized WGSL files are first-party
source and are included as Python package data. They are compiled at device
creation by the wgpu-native library already shipped in the wgpu-py wheel. The
live-dic wheel contains no Rust extension, GLSL source, or generated SPIR-V and
does not invoke Cargo, Maturin, `glslc`, shaderc, or a Vulkan SDK.

The native runtime still contains Rust internally because that is how the
upstream wgpu-py wheel implements WebGPU. It is an ordinary prebuilt Python
dependency; live-dic neither builds nor downloads that implementation at
runtime. GPU drivers remain a platform prerequisite.

## Standalone browser TIFF input

The PWA vendors [UTIF.js](https://github.com/photopea/UTIF.js) 3.1.0 under
`web-app/vendor/`. It is plain JavaScript; no project binary, compiler or CDN
is required. The pinned upstream source snapshot includes TIFF LZW decoder
corrections absent from the npm tarball, which otherwise truncated some strip
rows in the grayscale-plus-alpha mask used by the project. These PWA files are
not added to the Python wheel. The source commit, SHA-256 and MIT licence are
recorded in `web-app/vendor/tiff-provenance.json`.

| Component | Local file | SHA-256 |
| --- | --- | --- |
| UTIF.js 3.1.0 upstream snapshot `5883510` | `web-app/vendor/UTIF-3.1.0.js` | `e6239b23207e012bcf6c41664bba30ea9746fea32dbbee774f783d1ed60221f2` |
