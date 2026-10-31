# Third-party runtimes and browser assets

The browser and Python runtimes do not fetch JavaScript from a CDN. The
following files are kept in the repository so that the solver can run without
network access.

| Component | Version | Local file | SHA-256 |
| --- | --- | --- | --- |
| Tweakpane | 3.1.2 | `../../tweakpane.js` | `fe3395481ebfbc0eadb558260b6304d94c3f3c7b5b0b9282b152e408f92ef765` |
| Tweakpane Essentials | 0.1.0 | `vendor/tweakpane-plugin-essentials-0.1.0.min.js` | `1001dd26be6519682784fe34d4ef0c648ddc4da8fdf56b81fe51fe2b09dcf771` |
| GPU-IO | repository build | `../../gpu-io.js` | `8320a896f09375d948ed4c28ff6562b1949503d19631d5d9642c894d2bedfabb` |
| Three.js core | 0.186.1 | `three/vendor/three.core.js` | `9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6` |
| Three.js WebGPU | 0.186.1 | `three/vendor/three.webgpu.js` | `15cfce5c653541704fd9a3463c39d3e8b854bb6265ccd854d7cfe74090625cc6` |
| Three.js TSL | 0.186.1 | `three/vendor/three.tsl.js` | `6e6b92d8ce467818944f73815a96abadc6392fb34de577d479a23810b6aeeed0` |

Tweakpane and its Essentials plugin were retrieved from their official npm
packages. Their MIT licence texts are stored under `vendor/licenses/`. The
GPU-IO licence is included beside the copy embedded in the Python package.
The Three.js modules come from the official npm package and their MIT licence
is stored at `three/vendor/THREE-LICENSE.txt`.

The Python wheel contains private copies of `gpu-io.js`,
`LucasKanadeICGN.js`, `headless.html`, and the complete Three.js/TSL runtime
under `lucas_kanade_icgn/web/`. These are package data, not Python dependencies. They
allow the installed package to start the numerical backend without locating a
live-dic checkout.

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
