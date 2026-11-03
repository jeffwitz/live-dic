# Third-party runtimes and browser assets

The browser and Python runtimes do not fetch JavaScript from a CDN. The
following files are kept in the repository so that the solver can run without
network access.

| Component | Version | Local file | SHA-256 |
| --- | --- | --- | --- |
| Three.js core | 0.186.1 | `three/vendor/three.core.js` | `9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6` |
| Three.js WebGPU | 0.186.1 | `three/vendor/three.webgpu.js` | `15cfce5c653541704fd9a3463c39d3e8b854bb6265ccd854d7cfe74090625cc6` |
| Three.js TSL | 0.186.1 | `three/vendor/three.tsl.js` | `6e6b92d8ce467818944f73815a96abadc6392fb34de577d479a23810b6aeeed0` |

The Three.js modules come from the official npm package. Their MIT licence is
stored at `three/vendor/THREE-LICENSE.txt`.

The Python wheel contains the Three.js/TSL and raw-WebGPU browser adapters,
their local vendor modules, shared JavaScript core and WGSL shaders. No
GPU-IO runtime or legacy GPU-IO demonstration is maintained in this repository.

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
