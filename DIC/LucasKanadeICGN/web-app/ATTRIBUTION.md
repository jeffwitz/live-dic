# Attribution and scope

live-dic builds on Pierre Margerit's foundational work. His
`tests_gpuio` project established the original DIC prototype:
https://github.com/MargeritPierre/tests_gpuio
His later `Strobox` repository contains a Three.js/WebGPU camera prototype that
informed the browser execution path:
https://github.com/MargeritPierre/Strobox

This standalone application uses the live-dic portable WGSL solver. It does
not use GPU-IO, Three.js or Tweakpane; pinned browser codecs and their licences
are listed in `../THIRD_PARTY.md`.

The tensile demonstration is derived from the archived synthetic native-UHD
fixtures under `DIC/LucasKanadeICGN/assets/uhd_tensile`. Its manifest records
source and output hashes, image formation, encoding and analytic stretch
history. The original fixture provenance remains in that directory. The movie
is a compressed demonstration asset, not a numerical regression reference.

Camera processing runs locally. Hand-held use is not qualified for metrology.
