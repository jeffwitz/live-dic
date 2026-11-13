# Methodology acknowledgement and scope

Earlier GPU-based DIC experimentation by Pierre Margerit in his
`tests_gpuio` project informed the methodological exploration in live-dic:
https://github.com/MargeritPierre/tests_gpuio
His later `Strobox` repository contains a Three.js/WebGPU camera prototype that
informed the browser-GPU exploration:
https://github.com/MargeritPierre/Strobox

This is an acknowledgment of methodological influence. The maintained live-dic
solver is a separate implementation; this wording does not claim that Pierre
authored the current solver or that live-dic reuses Strobox source code.

This standalone application uses the live-dic portable WGSL solver. It does
not use GPU-IO or Three.js. Tweakpane has been retired. An optional CPM-HOG
matcher source snapshot from the maintainer's WGPU-Matching project is included
for Analysis initialization; see `vendor/wgpu-matching/PROVENANCE.md`. It does
not include source from the original CPM implementation. Pinned browser codecs
and third-party licences are listed in `../THIRD_PARTY.md`.

The tensile demonstration is derived from the archived synthetic native-UHD
fixtures under `DIC/LucasKanadeICGN/assets/uhd_tensile`. Its manifest records
source and output hashes, image formation, encoding and analytic stretch
history. The original fixture provenance remains in that directory. The movie
is a compressed demonstration asset, not a numerical regression reference.

Camera processing runs locally. Hand-held use is not qualified for metrology.
