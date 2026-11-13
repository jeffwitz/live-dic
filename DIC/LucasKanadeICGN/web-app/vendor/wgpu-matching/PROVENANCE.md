# WGPU-Matching browser source snapshot

This directory contains the CPM-HOG browser matcher and the WGSL sources it
loads. The snapshot was copied from `jeffwitz/WGPU-Matching` commit
`4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93` on 2026-11-13. It is kept local so
the GitHub Pages application and its offline cache need no CDN or runtime
network dependency.

Only `cpm.js`, `deepmatching.js`, `matcher-contracts.js`, the shared matcher
contract, and the five CPM/descriptor shaders used by those modules are
included. No code from the original CPM paper implementation is included;
WGPU-Matching documents its implementation as a CPM-inspired adaptation.
Changes to this snapshot should be made in WGPU-Matching first, then copied
with a new source revision recorded here.
