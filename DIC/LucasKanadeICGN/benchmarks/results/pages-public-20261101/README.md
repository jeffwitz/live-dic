# Public Pages origin verification — 1 November 2026

The owner requested publication at https://jeffwitz.github.io/live-dic/.
The release is code revision `a49de5b`, deployed by
[the successful Pages workflow](https://github.com/jeffwitz/live-dic/actions/runs/41297735042).
The repository's Pages build type is GitHub Actions and HTTPS is enforced.

`public-origin.json` records the actual public-origin test, browser version,
source/build information, online and offline field diagnostics, service-worker
scope, 48 cached asset URLs and nine verified source hashes. The test:

1. Opens `https://jeffwitz.github.io/live-dic/?smoke=1&case=plate&offline=1`
   in the existing Chromium harness, using the original UHD plate pair.
2. Waits for a real field and for service-worker installation/activation.
3. Verifies all six cached canonical WGSL files and all three original plate
   PNGs against the build's source hashes. Internal build markers are absent.
4. Disables the browser HTTP cache and simulates offline network conditions
   in both the page and the active service-worker CDP sessions. A cache-busted
   network probe must fail before the test can proceed.
5. Reloads with those sessions kept connected, requires a fresh document,
   an active controller and a newly completed UHD field, then captures the
   screenshot. It restores connectivity and closes the temporary browser.

Both fields accept 31,474 of 32,220 selected subsets, with no application error.
The screenshot is the actual public page after the offline reload. Its one-field
HUD age is not a steady-state FPS campaign. This qualifies this desktop Chromium
run, not physical cameras, mobile browsers or all GPU vendors.

## Failure controls and correction

The first deployed artifact exposed a real hosting difference: the `.nojekyll`
and `.live-dic-site` build markers returned HTTP 404 on Pages. Requesting them
in `cache.addAll` aborts installation. They remain local build markers but are
excluded from the browser cache list. The new portable regression fails before
that correction and passes afterwards; the local server-stopped UHD reload
and all 248 GPU/browser-enabled unit tests also pass.

Page-only offline emulation was rejected as a test setup: its uncached probe
remained reachable through the service worker. No offline success is claimed
for that attempt. The recorded passing run additionally emulates the service
worker's network and requires the uncached probe to fail. This is simulated
network isolation, not a physical network disconnection.
