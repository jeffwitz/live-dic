# Simulated UHD camera controls

`hardware.json` exercises the camera UI on the local Intel Vulkan adapter in
Chromium, with its explicitly enabled fake camera. It opens no physical camera.
A temporary Y4M stream cycles the archived 3840 x 2160 plate luma planes with
neutral chroma. Its source PNG hashes are recorded. The original images are
unchanged; YUV conversion means this capture is not a new metrology fixture.

The test requests 60 Hz and correctly reports the fake camera's negotiated
10 Hz. It runs for at least 65 seconds, keeps at most one image in flight,
counts drops, checks finite fields, pauses/exports, solves consecutive received
pairs, restarts the selected camera through the actual UI, exercises the video
callback fallback, and verifies all old tracks close on source replacement.
A denied permission is simulated before a successful acquisition.

```bash
python tools/build_pages.py --output _site
python tools/qualify_browser_camera.py --site _site --output /tmp/camera-check \
  --duration 65 --fallback
```

This is correctness and ownership evidence under development load, not an
isolated timing campaign, a 60 FPS claim, a physical camera test, or mobile
qualification. Manual exposure/focus/white balance are not exposed by this fake
device and are reported as unavailable.

`software.json` repeats the controls and ownership checks on SwiftShader,
including both capture paths. It needs a long observation interval: the retained
`software-short-negative.txt` shows an initial 10-second run in which an in-flight
UHD solve did not complete before the observation ended. The 65-second run
progressed from 3 to 7 completed results and passed the finite-field/lifecycle
checks. This is not hardware performance evidence. Different camera capture
phases choose different references/targets, so accepted counts between these
reports are not numerical parity measurements.

The local Python CI for `7f6fe65` completed successfully, including the UHD
local-source, results-workbench and mask-image SwiftShader jobs. This fake-camera
follow-up was tested locally; it is not asserted to be part of that CI run.
