# Local input and consecutive-pair correctness

The hardware report records the built working tree at b127e80 plus the local-input
changes, using Chromium 155, Linux/Vulkan, Intel Lunar Lake. It exercises actual
file inputs, full-resolution original 3840 x 2160 plate images, fixed 0→1 and
consecutive 1→2 correlation, black target/reference failure and subsequent
recovery, strict dimension rejection, and local 1080p video seek/play/pause.

The same geometric ROI requests 32,220 subsets throughout; exclusions remain
180. Both qualified pairs have zero status differences versus native Python,
parameter RMSE below 8e-6 and maximum difference below 0.0014. These are
correctness observations, not performance or physical-camera qualification.

Reproduce after building the site:

```bash
python3 tools/build_pages.py --output _site
python3 tools/qualify_local_sources.py --site _site --output /tmp/local-input
# Optional: --software, or --video /absolute/path/to/a/local/UHD/video.webm
```

Use the project Python environment. The helper stages original fixture bytes in
a temporary directory visible to Snap Chromium if applicable; it does not edit
or regenerate archived fixtures. NPZ field captures are generated in the chosen
output directory. The checked-in JSON retains pair identities and image hashes.
