# UHD mask image correctness

The actual PWA imports an opaque 3840 x 2160 PNG material mask, checks exact
binary geometry, applies it to the original plate pair, rejects wrong-size and
gray input without changing the old result, then explicitly thresholds a gray
control and obtains the same field. Pointer input adds another circular hole;
independent NumPy pixel/window selection matches every -3 status. PNG export
matches the editable net mask exactly. An unapplied undo preserves the previous
numerical field and its applied geometry.

Both hardware Chromium/Vulkan (Intel Lunar Lake) and Chromium SwiftShader pass.
The original mask selects 32,220 windows; the extra hole leaves 32,195. These
are full-window geometry/correctness observations, not live-FPS measurements.
Original analytical image fixtures and canonical numerical shaders are unchanged.

```bash
python3 tools/build_pages.py --output _site
python3 tools/qualify_browser_mask_images.py --site _site --output /tmp/mask-images
python3 tools/qualify_browser_mask_images.py --site _site --software --output /tmp/mask-images-cpu
```

Use the project Python environment. The generated mask inputs are geometric
controls; archived plate photographs are reused without modification. Source
build/content and mask hashes are in the reports. Later controls add keyboard
probe placement and optional saved-field arrows/contours without changing ROI
selection or calculated values.
