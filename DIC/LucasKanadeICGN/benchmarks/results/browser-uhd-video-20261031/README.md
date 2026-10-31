# Original UHD inputs and temporary UHD video — 31 October 2026

Both `qualify_local_sources.py` commands completed with exit code zero against
clean code revision `271b5e4`. `hardware.json` and `software.json` preserve the
raw reports, software versions, source/build identifiers, configuration and
input hashes. The browser output remained 3840 × 2160, with an explicit seek
to 0.5 seconds. The script then asserts two further completed fields during
playback, pauses and drains the bounded worker. The report's video state is the
paused seek observation before this playback assertion, not a throughput record.

The same runs cover original UHD PNG file-picker input, natural order, fixed and
consecutive references, whole-ROI failure/recovery and rejection of wrong-sized
images. Native/browser numerical comparisons apply to those original PNG fields;
they are not a metrology qualification of compressed video.

The temporary six-second, ten-frame/s VP9 movie uses the checked-in tensile
reference for 0.2 seconds and frame 40 for the remainder. It has 60 encoded
frames. `video-provenance.json` records the original image hashes, concat list,
FFmpeg version, generation arguments, probe result and movie hash. The original
PNG fixtures were verified unchanged. Lossless VP9 coding follows a grayscale
to YUV420P conversion; original grayscale-byte identity is not claimed. The
29 MB derived movie is intentionally not another maintained fixture or runtime
dependency. FFmpeg was used only to construct this optional developer input.

```bash
python3 tools/build_pages.py --output /tmp/live-dic-uhd-input-site
python3 tools/qualify_local_sources.py --site /tmp/live-dic-uhd-input-site --output /tmp/live-dic-uhd-video-hardware --video /path/to/uhd-six-seconds.webm
python3 tools/qualify_local_sources.py --site /tmp/live-dic-uhd-input-site --output /tmp/live-dic-uhd-video-software --video /path/to/uhd-six-seconds.webm --software
```

Recreate `concat.txt` with the original absolute PNG paths and the stored
`duration` lines, then use the recorded FFmpeg arguments (`-vf fps=10` performs
the explicit constant-rate repetition). On Snap Chromium, select a movie in a
Snap-visible home directory, as in these runs. Software playback deliberately
uses the existing ×1/16 control to observe progress before the finite movie ends.
These are functional I/O checks; ten encoded frames/s, slow playback and the
reported GUI counters do not qualify a live 4K60 acquisition pipeline. No
physical camera or additional OS/browser family was tested by this campaign.
