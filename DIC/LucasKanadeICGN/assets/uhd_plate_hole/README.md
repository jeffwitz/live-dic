# Original native-UHD analytical plate with a hole

Three original **3840 x 2160** grayscale PNGs are copied byte for byte from
`DIC_FROM_H264/experiments/decoded_control_filter/airy_plate_4k/results`, source
revision `77267f92920ea8984533371cc6a62fef149ff6e2`. These are native rendered
images, without resizing, tiling or video encoding. No sensor noise was added.
The manifest records original paths, SHA256 values, mechanics and load states.
Files are available offline; no source checkout is required to run the tests.

| File | Original frame | Primary load MPa | Secondary load MPa |
|---|---:|---:|---:|
| reference.png | 0 | 0 | 0 |
| frame_095.png | 95 | 250 | 0 |
| frame_225.png | 225 | 250 | 80 |

The secondary mode changes the heterogeneous field without changing the
generator's extension command. It is not a simple additional homogeneous strain.

## Geometry and physical boundary

The circular hole has centre `(1920,1080)` and radius `112` pixels. Its
boundary is traction free. The four image edges are an observation window in
an infinite isotropic medium, not free specimen edges. Pixel size is 0.01 mm,
Young's modulus 210000 MPa, Poisson ratio 0.3, with plane strain (`ezz=0`).

The exact nominal material mask is evaluated at integer reference pixels:

```python
y, x = np.ogrid[:2160, :3840]
mask = (x - 1920)**2 + (y - 1080)**2 >= 112**2
```

It has no additional erosion. Live-dic's `full` policy retains 32,220 squares
at size/stride 16/16. `partial` retains 32,244, including 24 partial supports.
The mask stays in reference coordinates while the rendered target hole moves.
Preprocessing/interpolation may still depend on pixels outside nominal supports;
this dataset does not add photometric isolation at a free boundary.

## Analytic truth and coordinates

`python/tests/plate_hole_reference.py` evaluates the original complex-potential
displacements, their zero-extension second mode, and analytic plane-strain
stress/strain in NumPy. Mechanics is used only for scoring and never passed to
registration. Source snippets and 16 C++ truth samples are archived in
`provenance/`. A standalone comparison against the original C++ implementation
found maximum differences below `6e-16` pixel and `5e-19` strain on those samples.
Tests additionally compare displacement derivatives to the stress-derived strain.

The generator's section calibration uses its historical `8+16*i` points.
Live-dic is scored at its actual nominal subset centres, `7.5+16*i` at size 16.
The half-pixel difference is not silently rounded away.

Targets use the original inverse cubic texture warp and uint8 rounding. The
dark hole is advected with the material. Analytic truth describes the continuous
mechanical field; discrete image formation and finite-window fitting still
contribute to measurement error, especially near the hole.

The displayed invariant is equivalent **strain**, not stress:

```text
exx = du/dx; eyy = dv/dy; exy = (du/dy + dv/dx)/2
eVM = (2/3) sqrt(exx² - exx*eyy + eyy² + 3*exy²), with ezz=0
```

It is evaluated from the direct local affine parameters. The old SYSCL report
used a 5x5 displacement-regression strain operator; its numerical errors cannot
be compared directly to these direct-gradient errors.

## Qualification and limits

The GPU regression runs the actual reference/frame-95 pair at native UHD,
with `full` and `partial`, affine 16/16, B-spline, two levels and explicit median.
It checks finite fields, exclusion status, all 24 partial points' acceptance,
and displacement RMSE <0.05 px / coverage >=90% on a **predeclared far-field
population** (`r>=288`, 32-pixel image margin). This is not a precision guarantee
for the new boundary gradients. Their larger errors remain in the complete
benchmark, plots and CSV, rather than being discarded.

See [the ROI/VM analysis](../../benchmarks/results/plate-hole-roi-20261030/README.md).
Run `test_plate_hole.py` with `ICGN_RUN_GPU_TESTS=1` for native GPU execution;
without opt-in the offline integrity/analytic checks run and the GPU test is
explicitly skipped. No compiler or external library is required at runtime.
The archived C++ checker documents the separate, one-time source validation.
