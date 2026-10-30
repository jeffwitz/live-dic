# Same-workload performance regression control

The 16.62 ms tensile-sequence median must not be interpreted as the new
universal speed of the engine, or compared directly to an older 10–12 ms
small-motion measurement. A separate native-UHD control was run to distinguish
implementation overhead from image content, motion and timing variability.

## Before and after on exactly the same image

Both revisions use the original 3840 × 2160 reference and `frame_040.png`,
affine 16/16, B-spline, uniform weights, explicit median smoothing, no ROI,
two levels `(8,16)` and one slot. The frame is deliberately held fixed for
this regression control; this does not replace changing-sequence qualification.
Twenty warmups precede 60 synchronous image-to-materialized-field calls.
The same Python 3.11/NumPy 2.4.6/wgpu 0.32 environment and Intel LNL/Mesa
25.2.8 adapter are used. A detached worktree supplies the baseline source;
each process records the actual imported package path.

Two repetitions, in order old/zero/temporal then temporal/zero/old:

| Implementation | Median, repetition 1 | Median, repetition 2 | p95 range |
|---|---:|---:|---:|
| Before temporal changes, `67e264f` | 12.547 ms | 13.099 ms | 15.523–15.621 ms |
| Current, zero initialization | 12.443 ms | 12.417 ms | 14.851–15.179 ms |
| Current, temporal initialization | 12.526 ms | 12.744 ms | 15.155–15.530 ms |

All controls accept 31,648 windows. Mean fine iterations are approximately
2.607 in every case. The existing pre-edit exact-field comparison remains
available in `default-parity.json`. There is no observed general slowdown
in these controls, and no significant several-millisecond fixed overhead
from temporal initialization. This is a measured control, not a statistical
non-regression guarantee across hardware and workloads.

## Why the long-sequence numbers differ

A second probe runs the 200-frame native-UHD trajectory twice, then holds
selected frames fixed in the same process. It confirms substantial variation:

| Frame interval | Sequence 1 median | Sequence 2 median |
|---|---:|---:|
| 0–39 | 12.17 ms | 14.84 ms |
| 40–79 | 14.44 ms | 15.61 ms |
| 80–119 | 18.22 ms | 16.59 ms |
| 120–159 | 19.67 ms | 21.71 ms |
| 160–199 | 17.85 ms | 24.56 ms |

Fine iterations grow from about 2.37 to 2.71, but these are accepted fine
estimates only. They do not measure coarse iterations or work spent on
rejected estimates. Holding the final image fixed subsequently measures
16.49 ms. The generated frame 40 measures 14.82 ms, and the same archived
frame 40 used in the control above measures 16.50 ms in that process.
Thus cumulative displacement alone does not explain the entire difference.
No claim of an isolated single cause is justified.

Additional sampled controls observed GPU power-limit reason `pl4`, actual
frequency around 1550 MHz while 1850 MHz was requested, and CPU-zone readings
near 85 °C. This is evidence of power constraints, not proof that thermal or
power limits explain every variation. Frequency, governor and power settings
were not changed. These telemetry controls measure 12.80 ms before and
12.89 ms with temporal initialization; they are separate from the table above.

A cumulative-pass ablation on archived frame 40 also indicates that the
projection and merge passes do not explain a several-millisecond fixed cost.
Its truncated outputs are not valid displacement measurements. Differences
between short-pass medians are too sensitive to timing variation to publish
a precise universal overhead. A timestamp probe failed a wall-time sanity
check and was discarded.

## Practical interpretation

Keep two different measurements visible: the stable small-motion control for
performance regression, and the full ordered sequence for capture range,
quality, sustained throughput and result age. The latter still does not
qualify an affine p95 below 16.67 ms. The former confirms that the previously
faster operating range remains available. Neither licenses replacing the
sequence measurements with repeated-image timings.

`performance-diagnostic.json` retains all timings, component measurements,
counts, telemetry, source paths and the exact scripts used. No runtime
implementation, algorithm parameter, default or machine power setting was
changed for this diagnosis.

To reproduce, extract the scripts to `/tmp` and recreate the baseline worktree
(use a different path if the named directory already exists):

```bash
git worktree add --detach /tmp/live-dic-before-temporal-perf 67e264f
python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("DIC/LucasKanadeICGN/benchmarks/results/temporal-init-20261030/performance-diagnostic.json").read_text())
for name, source in report["scripts"].items():
    Path("/tmp", name).write_text(source)
PY
PYTHONPATH=/tmp/live-dic-before-temporal-perf/DIC/LucasKanadeICGN/python/src \
  DIC/LucasKanadeICGN/python/.venv/bin/python /tmp/measure_temporal_ab.py old /tmp/old.json
DIC/LucasKanadeICGN/python/.venv/bin/python /tmp/measure_temporal_ab.py zero /tmp/zero.json
DIC/LucasKanadeICGN/python/.venv/bin/python /tmp/measure_temporal_ab.py temporal /tmp/temporal.json
```

The archived diagnostic scripts contain this machine's checkout path; adjust
their `root` variable when using another checkout. They are experimental
provenance, not installed runtime utilities. Run campaigns serially.
