"""Plot the archived paired UHD timings and asynchronous Linux observations.

Run beside report.json, monitor.json, timings.npz and monitor.csv.gz.
Requires NumPy and Matplotlib for this offline report only.
"""

import csv
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
report = json.loads((out / "report.json").read_text())
observer = json.loads((out / "monitor.json").read_text())
with np.load(out / "timings.npz", allow_pickle=False) as archive:
    timing = archive["timings"]
with gzip.open(out / "monitor.csv.gz", "rt") as stream:
    rows = list(csv.DictReader(stream))
rows = [
    row
    for row in rows
    if report["measurementStartedEpochSeconds"]
    <= float(row["epochSeconds"])
    <= report["measurementFinishedEpochSeconds"]
]


def numbers(key):
    return np.array([float(row[key]) if row[key] else np.nan for row in rows])


gt = "/sys/class/drm/card1/device/tile0/gt0/freq0/"

# One-minute bins retain long-run drift without plotting every individual call.
minutes = np.floor(timing[:, 0] / 60000).astype(int)
centres = []
values = []
for minute in np.unique(minutes):
    batch = timing[minutes == minute]
    centres.append(minute + 0.5)
    values.append(
        [
            np.median(batch[:, 1]),
            np.median(batch[:, 2]),
            np.percentile(batch[:, 2], 95),
            np.median(batch[:, 1] - batch[:, 2]),
        ]
    )
values = np.array(values)
t = (numbers("epochSeconds") - report["measurementStartedEpochSeconds"]) / 60
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
a = axes[0]
a.plot(centres, values[:, 0], label="Repeated dense search · median", color="#667784")
a.plot(centres, values[:, 1], label="Stop empty work · median", color="#138682")
a.plot(centres, values[:, 2], label="Stop empty work · p95", color="#138682", ls=":")
a.axhline(1000 / 60, color="#bd5340", ls="--", label="60 Hz period")
a.set_ylabel("Image → field (ms)")
a.legend(ncol=2, fontsize=8)
a = axes[1]
a.plot(centres, values[:, 3], color="#138682")
a.axhline(0, color="#667784", lw=0.7)
a.set_ylabel("Paired median\nsaving (ms)")
a = axes[2]
a.plot(t, numbers("rssBytes") / 2**20, label="Process RSS")
a.plot(t, numbers("residentBytes") / 2**20, label="DRM client resident sum")
a.set_ylabel("MiB · distinct counters")
a.legend(fontsize=8)
a = axes[3]
rp0 = observer["sensorReferences"][gt + "rp0_freq"]
a.plot(t, numbers(gt + "act_freq") / rp0, lw=0.5, alpha=0.5, label="Actual / rp0")
a.plot(t, numbers(gt + "cur_freq") / rp0, lw=0.8, label="Requested / rp0")
a.set_ylabel("GT0 clock / rp0")
a.set_xlabel("Minutes after paired measurement started")
a.legend(fontsize=8)
for a in axes:
    a.grid(alpha=0.2)
    a.set_xlim(0, float(timing[-1, 0] / 60000))
fig.suptitle(
    "Original UHD tensile pair · translation 32/16 · two propagation passes\nOne hardware device, per-frame AB/BA · retained empty-work optimization",
    fontsize=12,
)
fig.savefig(out / "overview.png", dpi=150)
plt.close(fig)
