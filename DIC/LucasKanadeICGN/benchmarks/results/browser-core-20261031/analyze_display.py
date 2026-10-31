"""Reproduce the display summary and figure from immutable raw frame records."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
report = json.loads((root / "display-final-70s-offline.json").read_text())
rows = report["worker_report"]["samples"]
elapsed = (np.array([row["workerAvailableMs"] for row in rows]) - rows[0]["workerAvailableMs"]) / 1000
age = np.array([row["ageMs"] for row in rows])
truth = np.array([row["truthRmse"] for row in rows])
blocks = []
for start in range(0, len(rows), 300):
    stop = min(start + 300, len(rows))
    blocks.append(dict(start_frame=start, stop_frame_exclusive=stop,
                       start_seconds=float(elapsed[start]), end_seconds=float(elapsed[stop - 1]),
                       age_p50_p95_p99_ms=np.percentile(age[start:stop], [50, 95, 99]).tolist()))
summary = dict(
    source="display-final-70s-offline.json", samples=len(rows),
    processing_span_seconds=report["processing_span_seconds"],
    completed_fps=(len(rows) - 1) / report["processing_span_seconds"],
    source_fps=30,
    worker_age_p50_p95_p99_max_ms=np.percentile(age, [50, 95, 99, 100]).tolist(),
    gpu_field_observable_p50_p95_p99_max_ms=np.percentile([r["gpuFieldObservableMs"] for r in rows], [50, 95, 99, 100]).tolist(),
    whole_accepted_truth_rmse_p50_p95_p99_max_px=np.percentile(truth, [50, 95, 99, 100]).tolist(),
    minimum_acceptance=report["minimum_acceptance"], offline_reload=report["offline_reload"],
    application_errors=report["console_errors"], blocks=blocks,
    interpretation="1080p compressed demo at 30 Hz, not 60 Hz camera qualification. Truth includes wrong basins near unobservable image boundaries. Age excludes main-to-worker delivery and display presentation.",
)
(root / "display-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
axes[0].plot(elapsed, age, ".", color="#386e7c", alpha=.25, ms=2, label="Each completed field")
mid = [(b["start_seconds"] + b["end_seconds"]) / 2 for b in blocks]
axes[0].plot(mid, [b["age_p50_p95_p99_ms"][0] for b in blocks], "o-", color="#18434d", label="Median / 300 fields")
axes[0].plot(mid, [b["age_p50_p95_p99_ms"][1] for b in blocks], "o-", color="#d17b25", label="p95 / 300 fields")
axes[0].axhline(1000 / 60, color="#9a3d37", ls="--", lw=1, label="16.67 ms reference")
axes[0].set(ylabel="Worker → observable field (ms)", ylim=(0, max(age) + 2))
axes[0].legend(ncol=2, fontsize=8)
axes[1].plot(elapsed, truth, color="#3b7577", lw=.9)
axes[1].set(xlabel="Time since first worker frame (s)", ylabel="Whole accepted truth RMSE (px)")
axes[1].text(.015, .92, "Retained boundary wrong basins; reproduced natively", transform=axes[1].transAxes, fontsize=9)
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("1080p worker PWA · Chromium 155 / Intel Lunar Lake · 70.5 s replay")
fig.savefig(root / "display-over-time.png", dpi=160)
plt.close(fig)
