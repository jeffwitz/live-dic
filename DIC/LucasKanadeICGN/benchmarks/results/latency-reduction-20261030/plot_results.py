"""Plot accuracy and all one-slot p95 samples, including rejected prototypes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.switch_backend("Agg")
    root = Path(__file__).resolve().parent
    cases = (
        ("runs", "baseline", "Baseline", "#555555"),
        ("runs", "adaptive", "Adaptive prototype", "#d55e00"),
        ("interpolation-control", "cubic-fixed", "Fixed cubic prototype", "#cc79a7"),
        ("finest-only-control", "abort", "Final: fine stop k=3", "#0072b2"),
        ("finest-only-control", "abort-fast", "Final: fine stop k=1", "#009e73"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for index, (collection, variant, label, color) in enumerate(cases):
        runs = [
            json.loads((root / collection / f"{variant}-d1-r{i}.json").read_text())
            for i in range(3)
        ]
        rows = runs[0]["quality"]
        frames = [r["frame"] for r in rows]
        for axis, key, scale in (
            (axes[0, 0], "fixed_coverage", 100),
            (axes[0, 1], "fixed_displacement_rmse_px", 1),
            (axes[1, 0], "fixed_strain_rmse", 1),
        ):
            axis.plot(frames, [r[key] * scale for r in rows], label=label, color=color, lw=1.6)
        values = [r["age_p95_ms"] for r in runs]
        axes[1, 1].scatter(index + np.array((-0.13, 0, 0.13)), values, color=color, s=38)
        axes[1, 1].plot([index - 0.2, index + 0.2], [np.median(values)] * 2, color=color, lw=2)
    axes[0, 0].set(
        title="Acceptance does not establish accuracy",
        ylabel="Fixed support accepted (%)",
        ylim=(90, 100.3),
        xlabel="Frame",
    )
    axes[0, 1].set(
        title="Displacement error on accepted fixed support",
        ylabel="RMSE (px)",
        yscale="log",
        xlabel="Frame",
    )
    axes[0, 1].axhline(0.05, color="black", ls="--", lw=1)
    axes[1, 0].set(
        title="Four affine-gradient component errors",
        ylabel="Aggregate gradient RMSE",
        yscale="log",
        xlabel="Frame",
    )
    axes[1, 0].axhline(0.002, color="black", ls="--", lw=1)
    axes[1, 1].axhline(16.67, color="black", ls="--", lw=1)
    axes[1, 1].set(
        title="One-slot p95: every repetition retained",
        ylabel="NumPy-to-field age (ms)",
        xticks=range(len(cases)),
        xticklabels=[c[2].replace(" ", "\n", 1) for c in cases],
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Native UHD tensile trajectory: 16/16 affine subsets, explicit median\n"
        "Historical prototypes and final fixed B-spline profiles; timing includes system variability",
        fontsize=13,
    )
    fig.savefig(root / "quality-and-latency.png", dpi=170)
    fig.savefig(root / "quality-and-latency.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
