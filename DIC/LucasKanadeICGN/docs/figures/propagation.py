"""Render original teaching schematics; no experimental images or timings.

Optional authoring dependency: matplotlib. Sphinx uses the checked-in SVGs
and does not run this script or need matplotlib. Run from any working directory.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

BLUE, GREEN, ORANGE, INK = "#275d94", "#167066", "#ac571a", "#223447"
LIGHT, PALE, WARM = "#eaf1fa", "#e7f4ef", "#fff0df"
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "live-dic-propagation-v1",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
    }
)


def box(ax, x, y, w, h, label, fill=LIGHT, color=BLUE, size=12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.5,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)


def arrow(ax, start, end, color=BLUE, **kwargs):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=16, color=color, linewidth=1.7, **kwargs
        )
    )


def canvas(title, height=4.3):
    fig, ax = plt.subplots(figsize=(11, height))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.04, top=0.88)
    ax.set(xlim=(0, 11), ylim=(0, height))
    ax.axis("off")
    fig.suptitle(title, fontsize=16, weight="bold", color=INK, y=0.98)
    return fig, ax


def basin():
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    u = np.linspace(-1.2, 8, 700)

    def cost(x):
        return (
            1
            - 0.84 * np.exp(-(((x - 0.3) / 0.8) ** 2))
            - 0.98 * np.exp(-(((x - 6.05) / 0.95) ** 2))
        )

    ax.plot(u, cost(u), color=INK, linewidth=2.2)
    ax.axvspan(-0.7, 1.3, color=WARM, alpha=0.8, zorder=-1)
    ax.axvspan(4.8, 7.3, color=PALE, alpha=0.8, zorder=-1)
    for x, color in ((0.3, ORANGE), (6.05, GREEN)):
        ax.plot(x, cost(x), "o", color=color, markersize=8)
    ax.annotate(
        "Current seed: 0.3 px\nLK stays in this basin",
        xy=(0.3, cost(0.3)),
        xytext=(1.9, 0.35),
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        ha="center",
    )
    ax.annotate(
        "Neighbour proposes 6.0 px\nScore here, then restart LK",
        xy=(6.05, cost(6.05)),
        xytext=(5.7, 0.5),
        color=GREEN,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 3},
        arrowprops={"arrowstyle": "->", "color": GREEN},
        ha="center",
    )
    ax.set(
        xlabel="Trial horizontal displacement u (pixels)",
        ylabel="Illustrative cost: 1 - correlation",
        xlim=(-1.2, 8),
        ylim=(-0.04, 1.1),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A local solver needs a seed in a useful attraction basin", fontsize=15, pad=16)
    fig.text(
        0.53,
        0.025,
        "Synthetic curve for explanation only; no measured DIC data.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.85)
    return fig


def pyramid():
    fig, ax = canvas("Three directions in which an initialization can travel", 5.1)
    box(ax, 0.2, 3.55, 3.1, 1.05, "Coarse level\n1920 x 1080\nu = 3 coarse pixels")
    box(ax, 6.4, 3.55, 4.15, 1.05, "Fine level\n3840 x 2160\nseed = 6 fine pixels")
    arrow(ax, (3.4, 4.07), (6.25, 4.07))
    ax.text(4.85, 4.45, "INTER-LEVEL", ha="center", weight="bold", color=BLUE)
    ax.text(
        4.85,
        3.65,
        "interpolate at the fine centre\nscale translation by 2",
        ha="center",
        fontsize=11,
    )
    box(ax, 0.2, 1.95, 3.1, 0.85, "Neighbour at the same level\nu = 6.0 pixels", PALE, GREEN)
    box(ax, 6.4, 1.95, 4.15, 0.85, "Destination patch\ntest that 6.0-pixel hypothesis", PALE, GREEN)
    arrow(ax, (3.4, 2.38), (6.25, 2.38), GREEN)
    ax.text(4.85, 2.83, "SPATIAL", ha="center", weight="bold", color=GREEN)
    ax.text(4.85, 1.97, "same pixel units", ha="center", fontsize=11)
    box(ax, 0.2, 0.45, 3.1, 0.85, "Previous target frame\nreference stays fixed", WARM, ORANGE)
    box(ax, 6.4, 0.45, 4.15, 0.85, "Next target frame\nseed, then solve again", WARM, ORANGE)
    arrow(ax, (3.4, 0.87), (6.25, 0.87), ORANGE)
    ax.text(4.85, 1.37, "TEMPORAL", ha="center", weight="bold", color=ORANGE)
    ax.text(4.85, 0.48, "time advances", ha="center", fontsize=11)
    return fig


def median():
    fig, ax = canvas("A median votes; a photometric test asks the destination image", 5.0)
    values = [
        [6.0, 6.1, 6.2, 5.9, -4.0, 6.0, 6.1, 6.0, 5.9],
        [0.2, 0.3, 6.0, 0.1, 0.3, 6.1, 0.2, 0.4, 6.2],
    ]
    assert np.median(values[0]) == 6.0 and np.median(values[1]) == 0.3
    for left, data, heading, footer in (
        (0.15, values[0], "One isolated bad value", "Median = 6.0 px: helpful"),
        (3.6, values[1], "A majority in the wrong basin", "Median = 0.3 px: still wrong"),
    ):
        ax.text(left + 1.35, 4.55, heading, ha="center", fontsize=11.5, weight="bold")
        for k, value in enumerate(data):
            x, y = left + (k % 3) * 0.9, 3.6 - (k // 3) * 0.9
            fill, color = (PALE, GREEN) if value >= 5 else (WARM, ORANGE)
            ax.add_patch(
                Rectangle(
                    (x, y),
                    0.86,
                    0.86,
                    facecolor=fill,
                    edgecolor=color,
                    linewidth=2.8 if k == 4 else 1,
                )
            )
            ax.text(x + 0.43, y + 0.43, f"{value:.1f}", ha="center", va="center", fontsize=14)
        ax.text(left + 1.3, 1.5, footer, ha="center", fontsize=11.5)
    ax.text(8.8, 4.55, "Test on the central patch", ha="center", weight="bold", fontsize=11.5)
    box(ax, 7.15, 3.35, 3.25, 0.85, "Current / median 0.3 px\ncorrelation = 0.84", WARM, ORANGE)
    box(ax, 7.15, 2.25, 3.25, 0.85, "Neighbour seed 6.0 px\ncorrelation = 0.98", PALE, GREEN)
    arrow(ax, (8.8, 2.15), (8.8, 1.65), GREEN)
    ax.text(8.8, 1.3, "Restart LK from 6.0 px\nthen check convergence", ha="center", fontsize=11.5)
    ax.text(
        5.4,
        0.35,
        "Numbers are illustrative horizontal displacements; bold border marks the destination.",
        ha="center",
        fontsize=10.5,
    )
    return fig


def schedules():
    fig, ax = canvas("Update order determines how far information travels", 5.0)
    starts = [0.15, 5.9]
    for left, heading in zip(
        starts, ("Parallel passes (Jacobi)", "Directional sweep (OpenCV idea)"), strict=True
    ):
        ax.text(left + 2.35, 4.6, heading, ha="center", weight="bold", fontsize=13)
        for row in range(3):
            good = row + 1 if left < 1 else (1 if row == 0 else 5)
            y = 3.5 - row * 1.12
            for col in range(5):
                x = left + 0.6 + col * 0.82
                fill = PALE if col < good else LIGHT
                ax.add_patch(
                    Rectangle(
                        (x, y), 0.7, 0.7, facecolor=fill, edgecolor=GREEN if col < good else BLUE
                    )
                )
                ax.text(
                    x + 0.35,
                    y + 0.35,
                    "6" if col < good else "?",
                    ha="center",
                    va="center",
                    fontsize=15,
                )
                if left > 1 and col and row:
                    start, end = (x - 0.15, y + 0.35), (x + 0.03, y + 0.35)
                    arrow(ax, start if row == 1 else end, end if row == 1 else start, GREEN)
                if left < 1 and row and col == row:
                    arrow(ax, (x - 0.27, y + 1.10), (x + 0.12, y + 0.73), GREEN)
            label = "Start" if row == 0 else f"Pass {row}"
            if left > 1:
                label = ("Start", "Forward", "Reverse")[row]
            ax.text(left + 0.4, y + 0.35, label, ha="right", va="center", fontsize=10)
    ax.text(
        2.6,
        0.55,
        "Read the previous pass only.\nOne neighbour link per pass.",
        ha="center",
        fontsize=11,
    )
    ax.text(
        8.2,
        0.55,
        "Reuse the just-refined neighbour.\nSeveral links within one sweep.",
        ha="center",
        fontsize=11,
    )
    ax.text(
        5.4,
        0.05,
        "Illustration assumes every proposed seed succeeds; neither schedule guarantees recovery.",
        ha="center",
        fontsize=10,
    )
    return fig


def compaction():
    fig, ax = canvas("GPU work list: launch LK only where a better seed was found", 5.4)
    box(ax, 0.15, 4.2, 2.5, 0.8, "Initial LK solve\ndense field of states")
    box(ax, 4.0, 4.2, 2.6, 0.8, "Eligibility\nROI + valid reference")
    box(ax, 8.0, 4.2, 2.6, 0.8, "Score candidates\non each destination")
    arrow(ax, (2.75, 4.6), (3.87, 4.6))
    arrow(ax, (6.7, 4.6), (7.87, 4.6))
    ax.text(
        5.4,
        3.75,
        "Failed-only policy also skips accepted points before scoring.",
        ha="center",
        fontsize=11,
    )
    arrow(ax, (9.3, 4.08), (9.3, 2.95))
    box(ax, 7.9, 1.85, 2.7, 1.0, "Append improved seeds\nIDs [7, 18, 42]\ncount = 3", PALE, GREEN)
    box(
        ax,
        3.95,
        1.85,
        2.7,
        1.0,
        "Indirect dispatch\n3 LK workgroups\nexisting solver equations",
        PALE,
        GREEN,
    )
    arrow(ax, (7.8, 2.35), (6.8, 2.35), GREEN)
    box(
        ax,
        0.15,
        1.85,
        2.6,
        1.0,
        "Dense result grid\nupdate 7, 18, 42\nretain unqueued states",
        PALE,
        GREEN,
    )
    arrow(ax, (3.85, 2.35), (2.9, 2.35), GREEN)
    ax.text(
        5.4,
        1.15,
        "No global sort. No intermediate CPU readback.\nEach queued patch must still pass the normal final solve.",
        ha="center",
        fontsize=12,
    )
    ax.text(
        5.4,
        0.32,
        "Toy IDs; not a measured active count. All-window search can remain expensive even with a short work list.",
        ha="center",
        fontsize=10.5,
    )
    return fig


def aggregation():
    fig, ax = canvas("DIS aggregation: weights belong to an image pixel", 5.0)
    ax.text(
        5.4,
        4.55,
        "Three overlapping patches cover the same image pixel x.",
        ha="center",
        fontsize=12,
    )
    vectors = [6.0, 0.3, 6.1]
    residuals = [1.0, 20.0, 2.0]
    weights = [1 / max(1.0, value) for value in residuals]
    combined = np.dot(weights, vectors) / sum(weights)
    assert round(combined, 2) == 5.85
    for name, u, error, weight, y in zip(
        "ABC", vectors, residuals, weights, (3.3, 2.0, 0.7), strict=True
    ):
        box(ax, 0.2, y, 2.4, 0.85, f"Patch {name}\nu = {u:.1f} px")
        arrow(ax, (2.72, y + 0.42), (3.48, y + 0.42))
        box(ax, 3.6, y, 3.15, 0.85, f"Residual at x = {error:g}\nweight = {weight:g}", PALE, GREEN)
        arrow(ax, (6.9, y + 0.42), (8.0, 2.4), GREEN)
    box(
        ax,
        8.1,
        1.55,
        2.45,
        1.7,
        f"Dense field at x\nu(x) = {combined:.2f} px\n\nweighted mean",
        PALE,
        GREEN,
    )
    ax.text(
        5.4,
        0.12,
        "Illustrative intensities/displacements. Repeat at every covered pixel; this is not a median or a best-seed selection.",
        ha="center",
        fontsize=10,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "_static/propagation"
    )
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in (
        ("basin", basin),
        ("pyramid", pyramid),
        ("median", median),
        ("schedules", schedules),
        ("compaction", compaction),
        ("aggregation", aggregation),
    ):
        fig = draw()
        fig.savefig(
            args.output / f"{name}.svg",
            facecolor="white",
            metadata={
                "Date": None,
                "Title": f"live-dic seed propagation: {name}",
                "Description": "Original illustrative schematic, not experimental data. See the propagation reference for a text explanation.",
            },
        )
        if args.preview_dir:
            fig.savefig(args.preview_dir / f"{name}.png", dpi=130, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
