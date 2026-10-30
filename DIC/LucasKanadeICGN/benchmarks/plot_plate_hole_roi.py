"""Matplotlib figures from archived UHD DIC fields; never run during timing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/tests"))
from plate_hole_reference import (
    CX,
    CY,
    RADIUS,
    strain_from_parameters,
    von_mises_strain,
)

VARIANTS = ("none", "full", "partial", "visibility")
TITLES = ("No ROI", "Full squares", "Partial supports", "Partial + visibility")
COLORS = ("#64748b", "#d97706", "#059669", "#2563eb")


def load_field(root, frame, policy, variant):
    with np.load(
        root / "fields" / f"{frame:03d}-{policy}-{variant}.npz", allow_pickle=False
    ) as archive:
        parameters = archive["parameters"]
        accepted = archive["status"] > 0.5
        selected = (
            archive["roi_selected"]
            if "roi_selected" in archive
            else np.ones(accepted.shape, bool)
        )
    return von_mises_strain(
        strain_from_parameters(parameters)
    ) * 1e6, accepted & selected


def configure(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-320, 320)
    ax.set_ylim(320, -320)
    ax.set_xticks((-256, 0, 256))
    ax.set_yticks((-256, 0, 256))
    ax.add_patch(
        Circle((0, 0), RADIUS, facecolor="white", edgecolor="#334155", lw=0.8, zorder=5)
    )
    ax.set_facecolor("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=180)
    fig.savefig(root / f"{name}.pdf", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    report = json.loads((root / "report.json").read_text())
    frames = (95, 225)
    truths = {}
    for frame in frames:
        with np.load(root / f"truth_{frame:03d}.npz", allow_pickle=False) as data:
            truths[frame] = {key: data[key] for key in ("x", "y", "physical", "vm")}
    peak = max(float(t["vm"][t["physical"]].max() * 1e6) for t in truths.values())
    scale = Normalize(0, peak)
    error_scale = TwoSlopeNorm(vmin=-peak, vcenter=0, vmax=peak)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e2e8f0")
    error_cmap = plt.get_cmap("RdBu_r").copy()
    error_cmap.set_bad("#e2e8f0")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42})
    display = {
        "resolution": [3840, 2160],
        "coordinates": "reference, y down",
        "view": "centre +/-320 px; computation remains full UHD",
        "vm_scale_ue": [0, peak],
        "signed_error_scale_ue": [-peak, peak],
        "frames": {},
    }
    for frame in frames:
        truth = truths[frame]
        x, y = truth["x"] - CX, truth["y"] - CY
        physical = truth["physical"]
        zoom = (np.abs(x) <= 320) & (np.abs(y) <= 320) & physical
        gt = truth["vm"] * 1e6
        phase = "uniaxial loading" if frame == 95 else "non-proportional loading"
        fig, axes = plt.subplots(
            2, 5, figsize=(17.8, 8.2), sharex=True, sharey=True, layout="constrained"
        )
        fig.get_layout_engine().set(hspace=0.10)
        errors, err_axes = plt.subplots(
            2, 4, figsize=(14.8, 7.8), sharex=True, sharey=True, layout="constrained"
        )
        fig.suptitle(
            f"Native UHD plate | frame {frame} | {phase}\nEquivalent strain (von Mises, $\\varepsilon_{{zz}}=0$), common scale",
            fontsize=15,
        )
        errors.suptitle(
            f"Frame {frame} | signed equivalent-strain error: DIC minus analytical point value",
            fontsize=14,
        )
        display["frames"][str(frame)] = {}
        for row, policy in enumerate(("disabled", "median")):
            label = (
                "Raw affine gradients" if policy == "disabled" else "Component median"
            )
            axes[row, 0].pcolormesh(
                x,
                y,
                np.ma.array(gt, mask=~physical),
                cmap=cmap,
                norm=scale,
                shading="nearest",
                rasterized=True,
            )
            axes[row, 0].set_title("Analytical point values")
            axes[row, 0].set_ylabel(f"{label}\ny - centre (px)", fontsize=10)
            err_axes[row, 0].set_ylabel(f"{label}\ny - centre (px)", fontsize=10)
            for col, (variant, title) in enumerate(zip(VARIANTS, TITLES, strict=True)):
                values, accepted = load_field(root, frame, policy, variant)
                valid = physical & accepted
                plotted = np.ma.array(values, mask=~valid)
                axes[row, col + 1].pcolormesh(
                    x,
                    y,
                    plotted,
                    cmap=cmap,
                    norm=scale,
                    shading="nearest",
                    rasterized=True,
                )
                record = next(
                    r
                    for r in report["records"]
                    if r["mask"] == variant and r["outlier_policy"] == policy
                )
                snapshot = next(s for s in record["snapshots"] if s["frame"] == frame)
                common = snapshot["regions"]["common_near_hole"]
                clipped = 100 * np.mean(values[zoom & valid] > peak)
                axes[row, col + 1].set_title(title)
                axes[row, col + 1].text(
                    0.03,
                    0.97,
                    f"Common RMSE: {common['vm_rmse_ue']:.0f} $\\mu\\varepsilon$",
                    va="top",
                    transform=axes[row, col + 1].transAxes,
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
                axes[row, col + 1].text(
                    0.03,
                    0.03,
                    f"Colour clipped: {clipped:.1f}%",
                    transform=axes[row, col + 1].transAxes,
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
                delta = values - gt
                err_axes[row, col].pcolormesh(
                    x,
                    y,
                    np.ma.array(delta, mask=~valid),
                    cmap=error_cmap,
                    norm=error_scale,
                    shading="nearest",
                    rasterized=True,
                )
                error_clipped = 100 * np.mean(np.abs(delta[zoom & valid]) > peak)
                err_axes[row, col].set_title(title)
                err_axes[row, col].text(
                    0.03,
                    0.03,
                    f"Colour clipped: {error_clipped:.1f}%",
                    transform=err_axes[row, col].transAxes,
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
                display["frames"][str(frame)][f"{policy}/{variant}"] = {
                    "field_colour_clipped_percent": float(clipped),
                    "error_colour_clipped_percent": float(error_clipped),
                }
        for ax in [*axes.ravel(), *err_axes.ravel()]:
            configure(ax)
        for ax in [*axes[-1], *err_axes[-1]]:
            ax.set_xlabel("x - centre (px)")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=scale, cmap=cmap),
            ax=axes.ravel().tolist(),
            location="right",
            shrink=0.85,
            extend="max",
            label="Equivalent strain ($\\mu\\varepsilon$)",
        )
        errors.colorbar(
            plt.cm.ScalarMappable(norm=error_scale, cmap=error_cmap),
            ax=err_axes.ravel().tolist(),
            location="right",
            shrink=0.85,
            extend="both",
            label="Signed error ($\\mu\\varepsilon$)",
        )
        footer = "3840 x 2160 registration; 16/16 affine subsets, B-spline, two levels. Zoom only. Grey = unavailable; white = hole.\nCommon-support scores exclude geometric differences; extra partial points are scored separately in metrics.csv."
        fig.supxlabel(footer, fontsize=9)
        errors.supxlabel(
            "Unclipped errors retained in NPZ/CSV. All panels and both frames share the same signed colour limits.",
            fontsize=9,
        )
        save(fig, root, f"von_mises_{frame:03d}")
        save(errors, root, f"von_mises_errors_{frame:03d}")
        # Distance bands retain a common physical denominator, so exclusion
        # cannot masquerade as improved near-boundary precision.
        radial, rad_axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
        for variant, title, color in zip(VARIANTS, TITLES, COLORS, strict=True):
            rec = next(
                r
                for r in report["records"]
                if r["mask"] == variant and r["outlier_policy"] == "median"
            )
            snap = next(s for s in rec["snapshots"] if s["frame"] == frame)
            bands = [
                snap["regions"][f"edge_{lo}_{hi}"]
                for lo, hi in ((0, 16), (16, 32), (32, 64), (64, 176))
            ]
            positions = [8, 24, 48, 120]
            rad_axes[0].plot(
                positions,
                [b["vm_rmse_ue"] for b in bands],
                "o-",
                color=color,
                label=title,
            )
            rad_axes[1].plot(
                positions,
                [100 * b["coverage_physical"] for b in bands],
                "o-",
                color=color,
                label=title,
            )
        radial.suptitle(f"Frame {frame} | median output by distance from the hole")
        rad_axes[0].set_ylabel("VM error RMSE ($\\mu\\varepsilon$)")
        rad_axes[1].set_ylabel("Accepted / physical grid centres (%)")
        rad_axes[1].set_ylim(0, 105)
        for ax in rad_axes:
            ax.set_xlabel("Distance from reference hole edge (px)")
            ax.grid(alpha=0.2)
        rad_axes[0].legend(fontsize=8)
        radial.supxlabel(
            "Errors use each method's accepted points; coverage uses a fixed physical population. Near-edge populations differ.",
            fontsize=8,
        )
        save(radial, root, f"boundary_bands_{frame:03d}")
    (root / "figure_scales.json").write_text(
        json.dumps(display, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
