"""Compare scored spatial seeds with equal-budget continuation (CPU research harness).

Uses unchanged native IC-GN kernels; CPU scoring/readbacks make this unsuitable
as a live pipeline. Install opencv-python-headless separately for spatial mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from live_dic import ICGNWGPU
from live_dic.roi import prepare_roi, reduce_mask, seed_permissions
from live_dic.wgpu_backend import _reduce_half, _transfer_pyramid_seed
from spatial_seed_reference import select_translation_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--mask", type=Path, help="Boolean reference-sized NPY; True retains pixels"
    )
    parser.add_argument(
        "--initial-affine",
        type=Path,
        help="JSON: dx, dy and linear=[du/dx,du/dy,dv/dx,dv/dy], centred at (W-1,H-1)/2",
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--mode", choices=("spatial", "continue"), default="spatial")
    parser.add_argument("--passes", type=int, choices=range(0, 9), default=2)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--subset-stride", type=int, default=16)
    args = parser.parse_args()

    def load(path):
        with Image.open(path) as image:
            if image.mode != "L":
                parser.error(
                    "Use explicit 8-bit grayscale inputs; this harness does not scale intensities"
                )
            return np.array(image)

    f = load(args.reference)
    g = load(args.target)
    if f.shape != g.shape:
        parser.error("Images must have the same shape")
    mask = (
        np.ones(f.shape, dtype=bool)
        if args.mask is None
        else np.load(args.mask, allow_pickle=False)
    )
    fit = (
        {"dx": 0.0, "dy": 0.0, "linear": [0.0, 0.0, 0.0, 0.0]}
        if args.initial_affine is None
        else json.loads(args.initial_affine.read_text())
    )
    if len(fit["linear"]) != 4 or not np.isfinite([fit["dx"], fit["dy"], *fit["linear"]]).all():
        parser.error("Affine values must be finite, with four linear coefficients")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(
        json.dumps(
            dict(
                command=sys.argv,
                sourceCommit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
                ).strip(),
                sourceDirty=bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"],
                        cwd=Path(__file__).resolve().parents[3],
                        text=True,
                    ).strip()
                ),
                inputSha256={
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (args.reference, args.target)
                },
                scriptSha256={
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__),
                        Path(__file__).with_name("spatial_seed_reference.py"),
                    )
                },
                shape=f.shape,
                initialAffine=fit,
                diagnosticCpuHost=True,
                iterations=[8, 16],
                tolerance=0.005,
                znccMin=0.8,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    size = args.subset_size
    stride = args.subset_stride
    fc = _reduce_half(f)
    gc = _reduce_half(g)

    def select_candidates(reference, target, parameters, roi, **kwargs):
        if args.mode == "spatial":
            return select_translation_candidates(
                reference, target, parameters, roi, size=size, stride=stride, **kwargs
            )
        seeds = np.zeros((*parameters.shape[:2], 8), np.float32)
        seeds[..., :6] = parameters
        return seeds, None

    def initial(engine, scale):
        y, x = np.indices((engine.grid_height, engine.grid_width))
        x = scale * (x * stride + (size - 1) / 2) + (scale - 1) * 0.5 - (f.shape[1] - 1) / 2
        y = scale * (y * stride + (size - 1) / 2) + (scale - 1) * 0.5 - (f.shape[0] - 1) / 2
        b = fit["linear"]
        s = np.zeros((*x.shape, 8), np.float32)
        s[..., 0] = (fit["dx"] + b[0] * x + b[1] * y) / scale
        s[..., 3] = (fit["dy"] + b[2] * x + b[3] * y) / scale
        return s

    rows = []
    options = dict(
        motion_model="translation",
        subset_size=size,
        subset_stride=stride,
        outlier_policy="disabled",
        prefer_subgroups=False,
        device_type="gpu",
    )
    with ICGNWGPU(**options) as fine, ICGNWGPU(**options) as coarse:
        print("PREPARING", flush=True)
        fine.set_reference(f, roi_mask=mask, iterations=16)
        coarse.set_reference(fc, roi_mask=reduce_mask(mask), iterations=8)
        configuration_path = out / "configuration.json"
        configuration = json.loads(configuration_path.read_text())
        configuration["adapter"] = fine.adapter_info
        configuration_path.write_text(json.dumps(configuration, indent=2, allow_nan=False) + "\n")
        fplan = fine.roi or prepare_roi(mask, size, stride)
        cplan = coarse.roi or prepare_roi(reduce_mask(mask), size, stride)
        print("PREPARED", flush=True)
        origc = initial(coarse, 2)
        origf = initial(fine, 1)
        cr = coarse._register_seeded_frame(gc, origc, cplan.selected.astype(np.uint32))
        perm = seed_permissions(
            cplan,
            fplan,
            source_scale=2,
            destination_scale=1,
            size=size,
            stride=stride,
            reference_mask=mask,
        )
        for stage in range(args.passes + 1):
            if stage:
                cs, _ = select_candidates(fc, gc, cr.parameters, cplan, fallback=origc)
                cr = coarse._register_seeded_frame(gc, cs, cplan.selected.astype(np.uint32))
            fs, _ = _transfer_pyramid_seed(
                cr,
                source_available=cplan.selected,
                source_scale=2,
                destination_scale=1,
                destination_grid=(fine.grid_width, fine.grid_height),
                requested=fplan.selected,
                subset_size=size,
                subset_stride=stride,
                permissions=perm,
            )
            fr = fine._register_seeded_frame(g, fs, fplan.selected.astype(np.uint32))
            row = dict(
                coarsePropagationPasses=stage,
                finePropagationPasses=0,
                accepted=fr.accepted,
                selected=int(fplan.selected.sum()),
                mode=args.mode,
            )
            rows.append(row)
            fr.config = {
                **dict(fr.config),
                "seedCandidateExperiment": dict(row),
                "initialAffine": fit,
            }
            fr.save_npz(out / f"{args.mode}-C{stage}-F0.npz")
            print(json.dumps(row), flush=True)
            for step in range(1, args.passes + 1):
                fs, _ = select_candidates(f, g, fr.parameters, fplan, fallback=origf)
                fr = fine._register_seeded_frame(g, fs, fplan.selected.astype(np.uint32))
                row = dict(
                    coarsePropagationPasses=stage,
                    finePropagationPasses=step,
                    accepted=fr.accepted,
                    selected=int(fplan.selected.sum()),
                    mode=args.mode,
                )
                rows.append(row)
                fr.config = {
                    **dict(fr.config),
                    "seedCandidateExperiment": dict(row),
                    "initialAffine": fit,
                }
                fr.save_npz(out / f"{args.mode}-C{stage}-F{step}.npz")
                print(json.dumps(row), flush=True)
            (out / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
