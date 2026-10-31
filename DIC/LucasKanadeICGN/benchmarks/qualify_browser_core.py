"""Compare browser/native temporal graphs through the existing DevTools harness."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWebGL, ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/tests"))
from temporal_cases import tensile_stretches, warp  # noqa: E402


def literal(array):
    encoded = base64.b64encode(array.tobytes()).decode()
    return f"Uint8Array.from(atob({json.dumps(encoded)}), c => c.charCodeAt(0))"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software-browser", action="store_true")
    parser.add_argument("--native-device", choices=("gpu", "cpu", "auto"), default="gpu")
    args = parser.parse_args()
    criteria = json.loads(
        (ROOT / "benchmarks/results/browser-core-20261031/criteria.json").read_text()
    )
    ref = np.array(Image.open(ROOT / "assets/uhd_tensile/reference.png"))[
        800:1056, 1500:1884
    ].copy()
    frames = [warp(ref, stretch=s) for s in tensile_stretches()[:60]]
    for i in (20, 21, 22):
        frames[i] = np.zeros_like(ref)
    mask = np.ones(ref.shape, bool)
    mask[:, 190:192] = False
    mask[100:130, 80:100] = False
    flags = (
        (
            "--use-angle=swiftshader",
            "--enable-unsafe-webgpu",
            "--use-webgpu-adapter=swiftshader",
            "--use-vulkan=swiftshader",
        )
        if args.software_browser
        else ()
    )
    rows = []
    with ICGNWebGL(backend="webgpu-wgsl", chromium_args=flags) as browser:
        for model in ("translation", "affine"):
            for depth in (1, 2):
                options = dict(
                    motion_model=model,
                    initialization="temporal",
                    temporal_prediction="velocity",
                    temporal_fine_correction=True,
                    temporal_max_age=2,
                    divergence_patience=3,
                    divergence_min_iterations=4,
                    outlier_policy="median",
                    device_type=args.native_device,
                )
                js_options = dict(
                    width=ref.shape[1],
                    height=ref.shape[0],
                    motionModel=model,
                    initialization="temporal",
                    temporalPrediction="velocity",
                    temporalFineCorrection=True,
                    temporalMaxAge=2,
                    divergencePatience=3,
                    divergenceMinIterations=4,
                    outlierPolicy="median",
                    pipelineDepth=depth,
                )
                browser._evaluate(f"""(async()=>{{
                  globalThis.core?.destroy();
                  const {{DICEngine}} = await import('../core/engine.js');
                  globalThis.core = await DICEngine.create({json.dumps(js_options)});
                  await core.setReference({literal(ref)}, {{roiMask:{literal(mask.astype(np.uint8))}}});
                  return true;
                }})()""")
                with ICGNWGPUFullPyramid(**options) as native:
                    native.set_reference(ref, roi_mask=mask)
                    for start, stop in ((0, 40), (40, 60)):
                        if start:
                            native.reset_temporal_state()
                            browser._evaluate("core.resetTemporalState(); true")
                        native_fields = list(
                            native.register_stream(frames[start:stop], pipeline_depth=depth)
                        )
                        # Submit actual batches to exercise ordering at the requested depth.
                        for offset in range(0, stop - start, depth):
                            batch = frames[start + offset : min(stop, start + offset + depth)]
                            expression = (
                                "(async()=>{const results=await Promise.all(["
                                + ",".join(f"core.register({literal(frame)})" for frame in batch)
                                + """]);
                              return results.map(r=>({ parameters:Array.from(r.parameters),status:Array.from(r.status),
                                iterations:Array.from(r.iterations),counters:r.counters }));})()"""
                            )
                            fields = browser._evaluate(expression)
                            for j, actual in enumerate(fields):
                                expected = native_fields[offset + j]
                                parameters = np.asarray(actual["parameters"], np.float32).reshape(
                                    expected.parameters.shape
                                )
                                status = np.asarray(actual["status"]).reshape(expected.status.shape)
                                iterations = np.asarray(actual["iterations"]).reshape(
                                    expected.iterations.shape
                                )
                                assert np.isfinite(parameters).all()
                                common = (status > 0) & expected.accepted_mask
                                delta = parameters[common] - expected.parameters[common]
                                row = dict(
                                    model=model,
                                    depth=depth,
                                    frame=start + offset + j,
                                    status_mismatches=int(
                                        np.count_nonzero(status != expected.status)
                                    ),
                                    iteration_mismatches=int(
                                        np.count_nonzero(iterations != expected.iterations)
                                    ),
                                    grid_total=status.size,
                                    common_accepted=int(common.sum()),
                                    parameter_rmse=float(np.sqrt(np.mean(delta**2)))
                                    if delta.size
                                    else None,
                                    parameter_max_abs=float(np.max(np.abs(delta)))
                                    if delta.size
                                    else None,
                                    parameter_all_max_abs=float(
                                        np.max(np.abs(parameters - expected.parameters))
                                    ),
                                    parameter_bit_equal=bool(
                                        np.array_equal(parameters, expected.parameters)
                                    ),
                                    browser_counters=actual["counters"],
                                    native_counters={
                                        k: int(expected.timing[k])
                                        for k in actual["counters"]
                                        if k in expected.timing
                                    },
                                )
                                gate = criteria["cross_runtime_tolerance"]
                                row["exceeds"] = (
                                    row["status_mismatches"] / status.size
                                    > gate["status_mismatch_fraction_max"]
                                    or row["iteration_mismatches"] / status.size
                                    > gate["iteration_mismatch_fraction_max"]
                                    or (
                                        delta.size > 0
                                        and (
                                            row["parameter_rmse"]
                                            > gate["common_accepted_parameter_rmse_max"]
                                            or row["parameter_max_abs"]
                                            > gate["common_accepted_parameter_max_abs_max"]
                                        )
                                    )
                                )
                                rows.append(row)
                                print(
                                    model,
                                    depth,
                                    row["frame"],
                                    row["status_mismatches"],
                                    row["iteration_mismatches"],
                                    row["parameter_max_abs"],
                                    row["exceeds"],
                                    flush=True,
                                )
        info = browser._evaluate("core.host.info()")
        browser._evaluate("core.destroy(); true")
    report = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        criteria=criteria,
        browser=info,
        reference_sha256=hashlib.sha256(ref).hexdigest(),
        mask_sha256=hashlib.sha256(mask).hexdigest(),
        frame_sha256=[hashlib.sha256(frame).hexdigest() for frame in frames],
        rows=rows,
        exceedances=sum(row["exceeds"] for row in rows),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["exceedances"]:
        raise SystemExit(
            f"{report['exceedances']} frames exceed predeclared criteria; report retained"
        )


if __name__ == "__main__":
    main()
