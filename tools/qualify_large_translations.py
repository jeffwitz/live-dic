"""Compare two/four levels on independent, known rigid shifts of the original UHD texture."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from qualify_local_sources import choose_files, export, wait

from live_dic import ICGNBrowser, ICGNWGPUFullPyramid
from lucas_kanade_icgn.engine import _StaticServer
from lucas_kanade_icgn.full_pyramid_backend import _DEFAULT_ITERATION_SCHEDULES

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "DIC/LucasKanadeICGN/assets/uhd_tensile/reference.png"
SHIFTS = [(8, -4), (16, -8), (32, -16), (64, -32)]


def shifted(reference, dx, dy):
    """Exact integer translation with zero-filled new content, never wrap/tiling."""
    height, width = reference.shape
    target = np.zeros_like(reference)
    target[max(0, dy) : min(height, height + dy), max(0, dx) : min(width, width + dx)] = reference[
        max(0, -dy) : min(height, height - dy), max(0, -dx) : min(width, width - dx)
    ]
    return target


def quality(parameters, status, truth, common_motion=False):
    grid_y, grid_x = np.indices((135, 240))
    x, y = grid_x * 16 + 7.5, grid_y * 16 + 7.5
    population = (x > 192) & (x < 3840 - 192) & (y > 192) & (y < 2160 - 192)
    if common_motion:
        population &= (
            (x + truth[0] > 192)
            & (x + truth[0] < 3840 - 192)
            & (y + truth[1] > 192)
            & (y + truth[1] < 2160 - 192)
        )
    accepted = population.ravel() & (status.ravel() > 0.5)
    assert np.isfinite(parameters).all()
    error = np.linalg.norm(parameters.reshape(-1, 6)[accepted][:, [0, 3]] - truth, axis=-1)
    coverage = float(accepted.sum() / population.sum())
    rmse = float(np.sqrt(np.mean(error**2))) if error.size else None
    p95 = float(np.percentile(error, 95)) if error.size else None
    return dict(
        population=int(population.sum()),
        accepted=int(accepted.sum()),
        accepted_coverage=coverage,
        accurate_coverage=float(np.count_nonzero(error < 0.1) / population.sum()),
        displacement_rmse_px=rmse,
        displacement_p95_px=p95,
        qualified=bool(coverage >= 0.90 and rmse is not None and rmse < 0.05 and p95 < 0.10),
    )


def native(reference, args):
    records = []
    for model in ("translation", "affine"):
        for policy in ("disabled", "median"):
            for levels in (2,) if args.fourier else (2, 4):
                schedule = (8, 16) if levels == 2 else (20, 8, 8, 16)
                with ICGNWGPUFullPyramid(
                    device_type="gpu",
                    motion_model=model,
                    outlier_policy=policy,
                    pyramid_levels=levels,
                    iterations_per_level=schedule,
                    initialization="fourier" if args.fourier else "zero",
                ) as engine:
                    preparation = engine.set_reference(reference)
                    for truth in SHIFTS:
                        target = shifted(reference, *truth)
                        for _ in range(args.warmups):
                            engine.register(target)
                        samples = []
                        for _ in range(args.repeats):
                            start = time.perf_counter()
                            result = engine.register(target)
                            samples.append((time.perf_counter() - start) * 1000)
                        metrics = quality(result.parameters, result.status, truth, args.fourier)
                        if args.fourier or (levels == 4 and truth == (16, -8)):
                            assert metrics["qualified"], (model, policy, metrics)
                        record = dict(
                            model=model,
                            policy=policy,
                            levels=levels,
                            shift=list(truth),
                            config=result.config,
                            adapter=engine.adapter_info,
                            preparation_wall_ms=preparation.total_wall_ms,
                            target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                            samples_ms=samples,
                            last_timing=result.timing,
                            median_ms=float(np.median(samples)),
                            p95_ms=float(np.percentile(samples, 95)),
                            metrics=metrics,
                        )
                        records.append(record)
                        print(
                            json.dumps(
                                {
                                    k: record[k]
                                    for k in (
                                        "model",
                                        "policy",
                                        "levels",
                                        "shift",
                                        "median_ms",
                                        "metrics",
                                    )
                                }
                            ),
                            flush=True,
                        )
    return records


def browser_cases(reference, args):
    site = args.site.resolve()
    server = _StaticServer(site.parent)
    server.start()
    flags = (
        (
            "--use-angle=swiftshader",
            "--enable-unsafe-webgpu",
            "--use-webgpu-adapter=swiftshader",
            "--use-vulkan=swiftshader",
        )
        if args.software
        else ()
    )
    try:
        visible = Path.home() / "snap/chromium/common"
        with (
            tempfile.TemporaryDirectory(dir=visible if visible.is_dir() else None) as temporary,
            ICGNBrowser(backend="webgpu-wgsl", chromium_args=flags, call_timeout=300) as browser,
        ):
            paths = []
            for index, image in enumerate(
                [reference, *[shifted(reference, *truth) for truth in SHIFTS]]
            ):
                path = Path(temporary) / f"frame_{index:03}.png"
                Image.fromarray(image).save(path)
                paths.append(path)
            browser._evaluate(
                f"location.assign({json.dumps(f'http://127.0.0.1:{server.port}/{quote(site.name)}/?smoke=1&case=manual')});true"
            )
            wait(browser, "globalThis.liveDicStatus?.ready")
            asyncio.run(choose_files(browser, "#image-files", paths))
            wait(
                browser,
                "liveDicSourceState().source==='images'&&!liveDicSourceState().loadingSource",
            )
            browser._evaluate("""document.getElementById('record-fields').checked=false;
              document.getElementById('custom-solver').checked=true;document.getElementById('temporal-start').checked=false;
              document.getElementById('outlier-policy').value='disabled';true""")
            if args.fourier:
                browser._evaluate(
                    "document.getElementById('fourier-start').checked=true;document.getElementById('fourier-start').onchange();true"
                )
            records = []
            for levels in (2,) if args.fourier else (2, 4):
                if levels == 4:
                    browser._evaluate(
                        "document.getElementById('large-shift-preset').onclick();true"
                    )
                else:
                    browser._evaluate(
                        """document.getElementById('pyramid-levels').value='2';document.getElementById('coarse-iterations').value='8';document.getElementById('custom-solver').onchange();true"""
                    )
                browser._evaluate(
                    "document.getElementById('image-reference').value='0';document.getElementById('use-image-reference').onclick().then(()=>true)"
                )
                applied = browser._evaluate(
                    "JSON.parse(document.getElementById('details').textContent).reference.config"
                )
                assert applied["pyramidLevels"] == levels and applied["initialization"] == (
                    "fourier" if args.fourier else "zero"
                ), applied
                for index, truth in enumerate(SHIFTS, 1):
                    browser._evaluate(
                        f"document.getElementById('local-seek').value={index};document.getElementById('local-seek').onchange().then(()=>true)"
                    )
                    browser._evaluate(
                        "document.getElementById('solve-range').value='single';document.getElementById('pause').onclick().then(()=>true)"
                    )
                    result = export(browser)
                    metrics = quality(result["parameters"], result["status"], truth, args.fourier)
                    if args.fourier or (levels == 4 and truth == (16, -8)):
                        assert metrics["qualified"], metrics
                    records.append(
                        dict(
                            levels=levels,
                            shift=list(truth),
                            metrics=metrics,
                            config=applied,
                            metadata=result["metadata"],
                        )
                    )
            return dict(
                build=json.loads((site / "build.json").read_text()),
                software=args.software,
                records=records,
                errors=browser._evaluate("liveDicStatus.errors"),
            )
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--browser-only", action="store_true")
    parser.add_argument("--fourier", action="store_true")
    parser.add_argument("--software", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    global SHIFTS
    if args.fourier:
        SHIFTS = [(64, -32), (256, -128), (512, -128), (700, -200)]
    if args.warmups < 0 or args.repeats < 1 or (args.browser_only and args.site is None):
        parser.error("Invalid warmups/repeats or missing --site for --browser-only")
    with Image.open(REFERENCE) as image:
        reference = np.array(image, dtype=np.uint8)
    assert reference.shape == (2160, 3840)
    report = dict(
        commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        dirty=bool(
            subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
        ),
        command=subprocess.list2cmdline(sys.argv),
        platform=platform.platform(),
        versions={name: version(name) for name in ("numpy", "wgpu", "Pillow")},
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        reference_pixel_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        mode="fourier" if args.fourier else "zero",
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *[
                    ROOT / "DIC/LucasKanadeICGN/python/src/lucas_kanade_icgn" / name
                    for name in [
                        "fourier.py",
                        "full_pyramid_backend.py",
                        "web/core/fourier.js",
                        "web/core/pyramid.js",
                        "web/core/temporal.js",
                    ]
                ],
            ]
        },
        schedules=dict(
            default_four=list(_DEFAULT_ITERATION_SCHEDULES[4]),
            trial_four=[20, 8, 8, 16],
            baseline_two=[8, 16],
        ),
        warmups=args.warmups,
        repeats=args.repeats,
        native=None if args.browser_only else native(reference, args),
        browser=browser_cases(reference, args) if args.site else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
