"""Compare uniform and Gaussian quadratic DIC against analytic truth.

The generated target is evaluated from the inverse of a known global
quadratic displacement field. Metrics are evaluated at each configuration's
exact subset centres, so odd and even supports can be compared without
silently shifting the ground truth.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from lucas_kanade_icgn import ICGNWGPUFullPyramid


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--gaussian-sigma", type=float, default=12.0)
    parser.add_argument(
        "--outlier-policy",
        choices=("replace", "flag", "disabled", "median"),
        default="replace",
    )
    args = parser.parse_args()
    if args.width < 128 or args.height < 128:
        parser.error("--width and --height must be at least 128")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups non-negative")
    if not np.isfinite(args.gaussian_sigma) or args.gaussian_sigma <= 0.0:
        parser.error("--gaussian-sigma must be finite and positive")
    return args


def _texture(x: np.ndarray, y: np.ndarray, phase: float) -> np.ndarray:
    return (
        127.0
        + 34.0 * np.sin(0.173 * x + 0.071 * y + phase)
        + 29.0 * np.cos(0.043 * x - 0.191 * y - 0.7 * phase)
        + 23.0 * np.sin(0.311 * x - 0.137 * y + 0.4 + 0.3 * phase)
        + 17.0 * np.cos(0.229 * x + 0.257 * y - 0.8 - phase)
    )


def _pair(
    width: int,
    height: int,
    parameters: np.ndarray,
    *,
    phase: float,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_y, target_x = np.mgrid[:height, :width].astype(np.float64)
    origin_x = 0.5 * (width - 1)
    origin_y = 0.5 * (height - 1)
    reference_x = target_x.copy()
    reference_y = target_y.copy()
    for _ in range(16):
        x = reference_x - origin_x
        y = reference_y - origin_y
        u = (
            parameters[0]
            + parameters[1] * x
            + parameters[2] * y
            + 0.5 * parameters[6] * x * x
            + parameters[7] * x * y
            + 0.5 * parameters[8] * y * y
        )
        v = (
            parameters[3]
            + parameters[4] * x
            + parameters[5] * y
            + 0.5 * parameters[9] * x * x
            + parameters[10] * x * y
            + 0.5 * parameters[11] * y * y
        )
        residual_x = reference_x + u - target_x
        residual_y = reference_y + v - target_y
        j11 = 1.0 + parameters[1] + parameters[6] * x + parameters[7] * y
        j12 = parameters[2] + parameters[7] * x + parameters[8] * y
        j21 = parameters[4] + parameters[9] * x + parameters[10] * y
        j22 = 1.0 + parameters[5] + parameters[10] * x + parameters[11] * y
        determinant = j11 * j22 - j12 * j21
        reference_x -= (j22 * residual_x - j12 * residual_y) / determinant
        reference_y -= (-j21 * residual_x + j11 * residual_y) / determinant

    reference = _texture(target_x, target_y, phase)
    target = _texture(reference_x, reference_y, phase)
    if noise_sigma:
        random = np.random.default_rng(seed)
        reference += random.normal(0.0, noise_sigma, reference.shape)
        target += random.normal(0.0, noise_sigma, target.shape)
    return (
        np.clip(np.rint(reference), 0, 255).astype(np.uint8),
        np.clip(np.rint(target), 0, 255).astype(np.uint8),
    )


def _truth(result, parameters: np.ndarray) -> np.ndarray:
    grid_y, grid_x = np.mgrid[: result.status.shape[0], : result.status.shape[1]]
    x = (
        0.5 * result.subset_size
        - 0.5
        + result.subset_stride * grid_x
        - 0.5 * (result.width - 1)
    )
    y = (
        0.5 * result.subset_size
        - 0.5
        + result.subset_stride * grid_y
        - 0.5 * (result.height - 1)
    )
    expected = np.empty((*result.status.shape, 12), dtype=np.float64)
    expected[..., 0] = (
        parameters[0]
        + parameters[1] * x
        + parameters[2] * y
        + 0.5 * parameters[6] * x * x
        + parameters[7] * x * y
        + 0.5 * parameters[8] * y * y
    )
    expected[..., 1] = parameters[1] + parameters[6] * x + parameters[7] * y
    expected[..., 2] = parameters[2] + parameters[7] * x + parameters[8] * y
    expected[..., 3] = (
        parameters[3]
        + parameters[4] * x
        + parameters[5] * y
        + 0.5 * parameters[9] * x * x
        + parameters[10] * x * y
        + 0.5 * parameters[11] * y * y
    )
    expected[..., 4] = parameters[4] + parameters[9] * x + parameters[10] * y
    expected[..., 5] = parameters[5] + parameters[10] * x + parameters[11] * y
    expected[..., 6:] = parameters[6:]
    return expected


def _metrics(result, parameters: np.ndarray) -> dict[str, object]:
    error = result.parameters.astype(np.float64) - _truth(result, parameters)
    mask = result.accepted_mask
    displacement = np.linalg.norm(error[..., (0, 3)], axis=-1)[mask]
    gradient = np.linalg.norm(error[..., (1, 2, 4, 5)], axis=-1)[mask]
    curvature = error[..., 6:][mask]
    return {
        "grid": list(result.status.shape[::-1]),
        "accepted_fraction": float(mask.mean()),
        "displacement_rmse_px": float(np.sqrt(np.mean(displacement**2))),
        "displacement_p95_px": float(np.percentile(displacement, 95)),
        "displacement_max_px": float(np.max(displacement)),
        "displacement_over_0_1_fraction": float(np.mean(displacement > 0.1)),
        "gradient_rmse": float(np.sqrt(np.mean(gradient**2))),
        "curvature_rmse_per_component": float(np.sqrt(np.mean(curvature**2))),
    }


def main() -> None:
    args = _arguments()
    cases = {
        "moderate_clean": (
            np.array(
                [
                    0.5,
                    0.0015,
                    -0.001,
                    -0.35,
                    0.0008,
                    0.0012,
                    4e-5,
                    -2.5e-5,
                    2e-5,
                    -2e-5,
                    3e-5,
                    -3.5e-5,
                ]
            ),
            0.0,
        ),
        "strong_clean": (
            np.array(
                [
                    0.5,
                    0.0015,
                    -0.001,
                    -0.35,
                    0.0008,
                    0.0012,
                    8e-5,
                    -5e-5,
                    4e-5,
                    -4e-5,
                    6e-5,
                    -7e-5,
                ]
            ),
            0.0,
        ),
        "mild_noise_sigma_1": (
            np.array(
                [
                    0.5,
                    0.0015,
                    -0.001,
                    -0.35,
                    0.0008,
                    0.0012,
                    2e-5,
                    -1.2e-5,
                    1e-5,
                    -1.1e-5,
                    1.5e-5,
                    -1.7e-5,
                ]
            ),
            1.0,
        ),
    }
    configurations = {
        "even_32_uniform": {"subset_size": 32, "spatial_weighting": "uniform"},
        "even_32_gaussian": {
            "subset_size": 32,
            "spatial_weighting": "gaussian",
            "gaussian_sigma": args.gaussian_sigma,
        },
        "odd_31_uniform": {"subset_size": 31, "spatial_weighting": "uniform"},
        "odd_31_gaussian": {
            "subset_size": 31,
            "spatial_weighting": "gaussian",
            "gaussian_sigma": args.gaussian_sigma,
        },
    }
    report: dict[str, object] = {
        "resolution": [args.width, args.height],
        "interpolation": "bspline",
        "motion_model": "quadratic",
        "outlier_policy": args.outlier_policy,
        "subset_stride": 16,
        "gaussian_sigma": args.gaussian_sigma,
        "cases": {},
    }
    case_report = report["cases"]
    assert isinstance(case_report, dict)
    for case_index, (case_name, (parameters, noise_sigma)) in enumerate(cases.items()):
        reference, target = _pair(
            args.width,
            args.height,
            parameters,
            phase=0.37 * case_index,
            noise_sigma=noise_sigma,
            seed=20261029 + case_index,
        )
        case_report[case_name] = {}
        for name, options in configurations.items():
            engine = ICGNWGPUFullPyramid(
                pyramid_levels=2,
                iterations_per_level=(10, 20),
                tolerance_per_level=(5e-3, 2e-3),
                interpolation="bspline",
                motion_model="quadratic",
                outlier_policy=args.outlier_policy,
                subset_stride=16,
                **options,
            )
            try:
                engine.set_reference(reference, zncc_min=0.7)
                for _ in range(args.warmups):
                    engine.register(target)
                timings = []
                result = None
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    result = engine.register(target)
                    timings.append(1_000.0 * (time.perf_counter() - started))
                assert result is not None
                case_report[case_name][name] = {
                    **_metrics(result, parameters),
                    "latency_median_ms": float(np.median(timings)),
                    "latency_p95_ms": float(np.percentile(timings, 95)),
                }
            finally:
                engine.close()
                engine.device.destroy()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
