from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lucas_kanade_icgn.cli import _save_benchmark, _save_registrations
from lucas_kanade_icgn.engine import BenchmarkResult, RegistrationResult


def _registration(policy: str = "median") -> RegistrationResult:
    parameters = np.zeros((1, 1, 6), dtype=np.float32)
    status = np.ones((1, 1), dtype=np.float32)
    iterations = np.full((1, 1), 3.0, dtype=np.float32)
    return RegistrationResult(
        parameters=parameters,
        status=status,
        iterations=iterations,
        accepted=1,
        replaced=0,
        mean_translation=(0.0, 0.0),
        mean_iterations=3.0,
        timing={"totalWallMs": 1.25},
        config={
            "motionModel": "affine",
            "outlierPolicy": policy,
            "subsetSize": 16,
        },
        width=16,
        height=16,
        subset_size=16,
        subset_stride=16,
    )


class CLIExportTests(unittest.TestCase):
    def test_sequence_archive_preserves_registration_configuration(self) -> None:
        result = _registration("median")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sequence.npz"
            _save_registrations(output, [Path("target.png")], [result])

            with np.load(output, allow_pickle=False) as archive:
                config = json.loads(str(archive["config_json"].item()))
                self.assertEqual(config, dict(result.config))
                np.testing.assert_array_equal(archive["parameters"], result.parameters[None])

    def test_benchmark_archive_preserves_registration_configuration(self) -> None:
        registration = _registration("flag")
        benchmark = BenchmarkResult(
            repeats=2,
            warmups=1,
            gpu_times_ms=np.asarray([0.8, 0.9]),
            wall_times_ms=np.asarray([1.1, 1.2]),
            gpu_median_ms=0.85,
            gpu_p95_ms=0.9,
            wall_median_ms=1.15,
            wall_p95_ms=1.2,
            device="test device",
            registration=registration,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark.npz"
            _save_benchmark(output, Path("target.png"), benchmark)

            with np.load(output, allow_pickle=False) as archive:
                config = json.loads(str(archive["config_json"].item()))
                self.assertEqual(config, dict(registration.config))
                np.testing.assert_array_equal(archive["gpu_times_ms"], [0.8, 0.9])


if __name__ == "__main__":
    unittest.main()
