"""Independent parser controls for optional Linux benchmark observation."""

import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "benchmarks/monitor_linux_process.py"
SPEC = importlib.util.spec_from_file_location("monitor_linux_process", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LinuxMonitorTests(unittest.TestCase):
    def test_duplicate_descriptors_do_not_double_count_a_drm_client(self):
        common = "drm-driver: xe\ndrm-client-id: 12\ndrm-pdev: 0000:00:02.0\n"
        records = [("3", common), ("4", common), ("5", common.replace("00:02.0", "01:00.0"))]
        self.assertEqual(len(MODULE.parse_clients(records)), 2)
        anonymous = "drm-driver: example\n"
        self.assertEqual(len(MODULE.parse_clients([("6", anonymous), ("7", anonymous)])), 2)

    def test_cycle_timestamps_are_not_misreported_as_memory(self):
        clients = MODULE.parse_clients(
            [
                (
                    "3",
                    "drm-driver: xe\ndrm-total-system: 2 KiB\ndrm-resident-system: 512\ndrm-total-cycles-rcs: 10000000000000\n",
                ),
                ("4", "drm-driver: other\ndrm-total-local: 1 MiB\ndrm-resident-local: 1 MiB\n"),
            ]
        )
        self.assertEqual(
            MODULE.memory_totals(clients),
            {
                "requestedBytes": 1024**2 + 2048,
                "residentBytes": 1024**2 + 512,
            },
        )
        self.assertEqual(MODULE.memory_totals([]), {"requestedBytes": None, "residentBytes": None})
        self.assertIsNone(MODULE.size_bytes("unknown"))
        self.assertIsNone(MODULE.size_bytes("1 bananas"))

    def test_process_names_with_spaces_and_parentheses_keep_the_correct_fields(self):
        # Linux fields 3..22: state, ten intervening fields, user/system ticks,
        # six intervening fields, then process start ticks.
        fields = ["S", *["0"] * 10, "17", "23", *["0"] * 6, "456"]
        actual = MODULE.process_stat("123 (python (worker)) " + " ".join(fields))
        self.assertEqual(actual, {"state": "S", "cpuTicks": 40, "startTicks": 456})
