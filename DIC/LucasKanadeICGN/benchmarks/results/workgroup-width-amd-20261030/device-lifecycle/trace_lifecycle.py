"""Trace the retained-engine temporal regression without changing its assertions."""

import os
import sys
from pathlib import Path

from lucas_kanade_icgn import ICGNWGPUFullPyramid


def snapshot(stage):
    device = Path("/sys/class/drm/card1/device")
    memory = {p.name: int(p.read_text()) for p in device.glob("mem_info_*_used")}
    print(stage, memory, flush=True)


def main():
    os.environ["ICGN_RUN_GPU_TESTS"] = "1"
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "python/tests"))
    from test_fused_pyramid import CorrectedTemporalTests

    original_engine = CorrectedTemporalTests.engine
    original_register = ICGNWGPUFullPyramid.register
    original_close = ICGNWGPUFullPyramid.close

    def engine(self, **kwargs):
        snapshot(f"CREATE {kwargs}")
        result = original_engine(self, **kwargs)
        snapshot("READY")
        return result

    def register(self, *args, **kwargs):
        snapshot(f"REGISTER {self.motion_model} {self.pyramid_levels} {self.initialization}")
        return original_register(self, *args, **kwargs)

    def close(self):
        original_close(self)
        snapshot("CLOSED")

    CorrectedTemporalTests.engine = engine
    ICGNWGPUFullPyramid.register = register
    ICGNWGPUFullPyramid.close = close
    try:
        CorrectedTemporalTests("test_zero_default_and_first_temporal_frame_match_bitwise").debug()
        snapshot("DONE")
    finally:
        CorrectedTemporalTests.engine = original_engine
        ICGNWGPUFullPyramid.register = original_register
        ICGNWGPUFullPyramid.close = original_close


if __name__ == "__main__":
    main()
