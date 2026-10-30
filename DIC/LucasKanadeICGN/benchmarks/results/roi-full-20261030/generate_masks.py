"""Reproduce the exact boolean UHD masks used by this qualification.

Usage: python generate_masks.py /tmp/live-dic-roi-masks
NPY files are experiment inputs, not extra runtime resources in the wheel.
"""
import sys
from pathlib import Path

import numpy as np

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
full = np.ones((2160, 3840), dtype=bool)
large = np.zeros_like(full)
large[160:2000, 160:3680] = True
small = np.zeros_like(full)
small[800:1360, 1600:2240] = True
fragmented = large.copy()
for x in range(512, 3500, 512):
    fragmented[320:1840, x:x+1] = False
fragmented[880:1200, 1800:2040] = False
for name, mask in (("full", full), ("large", large), ("small", small),
                   ("fragmented", fragmented)):
    np.save(destination / f"{name}.npy", mask, allow_pickle=False)
