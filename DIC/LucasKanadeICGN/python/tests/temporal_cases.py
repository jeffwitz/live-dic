"""Deterministic fixed-reference sequence formation, outside timed regions."""

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "assets/uhd_tensile"


def tensile_stretches():
    truth = np.genfromtxt(FIXTURES / "machine_truth.csv", delimiter=",", names=True)
    return np.column_stack((truth["stretch_x"], truth["stretch_y"]))


def warp(reference, stretch=(1.0, 1.0), translation=(0.0, 0.0), center=None):
    """Inverse affine map, float64 bilinear sampling, reflected pixels, nearest byte.

    Byte rounding differs from the exact bilinear interpolant by <=0.5 NG.
    This is discrete image formation, not an exact continuous-speckle oracle.
    Unlike the historical OpenCV remap, fractions are not quantized to 1/32.
    Native UHD uses precisely (1920,1080), not the pixel-centre midpoint.
    """
    height, width = reference.shape
    cx, cy = (width / 2, height / 2) if center is None else center
    lx, ly = stretch
    tx, ty = translation

    def reflect(index, size):
        folded = index % (2 * (size - 1))
        return np.minimum(folded, 2 * (size - 1) - folded)

    x = cx + (np.arange(width, dtype=np.float64) - cx - tx) / lx
    ix = np.floor(x).astype(int)
    fx = x - ix
    x0, x1 = reflect(ix, width), reflect(ix + 1, width)
    target = np.empty_like(reference)
    for start in range(0, height, 128):
        y = cy + (np.arange(start, min(start + 128, height), dtype=np.float64) - cy - ty) / ly
        iy = np.floor(y).astype(int)
        fy = (y - iy)[:, None]
        y0, y1 = reflect(iy, height), reflect(iy + 1, height)
        top = (1 - fx) * reference[y0[:, None], x0] + fx * reference[y0[:, None], x1]
        bottom = (1 - fx) * reference[y1[:, None], x0] + fx * reference[y1[:, None], x1]
        target[start : start + y.size] = np.clip(np.rint((1 - fy) * top + fy * bottom), 0, 255)
    return target
