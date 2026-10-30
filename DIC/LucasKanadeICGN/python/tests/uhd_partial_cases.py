"""Native-UHD ROI qualification fixtures, shared by tests and the benchmark.

No tiling, resizing, downloads or modifications of archived input images.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from test_gpu_regressions import _fractional_pair

UHD_SHAPE = (2160, 3840)
TRANSLATION = (0.35, -0.2)
FIXTURES = Path(__file__).resolve().parents[2] / "assets/uhd_tensile"


def boundary_mask() -> np.ndarray:
    y, x = np.indices(UHD_SHAPE)
    mask = ((x - 1920) / 1600) ** 2 + ((y - 1080) / 900) ** 2 <= 1
    mask[400:1205, 1930:1932] = False
    return mask


def fractional_shift(source: np.ndarray, translation=TRANSLATION) -> np.ndarray:
    """Render a known inverse translation by bilinear sampling and uint8 rounding.

    This is discrete image formation, unlike the continuous analytic-wave
    fixture. Reflected boundaries follow the archived tensile generator's
    convention. Row chunks bound setup memory; none of this runs in timing.
    """
    height, width = source.shape
    dx, dy = translation

    def reflect(indices, count):
        folded = indices % (2 * (count - 1))
        return np.minimum(folded, 2 * (count - 1) - folded)

    x = np.arange(width, dtype=np.float64) - dx
    ix = np.floor(x).astype(int)
    fx = x - ix
    x0, x1 = reflect(ix, width), reflect(ix + 1, width)
    target = np.empty_like(source)
    for start in range(0, height, 128):
        y = np.arange(start, min(start + 128, height), dtype=np.float64) - dy
        iy = np.floor(y).astype(int)
        fy = (y - iy)[:, None]
        y0, y1 = reflect(iy, height), reflect(iy + 1, height)
        top = (1 - fx) * source[y0[:, None], x0] + fx * source[y0[:, None], x1]
        bottom = (1 - fx) * source[y1[:, None], x0] + fx * source[y1[:, None], x1]
        values = (1 - fy) * top + fy * bottom
        target[start : start + y.size] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return target


def translation_pair(scene: str) -> tuple[np.ndarray, np.ndarray]:
    if scene == "analytic-wave":
        reference, target = _fractional_pair(3840, 2160, TRANSLATION)
    elif scene == "archived-speckle":
        with Image.open(FIXTURES / "reference.png") as image:
            reference = np.array(image, dtype=np.uint8)
        target = fractional_shift(reference)
    else:
        raise ValueError(f"Unknown UHD scene: {scene}")
    assert reference.shape == target.shape == UHD_SHAPE
    return reference, target


def crack_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Opposite fractional translations and a tip inside a central 17-square.

    The displacement jump is a constructed numerical case, not a physical
    fracture model. Its mask is fixed in reference coordinates.
    """
    reference, right = _fractional_pair(3840, 2160, TRANSLATION)
    _, left = _fractional_pair(3840, 2160, tuple(-v for v in TRANSLATION))
    gy, gx = 63, 113
    cut = gx * 17 + 5
    tip = gy * 17 + 14
    target = right.copy()
    target[:, :cut] = left[:, :cut]
    mask = boundary_mask()
    # Replace the generic slit with the one whose tip is inside this window.
    mask[400:1205, 1930:1932] = True
    mask[400:tip, cut] = False
    assert reference.shape == target.shape == mask.shape == UHD_SHAPE
    return reference, target, mask, (gy, gx)
