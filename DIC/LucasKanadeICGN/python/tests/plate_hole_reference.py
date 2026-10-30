"""Independent NumPy evaluation of the original SYSCL native-UHD plate fixture.

Generation/scoring only: these material constants and loads never enter DIC.
Reference: airy_plate_4k/common.hpp and airy_plate/kirsch.hpp at the source
revision recorded in assets/uhd_plate_hole/manifest.json. Units are MPa/mm,
with 0.01 mm/pixel. The image is a window in an infinite plane-strain medium.
"""

from functools import lru_cache

import numpy as np

WIDTH, HEIGHT = 3840, 2160
CX, CY, RADIUS, PIXEL_MM = 1920.0, 1080.0, 112.0, 0.01
YOUNG, POISSON = 210000.0, 0.3
MU = YOUNG / (2 * (1 + POISSON))
KAPPA = 3 - 4 * POISSON
A = RADIUS * PIXEL_MM


def von_mises_strain(strain: np.ndarray) -> np.ndarray:
    """Equivalent strain for (exx, eyy, tensorial exy), with ezz = 0."""
    exx, eyy, exy = np.moveaxis(np.asarray(strain), -1, 0)
    return (2 / 3) * np.sqrt(np.maximum(0, exx**2 - exx * eyy + eyy**2 + 3 * exy**2))


def _raw_displacement(x, y, horizontal=False):
    x, y = (np.asarray(x) - CX) * PIXEL_MM, (np.asarray(y) - CY) * PIXEL_MM
    if horizontal:
        x, y = y, x
    z = y - 1j * x
    psi = (z + 2 * A**2 / z) / 4
    derivative_psi = (1 - 2 * A**2 / z**2) / 4
    derivative_chi = -(z + A**2 / z - A**4 / z**3) / 2
    u = (KAPPA * psi - z * derivative_psi.conjugate() - derivative_chi.conjugate()) / (2 * MU)
    result = np.stack((-u.imag / PIXEL_MM, u.real / PIXEL_MM), axis=-1)
    return result[..., ::-1] if horizontal else result


@lru_cache(maxsize=1)
def calibration() -> tuple[float, float, float, float]:
    """Original pin translation and zero-extension secondary-mode projection.

    Keep the original generator's 8+16*i section coordinates here. Evaluating
    the live-dic result, separately, uses its actual 7.5+16*i subset centres.
    """
    x = np.arange(56, 3785, 16)
    low, high, hlow, hhigh = [], [], [], []
    for y in (120, 136):
        low.extend(_raw_displacement(x, y)[..., 1])
        hlow.extend(_raw_displacement(x, y, True)[..., 1])
    for y in (2024, 2040):
        high.extend(_raw_displacement(x, y)[..., 1])
        hhigh.extend(_raw_displacement(x, y, True)[..., 1])
    pin, hpin = float(np.mean(low)), float(np.mean(hlow))
    command = float(np.mean(high) - pin)
    ratio = float((np.mean(hhigh) - hpin) / command)
    return pin, hpin, ratio, command


def displacement(x, y, primary, secondary=0.0) -> np.ndarray:
    pin, hpin, ratio, _ = calibration()
    vertical = _raw_displacement(x, y).copy()
    vertical[..., 1] -= pin
    horizontal = _raw_displacement(x, y, True).copy()
    horizontal[..., 1] -= hpin
    horizontal -= ratio * vertical
    return np.asarray(primary)[..., None] * vertical + np.asarray(secondary)[..., None] * horizontal


def _stress(x, y, load):
    r = np.hypot(x, y)
    c, s, q = y / r, -x / r, A**2 / r**2
    c2, s2 = c**2 - s**2, 2 * c * s
    rr = load / 2 * ((1 - q) + (1 - 4 * q + 3 * q**2) * c2)
    tt = load / 2 * ((1 + q) - (1 + 3 * q**2) * c2)
    rt = -load / 2 * (1 + 2 * q - 3 * q**2) * s2
    xx = rr * s**2 + tt * c**2 + 2 * rt * c * s
    yy = rr * c**2 + tt * s**2 - 2 * rt * c * s
    xy = -((rr - tt) * c * s + rt * c2)
    return np.stack((xx, yy, xy), axis=-1)


def strain(x, y, primary, secondary=0.0) -> np.ndarray:
    _, _, ratio, _ = calibration()
    x, y = (np.asarray(x) - CX) * PIXEL_MM, (np.asarray(y) - CY) * PIXEL_MM
    v, h = _stress(x, y, primary - ratio * secondary), _stress(y, x, secondary)
    xx, yy, xy = v[..., 0] + h[..., 1], v[..., 1] + h[..., 0], v[..., 2] + h[..., 2]
    return np.stack(
        (
            ((1 - POISSON) * xx - POISSON * yy) / (2 * MU),
            ((1 - POISSON) * yy - POISSON * xx) / (2 * MU),
            xy / (2 * MU),
        ),
        axis=-1,
    )


def hole_mask() -> np.ndarray:
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    return (x - CX) ** 2 + (y - CY) ** 2 >= RADIUS**2


def grid(size=16, stride=16) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(((HEIGHT - size) // stride + 1, (WIDTH - size) // stride + 1))
    return x * stride + (size - 1) / 2, y * stride + (size - 1) / 2


def strain_from_parameters(parameters: np.ndarray) -> np.ndarray:
    return np.stack(
        (parameters[..., 1], parameters[..., 5], 0.5 * (parameters[..., 2] + parameters[..., 4])),
        axis=-1,
    )
