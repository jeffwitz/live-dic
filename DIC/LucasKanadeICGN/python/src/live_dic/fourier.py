"""Bounded, opt-in global translation initialization for separated image pairs.

A reference window of at most 1024² pixels is averaged by two, Hann tapered
and zero padded back to its original size. Linear phase correlation then
avoids circular displacement aliases while providing a two-pixel seed. IC-GN
performs the subpixel refinement; this module never replaces its measurements.
"""

from __future__ import annotations

import time

import numpy as np


class FourierInitializer:
    def __init__(self, reference: np.ndarray, mask: np.ndarray | None = None) -> None:
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
        height, width = reference.shape
        size = 1 << min(10, min(width, height).bit_length() - 1)
        if size < 32:
            raise ValueError("Fourier initialization needs an image of at least 32 pixels per side")
        best = None
        positions = [(0.5, 0.5)] + [(x, y) for y in (0, 0.5, 1) for x in (0, 0.5, 1)]
        for fx, fy in positions:
            x, y = int((width - size) * fx), int((height - size) * fy)
            patch = reference[y : y + size, x : x + size].astype(np.float64)
            allowed = (
                np.ones(patch.shape, bool) if mask is None else mask[y : y + size, x : x + size]
            )
            coverage = float(np.mean(allowed))
            std = float(np.std(patch[allowed])) if allowed.any() else 0.0
            if coverage >= 0.25 and std >= 1.0:
                candidate = (coverage * std, x, y, patch, allowed.copy())
                if best is None or candidate[0] > best[0]:
                    best = candidate
                if fx == fy == 0.5 and coverage >= 0.5:
                    break
        if best is None:
            raise ValueError("Fourier initialization has no sufficiently textured ROI window")
        _, x, y, patch, allowed = best
        self._window = dict(x=x, y=y, size=size, sampleStep=2)
        self.shape = reference.shape
        self.size = size
        half = size // 2
        hann = np.hanning(half)
        self.taper = hann[:, None] * hann[None, :]
        values = (patch - np.mean(patch[allowed])) * allowed
        reduced = values.reshape(half, 2, half, 2).mean(axis=(1, 3))
        self.reference_samples = reduced
        self.allowed_samples = allowed.reshape(half, 2, half, 2).all(axis=(1, 3))
        self.spectrum = np.fft.rfft2(reduced * self.taper, s=(size, size))

    @property
    def window(self) -> dict:
        return dict(self._window)

    def estimate(self, current: np.ndarray) -> dict:
        started = time.perf_counter()
        if current.shape != self.shape:
            raise ValueError("Fourier target dimensions must match the prepared reference")
        x, y, size = (self.window[key] for key in ("x", "y", "size"))
        patch = current[y : y + size, x : x + size].astype(np.float64)
        if np.std(patch) < 1.0:
            raise ValueError("Fourier initialization rejected a target window without texture")
        half = size // 2
        reduced = (patch - np.mean(patch)).reshape(half, 2, half, 2).mean(axis=(1, 3))
        spectrum = np.fft.rfft2(reduced * self.taper, s=(size, size))
        cross = spectrum * np.conj(self.spectrum)
        cross /= np.maximum(np.abs(cross), 1e-12)
        correlation = np.fft.irfft2(cross, s=(size, size))
        py, px = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak = float(correlation[py, px])
        side = np.ones(correlation.shape, bool)
        side[np.ix_(np.arange(py - 5, py + 6) % size, np.arange(px - 5, px + 6) % size)] = False
        background = correlation[side]
        score = float((peak - np.mean(background)) / max(float(np.std(background)), 1e-12))
        dx = 2 * int(px if px < half else px - size)
        dy = 2 * int(py if py < half else py - size)
        overlap = max(0, size - abs(dx)) * max(0, size - abs(dy)) / size**2
        sx, sy = dx // 2, dy // 2
        ref_y, ref_x = (
            slice(max(0, -sy), min(half, half - sy)),
            slice(max(0, -sx), min(half, half - sx)),
        )
        cur_y, cur_x = (
            slice(max(0, sy), min(half, half + sy)),
            slice(max(0, sx), min(half, half + sx)),
        )
        allowed = self.allowed_samples[ref_y, ref_x]
        a, b = self.reference_samples[ref_y, ref_x][allowed], reduced[cur_y, cur_x][allowed]
        if a.size:
            a, b = a - np.mean(a), b - np.mean(b)
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        correlation_score = float(np.sum(a * b) / norm) if norm > 0 else 0.0
        if score < 8.0 or overlap < 0.25 or correlation_score < 0.2:
            raise ValueError(
                f"Fourier initialization is ambiguous (peak score {score:.2f}, overlap {overlap:.3f}); "
                "no local field was calculated"
            )
        return dict(
            dx=dx,
            dy=dy,
            peakScore=score,
            overlap=overlap,
            correlation=correlation_score,
            window=self.window,
            method="linear-phase-correlation",
            wallMs=(time.perf_counter() - started) * 1000,
        )
