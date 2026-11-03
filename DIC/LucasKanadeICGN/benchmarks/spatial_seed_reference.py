"""CPU oracle for DIS-inspired seed-candidate selection experiments.

This is benchmark code, not a live engine or a DIS implementation. Final fields
must be recomputed by IC-GN; these candidates are never returned as measurements.
OpenCV is an optional experiment dependency, not a package dependency.
"""

import numpy as np


def select_translation_candidates(
    reference, target, parameters, roi, fallback=None, size=32, stride=16, batch_size=256
):
    import cv2

    if reference.ndim != 2 or target.shape != reference.shape:
        raise ValueError("Reference and target must be matching grayscale arrays")
    if max(target.shape) >= 32767:
        raise ValueError("This CPU oracle uses OpenCV remap and requires dimensions below 32767")
    selected = roi.selected
    bits = roi.neighbour_bits
    h, w = selected.shape
    if parameters.shape[:2] != selected.shape or parameters.shape[-1] not in (6, 8):
        raise ValueError(
            "Parameters must match the ROI grid with six parameters or eight state floats"
        )
    if not np.isfinite(parameters[..., :6]).all():
        raise ValueError("Parameters must remain finite")
    if fallback is not None and not np.isfinite(fallback[..., :6]).all():
        raise ValueError("Fallback must remain finite")
    if np.any(parameters[..., (1, 2, 4, 5)][selected] != 0):
        raise ValueError("This experiment supports translation states only")
    if fallback is not None and fallback.shape[:2] != selected.shape:
        raise ValueError("Fallback must match the ROI grid")
    if batch_size < 1 or size < 2 or stride < 1:
        raise ValueError("Batch size, subset size and stride must be positive")
    seeds = np.zeros((*selected.shape, 8), np.float32)
    scores = np.full(selected.shape, -np.inf)
    ids = np.flatnonzero(selected)
    ly, lx = np.mgrid[:size, :size]
    target = target.astype(np.float32)
    for start in range(0, len(ids), batch_size):
        ii = ids[start : start + batch_size]
        y, x = np.unravel_index(ii, selected.shape)
        sx = x[:, None, None] * stride + lx
        sy = y[:, None, None] * stride + ly
        F = reference[sy, sx].astype(np.float32)
        F -= F.mean(axis=(1, 2), keepdims=True)
        fn = (F * F).sum(axis=(1, 2))
        best = np.full(len(ii), -np.inf)
        vec = parameters[y, x][:, [0, 3]].copy()
        candidates = [(vec.copy(), np.ones(len(ii), bool))]
        if fallback is not None:
            candidates.append((fallback[y, x][:, [0, 3]], np.ones(len(ii), bool)))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == dy == 0:
                    continue
                nx = np.clip(x + dx, 0, w - 1)
                ny = np.clip(y + dy, 0, h - 1)
                allowed = (bits[y, x] & (1 << ((dy + 1) * 3 + dx + 1))) != 0
                candidates.append((parameters[ny, nx][:, [0, 3]], allowed))
        for p, allowed in candidates:
            mx = (sx + p[:, 0, None, None]).astype(np.float32)
            my = (sy + p[:, 1, None, None]).astype(np.float32)
            allowed = (
                allowed
                & np.isfinite(p).all(axis=1)
                & (mx.min(axis=(1, 2)) >= 0)
                & (mx.max(axis=(1, 2)) < target.shape[1] - 1)
                & (my.min(axis=(1, 2)) >= 0)
                & (my.max(axis=(1, 2)) < target.shape[0] - 1)
                & (fn > 1e-9)
            )
            G = cv2.remap(
                target,
                np.nan_to_num(mx).reshape(-1, size),
                np.nan_to_num(my).reshape(-1, size),
                cv2.INTER_LINEAR,
            ).reshape(-1, size, size)
            G -= G.mean(axis=(1, 2), keepdims=True)
            gn = (G * G).sum(axis=(1, 2))
            rho = (F * G).sum(axis=(1, 2)) / np.sqrt(np.maximum(fn * gn, 1e-30))
            rho[~allowed | (gn < 1e-9)] = -np.inf
            improve = rho > best
            best[improve] = rho[improve]
            vec[improve] = p[improve]
        seeds[y, x, 0] = vec[:, 0]
        seeds[y, x, 3] = vec[:, 1]
        scores[y, x] = best
    return seeds, scores
