"""Static reference-coordinate ROI geometry; no work here runs per target.

Neighbour/seed links use the closed pixel box enclosing the centre segment.
This conservative test includes every pixel touched by the segment, including
corner touches, and can reject additional links near oblique mask boundaries.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .subset import subset_grid_shape

PARTIAL_MAX_NORMALIZED_CONDITION = 1e8
PARTIAL_MIN_WEIGHTED_CONTRAST = 1.0


def immutable(array: np.ndarray) -> np.ndarray:
    """Own immutable bytes, rather than a writable alias to a caller's array."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def validate_roi(
    mask: np.ndarray | None,
    policy: str,
    shape: tuple[int, int],
    motion_model: str,
    truncation: str = "none",
) -> np.ndarray | None:
    """Validate even absent masks before replacing any reference resources."""
    if not isinstance(policy, str) or policy not in ("full", "partial"):
        raise ValueError("roi_policy must be 'full' or 'partial'")
    if not isinstance(truncation, str) or truncation not in ("none", "visibility"):
        raise ValueError("roi_truncation must be 'none' or 'visibility'")
    if truncation != "none" and (policy != "partial" or mask is None):
        raise ValueError("roi_truncation requires roi_policy='partial' and a mask")
    if mask is None:
        return None
    if motion_model not in ("translation", "affine"):
        raise ValueError("ROI supports only translation and affine motion models")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("roi_mask must be a two-dimensional boolean NumPy array")
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"roi_mask must have reference shape {shape}")
    # FullPyramid validates first, then passes the owned mask to its levels.
    # Reuse that immutable snapshot rather than copying it a second time.
    if not mask.flags.writeable and mask.flags.c_contiguous:
        owner = mask
        while isinstance(owner, np.ndarray):
            owner = owner.base
        if isinstance(owner, bytes) and np.all(mask.view(np.uint8) <= 1):
            return mask
    # Canonical True bytes make the hash independent of array storage history.
    return immutable(np.logical_and(mask, True))


def integral(mask: np.ndarray) -> np.ndarray:
    """Count forbidden pixels exactly with signed 64-bit prefix sums."""
    return np.pad((~mask).cumsum(0, dtype=np.int64).cumsum(1), ((1, 0), (1, 0)))


def boxes_clear(
    prefix: np.ndarray, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray
) -> np.ndarray:
    return prefix[y1, x1] - prefix[y0, x1] - prefix[y1, x0] + prefix[y0, x0] == 0


def select_windows(mask: np.ndarray, size: int, stride: int) -> np.ndarray:
    height, width = mask.shape
    gx, gy = subset_grid_shape(width, height, subset_size=size, subset_stride=stride)
    y, x = np.mgrid[:gy, :gx] * stride
    return boxes_clear(integral(mask), x, y, x + size, y + size)


def reduce_mask(mask: np.ndarray) -> np.ndarray:
    """All 25 reflected source pixels at phase (2*x, 2*y) must be allowed."""
    h, w = mask.shape
    padded = np.pad(mask, 2, mode="reflect")
    reduced = np.ones((h // 2, w // 2), dtype=bool)
    for dy in range(5):
        for dx in range(5):
            reduced &= padded[dy : dy + 2 * (h // 2) : 2, dx : dx + 2 * (w // 2) : 2]
    return immutable(reduced)


def _links_clear(
    prefix: np.ndarray, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray
) -> np.ndarray:
    h, w = np.array(prefix.shape) - 1
    x0 = np.maximum(0, np.ceil(np.minimum(ax, bx) - 0.5).astype(int))
    y0 = np.maximum(0, np.ceil(np.minimum(ay, by) - 0.5).astype(int))
    x1 = np.minimum(w, np.floor(np.maximum(ax, bx) + 0.5).astype(int) + 1)
    y1 = np.minimum(h, np.floor(np.maximum(ay, by) + 0.5).astype(int) + 1)
    return boxes_clear(prefix, x0, y0, x1, y1)


@dataclass(frozen=True, slots=True)
class RoiPlan:
    """Immutable arrays shared by references and all their stream results."""

    mask: np.ndarray
    selected: np.ndarray
    neighbour_bits: np.ndarray
    mask_hash: str
    effective: bool
    selected_count: int
    policy: str = "full"
    truncation: str = "none"
    partial_indices: np.ndarray | None = None
    support_words: np.ndarray | None = None
    support_boxes: np.ndarray | None = None
    coverage: np.ndarray | None = None
    support_condition: np.ndarray | None = None
    support_rank: np.ndarray | None = None
    support_contrast: np.ndarray | None = None

    @property
    def empty(self) -> bool:
        return self.selected_count == 0

    def summary(self) -> dict:
        count = self.selected_count
        return {
            "policy": self.policy,
            "truncation": self.truncation,
            "partial": 0 if self.partial_indices is None else int(self.partial_indices.size),
            **(
                {
                    "supportConditionLimit": PARTIAL_MAX_NORMALIZED_CONDITION,
                    "supportContrastRmsMinimum": PARTIAL_MIN_WEIGHTED_CONTRAST,
                    "supportConnectivity": 4,
                    "evenCenter": "all four central pixels required",
                }
                if self.policy == "partial"
                else {}
            ),
            "maskShape": list(self.mask.shape),
            "maskSha256": self.mask_hash,
            "gridTotal": self.selected.size,
            "selected": count,
            "excluded": self.selected.size - count,
            "hashFormat": "live-dic-roi-v1\\0 + uint64le(height,width) + C-order uint8 bool",
        }

    def archive_fields(self) -> dict[str, np.ndarray]:
        fields = {"roi_mask": self.mask, "roi_selected": self.selected}
        if self.partial_indices is not None:
            fields.update(
                roi_partial_indices=self.partial_indices,
                roi_support_words=self.support_words,
                roi_support_boxes=self.support_boxes,
                roi_coverage=self.coverage,
            )
            if self.support_condition is not None:
                fields.update(
                    roi_support_condition=self.support_condition,
                    roi_support_rank=self.support_rank,
                    roi_support_contrast=self.support_contrast,
                )
        return fields


def prepare_roi(
    mask: np.ndarray | None,
    size: int,
    stride: int,
    *,
    policy: str = "full",
    truncation: str = "none",
) -> RoiPlan | None:
    if mask is None:
        return None
    selected = select_windows(mask, size, stride)
    partial_indices = support_words = support_boxes = coverage = None
    if policy == "partial":
        selected, partial_indices, support_words, support_boxes, coverage = _partial_windows(
            mask, selected, size, stride, truncation=truncation
        )
    elif policy != "full" or truncation != "none":
        raise ValueError("Unknown ROI policy/truncation")
    bits = np.zeros(selected.shape, dtype=np.uint32)
    y, x = np.indices(selected.shape)
    prefix = integral(mask)
    ax = x * stride + (size - 1) / 2
    ay = y * stride + (size - 1) / 2
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = x + dx, y + dy
            inside = (nx >= 0) & (ny >= 0) & (nx < selected.shape[1]) & (ny < selected.shape[0])
            ny = ny.clip(0, selected.shape[0] - 1)
            nx = nx.clip(0, selected.shape[1] - 1)
            allowed = inside & selected & selected[ny, nx]
            allowed &= _links_clear(prefix, ax, ay, ax + dx * stride, ay + dy * stride)
            bits |= allowed.astype(np.uint32) << ((dy + 1) * 3 + dx + 1)
    digest = hashlib.sha256(
        b"live-dic-roi-v1\0" + struct.pack("<QQ", *mask.shape) + mask.tobytes(order="C")
    ).hexdigest()
    return RoiPlan(
        mask,
        immutable(selected),
        immutable(bits),
        digest,
        not bool(mask.all()),
        int(selected.sum()),
        policy,
        truncation,
        partial_indices,
        support_words,
        support_boxes,
        coverage,
    )


def pack_support(support: np.ndarray) -> np.ndarray:
    """Little-endian uint32 words, row-major pixels, bit 0 first."""
    packed = np.packbits(support.ravel(), bitorder="little")
    packed = np.pad(packed, (0, (-packed.size) % 4))
    return packed.view("<u4").astype(np.uint32, copy=False)


def unpack_support(words: np.ndarray, size: int) -> np.ndarray:
    packed = np.asarray(words, dtype="<u4").view(np.uint8)
    return np.unpackbits(packed, bitorder="little")[: size * size].reshape(size, size).astype(bool)


def _central_component(support: np.ndarray) -> np.ndarray:
    """4-connected component anchored on the central pixel(s), with no tie-break."""
    size = support.shape[0]
    low, high = (size - 1) // 2, size // 2
    result = np.zeros_like(support)
    # An even square has a geometric centre between four pixels. All four
    # must be allowed; this rejects a centre straddling a boundary/crack.
    if not support[low : high + 1, low : high + 1].all():
        return result
    pending = [(low, low)]
    result[low, low] = True
    while pending:
        y, x = pending.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < size and 0 <= nx < size and support[ny, nx] and not result[ny, nx]:
                result[ny, nx] = True
                pending.append((ny, nx))
    return result


@lru_cache(maxsize=7)
def _visibility_paths(size: int) -> np.ndarray:
    """Closed pixel cells touched by each centre-to-pixel segment, packed.

    Independent slab intersections include exact corner/boundary touches.
    This visibility heuristic is not Ncorr's contour/tangent truncation.
    """
    y, x = np.indices((size, size)).reshape(2, -1).astype(np.float64)
    center = (size - 1) / 2
    lower = np.zeros((size * size, size * size), np.float64)
    upper = np.ones_like(lower)
    for coordinates in (x, y):
        delta = coordinates - center
        moving = delta != 0
        a = np.full_like(lower, -np.inf)
        b = np.full_like(upper, np.inf)
        a[moving] = (coordinates[None, :] - 0.5 - center) / delta[moving, None]
        b[moving] = (coordinates[None, :] + 0.5 - center) / delta[moving, None]
        lower = np.maximum(lower, np.minimum(a, b))
        upper = np.minimum(upper, np.maximum(a, b))
        outside = np.abs(coordinates - center) > 0.5
        upper[np.ix_(~moving, outside)] = -1
    paths = lower <= upper
    return immutable(np.stack([pack_support(row) for row in paths]))


def _partial_windows(
    mask: np.ndarray, complete: np.ndarray, size: int, stride: int, *, truncation: str
) -> tuple:
    if truncation not in ("none", "visibility"):
        raise ValueError("roi_truncation must be 'none' or 'visibility'")
    selected = complete.copy()
    indices, words, boxes = [], [], []
    coverage = complete.astype(np.float32)
    for y, x in np.argwhere(~complete):
        local = mask[y * stride : y * stride + size, x * stride : x * stride + size]
        support = _central_component(local)
        if not support.any():
            continue
        if truncation == "visibility":
            visible = ~np.any(_visibility_paths(size) & ~pack_support(support), axis=1)
            support = _central_component(support & visible.reshape(size, size))
        if not support.any():
            continue
        selected[y, x] = True
        indices.append(y * complete.shape[1] + x)
        words.append(pack_support(support))
        sy, sx = np.nonzero(support)
        boxes.append((sx.min(), sy.min(), sx.max(), sy.max()))
        coverage[y, x] = support.sum() / (size * size)
    word_count = (size * size + 31) // 32
    return (
        selected,
        immutable(np.asarray(indices, np.uint32)),
        immutable(np.asarray(words, np.uint32).reshape(-1, word_count)),
        immutable(np.asarray(boxes, np.float32).reshape(-1, 4)),
        immutable(coverage),
    )


def seed_permissions(
    source: RoiPlan,
    destination: RoiPlan,
    *,
    source_scale: int,
    destination_scale: int,
    size: int,
    stride: int,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Four coarse-source link bits and destination selection bit 4."""
    y, x = np.indices(destination.selected.shape)
    center = size / 2
    # The historical seed layout uses edge coordinates and a half-pixel offset.
    sx = (
        destination_scale * (center + x * stride) + (source_scale - destination_scale) / 2
    ) / source_scale
    sy = (
        destination_scale * (center + y * stride) + (source_scale - destination_scale) / 2
    ) / source_scale
    lower_x = np.clip(
        np.floor((sx - center) / stride), 0, max(0, source.selected.shape[1] - 2)
    ).astype(int)
    lower_y = np.clip(
        np.floor((sy - center) / stride), 0, max(0, source.selected.shape[0] - 2)
    ).astype(int)
    bits = destination.selected.astype(np.uint32) << 4
    prefix = integral(reference_mask)
    for corner in range(4):
        nx = lower_x + (corner & 1)
        ny = lower_y + (corner >> 1)
        inside = (nx < source.selected.shape[1]) & (ny < source.selected.shape[0])
        nx = nx.clip(0, source.selected.shape[1] - 1)
        ny = ny.clip(0, source.selected.shape[0] - 1)
        allowed = inside & source.selected[ny, nx] & destination.selected
        allowed &= _links_clear(
            prefix,
            destination_scale * ((size - 1) / 2 + x * stride),
            destination_scale * ((size - 1) / 2 + y * stride),
            source_scale * ((size - 1) / 2 + nx * stride),
            source_scale * ((size - 1) / 2 + ny * stride),
        )
        bits |= allowed.astype(np.uint32) << corner
    return immutable(bits)
