"""2.5D slice selection and normalization for MRI volumes."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def percentile_normalize(x: np.ndarray, p1: float = 1, p99: float = 99) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, p1), np.percentile(x, p99)
    if hi <= lo:
        mn, mx = float(x.min()), float(x.max())
        return ((x - mn) / (mx - mn + 1e-8)).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def resize_2d(img: np.ndarray, out_size: int = 224) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for MRI preprocessing.")
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA).astype(np.float32)


def get_candidate_slice_indices(
    vol: np.ndarray,
    start_frac: float = 0.15,
    end_frac: float = 0.85,
    min_fg_ratio: float = 0.01,
) -> list[int]:
    if vol.ndim != 3 or vol.shape[-1] < 3:
        return []
    depth = vol.shape[-1]
    z0 = max(1, int(depth * start_frac))
    z1 = min(depth - 2, int(depth * end_frac))
    idxs = []
    for z in range(z0, z1 + 1):
        sl = vol[..., z]
        if float(np.mean(sl > sl.mean())) >= min_fg_ratio:
            idxs.append(z)
    if idxs:
        return idxs
    return list(range(max(1, depth // 4), min(depth - 2, 3 * depth // 4)))


def sample_slice_indices(idxs: list[int], max_samples: int) -> list[int]:
    if len(idxs) <= max_samples:
        return list(idxs)
    pick = np.linspace(0, len(idxs) - 1, max_samples).astype(int)
    return [idxs[i] for i in pick]


def build_25d_tensor(
    vol: np.ndarray,
    z: int,
    img_size: int,
    process_slice_fn,
) -> np.ndarray:
    """Stack adjacent slices (z-1, z, z+1) along last axis into (3, H, W)."""
    depth = vol.shape[-1]
    channels = []
    for zz in (max(0, z - 1), z, min(depth - 1, z + 1)):
        sl = vol[..., zz].astype(np.float32)
        processed = process_slice_fn(sl)
        channels.append(resize_2d(processed, img_size))
    return np.stack(channels, axis=0).astype(np.float32)


def compute_artifact_map(raw_slice: np.ndarray, cleaned_slice: np.ndarray) -> np.ndarray:
    """|raw − cleaned| map (notebook sequential visualization)."""
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for MRI preprocessing.")
    raw_n = percentile_normalize(raw_slice)
    h, w = raw_n.shape
    clean = cleaned_slice.astype(np.float32)
    if clean.shape != (h, w):
        clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA)
    clean = np.clip(clean, 0.0, 1.0)
    return np.abs(raw_n - clean).astype(np.float32)
