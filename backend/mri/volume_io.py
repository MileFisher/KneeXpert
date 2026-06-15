"""NIfTI volume loading with orientation / slice-axis metadata."""

from __future__ import annotations

import io
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_EXTENSIONS = {
    ".nii", ".nii.gz", ".gz", ".pkl", ".pck", ".pickle",
}

# Notebook convention: vol[..., z] — slice index runs along the last axis.
SLICE_AXIS = 2


def _nifti_metadata(img) -> dict[str, Any]:
    import nibabel as nib

    axcodes = nib.aff2axcodes(img.affine)
    shape = tuple(int(s) for s in img.shape[:3]) if len(img.shape) >= 3 else (*img.shape, 1)
    while len(shape) < 3:
        shape = (*shape, 1)
    in_plane = [shape[i] for i in range(3) if i != SLICE_AXIS]
    return {
        "shape": list(shape),
        "slice_axis": SLICE_AXIS,
        "slice_axis_label": "z",
        "num_slices": int(shape[SLICE_AXIS]),
        "in_plane_shape": in_plane,
        "orientation_axcodes": list(axcodes[:3]) if axcodes else [],
        "plane_description": f"2.5D axial stacks along axis {SLICE_AXIS} (depth={shape[SLICE_AXIS]})",
    }


def _load_nifti_path(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import nibabel as nib

    img = nib.load(str(path))
    try:
        vol = img.get_fdata(dtype=np.float32, caching="unchanged")
    except TypeError:
        vol = np.asarray(img.dataobj, dtype=np.float32)
    vol = np.asarray(vol, dtype=np.float32)
    meta = _nifti_metadata(img)
    return vol, meta


def _load_nifti_bytes(data: bytes, suffix: str = ".nii.gz") -> tuple[np.ndarray, dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return _load_nifti_path(Path(tmp.name))


def _load_pickle_bytes(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    obj = pickle.loads(data)
    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, dict):
        for key in ("volume", "vol", "data", "image", "array"):
            if key in obj and isinstance(obj[key], np.ndarray):
                arr = obj[key]
                break
        else:
            raise ValueError("Pickle dict does not contain a numpy volume array.")
    else:
        raise ValueError(f"Unsupported pickle content type: {type(obj)}")
    arr = np.asarray(arr, dtype=np.float32)
    meta = {
        "shape": list(arr.shape if arr.ndim == 3 else (*arr.shape, 1)),
        "slice_axis": SLICE_AXIS,
        "slice_axis_label": "z",
        "num_slices": int(arr.shape[SLICE_AXIS]) if arr.ndim >= 3 else 1,
        "in_plane_shape": list(arr.shape[:2]) if arr.ndim >= 2 else [1, 1],
        "orientation_axcodes": [],
        "plane_description": f"2.5D stacks along axis {SLICE_AXIS}",
    }
    return arr, meta


def _ensure_3d(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 2:
        return vol[..., np.newaxis]
    if vol.ndim == 3:
        return vol
    if vol.ndim == 4 and vol.shape[-1] in (1, 3):
        return vol[..., 0] if vol.shape[-1] == 1 else np.mean(vol, axis=-1)
    raise ValueError(f"Expected 2D/3D volume, got shape {vol.shape}")


def load_volume_from_bytes(data: bytes, filename: str = "scan.nii.gz") -> np.ndarray:
    vol, _ = load_volume_with_meta_from_bytes(data, filename=filename)
    return vol


def load_volume_with_meta_from_bytes(
    data: bytes,
    filename: str = "scan.nii.gz",
) -> tuple[np.ndarray, dict[str, Any]]:
    name = filename.lower()

    if name.endswith((".pkl", ".pck", ".pickle")):
        vol, meta = _load_pickle_bytes(data)
        return _ensure_3d(vol), meta

    if name.endswith((".nii", ".nii.gz")) or name.endswith(".gz"):
        suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
        vol, meta = _load_nifti_bytes(data, suffix=suffix)
        return _ensure_3d(vol), meta

    try:
        vol, meta = _load_nifti_bytes(data, suffix=".nii.gz")
        return _ensure_3d(vol), meta
    except Exception:
        pass
    try:
        vol, meta = _load_pickle_bytes(data)
        return _ensure_3d(vol), meta
    except Exception as exc:
        raise ValueError(f"Could not parse '{filename}' as NIfTI or pickle volume.") from exc


def load_volume_from_path(path: str | Path) -> np.ndarray:
    vol, _ = load_volume_with_meta_from_path(path)
    return vol


def load_volume_with_meta_from_path(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    name = path.name.lower()
    if name.endswith((".pkl", ".pck", ".pickle")):
        vol, meta = _load_pickle_bytes(path.read_bytes())
        return _ensure_3d(vol), meta
    vol, meta = _load_nifti_path(path)
    return _ensure_3d(vol), meta
