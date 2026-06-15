"""
MRI inference pipeline: raw 3D volume → MACS-Net artifact removal → 2.5D DeiT-S classification.

Matches notebooks/mri_pipeline.ipynb:
- Slice along last axis vol[..., z] (512×512 in-plane, depth on axis 2)
- 2.5D stack (z-1, z, z+1) → (3, 224, 224)
- Per-slice: raw | MACS cleaned | artifact map | DeiT Grad-CAM
- Study-level max aggregation over sampled slices
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import torch

from shared.clinical_feedback import findings_for_grade
from mri.config import (
    AGG_METHOD,
    DEIT_IMG_SIZE,
    DEIT_CHECKPOINT,
    MACS_CHECKPOINT,
    MAX_GALLERY_SLICES,
    MAX_SAMPLES_PER_STUDY,
    MIN_FOREGROUND_RATIO,
    PIPELINE_MODE,
    SLICE_END_FRAC,
    SLICE_START_FRAC,
)
from mri.deit_classifier import get_deit
from mri.gradcam import encode_artifact_map_png, generate_deit_gradcam, _encode_gray_png
from mri.grade_mapping import aggregate_slice_probs, multilabel_to_kl
from mri.label_summary import feedback_for_labels, lookup_ground_truth, parse_label_summary
from mri.macs_net import get_macs
from mri.preprocess import (
    build_25d_tensor,
    get_candidate_slice_indices,
    percentile_normalize,
    resize_2d,
    sample_slice_indices,
)
from mri.volume_io import (
    load_volume_from_bytes,
    load_volume_from_path,
    load_volume_with_meta_from_bytes,
    load_volume_with_meta_from_path,
)


def mri_models_available() -> dict[str, bool]:
    return {
        "macs_net": MACS_CHECKPOINT.exists(),
        "deit_small": DEIT_CHECKPOINT.exists(),
    }


def _encode_slice_png(slice_2d: np.ndarray) -> str | None:
    return _encode_gray_png(slice_2d)


def _labels_from_probs(category_names: list[str], probs: list[float], threshold: float) -> list[str]:
    return [name for name, p in zip(category_names, probs) if p >= threshold]


def _pick_primary_slice(idxs: list[int], all_probs: list[list[float]]) -> int:
    if not idxs or not all_probs:
        return 0
    scores = [max(p) if p else 0.0 for p in all_probs]
    return int(idxs[int(np.argmax(scores))])


def _build_slice_entry(
    vol: np.ndarray,
    z: int,
    probs: list[float],
    category_names: list[str],
    threshold: float,
    macs,
    deit,
) -> dict[str, Any]:
    raw = vol[..., z].astype(np.float32)
    cleaned = macs.clean_slice(raw)
    artifact_b64 = encode_artifact_map_png(raw, cleaned)

    tensor = build_25d_tensor(vol, z, DEIT_IMG_SIZE, process_slice_fn=macs.clean_slice)
    x = torch.from_numpy(tensor).unsqueeze(0).float()
    if torch.cuda.is_available():
        x = x.to(deit.device)
        if x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)

    gradcam_b64 = None
    if probs:
        target_idx = int(np.argmax(probs))
        gradcam_b64 = generate_deit_gradcam(deit.model, x, target_idx)

    return {
        "slice_idx": int(z),
        "slice_axis": 2,
        "raw_base64": _encode_slice_png(raw),
        "cleaned_base64": _encode_slice_png(cleaned),
        "artifact_map_base64": artifact_b64,
        "gradcam_base64": gradcam_b64,
        "predicted_labels": _labels_from_probs(category_names, probs, threshold),
        "probabilities": probs,
    }


def predict_mri(
    file_bytes: bytes | None = None,
    file_path: str | None = None,
    filename: str = "scan.nii.gz",
    threshold: float | None = None,
    max_slices: int | None = None,
) -> dict[str, Any]:
    if file_bytes is None and file_path is None:
        raise ValueError("Either file_bytes or file_path is required.")

    if file_bytes is not None:
        vol, volume_meta = load_volume_with_meta_from_bytes(file_bytes, filename=filename)
    else:
        vol, volume_meta = load_volume_with_meta_from_path(file_path)  # type: ignore[arg-type]
        filename = filename or str(file_path)

    macs = get_macs()
    deit = get_deit()
    macs.load()
    deit.load()

    thr = threshold if threshold is not None else deit.default_threshold
    max_n = max_slices or MAX_SAMPLES_PER_STUDY

    idxs = sample_slice_indices(
        get_candidate_slice_indices(
            vol,
            start_frac=SLICE_START_FRAC,
            end_frac=SLICE_END_FRAC,
            min_fg_ratio=MIN_FOREGROUND_RATIO,
        ),
        max_n,
    )
    if not idxs:
        raise ValueError(
            f"No valid slices along axis {volume_meta.get('slice_axis', 2)} "
            f"(volume shape {volume_meta.get('shape', list(vol.shape))})."
        )

    all_probs: list[list[float]] = []
    slice_meta: list[dict[str, Any]] = []

    for z in idxs:
        tensor = build_25d_tensor(vol, z, DEIT_IMG_SIZE, process_slice_fn=macs.clean_slice)
        x = torch.from_numpy(tensor).unsqueeze(0).float()
        probs = deit.predict_batch(x).squeeze(0).float().cpu().numpy().tolist()
        all_probs.append(probs)
        slice_meta.append({"slice_idx": int(z), "probabilities": probs})

    study_probs = aggregate_slice_probs(all_probs, method=AGG_METHOD)
    kl_grade, confidence, _raw_findings, label_details = multilabel_to_kl(
        deit.category_names,
        study_probs,
        threshold=thr,
    )

    summary_categories, scan_labels = parse_label_summary()
    positive_names = [d["name"] for d in label_details if d["predicted"]]
    pathology_findings = feedback_for_labels(positive_names)
    ground_truth_labels = lookup_ground_truth(filename, summary_categories, scan_labels)

    category_feedback = list(pathology_findings)
    if ground_truth_labels:
        category_feedback.append(f"SKM-TEA reference: {', '.join(ground_truth_labels)}.")

    kl_findings = findings_for_grade(kl_grade)
    findings = category_feedback if category_feedback else list(kl_findings[:2])

    primary_z = _pick_primary_slice(idxs, all_probs)
    gallery_idxs = sample_slice_indices(idxs, min(MAX_GALLERY_SLICES, len(idxs)))
    slice_gallery: list[dict[str, Any]] = []
    prob_by_z = {int(s["slice_idx"]): s["probabilities"] for s in slice_meta}

    for z in gallery_idxs:
        slice_gallery.append(
            _build_slice_entry(vol, z, prob_by_z[z], deit.category_names, thr, macs, deit)
        )

    primary_probs = prob_by_z.get(primary_z, study_probs)
    primary_entry = _build_slice_entry(
        vol, primary_z, primary_probs, deit.category_names, thr, macs, deit
    )

    return {
        "filename": filename,
        "pipeline_mode": PIPELINE_MODE,
        "artifact_removal": "MACS-Net (Swin-UNETR, frozen)",
        "classifier": "DeiT-Small (2.5D multi-label)",
        "volume_shape": list(vol.shape),
        "volume_meta": volume_meta,
        "slices_processed": len(idxs),
        "slice_indices": idxs,
        "gallery_slice_indices": gallery_idxs,
        "primary_slice_idx": int(primary_z),
        "grade": kl_grade,
        "confidence": confidence,
        "is_reliable": confidence >= 70.0,
        "threshold": thr,
        "findings": findings,
        "pathology_findings": pathology_findings,
        "category_feedback": category_feedback,
        "ground_truth_labels": ground_truth_labels,
        "ground_truth_feedback": feedback_for_labels(ground_truth_labels) if ground_truth_labels else [],
        "kl_findings": kl_findings,
        "skm_tea_categories": summary_categories,
        "multilabel_predictions": label_details,
        "slice_predictions": slice_meta,
        "slice_gallery": slice_gallery,
        "preview": {
            "center_slice_idx": int(primary_z),
            "slice_axis": volume_meta.get("slice_axis", 2),
            "raw_base64": primary_entry.get("raw_base64"),
            "cleaned_base64": primary_entry.get("cleaned_base64"),
            "artifact_map_base64": primary_entry.get("artifact_map_base64"),
            "gradcam_base64": primary_entry.get("gradcam_base64"),
        },
        "gradcam_base64": primary_entry.get("gradcam_base64"),
        "artifact_map_base64": primary_entry.get("artifact_map_base64"),
        "models_used": ["macs_net", "deit_small"],
    }


def predict_mri_file(path: str, **kwargs) -> dict[str, Any]:
    return predict_mri(file_path=path, filename=path.split("/")[-1], **kwargs)
