"""Map SKM-TEA multi-label probabilities to KL grade + clinical findings."""

from __future__ import annotations

import numpy as np

CARTILAGE_TO_KL: dict[str, int] = {
    "Cartilage Lesion (1)": 1,
    "Cartilage Lesion (2A)": 2,
    "Cartilage Lesion (2B)": 3,
    "Cartilage Lesion (3)": 4,
}

STRUCTURAL_LABELS = (
    "Meniscal Tear",
    "Ligament Tear",
    "Effusion",
)


def aggregate_slice_probs(all_probs: list[list[float]], method: str = "max") -> list[float]:
    if not all_probs:
        return []
    arr = np.array(all_probs, dtype=np.float32)
    if method == "mean":
        return arr.mean(axis=0).tolist()
    return arr.max(axis=0).tolist()


def multilabel_to_kl(
    category_names: list[str],
    study_probs: list[float],
    threshold: float = 0.5,
) -> tuple[int, float, list[str], list[dict]]:
    """
    Returns (kl_grade, confidence_percent, findings_strings, label_details).
    """
    pairs = list(zip(category_names, study_probs))
    positives = [(n, p) for n, p in pairs if p >= threshold]
    positives.sort(key=lambda x: -x[1])

    label_details = [
        {
            "name": name,
            "probability": round(float(prob) * 100, 2),
            "predicted": bool(prob >= threshold),
        }
        for name, prob in pairs
    ]

    kl_grade = 0
    for name, prob in positives:
        if name in CARTILAGE_TO_KL:
            kl_grade = max(kl_grade, CARTILAGE_TO_KL[name])

    if kl_grade == 0 and positives:
        has_structural = any(any(tag in name for tag in STRUCTURAL_LABELS) for name, _ in positives)
        kl_grade = 2 if has_structural else 1

    if positives:
        confidence = max(p * 100 for _, p in positives)
        findings = [f"{name} ({p * 100:.1f}%)" for name, p in positives]
    else:
        confidence = max(0.0, (1.0 - max(study_probs)) * 100) if study_probs else 0.0
        findings = ["No significant pathology above detection threshold."]

    return kl_grade, round(confidence, 2), findings, label_details
