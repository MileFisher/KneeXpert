"""Parse SKM-TEA categories and per-scan ground truth from scan_label_summary.txt."""

from __future__ import annotations

import re
from pathlib import Path

from mri.config import LABEL_SUMMARY_PATH

_CATEGORY_RE = re.compile(r"^\s*\[\s*(\d+)\]\s*(.+?)\s*$")
_SCAN_RE = re.compile(r"^(MTR_\d+)\s+\(source:")


def parse_label_summary(path: Path | None = None) -> tuple[list[dict], dict[str, list[int]]]:
    """
    Returns (categories, scan_labels).
    categories: [{id, name}, ...]
    scan_labels: scan_id -> [category_id, ...]
    """
    path = path or LABEL_SUMMARY_PATH
    if not path.is_file():
        return [], {}

    categories: list[dict] = []
    scan_labels: dict[str, list[int]] = {}
    current_scan: str | None = None
    in_categories = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("Categories:"):
            in_categories = True
            continue
        if in_categories and line.startswith("===="):
            in_categories = False
            continue

        cat_match = _CATEGORY_RE.match(line)
        if in_categories and cat_match:
            categories.append({"id": int(cat_match.group(1)), "name": cat_match.group(2).strip()})
            continue

        scan_match = _SCAN_RE.match(line)
        if scan_match:
            current_scan = scan_match.group(1)
            scan_labels[current_scan] = []
            continue

        if current_scan and cat_match:
            scan_labels[current_scan].append(int(cat_match.group(1)))
            continue

        if current_scan and "(no labels)" in line:
            scan_labels[current_scan] = []

    return categories, scan_labels


def category_names_by_id(categories: list[dict]) -> dict[int, str]:
    return {int(c["id"]): str(c["name"]) for c in categories}


def lookup_ground_truth(filename: str, categories: list[dict], scan_labels: dict[str, list[int]]) -> list[str]:
    """Resolve ground-truth label names for a volume filename."""
    id_to_name = category_names_by_id(categories)
    stem = Path(filename).name.replace(".nii.gz", "").replace(".nii", "")

    if stem in scan_labels:
        return [id_to_name[i] for i in scan_labels[stem] if i in id_to_name]

    if stem.upper().startswith("MTR_"):
        return [id_to_name[i] for i in scan_labels.get(stem.upper(), []) if i in id_to_name]

    # Dev sample: Effusion.nii.gz
    if "effusion" in stem.lower():
        return [id_to_name[16]] if 16 in id_to_name else ["Effusion"]

    if "cartilage" in stem.lower():
        hits = [id_to_name[i] for i in (12, 13, 14, 15) if i in id_to_name]
        return hits or ["Cartilage Lesion"]

    return []


def feedback_for_labels(label_names: list[str]) -> list[str]:
    """Turn category names into clinician-facing finding strings."""
    if not label_names:
        return ["No significant pathology detected above model threshold."]
    return [f"{name} identified on MRI." for name in label_names]
