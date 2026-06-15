"""DeiT-Small multi-label classifier for cleaned 2.5D MRI input."""

from __future__ import annotations

import json

import timm
import torch

from mri.config import CATEGORIES_JSON, DEIT_CHECKPOINT, DEIT_IMG_SIZE, DROPOUT_RATE, DROP_PATH_RATE
from mri.label_summary import parse_label_summary


def load_categories() -> tuple[list[int], list[str], float]:
    summary_cats, _ = parse_label_summary()
    if summary_cats:
        ordered = sorted(summary_cats, key=lambda c: int(c["id"]))
        return (
            [int(c["id"]) for c in ordered],
            [str(c["name"]) for c in ordered],
            0.5,
        )

    with open(CATEGORIES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    cats = data["categories"]
    ids = [int(c["id"]) for c in cats]
    names = [str(c["name"]) for c in cats]
    threshold = float(data.get("default_threshold", 0.5))
    return ids, names, threshold


class DeiTClassifier:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.category_ids, self.category_names, self.default_threshold = load_categories()
        self.num_classes = len(self.category_ids)
        self.img_size = DEIT_IMG_SIZE
        self.model = timm.create_model(
            "deit_small_patch16_224",
            pretrained=False,
            num_classes=self.num_classes,
            drop_rate=DROPOUT_RATE,
            drop_path_rate=DROP_PATH_RATE,
        ).to(self.device)
        if torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
        self._loaded = False
        self.use_amp = torch.cuda.is_available()
        self.checkpoint_threshold = self.default_threshold

    def load(self) -> None:
        if self._loaded:
            return
        if not DEIT_CHECKPOINT.exists():
            raise FileNotFoundError(f"DeiT checkpoint not found: {DEIT_CHECKPOINT}")
        try:
            ckpt = torch.load(DEIT_CHECKPOINT, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(DEIT_CHECKPOINT, map_location=self.device)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if isinstance(cfg, dict) and "threshold" in cfg:
            self.checkpoint_threshold = float(cfg["threshold"])
        self.model.eval()
        self._loaded = True
        print(f"[mri] DeiT-S loaded ({self.num_classes} labels) from {DEIT_CHECKPOINT.name}")

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        self.load()
        x = batch.to(self.device, non_blocking=True)
        if torch.cuda.is_available() and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            return torch.sigmoid(self.model(x))


_deit: DeiTClassifier | None = None


def get_deit() -> DeiTClassifier:
    global _deit
    if _deit is None:
        _deit = DeiTClassifier()
    return _deit
