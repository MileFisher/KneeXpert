"""MACS-Net (Swin-UNETR) artifact removal — frozen at inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

from mri.config import MACS_CHECKPOINT, MACS_IMG_SIZE


class MACSNet:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = MACS_IMG_SIZE
        self.model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=24,
            spatial_dims=2,
        ).to(self.device)
        self._loaded = False
        self.use_amp = torch.cuda.is_available()

    def load(self) -> None:
        if self._loaded:
            return
        if not MACS_CHECKPOINT.exists():
            raise FileNotFoundError(f"MACS checkpoint not found: {MACS_CHECKPOINT}")
        ckpt = torch.load(MACS_CHECKPOINT, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._loaded = True
        print(f"[mri] MACS-Net loaded from {MACS_CHECKPOINT.name}")

    @torch.no_grad()
    def clean_slice(self, slice_np: np.ndarray) -> np.ndarray:
        self.load()
        h_orig, w_orig = slice_np.shape
        arr = slice_np.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-8:
            return np.zeros((h_orig, w_orig), dtype=np.float32)
        arr = (arr - lo) / (hi - lo)

        if arr.shape != (self.img_size, self.img_size):
            t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            arr = t.squeeze().numpy()

        x = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            cleaned = self.model(x)
        out = np.clip(cleaned.squeeze().float().cpu().numpy(), 0.0, 1.0)

        if (h_orig, w_orig) != (self.img_size, self.img_size):
            t = torch.tensor(out).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            out = t.squeeze().numpy()
        return out.astype(np.float32)


_macs: MACSNet | None = None


def get_macs() -> MACSNet:
    global _macs
    if _macs is None:
        _macs = MACSNet()
    return _macs
