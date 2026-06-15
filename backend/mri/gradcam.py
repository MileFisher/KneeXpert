"""Grad-CAM (DeiT-S) and artifact-map encoding for MRI pipeline output."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from mri.preprocess import compute_artifact_map, percentile_normalize


def _encode_gray_png(arr: np.ndarray) -> str | None:
    try:
        norm = percentile_normalize(arr)
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _encode_heatmap_png(diff: np.ndarray) -> str | None:
    try:
        d = diff.astype(np.float32)
        if d.max() > d.min():
            d = (d - d.min()) / (d.max() - d.min())
        else:
            d = np.zeros_like(d)
        rgb = np.stack([d, d ** 2, d ** 4], axis=-1)
        img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def encode_artifact_map_png(raw_slice: np.ndarray, cleaned_slice: np.ndarray) -> str | None:
    diff = compute_artifact_map(raw_slice, cleaned_slice)
    return _encode_heatmap_png(diff)


def _deit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Drop CLS token and reshape ViT tokens to spatial feature map."""
    if tensor.ndim == 3 and tensor.shape[1] > 1:
        tokens = tensor[:, 1:, :]
        return tokens.reshape(tensor.shape[0], height, width, tensor.shape[2]).permute(0, 3, 1, 2)
    return tensor


def generate_deit_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_category_idx: int,
) -> str | None:
    try:
        target_layers = [model.blocks[-1].norm1]
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=_deit_reshape_transform,
        )
        targets = [ClassifierOutputTarget(target_category_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        vis = input_tensor.squeeze(0)[1].detach().float().cpu().numpy()
        vis = percentile_normalize(vis)
        vis_rgb = np.stack([vis, vis, vis], axis=-1)
        overlay = show_cam_on_image(vis_rgb, grayscale_cam, use_rgb=True)
        img = Image.fromarray(overlay)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[mri] DeiT Grad-CAM failed: {exc}")
        return None
