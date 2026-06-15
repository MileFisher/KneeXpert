"""Grad-CAM generation for X-ray classifiers (pytorch-grad-cam)."""

import base64
import io

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMGenerator:
    def __init__(self, use_cuda: bool = False):
        self.use_cuda = use_cuda

    def get_target_layer(self, model, family: str):
        name = family.lower()
        if "densenet" in name:
            if hasattr(model, "backbone"):
                return [model.backbone.features[-1]]
            return [model.features[-1]]
        if "resnet" in name:
            if hasattr(model, "backbone"):
                return [model.backbone.layer4[-1]]
            return [model.layer4[-1]]
        if "vgg" in name:
            if hasattr(model, "backbone"):
                return [model.backbone.features[-1]]
            return [model.features[-1]]
        raise ValueError(f"Grad-CAM target layer unknown for family: {family}")

    def generate(self, model, family: str, input_tensor: torch.Tensor, original_image: Image.Image) -> str | None:
        try:
            target_layers = self.get_target_layer(model, family)
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]

            vis_img = original_image.resize((224, 224))
            vis_np = np.array(vis_img, dtype=np.float32) / 255.0
            if len(vis_np.shape) == 2:
                vis_np = np.stack((vis_np,) * 3, axis=-1)
            elif vis_np.shape[2] == 4:
                vis_np = vis_np[:, :, :3]

            visualization = show_cam_on_image(vis_np, grayscale_cam, use_rgb=True)
            pil_vis = Image.fromarray(visualization)
            buffered = io.BytesIO()
            pil_vis.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error generating Grad-CAM for {family}: {e}")
            return None
