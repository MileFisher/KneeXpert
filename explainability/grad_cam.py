"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model
            target_layer: Target layer to generate CAM from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation maps."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def generate_gradcam_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    device: str = "cuda",
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for an input image.
    
    Args:
        model: Trained model
        input_tensor: Input image tensor (1, C, H, W)
        target_layer: Target layer for CAM generation
        target_class: Target class index
        device: Device to run on
        
    Returns:
        Heatmap as numpy array
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate_cam(input_tensor, target_class)
    
    return heatmap


def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, C) in RGB format, range [0, 255] or [0, 1]
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Transparency factor for heatmap overlay
        save_path: Optional path to save visualization
        
    Returns:
        Overlaid image
    """
    # Normalize image to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Resize heatmap to match image
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    # Save or display
    if save_path:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlaid)
        plt.title("Overlay")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Grad-CAM visualization saved to {save_path}")
    
    return overlaid


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM based on model architecture.
    
    Args:
        model: Model instance
        model_name: Name of the model architecture
        
    Returns:
        Target layer module
    """
    if "resnet" in model_name.lower():
        # For ResNet, use the last convolutional layer
        return model.backbone[-1][-1].conv3 if hasattr(model.backbone[-1][-1], "conv3") else model.backbone[-1]
    elif "efficientnet" in model_name.lower():
        # For EfficientNet, use the last block
        return list(model.backbone.blocks.children())[-1]
    elif "vit" in model_name.lower() or "transformer" in model_name.lower():
        # For ViT, use the last attention block
        return model.backbone.blocks[-1]
    else:
        # Default: try to find last conv layer
        for module in reversed(list(model.modules())):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                return module
        raise ValueError(f"Could not find suitable layer for {model_name}")

