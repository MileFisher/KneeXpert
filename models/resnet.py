"""
ResNet model implementation for knee disease classification.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetModel(nn.Module):
    """
    ResNet-based model for knee image classification.
    Supports transfer learning from ImageNet pre-trained weights.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Initialize ResNet model.
        
        Args:
            num_classes: Number of output classes
            model_name: ResNet variant (resnet18, resnet34, resnet50, resnet101, resnet152)
            pretrained: Whether to use ImageNet pre-trained weights
            dropout: Dropout rate for classifier head
            freeze_backbone: If True, freeze backbone layers for transfer learning
        """
        super(ResNetModel, self).__init__()
        
        # Load pre-trained ResNet backbone
        if model_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif model_name == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif model_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        elif model_name == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
            in_features = 2048
        elif model_name == "resnet152":
            backbone = models.resnet152(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Extract features (remove final fully connected layer)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification.
        Useful for Grad-CAM visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        return self.backbone(x)

