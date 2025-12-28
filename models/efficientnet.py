"""
EfficientNet model implementation for knee disease classification.
"""
import torch
import torch.nn as nn
import timm
from typing import Optional


class EfficientNetModel(nn.Module):
    """
    EfficientNet-based model for knee image classification.
    Uses timm library for pre-trained EfficientNet models.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Initialize EfficientNet model.
        
        Args:
            num_classes: Number of output classes
            model_name: EfficientNet variant (efficientnet_b0 to efficientnet_b7)
            pretrained: Whether to use ImageNet pre-trained weights
            dropout: Dropout rate for classifier head
            freeze_backbone: If True, freeze backbone layers for transfer learning
        """
        super(EfficientNetModel, self).__init__()
        
        # Load pre-trained EfficientNet backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool="",
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            in_features = features.shape[1] * features.shape[2] * features.shape[3]
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
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
        features = self.global_pool(features)
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
        features = self.backbone(x)
        return self.global_pool(features)

