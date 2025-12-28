"""
Vision Transformer (ViT) model implementation for knee disease classification.
"""
import torch
import torch.nn as nn
import timm
from typing import Optional


class VisionTransformerModel(nn.Module):
    """
    Vision Transformer model for knee image classification.
    Uses timm library for pre-trained ViT models.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Initialize Vision Transformer model.
        
        Args:
            num_classes: Number of output classes
            model_name: ViT variant (vit_base_patch16_224, vit_large_patch16_224, etc.)
            pretrained: Whether to use ImageNet pre-trained weights
            dropout: Dropout rate for classifier head
            freeze_backbone: If True, freeze backbone layers for transfer learning
        """
        super(VisionTransformerModel, self).__init__()
        
        # Load pre-trained ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            in_features = features.shape[-1]
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
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
        Useful for attention visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        return self.backbone(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from the transformer.
        Useful for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention maps
        """
        # This would require accessing internal attention layers
        # Implementation depends on timm's ViT structure
        # Placeholder for now - can be extended based on specific ViT architecture
        return self.backbone.get_intermediate_layers(x, n=1)[0]

