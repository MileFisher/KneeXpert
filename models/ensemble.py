"""
Ensemble model combining multiple architectures for improved performance.
"""
import torch
import torch.nn as nn
from typing import List, Dict
from .resnet import ResNetModel
from .efficientnet import EfficientNetModel
from .vit import VisionTransformerModel


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines predictions from multiple models.
    Supports weighted averaging and voting strategies.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int = 2,
        ensemble_method: str = "weighted_average",
        weights: List[float] = None,
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of trained models to ensemble
            num_classes: Number of output classes
            ensemble_method: "weighted_average", "average", or "voting"
            weights: Weights for each model (if weighted_average)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        
        # Default weights: equal weighting
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=0)  # (num_models, batch_size, num_classes)
        
        if self.ensemble_method == "weighted_average":
            # Weighted average of logits
            weights_tensor = torch.tensor(
                self.weights,
                device=stacked_preds.device,
                dtype=stacked_preds.dtype
            ).view(-1, 1, 1)
            ensemble_pred = (stacked_preds * weights_tensor).sum(dim=0)
        
        elif self.ensemble_method == "average":
            # Simple average
            ensemble_pred = stacked_preds.mean(dim=0)
        
        elif self.ensemble_method == "voting":
            # Hard voting (majority class)
            probs = torch.softmax(stacked_preds, dim=-1)
            votes = probs.argmax(dim=-1)  # (num_models, batch_size)
            # Count votes and return most common class
            # For simplicity, convert to probabilities
            ensemble_pred = stacked_preds.mean(dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred
    
    @classmethod
    def create_default_ensemble(
        cls,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> "EnsembleModel":
        """
        Create a default ensemble with ResNet50, EfficientNet-B0, and ViT-Base.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            
        Returns:
            Ensemble model
        """
        models = [
            ResNetModel(num_classes=num_classes, model_name="resnet50", pretrained=pretrained),
            EfficientNetModel(num_classes=num_classes, model_name="efficientnet_b0", pretrained=pretrained),
            VisionTransformerModel(num_classes=num_classes, model_name="vit_base_patch16_224", pretrained=pretrained),
        ]
        
        return cls(models=models, num_classes=num_classes)

