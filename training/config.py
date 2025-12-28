"""
Configuration file for training parameters and model settings.
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_dir: str = "data/processed"
    raw_data_dir: str = "data/raw"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_classes: int = 2  # Normal vs Pathological (can be extended)
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["Normal", "Pathological"]


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    model_name: str = "resnet50"  # Options: resnet50, efficientnet_b0, vit_base
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    early_stopping_patience: int = 10
    save_best_only: bool = True
    checkpoint_dir: str = "models/pretrained"
    
    # Transfer learning strategy
    freeze_backbone: bool = False  # If True, only train classifier head
    fine_tune_layers: Optional[int] = None  # Number of layers to fine-tune


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enable_augmentation: bool = True
    rotation_range: int = 15
    horizontal_flip: bool = True
    vertical_flip: bool = False
    brightness_range: tuple = (0.8, 1.2)
    contrast_range: tuple = (0.8, 1.2)
    elastic_transform: bool = True
    gaussian_noise: bool = True


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42
    log_dir: str = "logs"
    use_mixed_precision: bool = True
    gradient_clip: Optional[float] = 1.0
    
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]


# Global configuration instances
data_config = DataConfig()
model_config = ModelConfig()
augmentation_config = AugmentationConfig()
training_config = TrainingConfig()

