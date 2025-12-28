"""
PyTorch Dataset class for knee images.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Callable
from preprocessing.utils import load_image
from preprocessing.normalize import normalize_image
from preprocessing.augment import get_augmentation_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class KneeImageDataset(Dataset):
    """
    Dataset class for knee images with labels.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of integer labels
            class_names: List of class names
            image_size: Target image size (height, width)
            transform: Optional augmentation transforms
            normalize: Whether to normalize images
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Build transform pipeline
        if transform is None:
            # Default: resize and normalize
            transform_list = [
                A.Resize(image_size[0], image_size[1]),
            ]
            if normalize:
                transform_list.append(A.Normalize(mean=mean, std=std))
            transform_list.append(ToTensorV2())
            self.transform = A.Compose(transform_list)
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = load_image(image_path, target_size=self.image_size)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Manual normalization if transform is None
            image = image.astype(np.float32) / 255.0
            if self.normalize:
                image = (image - np.array(self.mean)) / np.array(self.std)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, label


def create_dataloaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    class_names: List[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: dict = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create data loaders for training, validation, and test sets.
    
    Args:
        train_paths: Training image paths
        train_labels: Training labels
        val_paths: Validation image paths
        val_labels: Validation labels
        test_paths: Optional test image paths
        test_labels: Optional test labels
        class_names: List of class names
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        augmentation_config: Configuration for data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if class_names is None:
        class_names = ["Normal", "Pathological"]
    
    # Training transforms with augmentation
    if augmentation_config and augmentation_config.get("enable_augmentation", True):
        train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=augmentation_config.get("rotation_range", 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=augmentation_config.get("brightness_range", (0.8, 1.2)),
                contrast_limit=augmentation_config.get("contrast_range", (0.8, 1.2)),
                p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Create datasets
    train_dataset = KneeImageDataset(
        train_paths, train_labels, class_names,
        image_size=image_size, transform=train_transform
    )
    val_dataset = KneeImageDataset(
        val_paths, val_labels, class_names,
        image_size=image_size, transform=val_transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Test loader (if provided)
    if test_paths and test_labels:
        test_dataset = KneeImageDataset(
            test_paths, test_labels, class_names,
            image_size=image_size, transform=val_transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

