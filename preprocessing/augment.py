"""
Data augmentation functions for medical images.
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    RandomBrightnessContrast,
    ElasticTransform,
    GaussianBlur,
    RandomGamma,
    ShiftScaleRotate,
)


def get_augmentation_transforms(
    image_size: Tuple[int, int] = (224, 224),
    rotation_range: int = 15,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    elastic_transform: bool = True,
    gaussian_noise: bool = True,
    training: bool = True,
) -> Compose:
    """
    Get augmentation transforms for training or validation.
    
    Args:
        image_size: Target image size (height, width)
        rotation_range: Maximum rotation angle in degrees
        horizontal_flip: Whether to apply horizontal flip
        vertical_flip: Whether to apply vertical flip
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment
        elastic_transform: Whether to apply elastic transformation
        gaussian_noise: Whether to add Gaussian noise
        training: If True, apply augmentations; if False, only resize
        
    Returns:
        Albumentations Compose object
    """
    if not training:
        # Validation: only resize and normalize
        transforms = Compose([
            # Resize will be handled separately
        ])
        return transforms
    
    # Training augmentations
    aug_list = []
    
    if horizontal_flip:
        aug_list.append(HorizontalFlip(p=0.5))
    
    if vertical_flip:
        aug_list.append(VerticalFlip(p=0.5))
    
    if rotation_range > 0:
        aug_list.append(Rotate(limit=rotation_range, p=0.5))
    
    aug_list.append(
        RandomBrightnessContrast(
            brightness_limit=brightness_range,
            contrast_limit=contrast_range,
            p=0.5
        )
    )
    
    if elastic_transform:
        aug_list.append(ElasticTransform(alpha=1, sigma=50, p=0.3))
    
    if gaussian_noise:
        aug_list.append(GaussianBlur(blur_limit=(3, 5), p=0.3))
    
    aug_list.append(RandomGamma(gamma_limit=(80, 120), p=0.3))
    aug_list.append(ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3))
    
    transforms = Compose(aug_list)
    return transforms


def apply_augmentation(
    image: np.ndarray,
    transforms: Compose,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply augmentation transforms to image.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        transforms: Albumentations transform object
        mask: Optional mask to apply same transforms to
        
    Returns:
        Augmented image and mask (if provided)
    """
    if mask is not None:
        augmented = transforms(image=image, mask=mask)
        return augmented["image"], augmented["mask"]
    else:
        augmented = transforms(image=image)
        return augmented["image"], None


def simple_augment(image: np.ndarray, augmentation_type: str = "rotation") -> np.ndarray:
    """
    Simple augmentation functions using OpenCV (fallback if albumentations not available).
    
    Args:
        image: Input image
        augmentation_type: Type of augmentation
        
    Returns:
        Augmented image
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    if augmentation_type == "rotation":
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(image, M, (w, h))
    
    elif augmentation_type == "flip_horizontal":
        augmented = cv2.flip(image, 1)
    
    elif augmentation_type == "flip_vertical":
        augmented = cv2.flip(image, 0)
    
    elif augmentation_type == "brightness":
        brightness = np.random.uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(image, alpha=1, beta=int((brightness - 1) * 50))
    
    else:
        augmented = image
    
    return augmented.astype(np.float32) / 255.0

