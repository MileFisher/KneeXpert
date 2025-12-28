"""
Image normalization functions for medical images.
"""
import numpy as np
import cv2
from typing import Tuple, Union


def normalize_image(
    image: np.ndarray,
    method: str = "min_max",
    mean: float = None,
    std: float = None,
) -> np.ndarray:
    """
    Normalize an image using various methods.
    
    Args:
        image: Input image as numpy array (uint8, 0-255)
        method: Normalization method ("min_max", "z_score", "histogram_equalization")
        mean: Mean value for z-score normalization
        std: Standard deviation for z-score normalization
        
    Returns:
        Normalized image (float32, typically 0-1 range)
    """
    if image.dtype != np.uint8:
        # Assume already normalized or convert
        if image.max() > 1.0:
            image = image.astype(np.uint8)
        else:
            return image.astype(np.float32)
    
    image_float = image.astype(np.float32)
    
    if method == "min_max":
        # Min-max normalization to [0, 1]
        min_val = image_float.min()
        max_val = image_float.max()
        if max_val > min_val:
            normalized = (image_float - min_val) / (max_val - min_val)
        else:
            normalized = image_float / 255.0
    
    elif method == "z_score":
        # Z-score normalization
        if mean is None:
            mean = image_float.mean()
        if std is None:
            std = image_float.std()
        
        if std > 0:
            normalized = (image_float - mean) / std
            # Clip to reasonable range
            normalized = np.clip(normalized, -3, 3)
            # Scale to [0, 1]
            normalized = (normalized + 3) / 6.0
        else:
            normalized = image_float / 255.0
    
    elif method == "histogram_equalization":
        # Histogram equalization for contrast enhancement
        if len(image.shape) == 3:
            # Convert to YUV, equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
        else:
            normalized = cv2.equalizeHist(image).astype(np.float32) / 255.0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def normalize_batch(
    images: np.ndarray,
    method: str = "min_max",
    mean: float = None,
    std: float = None,
) -> np.ndarray:
    """
    Normalize a batch of images.
    
    Args:
        images: Batch of images (N, H, W, C) or (N, H, W)
        method: Normalization method
        mean: Mean value for z-score
        std: Standard deviation for z-score
        
    Returns:
        Normalized batch
    """
    normalized = []
    for i in range(len(images)):
        norm_img = normalize_image(images[i], method=method, mean=mean, std=std)
        normalized.append(norm_img)
    return np.array(normalized)


def reduce_noise(
    image: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Reduce noise in medical images.
    
    Args:
        image: Input image
        method: Denoising method ("gaussian", "median", "bilateral")
        kernel_size: Size of filter kernel
        
    Returns:
        Denoised image
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    if method == "gaussian":
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        denoised = cv2.medianBlur(image, kernel_size)
    elif method == "bilateral":
        denoised = cv2.bilateralFilter(image, kernel_size, 80, 80)
    else:
        raise ValueError(f"Unknown denoising method: {method}")
    
    return denoised.astype(np.float32) / 255.0

