"""
Utility functions for image loading, saving, and data management.
"""
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import cv2


def load_image(image_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        target_size: Optional (width, height) to resize
        
    Returns:
        Image as numpy array (RGB format)
    """
    # Try PIL first
    try:
        img = Image.open(image_path).convert("RGB")
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception:
        # Fallback to OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array (RGB format)
        output_path: Path to save image
    """
    if isinstance(image, np.ndarray):
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        img = Image.fromarray(image)
        img.save(output_path)
    else:
        # Assume PIL Image
        image.save(output_path)


def create_data_splits(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create train/validation/test splits from data directory.
    Assumes data is organized by class in subdirectories.
    
    Args:
        data_dir: Directory containing class subdirectories
        output_dir: Directory to save split information
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to lists of file paths
    """
    np.random.seed(random_seed)
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    data_path = Path(data_dir)
    splits = {"train": [], "val": [], "test": []}
    
    # Get all class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        # Shuffle
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Add to splits with class label
        for file_path in train_files:
            splits["train"].append((str(file_path), class_name))
        for file_path in val_files:
            splits["val"].append((str(file_path), class_name))
        for file_path in test_files:
            splits["test"].append((str(file_path), class_name))
    
    # Save splits to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, file_list in splits.items():
        split_file = output_path / f"{split_name}.txt"
        with open(split_file, "w") as f:
            for file_path, class_name in file_list:
                f.write(f"{file_path}\t{class_name}\n")
    
    return splits


def get_image_statistics(data_dir: str) -> Dict[str, float]:
    """
    Calculate mean and std statistics for dataset normalization.
    
    Args:
        data_dir: Directory containing images
        
    Returns:
        Dictionary with 'mean' and 'std' for each channel
    """
    data_path = Path(data_dir)
    image_files = (
        list(data_path.rglob("*.jpg")) +
        list(data_path.rglob("*.png")) +
        list(data_path.rglob("*.jpeg"))
    )
    
    if not image_files:
        raise ValueError(f"No images found in {data_dir}")
    
    # Sample images for statistics (use subset for efficiency)
    sample_size = min(1000, len(image_files))
    sampled_files = np.random.choice(image_files, sample_size, replace=False)
    
    pixel_values = []
    for img_file in sampled_files:
        try:
            img = load_image(str(img_file))
            pixel_values.append(img.flatten())
        except Exception:
            continue
    
    if not pixel_values:
        raise ValueError("Could not load any images for statistics")
    
    all_pixels = np.concatenate(pixel_values)
    mean = np.mean(all_pixels) / 255.0
    std = np.std(all_pixels) / 255.0
    
    return {"mean": mean, "std": std}

