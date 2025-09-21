"""Data preprocessing utilities for color constancy."""

import numpy as np
from typing import List, Tuple, Optional

from utils import setup_logging, validate_image

logger = setup_logging()

def contrast_normalize(patch: np.ndarray) -> np.ndarray:
    """Global histogram stretching for contrast normalization."""
    patch = patch.astype(np.float32)
    min_val = np.min(patch)
    max_val = np.max(patch)
    patch = patch - min_val
    patch = patch / (max_val - min_val + 1e-8)
    return patch

def random_horizontal_flip(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally for data augmentation."""
    if np.random.random() < prob:
        return np.fliplr(image)
    return image

def random_color_jitter(image: np.ndarray, brightness: float = 0.1, contrast: float = 0.1) -> np.ndarray:
    """Apply random brightness and contrast changes for data augmentation."""
    if np.random.random() < 0.5:
        # Random brightness
        brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
        image = image * brightness_factor
        
        # Random contrast
        contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
        mean_val = np.mean(image)
        image = (image - mean_val) * contrast_factor + mean_val
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
    return image

def extract_patches(image: np.ndarray, patch_size: int = 64, random_crop: bool = False, augment: bool = False) -> np.ndarray:
    """
    Extract center or random 64x64 patch for training.
    Avoid edges where color checker might be located.
    Return: numpy array of shape (patch_size, patch_size, 3)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    h, w, _ = image.shape
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image too small for patch extraction: {image.shape}")
    
    # Apply data augmentation before cropping
    if augment:
        image = random_horizontal_flip(image)
        image = random_color_jitter(image)
    
    if random_crop:
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
    else:
        top = (h - patch_size) // 2
        left = (w - patch_size) // 2
    patch = image[top:top+patch_size, left:left+patch_size, :]
    patch = contrast_normalize(patch)
    return patch

def extract_multiple_patches(image: np.ndarray, patch_size: int = 64, num_patches: int = 5) -> np.ndarray:
    """
    Extract multiple random patches from an image for training data augmentation.
    Return: numpy array of shape (num_patches, patch_size, patch_size, 3)
    """
    patches = []
    for _ in range(num_patches):
        patch = extract_patches(image, patch_size=patch_size, random_crop=True, augment=True)
        patches.append(patch)
    return np.array(patches)

def extract_patch(image: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Alias for extracting center patch for test compatibility.
    """
    patch = extract_patches(image, patch_size=size, random_crop=False)
    return patch
