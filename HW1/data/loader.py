"""Data loading utilities for color constancy datasets."""

import os
import json
import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Any
from pathlib import Path

from utils import setup_logging, validate_image

logger = setup_logging()

def load_images(dataset_path: str) -> List[np.ndarray]:
    """
    Load RAW/linear RGB images (not sRGB!).
    Normalize to [0,1] range.
    Return: list of (H, W, 3) numpy arrays
    """
    images = []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    for fname in os.listdir(dataset_path):
        if fname.lower().endswith(('.npy', '.npz')):
            try:
                img = np.load(os.path.join(dataset_path, fname), allow_pickle=True)
                # Handle different data types and ranges
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                elif img.max() > 1.0:
                    img = img.astype(np.float32) / 65535.0
                else:
                    img = img.astype(np.float32)
                
                # Ensure 3D with 3 channels
                if img.ndim == 3 and img.shape[2] == 3:
                    images.append(img)
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")
                continue
    if not images:
        raise ValueError(f"No valid images found in '{dataset_path}'.")
    return images

def load_image(path: str) -> np.ndarray:
    """
    Load a single image and normalize to [0,1] float32.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file '{path}' does not exist.")
    img = np.load(path)
    img = img.astype(np.float32) / 65535.0 if img.max() > 1.0 else img.astype(np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Image at '{path}' must be 3D with 3 channels, got shape {img.shape}.")
    return img



def extract_ground_truth(metadata_file: str) -> Union[np.ndarray, None]:
    """
    Parse illuminant RGB values from dataset metadata.
    Supports .json (single vector) or .csv (multiple vectors).
    Returns: np.ndarray shape (3,) or (N,3)
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file '{metadata_file}' does not exist.")
    if metadata_file.endswith('.json'):
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            illum = np.array(data['illuminant'])
            if illum.shape != (3,):
                raise ValueError(f"Illuminant in '{metadata_file}' must be a 3D RGB vector.")
            illum = illum / np.linalg.norm(illum)
            if not np.all(illum > 0):
                raise ValueError(f"Illuminant components must be positive in '{metadata_file}'.")
            return illum
    else:
        illuminants = []
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    vals = [float(x) for x in line.strip().split(',')]
                    illum = np.array(vals)
                    if illum.shape != (3,):
                        raise ValueError(f"Illuminant in '{metadata_file}' must be a 3D RGB vector.")
                    illum = illum / np.linalg.norm(illum)
                    if not np.all(illum > 0):
                        raise ValueError(f"Illuminant components must be positive in '{metadata_file}'.")
                    illuminants.append(illum)
        if not illuminants:
            return None
        return np.array(illuminants)


class ColorConstancyDataset:
    """
    Dataset class for color constancy training and evaluation.
    
    Provides functionality to load images, illuminants, and extract patches
    for training neural networks on color constancy tasks.
    """
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing images and illuminant data
            image_size: Size to resize patches to (height, width)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.images = []
        self.illuminants = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load all images and corresponding illuminants."""
        logger.info(f"Loading data from {self.data_dir}")
        
        # Load images
        for img_file in self.data_dir.glob("*.npy"):
            if "cc" not in img_file.name:  # Skip color corrected images
                try:
                    img = load_image(str(img_file))
                    validate_image(img)
                    self.images.append(img)
                    
                    # Load corresponding illuminant
                    illum_file = img_file.with_suffix('.npy')
                    illum_file = illum_file.parent / f"{img_file.stem}_cc.npy"
                    
                    if illum_file.exists():
                        illum = np.load(str(illum_file), allow_pickle=True)
                        if isinstance(illum, np.ndarray) and illum.shape == (3,):
                            self.illuminants.append(illum)
                        else:
                            logger.warning(f"Invalid illuminant shape in {illum_file}")
                            self.images.pop()  # Remove corresponding image
                    else:
                        logger.warning(f"No illuminant found for {img_file}")
                        self.images.pop()  # Remove image without illuminant
                        
                except Exception as e:
                    logger.error(f"Error loading {img_file}: {e}")
        
        logger.info(f"Loaded {len(self.images)} image-illuminant pairs")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_patch, illuminant)
        """
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        
        image = self.images[idx]
        illuminant = self.illuminants[idx]
        
        # Extract random patch
        patch = self._extract_patch(image)
        
        return patch, illuminant
    
    def _extract_patch(self, image: np.ndarray) -> np.ndarray:
        """Extract a random patch from the image."""
        h, w, _ = image.shape
        patch_h, patch_w = self.image_size
        
        if h < patch_h or w < patch_w:
            # If image is smaller than patch size, just return resized using numpy interpolation
            # For simplicity, we'll just use the full image if too small
            logger.warning(f"Image too small ({h}x{w}) for patch size ({patch_h}x{patch_w}), using full image")
            return image
        
        # Random crop
        top = np.random.randint(0, h - patch_h + 1)
        left = np.random.randint(0, w - patch_w + 1)
        
        patch = image[top:top + patch_h, left:left + patch_w]
        return patch
