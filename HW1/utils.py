"""
Utilities for the color constancy project.
"""

import os
import logging
import numpy as np
import torch
from typing import List, Tuple, Optional
from config import LOGGING_CONFIG, PATHS


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format']
    )
    
    logger = logging.getLogger('color_constancy')
    
    if LOGGING_CONFIG['log_to_file']:
        os.makedirs(PATHS['logs_dir'], exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(PATHS['logs_dir'], LOGGING_CONFIG['log_filename'])
        )
        file_handler.setFormatter(
            logging.Formatter(LOGGING_CONFIG['format'])
        )
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def setup_directories() -> None:
    """Set up all required directories."""
    for path in PATHS.values():
        ensure_dir(path)


def normalize_illuminant(illuminant: np.ndarray) -> np.ndarray:
    """
    Normalize illuminant to unit vector.
    
    Args:
        illuminant: RGB illuminant vector
        
    Returns:
        Normalized unit vector
    """
    illuminant = np.array(illuminant, dtype=np.float32)
    norm = np.linalg.norm(illuminant)
    if norm < 1e-8:
        raise ValueError("Illuminant vector has zero norm")
    return illuminant / norm


def validate_image(image: np.ndarray) -> None:
    """
    Validate image format and properties.
    
    Args:
        image: Input image array
        
    Raises:
        ValueError: If image format is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D, got {image.ndim}D")
    
    if image.shape[2] != 3:
        raise ValueError(f"Image must have 3 channels, got {image.shape[2]}")
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_error_statistics(errors: List[float]) -> str:
    """
    Format error statistics for display.
    
    Args:
        errors: List of angular errors in degrees
        
    Returns:
        Formatted string with statistics
    """
    if not errors:
        return "No errors to display"
    
    errors_array = np.array(errors)
    stats = {
        'mean': np.mean(errors_array),
        'median': np.median(errors_array),
        'std': np.std(errors_array),
        'min': np.min(errors_array),
        'max': np.max(errors_array),
        'q25': np.percentile(errors_array, 25),
        'q75': np.percentile(errors_array, 75),
    }
    
    return f"""Error Statistics:
    Mean: {stats['mean']:.3f}°
    Median: {stats['median']:.3f}°
    Std: {stats['std']:.3f}°
    Range: [{stats['min']:.3f}°, {stats['max']:.3f}°]
    IQR: [{stats['q25']:.3f}°, {stats['q75']:.3f}°]
    """