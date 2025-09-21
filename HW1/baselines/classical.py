"""Classical baseline methods for color constancy estimation."""

import numpy as np
from typing import Optional, Tuple

from utils import setup_logging, validate_image, normalize_illuminant

logger = setup_logging()

def gray_world_estimate(image: np.ndarray) -> np.ndarray:
    """
    Gray-World algorithm: mean of image channels.
    Ignore very dark/bright pixels (top/bottom 5%).
    Return: unit vector (illuminant estimate)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    flat = image.reshape(-1, 3)
    percent = 5
    lower = np.percentile(flat, percent, axis=0)
    upper = np.percentile(flat, 100-percent, axis=0)
    mask = np.all((flat >= lower) & (flat <= upper), axis=1)
    if not np.any(mask):
        raise ValueError("No valid pixels for gray-world estimate.")
    mean_rgb = flat[mask].mean(axis=0)
    return mean_rgb / np.linalg.norm(mean_rgb)


def white_patch_estimate(image: np.ndarray) -> np.ndarray:
    """
    White-Patch algorithm: 99th percentile of each channel.
    Return: unit vector (illuminant estimate)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    flat = image.reshape(-1, 3)
    illum = np.percentile(flat, 99, axis=0)
    return illum / np.linalg.norm(illum)


def shades_of_gray_estimate(image: np.ndarray, p: int = 6) -> np.ndarray:
    """
    Shades-of-Gray algorithm: (mean(image^p))^(1/p).
    Return: unit vector (illuminant estimate)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    flat = image.reshape(-1, 3)
    # Avoid zero pixels to prevent numerical issues
    flat = np.maximum(flat, 1e-6)
    mean_p = np.mean(flat ** p, axis=0)
    illum = mean_p ** (1.0 / p)
    return illum / np.linalg.norm(illum)


def max_rgb_estimate(image: np.ndarray) -> np.ndarray:
    """
    Max RGB algorithm: maximum value in each channel.
    Return: unit vector (illuminant estimate)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    flat = image.reshape(-1, 3)
    illum = np.max(flat, axis=0)
    return illum / np.linalg.norm(illum)


def edge_based_estimate(image: np.ndarray) -> np.ndarray:
    """
    Edge-based color constancy: weighted by edge strength.
    Return: unit vector (illuminant estimate)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    
    # Simple edge detection using gradients
    dy, dx = np.gradient(image.astype(np.float32), axis=(0, 1))
    edge_strength = np.sqrt(dx**2 + dy**2).sum(axis=2)
    
    # Weight pixels by edge strength
    flat = image.reshape(-1, 3)
    weights = edge_strength.flatten()
    weights = weights / (weights.sum() + 1e-8)
    
    illum = np.average(flat, weights=weights, axis=0)
    return illum / np.linalg.norm(illum)


def robust_awb_estimate(image: np.ndarray, percentile: float = 95) -> np.ndarray:
    """
    Robust Auto White Balance: uses high percentile instead of max.
    Return: unit vector (illuminant estimate)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    flat = image.reshape(-1, 3)
    illum = np.percentile(flat, percentile, axis=0)
    return illum / np.linalg.norm(illum)
