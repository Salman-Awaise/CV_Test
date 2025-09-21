"""
Color constancy project package.

This package provides a comprehensive framework for color constancy estimation
using both classical algorithms and deep learning approaches.
"""

__version__ = "1.0.0"
__author__ = "Computer Vision Lab"
__email__ = "cv-lab@example.com"

# Import main classes and functions for convenience
from data.loader import ColorConstancyDataset, load_images, load_image
from data.preprocessing import contrast_normalize, random_horizontal_flip
from models.cnn import ColorConstancyCNN
from evaluation.metrics import angular_error, compute_statistics
from baselines.classical import (
    gray_world_estimate,
    white_patch_estimate,
    shades_of_gray_estimate,
    max_rgb_estimate,
    edge_based_estimate,
    robust_awb_estimate
)

__all__ = [
    "ColorConstancyDataset",
    "load_images",
    "load_image", 
    "contrast_normalize",
    "random_horizontal_flip",
    "ColorConstancyCNN",
    "angular_error",
    "compute_statistics",
    "gray_world_estimate",
    "white_patch_estimate", 
    "shades_of_gray_estimate",
    "max_rgb_estimate",
    "edge_based_estimate",
    "robust_awb_estimate"
]