import numpy as np
from typing import Any

def apply_color_correction(image: np.ndarray, estimated_illuminant: Any) -> np.ndarray:
    """
    Diagonal transform: corrected = image / illuminant
    Normalize to [0,1] and display
    Show: original | corrected_cnn | corrected_gray_world
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3D with 3 channels, got shape {image.shape}.")
    illum = np.array(estimated_illuminant)
    illum = illum / np.linalg.norm(illum)
    if illum.shape != (3,):
        raise ValueError(f"Illuminant must be a 3D vector, got shape {illum.shape}.")
    corrected = image / illum
    corrected = corrected / np.max(corrected)
    corrected = np.clip(corrected, 0, 1)
    return corrected
