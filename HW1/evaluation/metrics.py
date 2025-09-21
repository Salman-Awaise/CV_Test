"""Evaluation metrics for color constancy estimation."""

import numpy as np
from typing import Union, Dict, List

from utils import setup_logging

logger = setup_logging()

def angular_error(pred: Union[np.ndarray, list], true: Union[np.ndarray, list]) -> float:
    """
    Compute angular error (degrees) between two unit vectors.
    """
    pred = np.array(pred)
    true = np.array(true)
    if pred.shape != true.shape:
        raise ValueError(f"Predicted and true illuminants must have the same shape, got {pred.shape} and {true.shape}.")
    dot = np.clip(np.sum(pred * true, axis=-1), -1.0, 1.0)
    angle = np.arccos(dot)
    return float(angle * 180.0 / np.pi)


def compute_statistics(errors: Union[np.ndarray, list]) -> Dict[str, float]:
    """
    Compute mean, median, 95th percentile error, and percentages below thresholds.
    """
    errors = np.array(errors)
    if errors.size == 0:
        raise ValueError("Error array is empty.")
    mean = float(np.mean(errors))
    median = float(np.median(errors))
    perc95 = float(np.percentile(errors, 95))
    pct_lt3 = float(np.mean(errors < 3) * 100)
    pct_lt5 = float(np.mean(errors < 5) * 100)
    pct_lt10 = float(np.mean(errors < 10) * 100)
    return {
        'mean': mean,
        'median': median,
        'perc95': perc95,
        'pct_lt3': pct_lt3,
        'pct_lt5': pct_lt5,
        'pct_lt10': pct_lt10
    }
