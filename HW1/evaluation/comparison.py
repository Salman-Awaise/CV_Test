
import numpy as np
from typing import Dict, Any

def compare_methods(methods: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Compare multiple methods' errors and rank them.
    Args:
        methods: dict of {method_name: error_array}
    Returns:
        dict with mean_error, median_error, rankings
    """
    if not methods:
        raise ValueError("No methods provided for comparison.")
    mean_error = {}
    median_error = {}
    for k, v in methods.items():
        if not isinstance(v, (np.ndarray, list)):
            raise TypeError(f"Error array for method '{k}' must be a numpy array or list.")
        arr = np.array(v)
        if arr.size == 0:
            raise ValueError(f"Error array for method '{k}' is empty.")
        mean_error[k] = np.mean(arr)
        median_error[k] = np.median(arr)
    rankings = sorted(mean_error, key=lambda k: mean_error[k])
    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'rankings': rankings
    }
