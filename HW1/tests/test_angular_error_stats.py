import numpy as np
import pytest
from evaluation.metrics import angular_error, compute_statistics
from data.loader import load_image
from baselines.classical import gray_world_estimate

def test_angular_error_statistics():
    # Load test images (extend to batch for full dataset)
    img = load_image('data/building1.npy')
    img_cc = load_image('data/building1_cc.npy')
    # Use gray world as baseline
    pred = gray_world_estimate(img)
    true = gray_world_estimate(img_cc)
    error = angular_error(pred, true)
    stats = compute_statistics([error])
    print('Angular error:', error)
    print('Statistics:', stats)
    # Target: mean < 0.9 deg (as in original paper)
    assert stats['mean'] < 0.9
