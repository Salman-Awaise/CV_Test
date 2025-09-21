import numpy as np
from data.loader import load_image
from data.loader import extract_ground_truth
from data.preprocessing import extract_patch
from baselines.classical import gray_world_estimate, white_patch_estimate
from evaluation.metrics import angular_error

def test_full_pipeline():
    image = load_image("data/building1.npy")
    true_illuminant = gray_world_estimate(load_image("data/building1_cc.npy"))
    patch = extract_patch(image)
    mock_cnn_pred = np.array([0.8, 1.0, 0.9])
    mock_cnn_pred /= np.linalg.norm(mock_cnn_pred)
    gray_pred = gray_world_estimate(image)
    white_pred = white_patch_estimate(image)
    cnn_error = angular_error(mock_cnn_pred, true_illuminant)
    gray_error = angular_error(gray_pred, true_illuminant)
    white_error = angular_error(white_pred, true_illuminant)
    assert 0 <= cnn_error <= 180
    assert 0 <= gray_error <= 180
    assert 0 <= white_error <= 180
