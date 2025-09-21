import numpy as np
from baselines.classical import gray_world_estimate, white_patch_estimate, shades_of_gray_estimate

def test_gray_world_estimate():
    img = np.ones((64,64,3)) * [0.5, 0.2, 0.1]
    est = gray_world_estimate(img)
    assert est.shape == (3,)
    assert np.allclose(np.linalg.norm(est), 1)

def test_white_patch_estimate():
    img = np.ones((64,64,3)) * [0.5, 0.2, 0.1]
    est = white_patch_estimate(img)
    assert est.shape == (3,)
    assert np.allclose(np.linalg.norm(est), 1)

def test_shades_of_gray_estimate():
    img = np.ones((64,64,3)) * [0.5, 0.2, 0.1]
    est = shades_of_gray_estimate(img, p=6)
    assert est.shape == (3,)
    assert np.allclose(np.linalg.norm(est), 1)
