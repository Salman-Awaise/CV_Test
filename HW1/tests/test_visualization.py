import numpy as np
from evaluation.visualization import apply_color_correction

def test_apply_color_correction():
    img = np.ones((32,32,3)) * [0.5, 0.2, 0.1]
    illum = [0.5, 0.2, 0.1]
    corrected = apply_color_correction(img, illum)
    assert corrected.shape == (32,32,3)
    assert np.allclose(corrected.max(), 1)
