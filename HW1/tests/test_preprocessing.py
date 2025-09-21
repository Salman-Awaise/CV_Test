import numpy as np
from data.preprocessing import extract_patches

def test_extract_patches_center():
    img = np.random.rand(128,128,3)
    center_patch = img[32:96,32:96,:]
    from data.preprocessing import contrast_normalize
    norm_patch = contrast_normalize(center_patch)
    patch = extract_patches(img, patch_size=64)
    assert patch.shape == (64,64,3)
    assert np.allclose(patch, norm_patch)

def test_extract_patches_random():
    img = np.random.rand(128,128,3)
    patch = extract_patches(img, patch_size=64, random_crop=True)
    assert patch.shape == (64,64,3)
