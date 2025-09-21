import numpy as np
from data.loader import load_images, extract_ground_truth
import tempfile, os

def test_load_images():
    arr = np.ones((64,64,3), dtype=np.uint16) * 1000
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, 'img1.npy'), arr)
        imgs = load_images(tmpdir)
        assert len(imgs) == 1
        assert imgs[0].shape == (64,64,3)
        assert np.allclose(imgs[0].max(), 1000/65535, atol=1e-3)

def test_extract_ground_truth():
    with tempfile.NamedTemporaryFile('w+', delete=False) as f:
        f.write('1,2,2\n3,4,5\n')  # all positive values
        f.close()
    gt = extract_ground_truth(f.name)
    assert gt is not None
    assert gt.shape == (2,3)
    assert np.allclose(np.linalg.norm(gt, axis=1), 1)
    os.remove(f.name)
