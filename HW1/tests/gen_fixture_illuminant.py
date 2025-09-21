import numpy as np
arr = np.array([0.95, 1.0, 1.05], dtype=np.float32)
np.save('tests/fixtures/sample_illuminant.npy', arr)
print('sample_illuminant.npy written with shape', arr.shape)