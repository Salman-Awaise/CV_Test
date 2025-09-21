import numpy as np
arr = np.random.uniform(0.2, 1.0, (64, 64, 3)).astype(np.float32)
np.save('tests/fixtures/sample_image.npy', arr)
print('sample_image.npy written with shape', arr.shape)