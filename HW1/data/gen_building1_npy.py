from PIL import Image
import numpy as np
img = np.array(Image.open('data/building1.jpg'))
np.save('data/building1.npy', img)
img_cc = np.array(Image.open('data/building1_cc.jpg'))
np.save('data/building1_cc.npy', img_cc)
print('Saved building1.npy and building1_cc.npy with shapes', img.shape, img_cc.shape)