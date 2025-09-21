
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.loader import load_image
from baselines.classical import gray_world_estimate, white_patch_estimate
from evaluation.metrics import angular_error, compute_statistics
from evaluation.visualization import apply_color_correction

# Load images
img = load_image('data/building1.npy')
img_cc = load_image('data/building1_cc.npy')

# Estimate illuminants
gray_pred = gray_world_estimate(img)
white_pred = white_patch_estimate(img)
true_illum = gray_world_estimate(img_cc)

# Compute errors
gray_error = angular_error(gray_pred, true_illum)
white_error = angular_error(white_pred, true_illum)
errors = [gray_error, white_error]
labels = ['Gray World', 'White Patch']

# Visualize corrections
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img.astype(np.uint8))
axs[0].set_title('Original')
axs[1].imshow(apply_color_correction(img, gray_pred))
axs[1].set_title(f'Gray World\nError: {gray_error:.2f}')
axs[2].imshow(apply_color_correction(img, white_pred))
axs[2].set_title(f'White Patch\nError: {white_error:.2f}')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/building1_visualization.png')
plt.close()

# Print statistics
stats = compute_statistics(errors)
with open('results/building1_stats.txt', 'w') as f:
    f.write('Error statistics for building1 images:\n')
    for k, v in stats.items():
        f.write(f'{k}: {v}\n')
print('Visualization and statistics generated.')