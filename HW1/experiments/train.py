
"""
Integration script for CNN training.
"""
import logging
import numpy as np
import torch
from data.loader import load_images, extract_ground_truth
from data.preprocessing import extract_patches, contrast_normalize
from models.training import train_model, angular_error_loss, euclidean_loss
from models.cnn import ColorConstancyCNN
from models.training import train_model
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO)

def main():
	# Example paths (update as needed)
	dataset_path = "data/images"
	metadata_file = "data/metadata.json"
	images = load_images(dataset_path)
	gt = extract_ground_truth(metadata_file)
	patches = [extract_patches(img) for img in images]
	# Patches are already contrast normalized in extract_patches
	X = np.stack(patches)
	if gt is None:
		raise ValueError("No ground truth illuminant found.")
	if gt.ndim == 2:
		y = gt
	elif gt.ndim == 1:
		y = np.tile(gt, (len(X), 1))
	else:
		raise ValueError("Ground truth illuminant shape not supported.")
	X_tensor = torch.tensor(X.transpose(0,3,1,2), dtype=torch.float32)
	y_tensor = torch.tensor(y, dtype=torch.float32)
	from torch.utils.data import TensorDataset, DataLoader
	dataset = TensorDataset(X_tensor, y_tensor)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
	model = ColorConstancyCNN()
	logging.info("Starting training...")
	# Select loss function: 'angular' or 'euclidean'
	loss_fn = euclidean_loss  # Change to angular_error_loss if desired
	train_model(model, dataloader, epochs=10, loss_fn=loss_fn)
	logging.info("Training complete. Model saved as best_model.pth.")

if __name__ == "__main__":
	main()
