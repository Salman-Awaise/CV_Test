
"""
Integration script for evaluation and baseline comparison.
"""
from email import errors
import logging
import numpy as np
import torch
from data.loader import load_images, extract_ground_truth
from baselines.classical import gray_world_estimate, white_patch_estimate
from models.cnn import ColorConstancyCNN
from evaluation.metrics import angular_error, compute_statistics
from evaluation.comparison import compare_methods

logging.basicConfig(level=logging.INFO)

def main():
	# Example paths (update as needed)
	dataset_path = "data/images"
	metadata_file = "data/metadata.json"
	images = load_images(dataset_path)
	gt = extract_ground_truth(metadata_file)
	model = ColorConstancyCNN()
	model.load_state_dict(torch.load("best_model.pth"))
	model.eval()
	errors = {"cnn": [], "gray_world": [], "white_patch": []}
	for img in images:
		img_tensor = torch.tensor(img.transpose(2,0,1)[None], dtype=torch.float32)
		cnn_pred = model(img_tensor).detach().cpu().numpy()[0]
		gray_pred = gray_world_estimate(img)
		white_pred = white_patch_estimate(img)
		true_illum = gt if gt.ndim == 1 else gt[0]
		errors["cnn"].append(angular_error(cnn_pred, true_illum))
		errors["gray_world"].append(angular_error(gray_pred, true_illum))
		errors["white_patch"].append(angular_error(white_pred, true_illum))
	stats = {k: compute_statistics(v) for k, v in errors.items()}
	comparison = compare_methods({k: np.array(v) for k, v in errors.items()})
	logging.info(f"Stats: {stats}")
	logging.info(f"Comparison: {comparison}")

if __name__ == "__main__":
	main()
