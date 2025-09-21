"""
Configuration settings for the color constancy project.
"""

# Data configuration
DATA_CONFIG = {
    'patch_size': 64,
    'batch_size': 8,
    'num_patches_per_image': 10,
    'train_test_split': 0.8,
    'image_extensions': ['.npy', '.npz', '.jpg', '.png'],
}

# Model configuration
MODEL_CONFIG = {
    'dropout_rate': 0.4,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'early_stopping_patience': 10,
}

# Evaluation configuration
EVAL_CONFIG = {
    'target_angular_error': 0.9,  # degrees
    'error_thresholds': [3, 5, 10],  # degrees
    'percentiles': [95],
}

# Augmentation configuration
AUGMENT_CONFIG = {
    'horizontal_flip_prob': 0.5,
    'brightness_range': 0.1,
    'contrast_range': 0.1,
    'color_jitter_prob': 0.5,
}

# Classical algorithm configuration
CLASSICAL_CONFIG = {
    'shades_of_gray_p_values': [1, 6],
    'percentile_values': [90, 95, 99],
    'gray_world_percentile_exclude': 5,  # exclude top/bottom 5%
}

# Paths
PATHS = {
    'data_dir': 'data/',
    'results_dir': 'results/',
    'models_dir': 'models/checkpoints/',
    'logs_dir': 'logs/',
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_filename': 'color_constancy.log',
}