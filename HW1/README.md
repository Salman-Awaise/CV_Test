# CNN Color Constancy Project

[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](./tests/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A comprehensive deep learning and classical baseline pipeline for **color constancy estimation**. This project implements state-of-the-art CNN models alongside statistical methods for illuminant estimation, following research best practices from the computer vision literature.

### Key Features

- 🧠 **Deep Learning**: Improved CNN architecture with data augmentation and regularization
- 📊 **Classical Baselines**: 8 statistical algorithms (Gray World, White Patch, Shades of Gray, etc.)
- 🔬 **Comprehensive Evaluation**: Angular error metrics, statistical analysis, and visualization
- 🧪 **Testing Infrastructure**: Full pytest coverage with fixtures and integration tests
- 📈 **Performance Analysis**: Benchmarking against <0.9° target from research literature

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib pillow pytest

# Run comprehensive evaluation
python3 evaluation/batch_evaluation.py

# Train and evaluate CNN
python3 experiments/train_and_evaluate.py

# Run tests
pytest -v
```

## Project Structure

```
color_constancy/
├── 📁 data/                    # Data loading and preprocessing
│   ├── loader.py              # Image loading utilities
│   └── preprocessing.py       # Patch extraction & augmentation
├── 📁 models/                  # Deep learning models
│   ├── cnn.py                 # CNN architecture
│   └── training.py            # Training loops and loss functions
├── 📁 baselines/              # Classical algorithms
│   └── classical.py           # Statistical color constancy methods
├── 📁 evaluation/             # Analysis and metrics
│   ├── metrics.py             # Angular error and statistics
│   ├── visualization.py       # Color correction and plotting
│   └── batch_evaluation.py    # Comprehensive algorithm comparison
├── 📁 experiments/            # Training and evaluation scripts
│   ├── train.py               # Basic CNN training
│   └── train_and_evaluate.py  # Full pipeline
├── 📁 tests/                  # Test suite
│   └── fixtures/              # Test data and expected outputs
└── 📁 results/                # Generated outputs and analysis
```
│   ├── test_loader.py
│   ├── test_preprocessing.py
│   ├── test_classical.py
│   ├── test_metrics.py
│   ├── test_visualization.py
│   ├── test_cnn.py
│   ├── test_training.py
│   ├── test_integration.py
│   └── fixtures/
│       ├── sample_image.npy
│       ├── sample_illuminant.npy
│       ├── sample_metadata.json
│       └── expected_outputs.json
├── results/
└── README.md
```

## Setup
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install numpy torch pytest
   ```
3. Run tests:
   ```bash
   pytest -v
   ```

## Usage
- See `experiments/train.py` and `experiments/evaluate.py` for pipeline scripts.
- Place your dataset in `data/` and update paths as needed.
- Use provided test fixtures for development and validation.

## Contributing
- Follow PEP8 and use type hints.
- Add tests for new features in `tests/`.
- Document all public functions and classes.

## License
MIT
