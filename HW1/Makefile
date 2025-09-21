# Makefile for Color Constancy Project

.PHONY: help install install-dev test test-coverage lint format clean train evaluate all

# Default target
help:
	@echo "Available targets:"
	@echo "  install       Install package and dependencies"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint          Run code linting (flake8, mypy)"
	@echo "  format        Format code with black"
	@echo "  clean         Clean up generated files"
	@echo "  train         Train the CNN model"
	@echo "  evaluate      Evaluate all algorithms"
	@echo "  all           Run full pipeline (install, test, train, evaluate)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest -v

test-coverage:
	pytest --cov=. --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 .
	mypy .

format:
	black .

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".coverage" -delete
	find . -name "*.cover" -delete
	find . -name "*.log" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Model training and evaluation
train:
	python experiments/train.py

evaluate:
	python experiments/evaluate.py

train-and-evaluate:
	python experiments/train_and_evaluate.py

# Full pipeline
all: install-dev test train evaluate

# Development setup
setup-dev: install-dev
	pre-commit install
	@echo "Development environment set up successfully!"