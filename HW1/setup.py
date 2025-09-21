"""Installation configuration for the color constancy project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="color-constancy",
    version="1.0.0",
    author="Computer Vision Lab",
    author_email="cv-lab@example.com",
    description="A comprehensive framework for color constancy estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/color-constancy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "color-constancy-train=experiments.train:main",
            "color-constancy-evaluate=experiments.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
        "tests": ["fixtures/*"],
    },
)