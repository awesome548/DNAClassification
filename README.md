# DNA Classification from Nanopore Sequencing Data

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.8-blue.svg?style=for-the-badge&logo=appveyor" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.13-FF0000.svg?style=for-the-badge&logo=appveyor" />
  </a>
</p>

A deep learning framework for taxonomic classification of DNA sequences from Oxford Nanopore Technologies (ONT) raw signal data. This project implements and compares multiple neural network architectures for classifying nanopore sequencing signals directly from FAST5 files.

## Overview

This repository provides a comprehensive pipeline for DNA sequence classification, including data preprocessing, model training, and hyperparameter optimization. The framework processes raw nanopore electrical signals and applies various deep learning architectures to classify sequences at different taxonomic levels (species, genus, family).

## Implemented Models

### Convolutional Neural Networks
- **ResNet** - Based on [SquiggleNet](https://github.com/welch-lab/SquiggleNet), adapted for 1D signal classification
- **EfficientNetV2** - Efficient architecture with compound scaling for signal processing

### Recurrent Neural Networks
- **GRU** - Gated Recurrent Unit with bidirectional processing
- **LSTM** - Long Short-Term Memory networks for sequential signal modeling

### Transformer Architectures
- **Vision Transformer (ViT)** - Adapted from [vit-pytorch](https://github.com/lucidrains/vit-pytorch#simple-vit) for 1D signal sequences
- **Simple ViT** - Simplified variant of Vision Transformer
- **Cosformer** - Linear complexity transformer based on [cosformer-pytorch](https://github.com/davidsvy/cosformer-pytorch)

## Data Preprocessing

The preprocessing pipeline handles raw FAST5 files and applies several normalization and cleaning steps:

1. **Signal Extraction**: Reads raw electrical signals from FAST5 files using `ont_fast5_api`
2. **Length Filtering**: Filters sequences based on minimum length thresholds (`CUTOFF + MAXLEN`)
3. **MAD Normalization**: Applies Median Absolute Deviation (MAD) normalization for robust signal scaling:
   - Computes median and MAD for each sequence
   - Normalizes using: `(signal - median) / (1.4826 × MAD)`
4. **Outlier Removal**: Detects and removes outliers (>3.5 standard deviations) using interpolation

(**PAF Alignment Processing**: Optional alignment-based filtering using PAF (Pairwise mApping Format) files)

## Hyperparameter Optimization

The project implements multiple optimization strategies:

### Optuna
- Bayesian optimization for hyperparameter search
- Supports pruning of unpromising trials
- Integrated with Weights & Biases for experiment tracking

### Ray Tune
- Distributed hyperparameter tuning with ASHA scheduler
- Supports multi-GPU parallel trials
- OptunaSearch integration for efficient search space exploration

### Weights & Biases Sweeps
- Grid and random search configurations
- PyTorch Lightning integration for automated training loops
- Early stopping callbacks for efficient resource utilization

## MLOps & Experiment Tracking

- **Weights & Biases (wandb)**: Experiment tracking, hyperparameter logging, and visualization
- **PyTorch Lightning**: Structured training loops with automatic optimization
- **TensorBoard**: Additional logging and visualization support
- **Ray**: Distributed computing for parallel hyperparameter search

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies
- `torch`, `torchvision`, `torchaudio` - PyTorch ecosystem
- `optuna` - Hyperparameter optimization
- `ray[tune]` - Distributed hyperparameter tuning
- `wandb` - Experiment tracking
- `pytorch-lightning` - Training framework
- `ont_fast5_api` - FAST5 file processing
- `scikit-learn`, `scipy` - Data processing utilities
- `einops` - Tensor operations

## Usage

### Basic Training

```bash
python main.py --arch ResNet --batch 1000 --minepoch 20 --learningrate 1e-2 --hidden 64
```

### Hyperparameter Optimization with Optuna

```bash
python ML_optimization/optunaopt.py
```

### Distributed Tuning with Ray Tune

```bash
python ML_optimization/raytune.py
```

### Weights & Biases Sweep

```bash
python ML_optimization/sweep.py
```

## Project Structure

```
DNAClassification/
├── ML_model/           # Model architectures
├── ML_dataset/          # Data loading and formatting
├── ML_preparation/      # Preprocessing utilities
├── ML_processing/       # Training and evaluation loops
├── ML_optimization/     # Hyperparameter optimization scripts
├── main.py              # Main training script
├── main_multi.py        # Multi-class classification
├── main_zymo.py         # Zymo dataset specific
└── inference.py         # Inference script
```

## Configuration

Configuration is managed through environment variables (`.env` file):
- `CUTLEN`: Length of signal segments
- `CUTOFF`: Starting position offset
- `MAXLEN`: Maximum sequence length
- `DATASETSIZE`: Number of samples per class
- `STRIDE`: Sliding window stride
- `DATAPATH`: Data storage path

## Citation

If you use this code in your research, please cite the original model implementations:
- SquiggleNet: [Welch Lab](https://github.com/welch-lab/SquiggleNet)
- ViT: [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- Cosformer: [davidsvy/cosformer-pytorch](https://github.com/davidsvy/cosformer-pytorch)
