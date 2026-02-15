<a id="readme-top"></a>

[![Python][python-shield]][python-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![Optuna][optuna-shield]][optuna-url]
[![Ray Tune][ray-shield]][ray-url]
[![ONT FAST5 API][fast5-shield]][fast5-url]

<br />
<div align="center">
  <h3 align="center">DNAClassification</h3>
  <p align="center">
    Deep learning pipeline for taxonomic classification from Oxford Nanopore raw electrical signals.
    <br />
    <a href="https://drive.google.com/file/d/11i1k8UQzf9-fSThy2creKTBrJSdhgkAP/view?usp=sharing"><strong>Repository »</strong></a>
    <br />
    <br />
    <a href="https://github.com/awesome548/DNAClassification/issues">Report Bug</a>
    ·
    <a href="https://github.com/awesome548/DNAClassification/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#problem-space">Problem Space</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#implemented-models">Implemented Models</a></li>
    <li><a href="#optimization--tracking">Optimization & Tracking</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This repository trains neural networks directly on ONT FAST5-derived current traces, without basecalling first. It provides:

- signal preprocessing with MAD normalization and outlier correction
- sliding-window signal shaping for fixed-length model input
- single-stage and category-filtered classification workflows
- multiple model backbones (CNN/RNN/Transformer variants)

Core training entrypoint: `main.py`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Problem Space

Nanopore sequencing produces long, noisy electrical signals. Traditional pipelines depend on basecalling and alignment, which can introduce additional error and latency. This project addresses that by learning directly from raw signal windows, enabling fast taxonomic discrimination workflows (species/genus/family modes and two-stage category analysis).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

### Directory Layout

```text
DNAClassification/
├── main.py                    # Primary training CLI (Click)
├── inference.py               # Two-stage evaluation/inference script
├── ML_preparation/
│   ├── preprocess.py          # FAST5 -> normalized tensors
│   └── utils.py               # MAD normalization, window slicing
├── ML_dataset/
│   ├── dataformat.py          # Loader construction from FAST5 folders
│   ├── dataset.py             # MultiDataset labels and class mapping
│   └── in_category_data.py    # Stage-2 category dataset handling
├── ML_model/
│   ├── preference.py          # Architecture dispatch
│   ├── resnet.py              # CNN baseline
│   ├── effnetv2.py            # EfficientNetV2 variant
│   ├── gru.py / lstm.py       # Sequence models
│   └── cosformer.py           # Transformer variant
├── ML_processing/
│   └── train.py               # Training and validation loops
└── ML_optimization/           # Optuna / Ray Tune / W&B sweep scripts
```

### Module Interaction

```text
FAST5 directories
  -> ML_preparation.preprocess (normalization + filtering)
  -> ML_dataset.dataformat (train/val/test DataLoader)
  -> ML_model.preference (select architecture)
  -> ML_processing.train.train_loop (fit + validate)
  -> saved model (.pth) and metric logs
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- Python `3.10+`
- `pip`
- CUDA-enabled GPU (optional, but recommended)
- **FAST5 input data** -- a directory containing one subfolder per species, each holding Oxford Nanopore `.fast5` files (see [Prepare Input Data](#prepare-input-data) below)

### Installation

1. Clone the repository.
   ```sh
   git clone git@github.com:awesome548/DNAClassification.git
   cd DNAClassification
   ```
2. Create and activate a virtual environment.
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies.
   ```sh
   pip install -r requirements.txt
   ```

### Prepare Input Data

This project trains directly on raw electrical signals from Oxford Nanopore sequencing. You must provide a **FAST5 directory** organized by species before running any training or inference.

Required directory layout (pointed to by the `FAST5` environment variable):

```text
/path/to/fast5_root/
├── SpeciesA/          # Folder name must start with A-Z (uppercase)
│   ├── batch_0.fast5
│   ├── batch_1.fast5
│   └── ...
├── SpeciesB/
│   ├── batch_0.fast5
│   └── ...
└── _excluded/         # Folders starting with "_" are ignored
```

Key requirements:
- Each species folder **must start with an uppercase letter** (`A-Z`). Folders starting with `_` or lowercase are skipped.
- Each `.fast5` file must contain reads of at least `CUTOFF + MAXLEN` samples in length (shorter reads are filtered out).
- You need at least `DATASETSIZE` valid reads per species.
- The pipeline will normalize signals (MAD normalization), apply a sliding window of length `CUTLEN` with stride `STRIDE`, and split data into train/val/test (80%/10%/10%).

If preprocessed `.pt` tensor files already exist in `DATAPATH`, they will be loaded directly instead of re-reading the raw FAST5 files.

### Environment Configuration

Create a `.env` file in the project root:

```env
# Required -- paths
FAST5=/absolute/path/to/species_fast5_dirs
DATAPATH=/absolute/path/to/preprocessed_tensors
MODEL=/absolute/path/to/model_output

# Required -- signal processing parameters
DATASETSIZE=1000    # Number of reads to sample per species
CUTOFF=1500         # Trim this many samples from the start of each read
MAXLEN=6000         # Length of signal window after cutoff
CUTLEN=3000         # Sliding window length fed to the model
STRIDE=500          # Step size for the sliding window

# Optional (used by auxiliary scripts)
RESULT=/absolute/path/to/results
IDLIST=/absolute/path/to/id_lists
MISC=/absolute/path/to/misc
ZYMO=/absolute/path/to/zymo_fast5
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Train a Model

```sh
python main.py \
  --arch ResNet \
  --batch 1000 \
  --minepoch 20 \
  --learningrate 1e-2 \
  --hidden 64 \
  --cls_type base
```

Supported architecture keys (via `ML_model/preference.py`):
- `ResNet`
- `Effnet`
- `GRU`
- `LSTM`
- `Transformer`

### Run Two-Stage Inference

```sh
python inference.py
```

Note: `inference.py` currently contains hardcoded checkpoint and dataset paths. Update those paths before execution.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Implemented Models

- CNN: ResNet, EfficientNetV2
- RNN: GRU, LSTM
- Transformer-family: Cosformer-based classifier

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Optimization & Tracking

Optional scripts in `ML_optimization/` support:

- Optuna-based hyperparameter search: `python ML_optimization/optunaopt.py`
- Ray Tune workflows: `python ML_optimization/raytune.py`
- Weights & Biases sweeps: `python ML_optimization/sweep.py`

These scripts rely on additional runtime setup (e.g., `wandb`, cluster/GPU configuration, and dataset path adjustments).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Open a pull request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [Oxford Nanopore FAST5 API](https://github.com/nanoporetech/ont_fast5_api)
- [Optuna](https://optuna.org)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
- [Weights & Biases](https://wandb.ai)
- [Shields.io](https://shields.io)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS -->
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[optuna-shield]: https://img.shields.io/badge/Optuna-HPO-5C4EE5?style=for-the-badge
[optuna-url]: https://optuna.org/
[ray-shield]: https://img.shields.io/badge/Ray-Tune-028CF0?style=for-the-badge
[ray-url]: https://docs.ray.io/en/latest/tune/index.html
[fast5-shield]: https://img.shields.io/badge/ONT-FAST5%20API-00A3E0?style=for-the-badge
[fast5-url]: https://github.com/nanoporetech/ont_fast5_api
[context7-shield]: https://img.shields.io/badge/Context7-MCP-111827?style=for-the-badge
[context7-url]: https://context7.com/
[contributors-shield]: https://img.shields.io/github/contributors/awesome548/DNAClassification.svg?style=for-the-badge
[contributors-url]: https://github.com/awesome548/DNAClassification/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/awesome548/DNAClassification.svg?style=for-the-badge
[forks-url]: https://github.com/awesome548/DNAClassification/network/members
[stars-shield]: https://img.shields.io/github/stars/awesome548/DNAClassification.svg?style=for-the-badge
[stars-url]: https://github.com/awesome548/DNAClassification/stargazers
[issues-shield]: https://img.shields.io/github/issues/awesome548/DNAClassification.svg?style=for-the-badge
[issues-url]: https://github.com/awesome548/DNAClassification/issues
