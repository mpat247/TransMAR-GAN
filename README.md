# TransMAR-GAN

PyTorch implementation of TransMAR-GAN for CT metal artifact reduction.

## Overview

TransMAR-GAN is a transformer-based adversarial network for reducing metal artifacts in CT images. The model combines an NGswin generator (based on N-gram Swin Transformers) with a multi-scale PatchGAN discriminator and metal-aware loss functions.

## Repository Structure

```
TransMAR-GAN/
├── models/                    # Model architectures
│   ├── generator/            # NGswin generator and components
│   ├── discriminator/        # Multi-scale and single-scale discriminators
│   └── baseline/             # Baseline models for comparison
├── losses/                    # Loss function implementations
├── data/                      # Dataset classes and preprocessing
├── training/                  # Training scripts
├── testing/                   # Testing and inference scripts
├── evaluation/                # Evaluation and benchmark comparison
├── scripts/                   # Utility scripts (ablation, figure generation)
├── configs/                   # Configuration files
├── checkpoints/               # Pre-trained model weights
├── results/                   # Experimental results
├── docs/                      # Documentation
└── external/                  # External dependencies (torch-radon)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mpat247/TransMAR-GAN.git
cd TransMAR-GAN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install torch-radon for physics loss (optional)
cd external/torch-radon
python setup.py install
cd ../..
```

## Pre-trained Models

Pre-trained weights are available in the `checkpoints/` directory:

- `best_model_syndeeplesion.pth` - Trained on SynDeepLesion dataset
- `best_finetuned_spineweb.pth` - Fine-tuned on SpineWeb dataset

## Usage

### Inference

```python
from models.generator.ngswin import NGswin
import torch

# Load model
model = NGswin()
checkpoint = torch.load('checkpoints/best_model_syndeeplesion.pth')
model.load_state_dict(checkpoint['generator_state_dict'])
model.eval()
```

### Training

Train on SynDeepLesion:

```bash
python training/train_combined.py --config configs/train_syndeeplesion.yaml
```

Fine-tune on SpineWeb:

```bash
python scripts/finetune_all_benchmarks.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --config configs/finetune_spineweb.yaml
```

### Ablation Studies

```bash
python scripts/run_ablation_studies.py --output_dir results/ablation_studies
```

## Model Architecture

### Generator

The generator uses an NGswin architecture based on N-gram Swin Transformers with:
- Multi-scale hierarchical feature extraction
- Window-based self-attention
- Skip connections for detail preservation

### Discriminator

The discriminator is a multi-scale PatchGAN with:
- Three discrimination scales (1x, 1/2x, 1/4x)
- Spectral normalization for training stability
- Feature matching loss

### Loss Functions

- Metal-aware weighted L1 loss
- Metal-aware edge loss
- Physics-consistency loss (sinogram domain)
- Metal consistency loss
- Adversarial loss with feature matching

## Model Variants

Training scripts for different model configurations:

```bash
# Baseline (MSE only)
python scripts/train_model_variants.py baseline

# With adversarial loss
python scripts/train_model_variants.py v1

# With multi-scale discriminator
python scripts/train_model_variants.py v2

# With feature matching
python scripts/train_model_variants.py v3

# With metal-aware reconstruction loss
python scripts/train_model_variants.py v4

# With metal-aware edge loss
python scripts/train_model_variants.py v5

# Full model (all losses)
python scripts/train_model_variants.py full
```

## Evaluation

Test on clinical data:

```bash
python testing/test_finetuned_model.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --data_path /path/to/data \
    --output_dir results/
```

Benchmark comparison:

```bash
python evaluation/benchmark_comparison_syndeeplesion.py
python evaluation/finetuned_comparison_spineweb.py
```

## Documentation

- [Installation Guide](docs/INSTALL.md)
- [Training Guide](docs/TRAINING.md)
- [Testing Guide](docs/TESTING.md)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (for GPU training)

See `requirements.txt` for the full list of dependencies.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- NGswin architecture based on [SwinIR](https://github.com/JingyunLiang/SwinIR)
- Multi-Scale PatchGAN inspired by [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
- SynDeepLesion dataset from [DeepLesion](https://nihcc.app.box.com/v/DeepLesion)
- SpineWeb dataset from UW Spine Research
