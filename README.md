# TransMAR-GAN: Transformer-based Multi-Scale Adversarial Reconstruction GAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **TransMAR-GAN** for CT metal artifact reduction, submitted to IEEE Transactions on Biomedical Engineering (TBME).

## 🔥 Highlights

- **NGswin Generator**: N-gram Swin Transformer-based generator for high-quality artifact reduction
- **Multi-Scale PatchGAN Discriminator**: Three-scale discriminator with spectral normalization
- **Metal-Aware Loss Functions**: Physics-consistent losses tailored for metal artifact reduction
- **Comprehensive Ablation Studies**: All model variants and loss function combinations
- **State-of-the-Art Performance**: Outperforms existing methods on SynDeepLesion and SpineWeb datasets

## 📁 Repository Structure

```
TransMAR-GAN/
├── models/                    # Model architectures
│   ├── generator/            # NGswin generator and components
│   ├── discriminator/        # Multi-scale and single-scale discriminators
│   ├── baseline/             # Baseline models for comparison
│   └── variants/             # Model variants for ablation
├── losses/                    # Loss function implementations
├── data/                      # Dataset classes and preprocessing
├── training/                  # Training scripts (single-GPU, DDP, DP)
├── testing/                   # Testing and inference scripts
├── evaluation/                # Evaluation and benchmark comparison
├── scripts/                   # Utility scripts (ablation, figure generation)
├── configs/                   # Configuration files
├── checkpoints/               # Pre-trained model weights
├── results/                   # Experimental results
│   ├── ablation_studies/     # Ablation study results
│   ├── benchmark_comparisons/ # Comparison with other methods
│   ├── figures/              # Paper figures
│   └── metrics/              # Quantitative metrics
├── docs/                      # Documentation
└── external/                  # External dependencies (torch-radon)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TransMAR-GAN.git
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

### Download Pre-trained Models

Download pre-trained weights from [Google Drive/Zenodo link] and place them in `checkpoints/`:

```bash
checkpoints/
├── best_model_syndeeplesion.pth     # Trained on SynDeepLesion
└── best_model_spineweb_finetuned.pth # Fine-tuned on SpineWeb
```

### Quick Inference

```python
from models.generator.ngswin import NGswin
from testing.inference_test import load_model, run_inference

# Load model
model = load_model('checkpoints/best_model_syndeeplesion.pth')

# Run inference
output = run_inference(model, 'path/to/artifact_image.png')
```

## 📊 Training

### Train on SynDeepLesion (Full Model)

```bash
# Single GPU
python training/train_combined.py --config configs/train_syndeeplesion.yaml

# Multi-GPU (DDP)
python training/engines/ddp_train.py --gpus_per_node 4 --config configs/train_syndeeplesion.yaml

# Multi-GPU (DataParallel)
python training/engines/dp_train.py --config configs/train_syndeeplesion.yaml
```

### Fine-tune on SpineWeb

```bash
python training/engines/ddp_finetune.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --config configs/finetune_spineweb.yaml
```

### Run Ablation Studies

```bash
# Run all loss function ablations
python scripts/run_ablation_studies.py --output_dir results/ablation_studies

# Results saved to results/ablation_studies/
```

## 🧪 Model Variants

This repository includes multiple model configurations:

### Generator Options
- **NGswin** (default): N-gram Swin Transformer generator

### Discriminator Options
- **Multi-Scale PatchGAN** (default): 3-scale discriminator with spectral normalization
- **Single-Scale PatchGAN**: Single-scale baseline
- **DCGAN-style**: Simple convolutional discriminator

### Training Variants

```bash
# Baseline: NGswin + Single-scale + MSE only
python scripts/train_model_variants.py baseline

# V1: NGswin + Single-scale + Adversarial
python scripts/train_model_variants.py v1

# V2: NGswin + MS-PatchGAN + Adversarial
python scripts/train_model_variants.py v2

# V3: V2 + Feature Matching
python scripts/train_model_variants.py v3

# V4: V3 + Metal-Aware Reconstruction (Eq 3)
python scripts/train_model_variants.py v4

# V5: V4 + Metal-Aware Edge (Eq 4)
python scripts/train_model_variants.py v5

# Full Model: All Losses
python scripts/train_model_variants.py full
```

### Loss Function Combinations
See `scripts/run_ablation_studies.py` for all tested combinations:
- A0: MSE only (baseline)
- A1: Without physics loss
- A2: Without metal consistency loss
- A3: Without metal-aware weighting
- A4: Without adversarial loss
- A5: Without feature matching
- A6: Without edge loss
- A7: Hinge GAN loss
- A8: Vanilla GAN loss
- B1: Single-scale discriminator
- B2: No spectral normalization
- B3: Different dilation radii (r=0,3,5,7)

## 📈 Evaluation

### Test on Clinical Data

```bash
python testing/test_finetuned_model.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --data_path /path/to/clinical/data \
    --output_dir results/clinical_evaluation
```

### Benchmark Comparison

```bash
# Compare with other methods on SynDeepLesion
python evaluation/compare_benchmarks_syndeeplesion.py

# Compare on SpineWeb
python evaluation/compare_benchmarks_spineweb.py
```

## 📊 Results

### Quantitative Results (SynDeepLesion)

| Method | PSNR ↑ | SSIM ↑ | MAE ↓ |
|--------|--------|--------|-------|
| DICDNet | XX.XX | X.XXX | X.XXX |
| InDuDoNet+ | XX.XX | X.XXX | X.XXX |
| **TransMAR-GAN (Ours)** | **XX.XX** | **X.XXX** | **X.XXX** |

### Ablation Study Results

See `results/ablation_studies/` for detailed metrics on each configuration.

## 🔧 Key Features

### NGswin Generator
- N-gram context in Swin Transformer blocks
- Multi-scale hierarchical feature extraction
- Efficient window-based self-attention
- Skip connections for detail preservation

### Multi-Scale PatchGAN Discriminator
- Three discrimination scales (1×, 1/2×, 1/4×)
- Spectral normalization for training stability
- Feature matching loss for perceptual quality

### Metal-Aware Losses
- **Metal-aware weighted L1**: Emphasizes regions near metal (Eq. 3)
- **Metal-aware edge loss**: Preserves tissue boundaries (Eq. 4)
- **Physics-consistency loss**: Enforces CT physics in sinogram domain (Eq. 6)
- **Metal consistency loss**: Accurate reconstruction in metal regions

## 📚 Documentation

- [Installation Guide](docs/INSTALL.md)
- [Training Guide](docs/TRAINING.md)
- [Testing Guide](docs/TESTING.md)
- [SpineWeb Implementation Notes](docs/SPINEWEB_NOTES.md)

## 🎯 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yourname2025transmargan,
  title={TransMAR-GAN: Transformer-based Multi-Scale Adversarial Reconstruction GAN for Metal Artifact Reduction},
  author={Your Name and Co-authors},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NGswin architecture based on [SwinIR](https://github.com/JingyunLiang/SwinIR)
- Multi-Scale PatchGAN inspired by [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
- SynDeepLesion dataset from [DeepLesion](https://nihcc.app.box.com/v/DeepLesion)
- SpineWeb dataset from UW Spine Research

## 📧 Contact

For questions and feedback, please contact: your.email@university.edu

---

**Status**: Submitted to IEEE TBME (December 2025)
