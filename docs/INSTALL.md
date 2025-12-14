# Installation Guide

## System Requirements

- **OS**: Linux (Ubuntu 18.04+), macOS, or Windows 10+
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3090 or better)
- **CUDA**: 11.0 or higher
- **Python**: 3.8 or higher
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for datasets and results

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/TransMAR-GAN.git
cd TransMAR-GAN
```

## Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n transmargan python=3.8
conda activate transmargan
```

## Step 3: Install PyTorch

Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Check installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Install torch-radon (Optional, for Physics Loss)

```bash
cd external/torch-radon
python setup.py install
cd ../..

# Verify installation
python -c "from torch_radon import Radon; print('torch-radon installed successfully')"
```

## Step 6: Download Datasets

### SynDeepLesion Dataset

1. Download from [DeepLesion website](https://nihcc.app.box.com/v/DeepLesion)
2. Extract to `/path/to/SynDeepLesion/`
3. Update data path in `configs/train_syndeeplesion.yaml`

### SpineWeb Dataset

1. Request access from UW Spine Research
2. Extract to `/path/to/UWSpine_adn/`
3. Update data path in `configs/finetune_spineweb.yaml`

## Step 7: Download Pre-trained Models (Optional)

Download pre-trained checkpoints:

```bash
# Download from Google Drive/Zenodo
wget [link-to-pretrained-models] -O checkpoints.zip
unzip checkpoints.zip -d checkpoints/
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config files
- Use gradient accumulation
- Use mixed precision training

### torch-radon Installation Issues
- Ensure CUDA toolkit is installed
- Check compiler compatibility
- See [torch-radon documentation](https://github.com/matteo-ronchetti/torch-radon)

### ImportError: No module named 'timm'
```bash
pip install timm
```

## Verification

Run quick test to verify installation:

```bash
python -c "
from models.generator.ngswin import NGswin
from models.discriminator.ms_patchgan import MultiScaleDiscriminator
print('Installation successful!')
"
```
