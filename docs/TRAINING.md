# Training Guide

## Table of Contents
1. [Dataset Preparation](#dataset-preparation)
2. [Training Configuration](#training-configuration)
3. [Single-GPU Training](#single-gpu-training)
4. [Multi-GPU Training](#multi-gpu-training)
5. [Fine-tuning](#fine-tuning)
6. [Ablation Studies](#ablation-studies)
7. [Monitoring Training](#monitoring-training)

## Dataset Preparation

### SynDeepLesion

Prepare the SynDeepLesion dataset:

```bash
python scripts/prepare_data.py \
    --dataset syndeeplesion \
    --input_dir /path/to/raw/data \
    --output_dir /path/to/SynDeepLesion
```

### SpineWeb

Prepare the SpineWeb dataset:

```bash
python scripts/prepare_spineweb_data.py \
    --artifact_dir /path/to/metal_artifacts \
    --clean_dir /path/to/clean_images \
    --output_dir /path/to/UWSpine_adn
```

## Training Configuration

Edit `configs/train_syndeeplesion.yaml`:

```yaml
model:
  generator:
    type: NGswin
    embed_dim: 64
    depths: [6, 4, 4]
    num_heads: [6, 4, 4]
    window_size: 8
    ngrams: [2, 2, 2, 2]
  
  discriminator:
    type: MultiScalePatchGAN
    num_scales: 3
    base_channels: 64
    use_spectral_norm: true

training:
  num_epochs: 100
  batch_size: 4
  patch_size: 128
  
  optimizer:
    generator:
      lr: 1.0e-4
      betas: [0.5, 0.999]
    discriminator:
      lr: 2.0e-4  # TTUR: 2x generator LR
      betas: [0.5, 0.999]
  
  loss_weights:
    adversarial: 0.1
    feature_matching: 10.0
    reconstruction: 1.0
    edge: 0.2
    physics: 0.02
    metal_consistency: 0.5
  
  metal_aware:
    threshold: 0.6
    dilation_radius: 5
    beta_weight: 1.0
    w_max: 3.0

data:
  train_path: "/path/to/SynDeepLesion/"
  patch_size: 128
  workers: 4
  
checkpointing:
  save_every: 5
  keep_last: 3
```

## Single-GPU Training

### Full Model (All Losses)

```bash
python training/train_combined.py \
    --config configs/train_syndeeplesion.yaml \
    --output_dir results/full_model \
    --gpu 0
```

### Baseline (DCGAN Discriminator)

```bash
python training/train_baseline_variants.py \
    --config configs/train_baseline.yaml \
    --output_dir results/dcgan_baseline \
    --gpu 0
```

### Model Variants

Train specific model variants with different components:

```bash
# Baseline: NGswin + Single-scale + MSE only
python scripts/train_model_variants.py baseline --gpu 0

# V1: NGswin + Single-scale + Adversarial
python scripts/train_model_variants.py v1 --gpu 0

# V2: NGswin + MS-PatchGAN + Adversarial  
python scripts/train_model_variants.py v2 --gpu 0

# V3: NGswin + MS-PatchGAN + Adv + Feature Matching
python scripts/train_model_variants.py v3 --gpu 0

# V4: NGswin + MS-PatchGAN + Adv + FM + Metal-aware Reconstruction
python scripts/train_model_variants.py v4 --gpu 0

# V5: NGswin + MS-PatchGAN + Adv + FM + MAR + Edge
python scripts/train_model_variants.py v5 --gpu 0

# Full Model: All components
python scripts/train_model_variants.py full --gpu 0
```

## Multi-GPU Training

### DistributedDataParallel (Recommended)

```bash
# 4 GPUs on single node
python training/engines/ddp_train.py \
    --total_nodes 1 \
    --gpus_per_node 4 \
    --node_rank 0 \
    --ip_address localhost \
    --backend nccl \
    --config configs/train_syndeeplesion.yaml

# Multi-node setup (e.g., 2 nodes, 4 GPUs each)
# Node 0:
python training/engines/ddp_train.py \
    --total_nodes 2 \
    --gpus_per_node 4 \
    --node_rank 0 \
    --ip_address 192.168.1.100 \
    --backend nccl

# Node 1:
python training/engines/ddp_train.py \
    --total_nodes 2 \
    --gpus_per_node 4 \
    --node_rank 1 \
    --ip_address 192.168.1.100 \
    --backend nccl
```

### DataParallel

```bash
python training/engines/dp_train.py \
    --gpus 0,1,2,3 \
    --config configs/train_syndeeplesion.yaml
```

## Fine-tuning

### Fine-tune on SpineWeb

```bash
python training/engines/ddp_finetune.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --config configs/finetune_spineweb.yaml \
    --gpus_per_node 4 \
    --output_dir results/spineweb_finetuned
```

Configuration for fine-tuning (`configs/finetune_spineweb.yaml`):

```yaml
training:
  num_epochs: 25  # Fewer epochs for fine-tuning
  batch_size: 4
  
  optimizer:
    generator:
      lr: 1.0e-5  # Lower LR for fine-tuning (10x smaller)
    discriminator:
      lr: 2.0e-5

data:
  train_path: "/path/to/UWSpine_adn/train/"
  test_path: "/path/to/UWSpine_adn/test/"
```

## Ablation Studies

Run all loss ablation experiments:

```bash
python scripts/run_ablation_studies.py \
    --output_dir results/ablation_studies \
    --num_epochs 10 \
    --gpus_per_node 4
```

This will run all configurations:
- A0: MSE only
- A1-A8: Various loss combinations
- B1-B2: Discriminator ablations
- B3: Dilation radius ablations

Results saved to `results/ablation_studies/`

## Monitoring Training

### TensorBoard

Launch TensorBoard:

```bash
tensorboard --logdir results/full_model/tb --port 6006
```

View at: http://localhost:6006

### Check Training Progress

```bash
# View latest loss
tail -f results/full_model/training.log

# Plot loss curves
python utils/plot_training_curves.py \
    --input results/full_model/training_history.csv \
    --output results/full_model/loss_curves.png
```

## Resume Training

Resume from checkpoint:

```bash
python training/train_combined.py \
    --config configs/train_syndeeplesion.yaml \
    --resume results/full_model/checkpoints/latest.pth \
    --output_dir results/full_model
```

## Advanced Options

### Mixed Precision Training

```bash
python training/train_combined.py \
    --config configs/train_syndeeplesion.yaml \
    --amp \
    --output_dir results/full_model_fp16
```

### Gradient Accumulation

For effective larger batch sizes:

```bash
python training/train_combined.py \
    --config configs/train_syndeeplesion.yaml \
    --batch_size 2 \
    --accumulation_steps 4  # Effective batch size: 2*4=8
```

## Tips for Best Results

1. **Start with pre-training on SynDeepLesion** (100 epochs)
2. **Fine-tune on SpineWeb** with lower learning rate (25 epochs)
3. **Use TTUR**: Discriminator LR = 2× Generator LR
4. **Monitor discriminator**: If D_loss → 0, reduce lambda_adv
5. **Check physics loss**: Should decrease steadily, not oscillate
6. **Save checkpoints frequently**: Best model may not be the last epoch
