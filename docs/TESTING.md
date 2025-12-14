# Testing and Evaluation Guide

## Quick Inference

### Single Image

```python
import torch
from models.generator.ngswin import NGswin
from PIL import Image
import numpy as np

# Load model
model = NGswin()
checkpoint = torch.load('checkpoints/best_model_syndeeplesion.pth')
model.load_state_dict(checkpoint['netG_state_dict'])
model.eval()
model.cuda()

# Load and preprocess image
from data.datasets import normalize
image = Image.open('artifact_image.png').convert('L')
image = np.array(image)
image_tensor = normalize(image, (0.0, 1.0))
image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    output = model(image_tensor)

# Save result
output_np = output.cpu().numpy().squeeze()
output_np = (output_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
output_img = Image.fromarray((output_np * 255).astype(np.uint8))
output_img.save('result.png')
```

### Batch Inference

```bash
python testing/inference_test.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --input_dir /path/to/artifact/images \
    --output_dir results/inference_outputs
```

## Evaluate on Test Set

### SynDeepLesion

```bash
python testing/test_finetuned_model.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --data_path /path/to/SynDeepLesion/test \
    --output_dir results/syndeeplesion_test \
    --save_images
```

### SpineWeb

```bash
python testing/test_finetuned_model.py \
    --checkpoint checkpoints/best_model_spineweb_finetuned.pth \
    --data_path /path/to/UWSpine_adn/test \
    --output_dir results/spineweb_test
```

## Compute Metrics

```bash
python evaluation/compute_metrics.py \
    --predictions results/syndeeplesion_test/predictions \
    --ground_truth /path/to/ground/truth \
    --output results/syndeeplesion_test/metrics.json
```

Computed metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Metal-region specific metrics

## Benchmark Comparison

### Compare with Baselines

```bash
# SynDeepLesion comparison
python evaluation/compare_benchmarks_syndeeplesion.py \
    --our_model checkpoints/best_model_syndeeplesion.pth \
    --data_path /path/to/SynDeepLesion/test \
    --output_dir results/benchmark_comparison

# SpineWeb comparison
python evaluation/compare_benchmarks_spineweb.py \
    --our_model checkpoints/best_model_spineweb_finetuned.pth \
    --data_path /path/to/UWSpine_adn/test \
    --output_dir results/spineweb_comparison
```

This compares against:
- DICDNet
- InDuDoNet
- InDuDoNet+
- FIND-Net
- MEPNet
- ADN (baseline)

## Distributed Testing (Multi-GPU)

```bash
python testing/ddp_test.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --data_path /path/to/test/data \
    --gpus_per_node 4 \
    --output_dir results/test_ddp
```

## Generate Paper Figures

### Figure 1: MSE Limitation

```bash
python scripts/figure_generation/figure1_mse_limitation.py \
    --output_dir results/figures/figure1
```

### Figure 2: Physics Consistency

```bash
python scripts/figure_generation/figure2_physics_consistency_v2.py \
    --checkpoint checkpoints/best_model_syndeeplesion.pth \
    --output_dir results/figures/figure2
```

### Figure 3: Metal-Aware Masks

```bash
python scripts/figure_generation/figure3_metal_aware_masks.py \
    --output_dir results/figures/figure3
```

### Figure 4: Ablation Visualization

```bash
python scripts/figure_generation/figure4_ablation_comparison.py \
    --ablation_dir results/ablation_studies \
    --output_dir results/figures/figure4
```

## Export Results Table

Generate LaTeX table for paper:

```bash
python evaluation/export_results_table.py \
    --results_dir results/benchmark_comparison \
    --output results/benchmark_table.tex
```
