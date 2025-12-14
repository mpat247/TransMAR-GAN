#!/usr/bin/env python3
"""
Figure 7: Error Heatmaps

Purpose:
    Show the spatial distribution of reconstruction error for different methods.

Data and Methods:
    - Use synthetic cases with ground-truth
    - Compare: Baseline MSE, w/o Physics, w/o Metal, SGAMARN (Full Model)

Panel Layout:
    - Top row: Reconstructed images
    - Bottom row: |x - x_hat| error maps (5x scaled) for each method

Implementation Notes:
    - Use hot/jet colormap for error visualization
    - 5x intensity scaling for visibility
    - Save individual panels and composite figure

Author: NgSwinGAN Paper
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared config
from shared_config import (
    PATHS,
    get_device,
    load_test_dataset,
    get_selected_slices,
    load_generator,
    denormalize,
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    'output_dir': 'figure7_error_heatmaps',
    
    # Model checkpoints to compare
    'models': {
        'baseline_mse': {
            'name': 'Baseline (MSE Only)',
            'short_name': 'MSE Only',
            'path': '/home/grad/mppatel/Documents/DCGAN/ablation_results/loss_ablations_20251207_155318/A0_mse_only/checkpoints/best_model.pth',
        },
        'no_physics': {
            'name': 'w/o Physics Loss',
            'short_name': 'w/o Physics',
            'path': '/home/grad/mppatel/Documents/DCGAN/ablation_results/loss_ablations_20251207_155318/A1_no_physics/checkpoints/best_model.pth',
        },
        'no_metal': {
            'name': 'w/o Metal Consistency',
            'short_name': 'w/o Metal',
            'path': '/home/grad/mppatel/Documents/DCGAN/ablation_results/loss_ablations_20251207_155318/A2_no_metal_consistency/checkpoints/best_model.pth',
        },
        'full_model': {
            'name': 'TransMAR-GAN (Full Model)',
            'short_name': 'TransMAR-GAN',
            'path': '/home/grad/mppatel/Documents/DCGAN/combined_results/run_20251202_211759/checkpoints/best_model.pth',
        },
    },
    
    # Visualization settings
    'error_scale': 5.0,       # 5x intensity scaling
    'error_cmap': 'jet',      # Colormap for error maps
    'error_vmax': 1.0,        # Max value for error colormap (after scaling)
    'dpi': 300,
    'figsize_composite': (16, 8),  # 4 columns x 2 rows
    'figsize_individual': (6, 6),
    
    # Number of example slices
    'num_examples': 10,
}

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, 'figure7_error_heatmaps.log')
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('figure7')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_all_models(device, logger):
    """Load all models for comparison."""
    models = {}
    
    for model_id, model_info in CONFIG['models'].items():
        checkpoint_path = model_info['path']
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"  Checkpoint not found for {model_id}: {checkpoint_path}")
            continue
        
        logger.info(f"  Loading {model_id}...")
        try:
            model = load_generator(checkpoint_path, device)
            models[model_id] = {
                'model': model,
                'name': model_info['name'],
                'short_name': model_info['short_name'],
            }
            logger.info(f"    ✓ Loaded successfully")
        except Exception as e:
            logger.error(f"    ✗ Failed to load: {e}")
    
    return models

# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, input_tensor, device):
    """Run inference on a single input."""
    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    output = model(input_tensor)
    return output

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def tensor_to_display(tensor):
    """Convert tensor from [-1,1] to [0,1] for display."""
    img = tensor.squeeze().cpu().numpy()
    return np.clip((img + 1) / 2, 0, 1)


def compute_error_map(pred, gt, scale=5.0):
    """
    Compute absolute error map with scaling.
    
    Args:
        pred: Prediction tensor in [-1, 1]
        gt: Ground truth tensor in [-1, 1]
        scale: Intensity scaling factor
    
    Returns:
        Error map in [0, 1] range (after scaling and clipping)
    """
    pred_np = tensor_to_display(pred)
    gt_np = tensor_to_display(gt)
    
    error = np.abs(pred_np - gt_np)
    error_scaled = error * scale
    error_scaled = np.clip(error_scaled, 0, 1)
    
    return error_scaled


def create_composite_figure(results, slice_idx, output_dir, input_tensor, gt_tensor, logger=None):
    """
    Create composite figure with input, reconstructions, and error maps.
    
    Layout:
        Row 0: Input (corrupted) + Reconstructed images from each method
        Row 1: Input error map (|GT - corrupted|) + Error maps (5x scaled) for each method
        Bottom: Metrics (PSNR, SSIM, MAE) for each method
    """
    n_methods = len(results)
    n_cols = n_methods + 1  # +1 for input column
    
    # Create figure: 2 rows, n_cols columns (extra space at bottom for metrics)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))
    
    # Get input image and ground truth
    input_np = tensor_to_display(input_tensor)
    gt_np = tensor_to_display(gt_tensor)
    
    # Compute corrupted-GT difference map (shows artifact locations)
    input_error = np.abs(input_np - gt_np) * CONFIG['error_scale']
    input_error = np.clip(input_error, 0, 1)
    
    # Column 0: Input (corrupted) image
    axes[0, 0].imshow(input_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Corrupted Input', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Column 0, Row 1: Corrupted-GT difference map (shows where artifacts are)
    axes[1, 0].imshow(input_error, cmap=CONFIG['error_cmap'], vmin=0, vmax=CONFIG['error_vmax'])
    axes[1, 0].axis('off')
    
    method_order = ['baseline_mse', 'no_physics', 'no_metal', 'full_model']
    ordered_results = [(k, results[k]) for k in method_order if k in results]
    
    for col_idx, (model_id, data) in enumerate(ordered_results):
        col = col_idx + 1  # Shift by 1 for input column
        pred_np = data['pred_display']
        error_np = data['error_map']
        short_name = data['short_name']
        psnr_val = data['psnr']
        ssim_val = data['ssim']
        mae_val = data['mae']
        
        # Row 0: Reconstruction
        axes[0, col].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(short_name, fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
        
        # Row 1: Error map (5x scaled)
        im = axes[1, col].imshow(error_np, cmap=CONFIG['error_cmap'], 
                                  vmin=0, vmax=CONFIG['error_vmax'])
        axes[1, col].axis('off')
        
        # Add metrics below error map
        metrics_text = f'PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}\nMAE: {mae_val:.4f}'
        axes[1, col].text(0.5, -0.08, metrics_text, transform=axes[1, col].transAxes,
                          ha='center', va='top', fontsize=10, fontfamily='monospace',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Tight layout with extra space at bottom for metrics
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.08, left=0.06, bottom=0.12)
    
    # Save
    composite_path = os.path.join(output_dir, f'error_heatmap_slice_{slice_idx}.png')
    plt.savefig(composite_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    pdf_path = composite_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    
    plt.close()
    
    if logger:
        logger.info(f"  Saved composite: error_heatmap_slice_{slice_idx}.png")
    
    return composite_path


def save_individual_panels(results, gt_tensor, input_tensor, slice_idx, output_dir, logger=None):
    """Save each panel as individual high-res image."""
    
    panel_dir = os.path.join(output_dir, 'individual_panels', f'slice_{slice_idx}')
    os.makedirs(panel_dir, exist_ok=True)
    
    gt_np = tensor_to_display(gt_tensor)
    input_np = tensor_to_display(input_tensor)
    
    # Compute corrupted-GT error map (artifact locations)
    input_error = np.abs(input_np - gt_np) * CONFIG['error_scale']
    input_error = np.clip(input_error, 0, 1)
    
    # Save ground truth
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    ax.imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(panel_dir, 'ground_truth.png'),
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save input (corrupted)
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    ax.imshow(input_np, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(panel_dir, 'input_corrupted.png'),
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save input error map (corrupted - GT)
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    ax.imshow(input_error, cmap=CONFIG['error_cmap'], vmin=0, vmax=CONFIG['error_vmax'])
    ax.axis('off')
    plt.savefig(os.path.join(panel_dir, 'input_error_map.png'),
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save each method's outputs
    for model_id, data in results.items():
        pred_np = data['pred_display']
        error_np = data['error_map']
        
        # Save reconstruction
        fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
        ax.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        plt.savefig(os.path.join(panel_dir, f'{model_id}_reconstruction.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save error map
        fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
        ax.imshow(error_np, cmap=CONFIG['error_cmap'], vmin=0, vmax=CONFIG['error_vmax'])
        ax.axis('off')
        plt.savefig(os.path.join(panel_dir, f'{model_id}_error_map.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save numpy arrays
        np.save(os.path.join(panel_dir, f'{model_id}_reconstruction.npy'), pred_np)
        np.save(os.path.join(panel_dir, f'{model_id}_error_map.npy'), error_np)
    
    # Save ground truth and input numpy
    np.save(os.path.join(panel_dir, 'ground_truth.npy'), gt_np)
    np.save(os.path.join(panel_dir, 'input_corrupted.npy'), input_np)
    np.save(os.path.join(panel_dir, 'input_error_map.npy'), input_error)
    
    if logger:
        logger.debug(f"    Saved individual panels to: individual_panels/slice_{slice_idx}/")


def create_summary_figure(all_results, output_dir, logger=None):
    """
    Create a summary figure showing average error metrics across all slices.
    """
    # Compute average metrics per method
    method_metrics = {}
    
    for slice_idx, slice_results in all_results.items():
        for model_id, data in slice_results.items():
            if model_id not in method_metrics:
                method_metrics[model_id] = {
                    'mae': [],
                    'short_name': data['short_name'],
                }
            method_metrics[model_id]['mae'].append(data['mae'])
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    method_order = ['baseline_mse', 'no_physics', 'no_metal', 'full_model']
    ordered_methods = [(k, method_metrics[k]) for k in method_order if k in method_metrics]
    
    names = [m[1]['short_name'] for m in ordered_methods]
    mae_means = [np.mean(m[1]['mae']) for m in ordered_methods]
    mae_stds = [np.std(m[1]['mae']) for m in ordered_methods]
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']  # Red, Orange, Blue, Green
    
    bars = ax.bar(names, mae_means, yerr=mae_stds, capsize=5, color=colors, edgecolor='black')
    
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title('Reconstruction Error Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    
    # Add value labels on bars
    for bar, mean in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'error_summary.png')
    plt.savefig(summary_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.savefig(summary_path.replace('.png', '.pdf'), dpi=CONFIG['dpi'], 
                bbox_inches='tight', format='pdf')
    plt.close()
    
    if logger:
        logger.info(f"  Saved summary figure: error_summary.png")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("FIGURE 7: Error Heatmaps")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    
    # Log configuration
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Error scale: {CONFIG['error_scale']}x")
    logger.info(f"  Error colormap: {CONFIG['error_cmap']}")
    logger.info(f"  Number of examples: {CONFIG['num_examples']}")
    
    logger.info("")
    logger.info("Models to compare:")
    for model_id, model_info in CONFIG['models'].items():
        logger.info(f"  {model_info['short_name']}: {model_info['path']}")
    
    # Get device
    device = get_device()
    logger.info(f"\nUsing device: {device}")
    
    # Load all models
    logger.info("")
    logger.info("Loading models...")
    models = load_all_models(device, logger)
    
    if len(models) == 0:
        logger.error("No models loaded! Exiting.")
        return
    
    logger.info(f"  Loaded {len(models)} models")
    
    # Load test dataset
    logger.info("")
    logger.info("Loading test dataset...")
    test_dataset = load_test_dataset()
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Get selected slices
    selected_indices = get_selected_slices()
    logger.info(f"  Using {len(selected_indices)} selected slices")
    
    # Statistics
    all_results = {}
    processed = 0
    skipped = 0
    
    # Process slices
    logger.info("")
    logger.info(f"Processing {CONFIG['num_examples']} example slices...")
    
    for i, idx in enumerate(selected_indices[:CONFIG['num_examples']]):
        logger.info("")
        logger.info(f"Processing slice {idx} ({i+1}/{CONFIG['num_examples']})...")
        
        # Get data
        batch = test_dataset[idx]
        input_tensor = batch[0].unsqueeze(0).to(device)  # Corrupted CT [1, 1, H, W]
        gt_tensor = batch[1].unsqueeze(0).to(device)      # Ground truth [1, 1, H, W]
        
        # Run inference for each model
        slice_results = {}
        
        for model_id, model_data in models.items():
            model = model_data['model']
            
            # Run inference
            pred_tensor = run_inference(model, input_tensor, device)
            
            # Compute metrics
            pred_np = tensor_to_display(pred_tensor)
            gt_np = tensor_to_display(gt_tensor)
            
            error_map = compute_error_map(pred_tensor, gt_tensor, scale=CONFIG['error_scale'])
            mae = np.mean(np.abs(pred_np - gt_np))
            psnr_val = psnr_func(gt_np, pred_np, data_range=1.0)
            ssim_val = ssim_func(gt_np, pred_np, data_range=1.0)
            
            slice_results[model_id] = {
                'pred_tensor': pred_tensor,
                'pred_display': pred_np,
                'error_map': error_map,
                'mae': mae,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'short_name': model_data['short_name'],
            }
            
            logger.debug(f"    {model_data['short_name']}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, MAE={mae:.6f}")
        
        all_results[idx] = slice_results
        
        # Create composite figure
        create_composite_figure(slice_results, idx, output_dir, input_tensor, gt_tensor, logger)
        
        # Save individual panels
        save_individual_panels(slice_results, gt_tensor, input_tensor, idx, output_dir, logger)
        
        processed += 1
    
    # Create summary figure
    logger.info("")
    logger.info("Creating summary figure...")
    create_summary_figure(all_results, output_dir, logger)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'figure': 'Figure 7: Error Heatmaps',
            'methods': {k: v['name'] for k, v in CONFIG['models'].items()},
            'error_scale': CONFIG['error_scale'],
            'num_processed': processed,
            'selected_indices': [int(x) for x in selected_indices[:CONFIG['num_examples']]],
        }, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("FIGURE 7 GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {processed} slices")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - error_heatmap_slice_*.png/pdf: Composite figures")
    logger.info("  - individual_panels/slice_*/: Individual high-res panels")
    logger.info("  - error_summary.png/pdf: Average error comparison")
    logger.info("  - config.json")
    logger.info("  - figure7_error_heatmaps.log")


if __name__ == "__main__":
    main()
