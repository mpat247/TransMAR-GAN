"""
Figure 1: Limitation of Pixel-Wise Loss Only

Purpose:
    Demonstrate that training with only MSE/MAE loss leads to blurry or 
    residual-artifact solutions, motivating the multi-term loss.

Comparison:
    (A) Ground Truth clean image
    (B) MSE-only model output
    (C) Full SGAMARN model output (all losses)

Outputs:
    - Main grid figure (25 slices × 3 columns)
    - Individual slice comparisons with zoom ROIs
    - Error maps showing where MSE fails
    - Intensity profiles through metal regions
    - Metrics comparison (PSNR, SSIM per slice)
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torchvision.utils as vutils
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared config
from shared_config import (
    PATHS, MODEL_CONFIG, VIS_CONFIG,
    denormalize, normalize_to_model, get_device, load_generator,
    load_test_dataset, get_selected_slices, check_checkpoint_exists,
    print_checkpoint_status
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - UPDATE THESE PATHS WHEN CHECKPOINTS ARE READY
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Checkpoint paths - UPDATE WHEN READY
    'mse_only_checkpoint': PATHS['checkpoints'].get('A0_mse_only'),  # Set when ablation finishes
    'sgamarn_checkpoint': PATHS['checkpoints'].get('A0_baseline'),   # Full model with all losses
    
    # Output settings
    'output_dir': os.path.join(PATHS['output_root'], 'figure1_mse_limitation'),
    
    # Visualization settings
    'cmap': VIS_CONFIG['cmap'],
    'vmin': VIS_CONFIG['vmin'],
    'vmax': VIS_CONFIG['vmax'],
    'dpi': VIS_CONFIG['dpi'],
    'figsize_grid': VIS_CONFIG['figsize_grid'],
    'figsize_individual': VIS_CONFIG['figsize_individual'],
    
    # Metal detection for ROI
    'metal_threshold': VIS_CONFIG['metal_threshold'],
}

def compute_metrics(pred, target):
    """Compute PSNR and SSIM between prediction and target."""
    pred_np = pred.squeeze().cpu().numpy() if torch.is_tensor(pred) else pred.squeeze()
    target_np = target.squeeze().cpu().numpy() if torch.is_tensor(target) else target.squeeze()
    
    # Clip to valid range
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    
    return {
        'PSNR': float(psnr_val),
        'SSIM': float(ssim_val),
        'MSE': float(np.mean((pred_np - target_np) ** 2)),
        'MAE': float(np.mean(np.abs(pred_np - target_np))),
    }

def detect_metal_roi(image, threshold=0.9, min_size=50):
    """
    Detect metal regions and return ROI bounding box.
    Returns (x, y, width, height) or None if no metal found.
    """
    img_np = image.squeeze().cpu().numpy() if torch.is_tensor(image) else image.squeeze()
    
    # Threshold for metal
    metal_mask = img_np > threshold
    
    # Find connected components
    labeled, num_features = ndimage.label(metal_mask)
    
    if num_features == 0:
        return None
    
    # Find largest metal region
    sizes = ndimage.sum(metal_mask, labeled, range(1, num_features + 1))
    if max(sizes) < min_size:
        return None
    
    largest_label = np.argmax(sizes) + 1
    metal_coords = np.where(labeled == largest_label)
    
    # Get bounding box with padding
    y_min, y_max = metal_coords[0].min(), metal_coords[0].max()
    x_min, x_max = metal_coords[1].min(), metal_coords[1].max()
    
    # Add padding
    pad = 20
    h, w = img_np.shape
    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_main_grid(results, save_path, config):
    """
    Create main publication figure: N rows × 3 columns grid.
    Columns: (A) Ground Truth, (B) MSE-only, (C) SGAMARN
    """
    n_slices = len(results)
    
    fig, axes = plt.subplots(n_slices, 3, figsize=(12, 4 * n_slices))
    fig.suptitle('Figure 1: Limitation of Pixel-Wise Loss Only\n' +
                 '(A) Ground Truth | (B) MSE-Only | (C) SGAMARN (Full Model)',
                 fontsize=14, fontweight='bold', y=1.002)
    
    column_titles = ['(A) Ground Truth', '(B) MSE-Only Output', '(C) SGAMARN Output']
    
    for row_idx, result in enumerate(results):
        gt = result['gt']
        mse_out = result['mse_output']
        sgamarn_out = result['sgamarn_output']
        
        images = [gt, mse_out, sgamarn_out]
        
        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx] if n_slices > 1 else axes[col_idx]
            
            ax.imshow(img, cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'])
            ax.axis('off')
            
            # Add column titles to first row
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=11, fontweight='bold')
            
            # Add metrics to MSE and SGAMARN columns
            if col_idx == 1:  # MSE-only
                ax.text(0.02, 0.98, f"PSNR: {result['mse_metrics']['PSNR']:.2f} dB\nSSIM: {result['mse_metrics']['SSIM']:.4f}",
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            elif col_idx == 2:  # SGAMARN
                ax.text(0.02, 0.98, f"PSNR: {result['sgamarn_metrics']['PSNR']:.2f} dB\nSSIM: {result['sgamarn_metrics']['SSIM']:.4f}",
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add slice index label
        ax_first = axes[row_idx, 0] if n_slices > 1 else axes[0]
        ax_first.text(-0.1, 0.5, f'Slice {result["slice_idx"]}',
                     transform=ax_first.transAxes, fontsize=10, fontweight='bold',
                     verticalalignment='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved main grid: {save_path}")

def create_individual_slice_comparison(result, save_dir, config):
    """
    Create individual slice comparison with zoom ROIs.
    """
    slice_idx = result['slice_idx']
    gt = result['gt']
    ma_ct = result['ma_ct']
    mse_out = result['mse_output']
    sgamarn_out = result['sgamarn_output']
    
    # Detect ROI around metal
    roi = detect_metal_roi(ma_ct, threshold=config['metal_threshold'])
    
    fig = plt.figure(figsize=(20, 10))
    
    # Main row: Input, GT, MSE-only, SGAMARN
    gs = GridSpec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.15)
    
    images = [ma_ct, gt, mse_out, sgamarn_out]
    titles = ['Input (Metal Artifact)', 'Ground Truth', 
              f"MSE-Only\nPSNR: {result['mse_metrics']['PSNR']:.2f} dB",
              f"SGAMARN\nPSNR: {result['sgamarn_metrics']['PSNR']:.2f} dB"]
    
    # Top row: Full images
    for col_idx, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(img, cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Draw ROI rectangle if found
        if roi is not None:
            x, y, w, h = roi
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                     edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    
    # Bottom row: Zoomed ROIs
    if roi is not None:
        x, y, w, h = roi
        for col_idx, (img, title) in enumerate(zip(images, ['Input (Zoom)', 'GT (Zoom)', 
                                                            'MSE-Only (Zoom)', 'SGAMARN (Zoom)'])):
            ax = fig.add_subplot(gs[1, col_idx])
            
            # Extract ROI
            img_np = img.squeeze().cpu().numpy() if torch.is_tensor(img) else img.squeeze()
            roi_img = img_np[y:y+h, x:x+w]
            
            ax.imshow(roi_img, cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'])
            ax.set_title(title, fontsize=10)
            ax.axis('off')
    else:
        # No ROI found - show error maps instead
        ax = fig.add_subplot(gs[1, 0])
        ax.text(0.5, 0.5, 'No metal ROI\ndetected', ha='center', va='center', fontsize=12)
        ax.axis('off')
        
        # Error maps
        mse_error = np.abs(mse_out - gt)
        sgamarn_error = np.abs(sgamarn_out - gt)
        
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(mse_error, cmap='hot', vmin=0, vmax=0.3)
        ax.set_title('MSE-Only Error', fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(sgamarn_error, cmap='hot', vmin=0, vmax=0.3)
        ax.set_title('SGAMARN Error', fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 3])
        diff = mse_error - sgamarn_error
        ax.imshow(diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        ax.set_title('Difference\n(Red=MSE worse)', fontsize=10)
        ax.axis('off')
    
    fig.suptitle(f'Slice {slice_idx} Comparison', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, f'slice_{slice_idx:04d}_comparison.png')
    plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

def create_error_map_comparison(results, save_path, config):
    """
    Create error map comparison showing where MSE-only fails.
    """
    n_slices = min(9, len(results))  # Show up to 9 slices
    
    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    fig.suptitle('Error Map Comparison: MSE-Only vs SGAMARN\n' +
                 '(Brighter = Higher Error)', fontsize=14, fontweight='bold', y=1.002)
    
    for row_idx in range(n_slices):
        result = results[row_idx]
        gt = result['gt']
        mse_out = result['mse_output']
        sgamarn_out = result['sgamarn_output']
        
        mse_error = np.abs(mse_out - gt)
        sgamarn_error = np.abs(sgamarn_out - gt)
        diff = mse_error - sgamarn_error
        
        # Ground Truth
        axes[row_idx, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 0].set_title('Ground Truth' if row_idx == 0 else '', fontsize=10)
        axes[row_idx, 0].axis('off')
        
        # MSE-only error
        im1 = axes[row_idx, 1].imshow(mse_error, cmap='hot', vmin=0, vmax=0.3)
        axes[row_idx, 1].set_title(f'MSE-Only Error\nMAE: {np.mean(mse_error):.4f}' if row_idx == 0 
                                   else f'MAE: {np.mean(mse_error):.4f}', fontsize=10)
        axes[row_idx, 1].axis('off')
        
        # SGAMARN error
        im2 = axes[row_idx, 2].imshow(sgamarn_error, cmap='hot', vmin=0, vmax=0.3)
        axes[row_idx, 2].set_title(f'SGAMARN Error\nMAE: {np.mean(sgamarn_error):.4f}' if row_idx == 0 
                                   else f'MAE: {np.mean(sgamarn_error):.4f}', fontsize=10)
        axes[row_idx, 2].axis('off')
        
        # Difference (positive = MSE worse)
        im3 = axes[row_idx, 3].imshow(diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        axes[row_idx, 3].set_title('Difference\n(Red = MSE Worse)' if row_idx == 0 else '', fontsize=10)
        axes[row_idx, 3].axis('off')
        
        # Slice label
        axes[row_idx, 0].text(-0.1, 0.5, f'Slice {result["slice_idx"]}',
                             transform=axes[row_idx, 0].transAxes, fontsize=10, fontweight='bold',
                             verticalalignment='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved error maps: {save_path}")

def create_intensity_profiles(results, save_path, config):
    """
    Create intensity profile comparison through metal regions.
    Shows blurring in MSE-only vs sharper SGAMARN.
    """
    n_slices = min(6, len(results))  # Show up to 6 slices
    
    fig, axes = plt.subplots(n_slices, 2, figsize=(14, 4 * n_slices))
    fig.suptitle('Intensity Profiles Through Metal Regions\n' +
                 '(MSE-only shows smoothed/blurred profiles)', fontsize=14, fontweight='bold', y=1.002)
    
    for row_idx in range(n_slices):
        result = results[row_idx]
        gt = result['gt']
        ma_ct = result['ma_ct']
        mse_out = result['mse_output']
        sgamarn_out = result['sgamarn_output']
        
        H, W = gt.shape
        
        # Horizontal profile (middle row)
        ax_img = axes[row_idx, 0]
        ax_profile = axes[row_idx, 1]
        
        # Show image with profile line
        ax_img.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax_img.axhline(y=H//2, color='red', linestyle='--', linewidth=1.5, label='Profile line')
        ax_img.set_title(f'Slice {result["slice_idx"]} - GT with profile line', fontsize=10)
        ax_img.axis('off')
        
        # Extract profiles
        h_gt = gt[H//2, :]
        h_ma = ma_ct[H//2, :]
        h_mse = mse_out[H//2, :]
        h_sgamarn = sgamarn_out[H//2, :]
        x_pos = np.arange(W)
        
        # Plot profiles
        ax_profile.plot(x_pos, h_gt, 'g-', linewidth=2, alpha=0.8, label='Ground Truth')
        ax_profile.plot(x_pos, h_ma, 'b--', linewidth=1.5, alpha=0.5, label='Input (Artifact)')
        ax_profile.plot(x_pos, h_mse, 'r-', linewidth=1.5, alpha=0.8, label='MSE-Only')
        ax_profile.plot(x_pos, h_sgamarn, 'm-', linewidth=1.5, alpha=0.8, label='SGAMARN')
        
        ax_profile.set_xlabel('Pixel Position', fontsize=10)
        ax_profile.set_ylabel('Intensity', fontsize=10)
        ax_profile.set_title(f'Horizontal Intensity Profile (Row {H//2})', fontsize=10)
        ax_profile.legend(fontsize=8, loc='upper right')
        ax_profile.grid(True, alpha=0.3)
        ax_profile.set_ylim(0, 1)
        ax_profile.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved intensity profiles: {save_path}")

def create_histogram_comparison(results, save_path, config):
    """
    Create intensity histogram comparison.
    MSE-only tends to have narrower distribution (blurry).
    """
    # Aggregate all pixels
    gt_all = []
    mse_all = []
    sgamarn_all = []
    
    for result in results:
        gt_all.extend(result['gt'].flatten())
        mse_all.extend(result['mse_output'].flatten())
        sgamarn_all.extend(result['sgamarn_output'].flatten())
    
    gt_all = np.array(gt_all)
    mse_all = np.array(mse_all)
    sgamarn_all = np.array(sgamarn_all)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Intensity Distribution Comparison\n' +
                 '(MSE-only typically shows narrower distribution = blurring)', 
                 fontsize=14, fontweight='bold')
    
    # Individual histograms
    axes[0].hist(gt_all, bins=100, alpha=0.7, color='green', density=True, label='Ground Truth')
    axes[0].set_xlabel('Intensity', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].set_title(f'Ground Truth\nStd: {np.std(gt_all):.4f}', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(mse_all, bins=100, alpha=0.7, color='red', density=True, label='MSE-Only')
    axes[1].set_xlabel('Intensity', fontsize=10)
    axes[1].set_ylabel('Density', fontsize=10)
    axes[1].set_title(f'MSE-Only\nStd: {np.std(mse_all):.4f}', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(sgamarn_all, bins=100, alpha=0.7, color='purple', density=True, label='SGAMARN')
    axes[2].set_xlabel('Intensity', fontsize=10)
    axes[2].set_ylabel('Density', fontsize=10)
    axes[2].set_title(f'SGAMARN\nStd: {np.std(sgamarn_all):.4f}', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved histogram comparison: {save_path}")

def create_metrics_bar_chart(results, save_path, config):
    """
    Create bar chart comparing metrics between MSE-only and SGAMARN.
    """
    mse_psnr = [r['mse_metrics']['PSNR'] for r in results]
    sgamarn_psnr = [r['sgamarn_metrics']['PSNR'] for r in results]
    mse_ssim = [r['mse_metrics']['SSIM'] for r in results]
    sgamarn_ssim = [r['sgamarn_metrics']['SSIM'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Quantitative Comparison: MSE-Only vs SGAMARN', fontsize=14, fontweight='bold')
    
    x = np.arange(len(results))
    width = 0.35
    
    # PSNR comparison
    bars1 = axes[0].bar(x - width/2, mse_psnr, width, label='MSE-Only', color='indianred', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, sgamarn_psnr, width, label='SGAMARN', color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Slice Index', fontsize=10)
    axes[0].set_ylabel('PSNR (dB)', fontsize=10)
    axes[0].set_title(f'PSNR Comparison\nMSE-Only Avg: {np.mean(mse_psnr):.2f} dB | SGAMARN Avg: {np.mean(sgamarn_psnr):.2f} dB',
                     fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r['slice_idx'] for r in results], fontsize=8, rotation=45)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # SSIM comparison
    bars3 = axes[1].bar(x - width/2, mse_ssim, width, label='MSE-Only', color='indianred', alpha=0.8)
    bars4 = axes[1].bar(x + width/2, sgamarn_ssim, width, label='SGAMARN', color='steelblue', alpha=0.8)
    axes[1].set_xlabel('Slice Index', fontsize=10)
    axes[1].set_ylabel('SSIM', fontsize=10)
    axes[1].set_title(f'SSIM Comparison\nMSE-Only Avg: {np.mean(mse_ssim):.4f} | SGAMARN Avg: {np.mean(sgamarn_ssim):.4f}',
                     fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r['slice_idx'] for r in results], fontsize=8, rotation=45)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved metrics bar chart: {save_path}")

def save_metrics_csv(results, save_path):
    """Save per-slice metrics to CSV."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Slice_Index', 'MSE_PSNR', 'MSE_SSIM', 'MSE_MSE', 'MSE_MAE',
                        'SGAMARN_PSNR', 'SGAMARN_SSIM', 'SGAMARN_MSE', 'SGAMARN_MAE',
                        'PSNR_Improvement', 'SSIM_Improvement'])
        
        for r in results:
            psnr_imp = r['sgamarn_metrics']['PSNR'] - r['mse_metrics']['PSNR']
            ssim_imp = r['sgamarn_metrics']['SSIM'] - r['mse_metrics']['SSIM']
            
            writer.writerow([
                r['slice_idx'],
                f"{r['mse_metrics']['PSNR']:.4f}",
                f"{r['mse_metrics']['SSIM']:.4f}",
                f"{r['mse_metrics']['MSE']:.6f}",
                f"{r['mse_metrics']['MAE']:.6f}",
                f"{r['sgamarn_metrics']['PSNR']:.4f}",
                f"{r['sgamarn_metrics']['SSIM']:.4f}",
                f"{r['sgamarn_metrics']['MSE']:.6f}",
                f"{r['sgamarn_metrics']['MAE']:.6f}",
                f"{psnr_imp:.4f}",
                f"{ssim_imp:.4f}",
            ])
    
    print(f"  ✓ Saved metrics CSV: {save_path}")

# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FIGURE 1: Limitation of Pixel-Wise Loss Only")
    print("=" * 70)
    
    config = CONFIG.copy()
    
    # Print checkpoint status
    print_checkpoint_status()
    
    # Check checkpoint paths
    if config['mse_only_checkpoint'] is None:
        print("\n⚠️  ERROR: MSE-only checkpoint path not set!")
        print("   Please update PATHS['checkpoints']['A0_mse_only'] in shared_config.py")
        print("   with the path to the A0_mse_only checkpoint from the ablation study.")
        return
    
    if not os.path.exists(config['mse_only_checkpoint']):
        print(f"\n⚠️  ERROR: MSE-only checkpoint not found: {config['mse_only_checkpoint']}")
        return
    
    if config['sgamarn_checkpoint'] is None or not os.path.exists(config['sgamarn_checkpoint']):
        print(f"\n⚠️  ERROR: SGAMARN checkpoint not found: {config['sgamarn_checkpoint']}")
        return
    
    # Create output directories
    output_dir = config['output_dir']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'run_{timestamp}')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'individual_slices'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'error_maps'), exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}")
    
    # Save config
    config_save = config.copy()
    config_save['timestamp'] = timestamp
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Load models
    print("\nLoading models...")
    device = get_device()
    
    print(f"  Loading MSE-only model from: {config['mse_only_checkpoint']}")
    mse_model = load_generator(config['mse_only_checkpoint'], device)
    
    print(f"  Loading SGAMARN model from: {config['sgamarn_checkpoint']}")
    sgamarn_model = load_generator(config['sgamarn_checkpoint'], device)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = load_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"  Test samples: {len(test_dataset)}")
    
    # Get selected slice indices (shared across all figures)
    selected_indices = get_selected_slices()
    print(f"\nUsing {len(selected_indices)} pre-selected slices (shared across all figures)")
    
    # Run inference on selected slices
    print("\nRunning inference on selected slices...")
    results = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            if idx not in selected_indices:
                continue
            
            ma_ct = batch[0].to(device)  # Input with artifacts
            gt = batch[1].to(device)     # Ground truth
            
            # Run both models
            mse_output = mse_model(ma_ct)
            sgamarn_output = sgamarn_model(ma_ct)
            
            # Denormalize all
            ma_ct_dn = denormalize(ma_ct).squeeze().cpu().numpy()
            gt_dn = denormalize(gt).squeeze().cpu().numpy()
            mse_output_dn = denormalize(mse_output).squeeze().cpu().numpy()
            sgamarn_output_dn = denormalize(sgamarn_output).squeeze().cpu().numpy()
            
            # Clip to valid range
            mse_output_dn = np.clip(mse_output_dn, 0, 1)
            sgamarn_output_dn = np.clip(sgamarn_output_dn, 0, 1)
            
            # Compute metrics
            mse_metrics = compute_metrics(torch.tensor(mse_output_dn), torch.tensor(gt_dn))
            sgamarn_metrics = compute_metrics(torch.tensor(sgamarn_output_dn), torch.tensor(gt_dn))
            
            results.append({
                'slice_idx': idx,
                'ma_ct': ma_ct_dn,
                'gt': gt_dn,
                'mse_output': mse_output_dn,
                'sgamarn_output': sgamarn_output_dn,
                'mse_metrics': mse_metrics,
                'sgamarn_metrics': sgamarn_metrics,
            })
    
    # Sort results by slice index for consistent ordering
    results.sort(key=lambda x: x['slice_idx'])
    
    print(f"\nGenerated results for {len(results)} slices")
    
    # Generate all visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Main grid figure
    print("\n1. Creating main grid figure...")
    create_main_grid(results, os.path.join(run_dir, 'figure1_main_grid.png'), config)
    
    # 2. Individual slice comparisons with zoom
    print("\n2. Creating individual slice comparisons...")
    for result in tqdm(results, desc="Individual slices"):
        create_individual_slice_comparison(result, os.path.join(run_dir, 'individual_slices'), config)
    
    # 3. Error map comparison
    print("\n3. Creating error map comparison...")
    create_error_map_comparison(results, os.path.join(run_dir, 'error_maps', 'error_map_comparison.png'), config)
    
    # 4. Intensity profiles
    print("\n4. Creating intensity profiles...")
    create_intensity_profiles(results, os.path.join(run_dir, 'intensity_profiles.png'), config)
    
    # 5. Histogram comparison
    print("\n5. Creating histogram comparison...")
    create_histogram_comparison(results, os.path.join(run_dir, 'histogram_comparison.png'), config)
    
    # 6. Metrics bar chart
    print("\n6. Creating metrics bar chart...")
    create_metrics_bar_chart(results, os.path.join(run_dir, 'metrics_bar_chart.png'), config)
    
    # 7. Save metrics CSV
    print("\n7. Saving metrics CSV...")
    save_metrics_csv(results, os.path.join(run_dir, 'metrics_comparison.csv'))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    mse_psnr_avg = np.mean([r['mse_metrics']['PSNR'] for r in results])
    sgamarn_psnr_avg = np.mean([r['sgamarn_metrics']['PSNR'] for r in results])
    mse_ssim_avg = np.mean([r['mse_metrics']['SSIM'] for r in results])
    sgamarn_ssim_avg = np.mean([r['sgamarn_metrics']['SSIM'] for r in results])
    
    print(f"\nMSE-Only Average:")
    print(f"  PSNR: {mse_psnr_avg:.3f} dB")
    print(f"  SSIM: {mse_ssim_avg:.4f}")
    
    print(f"\nSGAMARN Average:")
    print(f"  PSNR: {sgamarn_psnr_avg:.3f} dB")
    print(f"  SSIM: {sgamarn_ssim_avg:.4f}")
    
    print(f"\nImprovement (SGAMARN over MSE-Only):")
    print(f"  PSNR: +{sgamarn_psnr_avg - mse_psnr_avg:.3f} dB")
    print(f"  SSIM: +{sgamarn_ssim_avg - mse_ssim_avg:.4f}")
    
    # Save summary
    summary = {
        'num_slices': len(results),
        'mse_only': {
            'avg_psnr': mse_psnr_avg,
            'avg_ssim': mse_ssim_avg,
            'checkpoint': config['mse_only_checkpoint'],
        },
        'sgamarn': {
            'avg_psnr': sgamarn_psnr_avg,
            'avg_ssim': sgamarn_ssim_avg,
            'checkpoint': config['sgamarn_checkpoint'],
        },
        'improvement': {
            'psnr_db': sgamarn_psnr_avg - mse_psnr_avg,
            'ssim': sgamarn_ssim_avg - mse_ssim_avg,
        },
    }
    
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ All outputs saved to: {run_dir}")
    print("\nFiles generated:")
    print("  - figure1_main_grid.png (main publication figure)")
    print("  - individual_slices/ (per-slice comparisons with zoom)")
    print("  - error_maps/error_map_comparison.png")
    print("  - intensity_profiles.png")
    print("  - histogram_comparison.png")
    print("  - metrics_bar_chart.png")
    print("  - metrics_comparison.csv")
    print("  - summary.json")
    print("  - config.json")

if __name__ == "__main__":
    main()
