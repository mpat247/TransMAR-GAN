"""
Figure 2: Physics Consistency Visualization

Purpose:
    Show how physics consistency loss reduces sinogram mismatch compared to 
    an image-only baseline (no L_phys).

Comparison:
    - Baseline: A1_no_physics (λ_phys = 0)
    - Full Model: A0_baseline (with physics loss)

Panel Layout:
    (A) Clean sinogram y_clean = P(GT)
    (B) Forward projection of baseline output: P(x_base)
    (C) Forward projection of full model output: P(x_full)
    (D) Residual |P(x_base) - y_clean|
    (E) Residual |P(x_full) - y_clean|

Additional Outputs:
    - Line profiles (detector & angle)
    - Per-angle error analysis
    - Sinogram residual histograms
    - Metal trace visualization
    - Model pipeline visualization (stage-by-stage)
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
from matplotlib.colors import Normalize
import torchvision.utils as vutils
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared config
from shared_config import (
    PATHS, MODEL_CONFIG, RADON_CONFIG, VIS_CONFIG,
    denormalize, normalize_to_model, get_device, load_generator,
    load_test_dataset, get_selected_slices, check_checkpoint_exists,
    print_checkpoint_status
)

# TorchRadon for forward projection
try:
    from torch_radon import Radon
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    print("WARNING: torch_radon not available. Physics consistency cannot be computed.")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Output settings
    'output_dir': os.path.join(PATHS['output_root'], 'figure2_physics_consistency'),
    
    # Radon transform settings
    'num_angles': 180,
    'angle_range': (0, np.pi),  # 0 to 180 degrees
    
    # Visualization
    'sinogram_cmap': 'gray',
    'residual_cmap': 'hot',
    'dpi': 200,
    
    # Analysis settings
    'profile_angle_idx': 90,  # Which angle to use for detector profile (90° = horizontal)
    'profile_detector_idx': None,  # Will be set to center detector
}

# ═══════════════════════════════════════════════════════════════
# FORWARD PROJECTION UTILITIES
# ═══════════════════════════════════════════════════════════════

def create_radon_projector(img_size, device):
    """Create TorchRadon forward projector."""
    if not RADON_AVAILABLE:
        raise RuntimeError("torch_radon is required for physics consistency visualization")
    
    # TorchRadon uses numpy angles, not torch tensors
    angles = np.linspace(0, np.pi, CONFIG['num_angles'], dtype=np.float32)
    # Radon projector doesn't have .to() method - it operates on input tensors
    projector = Radon(img_size, angles)
    
    return projector, angles

def forward_project(projector, image):
    """
    Forward project image to sinogram domain.
    
    Args:
        projector: TorchRadon Radon projector
        image: Image tensor [B, C, H, W] or [B, H, W]
    
    Returns:
        Sinogram tensor [B, num_angles, det_count]
    """
    # TorchRadon expects [B, H, W]
    if image.dim() == 4:
        image = image.squeeze(1)  # Remove channel dim
    
    sinogram = projector.forward(image)
    
    return sinogram

def compute_sinogram_metrics(sino_pred, sino_clean):
    """Compute metrics between predicted and clean sinograms."""
    pred_np = sino_pred.squeeze().cpu().numpy()
    clean_np = sino_clean.squeeze().cpu().numpy()
    
    mse = np.mean((pred_np - clean_np) ** 2)
    mae = np.mean(np.abs(pred_np - clean_np))
    
    # Per-angle error
    per_angle_mse = np.mean((pred_np - clean_np) ** 2, axis=1)
    per_angle_mae = np.mean(np.abs(pred_np - clean_np), axis=1)
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'per_angle_MSE': per_angle_mse.tolist(),
        'per_angle_MAE': per_angle_mae.tolist(),
    }

# ═══════════════════════════════════════════════════════════════
# MODEL PIPELINE VISUALIZATION
# ═══════════════════════════════════════════════════════════════

class GeneratorWithIntermediates(nn.Module):
    """
    Wrapper around NGswin generator to capture intermediate activations.
    """
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.intermediates = {}
        
        # Register hooks to capture intermediate outputs
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on key layers."""
        self.hooks = []
        
        # Hook on conv_first (initial embedding)
        if hasattr(self.generator, 'conv_first'):
            hook = self.generator.conv_first.register_forward_hook(
                lambda m, inp, out: self._save_intermediate('after_conv_first', out)
            )
            self.hooks.append(hook)
        
        # Hook on each RSTB (Residual Swin Transformer Block)
        if hasattr(self.generator, 'layers'):
            for i, layer in enumerate(self.generator.layers):
                hook = layer.register_forward_hook(
                    lambda m, inp, out, idx=i: self._save_intermediate(f'after_rstb_{idx}', out)
                )
                self.hooks.append(hook)
        
        # Hook before final conv
        if hasattr(self.generator, 'conv_before_upsample'):
            hook = self.generator.conv_before_upsample.register_forward_hook(
                lambda m, inp, out: self._save_intermediate('before_final_conv', out)
            )
            self.hooks.append(hook)
    
    def _save_intermediate(self, name, tensor):
        """Save intermediate tensor."""
        self.intermediates[name] = tensor.detach().clone()
    
    def forward(self, x):
        self.intermediates = {'input': x.detach().clone()}
        output = self.generator(x)
        self.intermediates['output'] = output.detach().clone()
        return output
    
    def get_intermediates(self):
        """Return captured intermediate activations."""
        return self.intermediates
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()

def visualize_generator_stages(intermediates, save_path, slice_idx):
    """
    Visualize generator intermediate stages.
    
    Args:
        intermediates: Dict of intermediate tensors from GeneratorWithIntermediates
        save_path: Path to save the visualization
        slice_idx: Slice index for labeling
    """
    # Filter to key stages
    stage_order = ['input', 'after_conv_first', 'after_rstb_0', 'after_rstb_1', 
                   'after_rstb_2', 'after_rstb_3', 'before_final_conv', 'output']
    
    available_stages = [s for s in stage_order if s in intermediates]
    n_stages = len(available_stages)
    
    if n_stages == 0:
        print(f"  Warning: No intermediate stages captured for slice {slice_idx}")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, (n_stages + 1) // 2, figsize=(4 * ((n_stages + 1) // 2), 8))
    axes = axes.flatten() if n_stages > 1 else [axes]
    
    fig.suptitle(f'Generator Pipeline - Slice {slice_idx}', fontsize=14, fontweight='bold')
    
    for idx, stage_name in enumerate(available_stages):
        if idx >= len(axes):
            break
            
        tensor = intermediates[stage_name]
        
        # Get visualization (first channel or mean across channels)
        if tensor.dim() == 4:
            # [B, C, H, W] - take first sample, visualize first channel or mean
            vis = tensor[0]
            if vis.shape[0] > 1:
                vis = vis.mean(dim=0)  # Mean across channels
            else:
                vis = vis.squeeze(0)
        elif tensor.dim() == 3:
            vis = tensor[0] if tensor.shape[0] > 1 else tensor.squeeze(0)
        else:
            vis = tensor.squeeze()
        
        vis_np = vis.cpu().numpy()
        
        # Normalize for display
        if stage_name in ['input', 'output']:
            # These are in [-1, 1], convert to [0, 1]
            vis_np = (vis_np + 1) / 2
            vmin, vmax = 0, 1
        else:
            # Feature maps - normalize per-image
            vmin, vmax = vis_np.min(), vis_np.max()
        
        axes[idx].imshow(vis_np, cmap='gray', vmin=vmin, vmax=vmax)
        axes[idx].set_title(stage_name.replace('_', ' ').title(), fontsize=10)
        axes[idx].axis('off')
        
        # Add shape info
        shape_str = f"{list(tensor.shape)}"
        axes[idx].text(0.5, -0.05, shape_str, transform=axes[idx].transAxes,
                      fontsize=8, ha='center', va='top')
    
    # Hide unused axes
    for idx in range(len(available_stages), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_main_physics_figure(result, save_path):
    """
    Create main Figure 2 with panels A-E.
    
    Panels:
        (A) Clean sinogram y_clean
        (B) Forward projection of baseline output P(x_base)
        (C) Forward projection of full model output P(x_full)
        (D) Residual |P(x_base) - y_clean|
        (E) Residual |P(x_full) - y_clean|
    """
    sino_clean = result['sino_clean']
    sino_base = result['sino_baseline']
    sino_full = result['sino_full']
    
    residual_base = np.abs(sino_base - sino_clean)
    residual_full = np.abs(sino_full - sino_clean)
    
    # Normalize sinograms for display
    sino_vmin = min(sino_clean.min(), sino_base.min(), sino_full.min())
    sino_vmax = max(sino_clean.max(), sino_base.max(), sino_full.max())
    
    residual_vmax = max(residual_base.max(), residual_full.max())
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 5, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
    
    # Row 1: Sinograms
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(sino_clean, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    ax_a.set_title('(A) Clean Sinogram\ny_clean = P(GT)', fontsize=11, fontweight='bold')
    ax_a.set_xlabel('Detector Index', fontsize=9)
    ax_a.set_ylabel('Angle Index', fontsize=9)
    plt.colorbar(im_a, ax=ax_a, fraction=0.046)
    
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(sino_base, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    ax_b.set_title('(B) Baseline Sinogram\nP(x_base) [No Physics]', fontsize=11, fontweight='bold')
    ax_b.set_xlabel('Detector Index', fontsize=9)
    ax_b.set_ylabel('Angle Index', fontsize=9)
    plt.colorbar(im_b, ax=ax_b, fraction=0.046)
    
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(sino_full, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    ax_c.set_title('(C) Full Model Sinogram\nP(x_full) [With Physics]', fontsize=11, fontweight='bold')
    ax_c.set_xlabel('Detector Index', fontsize=9)
    ax_c.set_ylabel('Angle Index', fontsize=9)
    plt.colorbar(im_c, ax=ax_c, fraction=0.046)
    
    # Row 1 continued: Residuals
    ax_d = fig.add_subplot(gs[0, 3])
    im_d = ax_d.imshow(residual_base, cmap='hot', aspect='auto', vmin=0, vmax=residual_vmax)
    ax_d.set_title(f'(D) Baseline Residual\n|P(x_base) - y_clean|\nMAE: {residual_base.mean():.4f}', 
                   fontsize=11, fontweight='bold')
    ax_d.set_xlabel('Detector Index', fontsize=9)
    ax_d.set_ylabel('Angle Index', fontsize=9)
    plt.colorbar(im_d, ax=ax_d, fraction=0.046)
    
    ax_e = fig.add_subplot(gs[0, 4])
    im_e = ax_e.imshow(residual_full, cmap='hot', aspect='auto', vmin=0, vmax=residual_vmax)
    ax_e.set_title(f'(E) Full Model Residual\n|P(x_full) - y_clean|\nMAE: {residual_full.mean():.4f}', 
                   fontsize=11, fontweight='bold')
    ax_e.set_xlabel('Detector Index', fontsize=9)
    ax_e.set_ylabel('Angle Index', fontsize=9)
    plt.colorbar(im_e, ax=ax_e, fraction=0.046)
    
    # Row 2: Images for reference
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.imshow(result['gt'], cmap='gray', vmin=0, vmax=1)
    ax_gt.set_title('Ground Truth', fontsize=10)
    ax_gt.axis('off')
    
    ax_input = fig.add_subplot(gs[1, 1])
    ax_input.imshow(result['ma_ct'], cmap='gray', vmin=0, vmax=1)
    ax_input.set_title('Input (Metal Artifact)', fontsize=10)
    ax_input.axis('off')
    
    ax_base = fig.add_subplot(gs[1, 2])
    ax_base.imshow(result['output_baseline'], cmap='gray', vmin=0, vmax=1)
    ax_base.set_title(f"Baseline Output\nPSNR: {result['baseline_metrics']['PSNR']:.2f} dB", fontsize=10)
    ax_base.axis('off')
    
    ax_full = fig.add_subplot(gs[1, 3])
    ax_full.imshow(result['output_full'], cmap='gray', vmin=0, vmax=1)
    ax_full.set_title(f"Full Model Output\nPSNR: {result['full_metrics']['PSNR']:.2f} dB", fontsize=10)
    ax_full.axis('off')
    
    # Residual comparison (image domain)
    ax_diff = fig.add_subplot(gs[1, 4])
    img_residual_base = np.abs(result['output_baseline'] - result['gt'])
    img_residual_full = np.abs(result['output_full'] - result['gt'])
    diff = img_residual_base - img_residual_full  # Positive = baseline worse
    ax_diff.imshow(diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax_diff.set_title('Image Error Difference\n(Red = Baseline Worse)', fontsize=10)
    ax_diff.axis('off')
    
    fig.suptitle(f'Figure 2: Physics Consistency Visualization - Slice {result["slice_idx"]}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

def create_line_profiles(result, save_path):
    """
    Create line profile plots for sinograms.
    Shows intensity vs detector index for fixed angle, and vs angle for fixed detector.
    """
    sino_clean = result['sino_clean']
    sino_base = result['sino_baseline']
    sino_full = result['sino_full']
    
    num_angles, num_detectors = sino_clean.shape
    
    # Settings
    angle_idx = CONFIG['profile_angle_idx']
    detector_idx = num_detectors // 2  # Center detector
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Sinogram Line Profiles - Slice {result["slice_idx"]}', fontsize=14, fontweight='bold')
    
    # Top-left: Detector profile at fixed angle
    ax1 = axes[0, 0]
    detector_pos = np.arange(num_detectors)
    ax1.plot(detector_pos, sino_clean[angle_idx, :], 'g-', linewidth=2, label='Clean (GT)', alpha=0.8)
    ax1.plot(detector_pos, sino_base[angle_idx, :], 'r-', linewidth=1.5, label='Baseline (No Physics)', alpha=0.7)
    ax1.plot(detector_pos, sino_full[angle_idx, :], 'b-', linewidth=1.5, label='Full Model', alpha=0.7)
    ax1.set_xlabel('Detector Index', fontsize=10)
    ax1.set_ylabel('Projection Value', fontsize=10)
    ax1.set_title(f'Detector Profile at Angle {angle_idx}° ({angle_idx * 180 // num_angles}°)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Angle profile at fixed detector
    ax2 = axes[0, 1]
    angle_pos = np.arange(num_angles)
    ax2.plot(angle_pos, sino_clean[:, detector_idx], 'g-', linewidth=2, label='Clean (GT)', alpha=0.8)
    ax2.plot(angle_pos, sino_base[:, detector_idx], 'r-', linewidth=1.5, label='Baseline', alpha=0.7)
    ax2.plot(angle_pos, sino_full[:, detector_idx], 'b-', linewidth=1.5, label='Full Model', alpha=0.7)
    ax2.set_xlabel('Angle Index', fontsize=10)
    ax2.set_ylabel('Projection Value', fontsize=10)
    ax2.set_title(f'Angle Profile at Detector {detector_idx} (Center)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Residual detector profile
    ax3 = axes[1, 0]
    residual_base = np.abs(sino_base[angle_idx, :] - sino_clean[angle_idx, :])
    residual_full = np.abs(sino_full[angle_idx, :] - sino_clean[angle_idx, :])
    ax3.plot(detector_pos, residual_base, 'r-', linewidth=1.5, label='Baseline Residual', alpha=0.8)
    ax3.plot(detector_pos, residual_full, 'b-', linewidth=1.5, label='Full Model Residual', alpha=0.8)
    ax3.fill_between(detector_pos, residual_base, residual_full, 
                     where=(residual_base > residual_full), color='red', alpha=0.2, label='Baseline Worse')
    ax3.fill_between(detector_pos, residual_base, residual_full,
                     where=(residual_full > residual_base), color='blue', alpha=0.2, label='Full Worse')
    ax3.set_xlabel('Detector Index', fontsize=10)
    ax3.set_ylabel('Absolute Residual', fontsize=10)
    ax3.set_title(f'Residual at Angle {angle_idx}°', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Per-angle mean error
    ax4 = axes[1, 1]
    per_angle_base = np.mean(np.abs(sino_base - sino_clean), axis=1)
    per_angle_full = np.mean(np.abs(sino_full - sino_clean), axis=1)
    ax4.plot(angle_pos, per_angle_base, 'r-', linewidth=1.5, label=f'Baseline (Mean: {per_angle_base.mean():.4f})', alpha=0.8)
    ax4.plot(angle_pos, per_angle_full, 'b-', linewidth=1.5, label=f'Full Model (Mean: {per_angle_full.mean():.4f})', alpha=0.8)
    ax4.set_xlabel('Angle Index', fontsize=10)
    ax4.set_ylabel('Mean Absolute Error', fontsize=10)
    ax4.set_title('Per-Angle Mean Error', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

def create_residual_histogram(results, save_path):
    """
    Create histogram of sinogram residuals across all slices.
    """
    # Collect all residuals
    residuals_base = []
    residuals_full = []
    
    for r in results:
        residual_base = np.abs(r['sino_baseline'] - r['sino_clean']).flatten()
        residual_full = np.abs(r['sino_full'] - r['sino_clean']).flatten()
        residuals_base.extend(residual_base)
        residuals_full.extend(residual_full)
    
    residuals_base = np.array(residuals_base)
    residuals_full = np.array(residuals_full)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Sinogram Residual Distribution (All Slices)', fontsize=14, fontweight='bold')
    
    # Histogram comparison
    bins = np.linspace(0, max(residuals_base.max(), residuals_full.max()), 100)
    
    axes[0].hist(residuals_base, bins=bins, alpha=0.6, color='red', density=True, label='Baseline (No Physics)')
    axes[0].hist(residuals_full, bins=bins, alpha=0.6, color='blue', density=True, label='Full Model')
    axes[0].set_xlabel('Absolute Residual', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].set_title('Residual Distribution', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # CDF comparison
    axes[1].hist(residuals_base, bins=bins, alpha=0.6, color='red', density=True, 
                 cumulative=True, histtype='step', linewidth=2, label='Baseline')
    axes[1].hist(residuals_full, bins=bins, alpha=0.6, color='blue', density=True,
                 cumulative=True, histtype='step', linewidth=2, label='Full Model')
    axes[1].set_xlabel('Absolute Residual', fontsize=10)
    axes[1].set_ylabel('Cumulative Density', fontsize=10)
    axes[1].set_title('Cumulative Distribution', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Statistics
    ax3 = axes[2]
    stats = {
        'Baseline': {
            'Mean': np.mean(residuals_base),
            'Std': np.std(residuals_base),
            'Median': np.median(residuals_base),
            'Max': np.max(residuals_base),
        },
        'Full Model': {
            'Mean': np.mean(residuals_full),
            'Std': np.std(residuals_full),
            'Median': np.median(residuals_full),
            'Max': np.max(residuals_full),
        },
    }
    
    ax3.axis('off')
    text = "Sinogram Residual Statistics\n" + "=" * 40 + "\n\n"
    
    for model_name, model_stats in stats.items():
        text += f"{model_name}:\n"
        for stat_name, stat_val in model_stats.items():
            text += f"  {stat_name}: {stat_val:.6f}\n"
        text += "\n"
    
    # Improvement
    improvement = (stats['Baseline']['Mean'] - stats['Full Model']['Mean']) / stats['Baseline']['Mean'] * 100
    text += f"Improvement: {improvement:.2f}% reduction in mean residual"
    
    ax3.text(0.1, 0.9, text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

def create_metal_trace_visualization(result, save_path):
    """
    Visualize metal trace in sinogram domain and overlay with residuals.
    """
    ma_ct = result['ma_ct']
    sino_clean = result['sino_clean']
    sino_base = result['sino_baseline']
    sino_full = result['sino_full']
    
    # Create metal mask from input (high intensity = metal)
    metal_mask = (ma_ct > VIS_CONFIG['metal_threshold']).astype(np.float32)
    
    # Forward project metal mask to get metal trace in sinogram
    # We'll approximate by thresholding the input sinogram
    sino_input = result.get('sino_input', None)
    
    residual_base = np.abs(sino_base - sino_clean)
    residual_full = np.abs(sino_full - sino_clean)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Metal Trace Analysis - Slice {result["slice_idx"]}', fontsize=14, fontweight='bold')
    
    # Row 1: Images
    axes[0, 0].imshow(ma_ct, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Input Image\n(Metal Artifact)', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(metal_mask, cmap='hot')
    axes[0, 1].set_title('Metal Mask\n(High Intensity)', fontsize=10)
    axes[0, 1].axis('off')
    
    # Clean sinogram
    axes[0, 2].imshow(sino_clean, cmap='gray', aspect='auto')
    axes[0, 2].set_title('Clean Sinogram', fontsize=10)
    axes[0, 2].set_xlabel('Detector')
    axes[0, 2].set_ylabel('Angle')
    
    # Row 2: Residuals with high-error regions highlighted
    axes[1, 0].imshow(residual_base, cmap='hot', aspect='auto')
    axes[1, 0].set_title(f'Baseline Residual\nMAE: {residual_base.mean():.4f}', fontsize=10)
    axes[1, 0].set_xlabel('Detector')
    axes[1, 0].set_ylabel('Angle')
    
    axes[1, 1].imshow(residual_full, cmap='hot', aspect='auto')
    axes[1, 1].set_title(f'Full Model Residual\nMAE: {residual_full.mean():.4f}', fontsize=10)
    axes[1, 1].set_xlabel('Detector')
    axes[1, 1].set_ylabel('Angle')
    
    # Difference
    diff = residual_base - residual_full
    axes[1, 2].imshow(diff, cmap='RdBu_r', aspect='auto', 
                      vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[1, 2].set_title('Residual Difference\n(Red = Baseline Worse)', fontsize=10)
    axes[1, 2].set_xlabel('Detector')
    axes[1, 2].set_ylabel('Angle')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

def create_per_angle_error_plot(results, save_path):
    """
    Create per-angle error analysis across all slices.
    """
    num_angles = CONFIG['num_angles']
    
    # Collect per-angle errors
    per_angle_base = np.zeros(num_angles)
    per_angle_full = np.zeros(num_angles)
    
    for r in results:
        per_angle_base += np.mean(np.abs(r['sino_baseline'] - r['sino_clean']), axis=1)
        per_angle_full += np.mean(np.abs(r['sino_full'] - r['sino_clean']), axis=1)
    
    per_angle_base /= len(results)
    per_angle_full /= len(results)
    
    angles_deg = np.linspace(0, 180, num_angles)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Per-Angle Error Analysis (Averaged Across All Slices)', fontsize=14, fontweight='bold')
    
    # Line plot
    axes[0].plot(angles_deg, per_angle_base, 'r-', linewidth=2, label='Baseline (No Physics)', alpha=0.8)
    axes[0].plot(angles_deg, per_angle_full, 'b-', linewidth=2, label='Full Model', alpha=0.8)
    axes[0].set_xlabel('Projection Angle (degrees)', fontsize=10)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=10)
    axes[0].set_title('Per-Angle MAE', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Improvement per angle
    improvement = (per_angle_base - per_angle_full) / per_angle_base * 100
    axes[1].bar(angles_deg, improvement, width=1.0, color='green', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Projection Angle (degrees)', fontsize=10)
    axes[1].set_ylabel('Improvement (%)', fontsize=10)
    axes[1].set_title(f'Physics Loss Improvement per Angle\n(Mean: {improvement.mean():.1f}%)', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

def save_metrics_csv(results, save_path):
    """Save sinogram metrics to CSV."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Slice_Index', 
                        'Sino_Base_MSE', 'Sino_Base_MAE',
                        'Sino_Full_MSE', 'Sino_Full_MAE',
                        'MSE_Improvement_%', 'MAE_Improvement_%',
                        'Image_Base_PSNR', 'Image_Full_PSNR'])
        
        for r in results:
            base_mse = r['sino_baseline_metrics']['MSE']
            base_mae = r['sino_baseline_metrics']['MAE']
            full_mse = r['sino_full_metrics']['MSE']
            full_mae = r['sino_full_metrics']['MAE']
            
            mse_imp = (base_mse - full_mse) / base_mse * 100 if base_mse > 0 else 0
            mae_imp = (base_mae - full_mae) / base_mae * 100 if base_mae > 0 else 0
            
            writer.writerow([
                r['slice_idx'],
                f"{base_mse:.6f}", f"{base_mae:.6f}",
                f"{full_mse:.6f}", f"{full_mae:.6f}",
                f"{mse_imp:.2f}", f"{mae_imp:.2f}",
                f"{r['baseline_metrics']['PSNR']:.3f}",
                f"{r['full_metrics']['PSNR']:.3f}",
            ])
    
    print(f"  ✓ Saved metrics CSV: {save_path}")

# ═══════════════════════════════════════════════════════════════
# IMAGE METRICS
# ═══════════════════════════════════════════════════════════════

def compute_image_metrics(pred, target):
    """Compute PSNR and SSIM between prediction and target."""
    pred_np = np.clip(pred, 0, 1)
    target_np = np.clip(target, 0, 1)
    
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    
    return {
        'PSNR': float(psnr_val),
        'SSIM': float(ssim_val),
    }

# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FIGURE 2: Physics Consistency Visualization")
    print("=" * 70)
    
    # Check prerequisites
    if not RADON_AVAILABLE:
        print("\n❌ ERROR: torch_radon is required but not installed.")
        print("   Install with: pip install torch-radon")
        return
    
    # Print checkpoint status
    print_checkpoint_status()
    
    # Check required checkpoints
    baseline_exists, baseline_path = check_checkpoint_exists('A1_no_physics')
    full_exists, full_path = check_checkpoint_exists('A0_baseline')
    
    if not baseline_exists:
        print(f"\n❌ ERROR: A1_no_physics checkpoint not found: {baseline_path}")
        return
    
    if not full_exists:
        print(f"\n❌ ERROR: A0_baseline checkpoint not found: {full_path}")
        return
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'main_figures'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'line_profiles'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'metal_trace'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'model_pipeline'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'sinogram_analysis'), exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}")
    
    # Save config
    config_save = {**CONFIG, 'timestamp': timestamp, 
                   'baseline_checkpoint': baseline_path,
                   'full_checkpoint': full_path}
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=2, default=str)
    
    # Load models
    print("\nLoading models...")
    device = get_device()
    
    print(f"  Loading Baseline (no physics) from: {baseline_path}")
    baseline_model = load_generator(baseline_path, device)
    baseline_with_intermediates = GeneratorWithIntermediates(baseline_model)
    
    print(f"  Loading Full Model from: {full_path}")
    full_model = load_generator(full_path, device)
    full_with_intermediates = GeneratorWithIntermediates(full_model)
    
    # Create Radon projector
    print("\nInitializing Radon projector...")
    img_size = MODEL_CONFIG['img_size']
    projector, angles = create_radon_projector(img_size, device)
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Number of angles: {CONFIG['num_angles']}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = load_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"  Test samples: {len(test_dataset)}")
    
    # Get selected slice indices
    selected_indices = get_selected_slices()
    print(f"\nUsing {len(selected_indices)} selected slices")
    
    # Run inference
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)
    
    results = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Processing")):
            if idx not in selected_indices:
                continue
            
            ma_ct = batch[0].to(device)  # Input with artifacts
            gt = batch[1].to(device)     # Ground truth
            
            # Run both models (with intermediate capture)
            output_baseline = baseline_with_intermediates(ma_ct)
            baseline_intermediates = baseline_with_intermediates.get_intermediates()
            
            output_full = full_with_intermediates(ma_ct)
            full_intermediates = full_with_intermediates.get_intermediates()
            
            # Denormalize
            ma_ct_dn = denormalize(ma_ct).squeeze().cpu().numpy()
            gt_dn = denormalize(gt).squeeze().cpu().numpy()
            output_baseline_dn = np.clip(denormalize(output_baseline).squeeze().cpu().numpy(), 0, 1)
            output_full_dn = np.clip(denormalize(output_full).squeeze().cpu().numpy(), 0, 1)
            
            # Forward project to sinogram domain
            # Convert to [0, 1] range for projection (Radon expects non-negative)
            gt_for_proj = denormalize(gt)
            baseline_for_proj = denormalize(output_baseline).clamp(0, 1)
            full_for_proj = denormalize(output_full).clamp(0, 1)
            
            sino_clean = forward_project(projector, gt_for_proj).squeeze().cpu().numpy()
            sino_baseline = forward_project(projector, baseline_for_proj).squeeze().cpu().numpy()
            sino_full = forward_project(projector, full_for_proj).squeeze().cpu().numpy()
            
            # Compute metrics
            sino_baseline_metrics = compute_sinogram_metrics(
                torch.tensor(sino_baseline), torch.tensor(sino_clean))
            sino_full_metrics = compute_sinogram_metrics(
                torch.tensor(sino_full), torch.tensor(sino_clean))
            
            baseline_img_metrics = compute_image_metrics(output_baseline_dn, gt_dn)
            full_img_metrics = compute_image_metrics(output_full_dn, gt_dn)
            
            result = {
                'slice_idx': idx,
                'ma_ct': ma_ct_dn,
                'gt': gt_dn,
                'output_baseline': output_baseline_dn,
                'output_full': output_full_dn,
                'sino_clean': sino_clean,
                'sino_baseline': sino_baseline,
                'sino_full': sino_full,
                'sino_baseline_metrics': sino_baseline_metrics,
                'sino_full_metrics': sino_full_metrics,
                'baseline_metrics': baseline_img_metrics,
                'full_metrics': full_img_metrics,
                'baseline_intermediates': baseline_intermediates,
                'full_intermediates': full_intermediates,
            }
            
            results.append(result)
    
    print(f"\nProcessed {len(results)} slices")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Main physics figure for each slice
    print("\n1. Creating main physics figures...")
    for result in tqdm(results, desc="Main figures"):
        save_path = os.path.join(run_dir, 'main_figures', f'slice_{result["slice_idx"]:04d}_physics.png')
        create_main_physics_figure(result, save_path)
    
    # 2. Line profiles for each slice
    print("\n2. Creating line profiles...")
    for result in tqdm(results, desc="Line profiles"):
        save_path = os.path.join(run_dir, 'line_profiles', f'slice_{result["slice_idx"]:04d}_profiles.png')
        create_line_profiles(result, save_path)
    
    # 3. Metal trace visualization
    print("\n3. Creating metal trace visualizations...")
    for result in tqdm(results, desc="Metal trace"):
        save_path = os.path.join(run_dir, 'metal_trace', f'slice_{result["slice_idx"]:04d}_metal.png')
        create_metal_trace_visualization(result, save_path)
    
    # 4. Generator pipeline visualization
    print("\n4. Creating model pipeline visualizations...")
    for result in tqdm(results, desc="Pipeline"):
        # Baseline pipeline
        save_path = os.path.join(run_dir, 'model_pipeline', f'slice_{result["slice_idx"]:04d}_baseline_pipeline.png')
        visualize_generator_stages(result['baseline_intermediates'], save_path, result['slice_idx'])
        
        # Full model pipeline
        save_path = os.path.join(run_dir, 'model_pipeline', f'slice_{result["slice_idx"]:04d}_full_pipeline.png')
        visualize_generator_stages(result['full_intermediates'], save_path, result['slice_idx'])
    
    # 5. Aggregated analysis
    print("\n5. Creating aggregated analysis...")
    
    # Residual histogram
    create_residual_histogram(results, os.path.join(run_dir, 'sinogram_analysis', 'residual_histogram.png'))
    
    # Per-angle error
    create_per_angle_error_plot(results, os.path.join(run_dir, 'sinogram_analysis', 'per_angle_error.png'))
    
    # 6. Save metrics
    print("\n6. Saving metrics...")
    save_metrics_csv(results, os.path.join(run_dir, 'sinogram_metrics.csv'))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    sino_base_mae_avg = np.mean([r['sino_baseline_metrics']['MAE'] for r in results])
    sino_full_mae_avg = np.mean([r['sino_full_metrics']['MAE'] for r in results])
    img_base_psnr_avg = np.mean([r['baseline_metrics']['PSNR'] for r in results])
    img_full_psnr_avg = np.mean([r['full_metrics']['PSNR'] for r in results])
    
    print(f"\nSinogram Domain (MAE):")
    print(f"  Baseline (No Physics): {sino_base_mae_avg:.6f}")
    print(f"  Full Model:            {sino_full_mae_avg:.6f}")
    print(f"  Improvement:           {(sino_base_mae_avg - sino_full_mae_avg) / sino_base_mae_avg * 100:.2f}%")
    
    print(f"\nImage Domain (PSNR):")
    print(f"  Baseline (No Physics): {img_base_psnr_avg:.3f} dB")
    print(f"  Full Model:            {img_full_psnr_avg:.3f} dB")
    print(f"  Improvement:           +{img_full_psnr_avg - img_base_psnr_avg:.3f} dB")
    
    # Save summary
    summary = {
        'num_slices': len(results),
        'sinogram_domain': {
            'baseline_MAE': sino_base_mae_avg,
            'full_MAE': sino_full_mae_avg,
            'improvement_percent': (sino_base_mae_avg - sino_full_mae_avg) / sino_base_mae_avg * 100,
        },
        'image_domain': {
            'baseline_PSNR': img_base_psnr_avg,
            'full_PSNR': img_full_psnr_avg,
            'improvement_dB': img_full_psnr_avg - img_base_psnr_avg,
        },
    }
    
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Cleanup hooks
    baseline_with_intermediates.remove_hooks()
    full_with_intermediates.remove_hooks()
    
    print(f"\n✓ All outputs saved to: {run_dir}")
    print("\nFiles generated:")
    print("  - main_figures/       (Panel A-E for each slice)")
    print("  - line_profiles/      (Detector & angle profiles)")
    print("  - metal_trace/        (Metal trace analysis)")
    print("  - model_pipeline/     (Generator stage-by-stage)")
    print("  - sinogram_analysis/  (Histograms, per-angle error)")
    print("  - sinogram_metrics.csv")
    print("  - summary.json")
    print("  - config.json")

if __name__ == "__main__":
    main()
