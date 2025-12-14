#!/usr/bin/env python3
"""
Figure 2: Physics Consistency Visualization (V2 - Improved)

Purpose:
    Show how physics consistency loss reduces sinogram mismatch and demonstrate
    the complete forward/back projection pipeline.

Key Improvements over V1:
    - Clear visualization of domain conversions (Image ↔ Sinogram)
    - Proper back-projection to verify reconstruction quality
    - Complete pipeline showing: Input → Network → Forward Project → Compare → Back Project
    - Metrics in both image and sinogram domains

Comparison:
    - Baseline: A1_no_physics (λ_phys = 0)  
    - Full Model: TransMAR-GAN (with physics loss)

Figure Layouts:
    1. Main Pipeline Figure: Shows complete forward/back projection flow
    2. Sinogram Comparison: Clean vs Baseline vs Full Model sinograms
    3. Residual Analysis: Sinogram and image domain residuals
    4. Domain Conversion Demo: Step-by-step conversion visualization

Author: TransMAR-GAN Paper
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared config
from shared_config import (
    PATHS, MODEL_CONFIG, VIS_CONFIG,
    denormalize, get_device, load_generator,
    load_test_dataset, get_selected_slices,
)

# TorchRadon for forward/back projection
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
    'output_dir': 'figure2_physics_consistency_v2',
    
    # Radon transform settings
    'num_angles': 180,
    'det_count': None,  # Will be set to image size
    
    # Dataset image size (TestDataset returns 416x416 images)
    # The Radon transform should match this
    'dataset_img_size': 416,
    
    # Model checkpoints
    'models': {
        'baseline': {
            'name': 'Baseline (No Physics Loss)',
            'short_name': 'No Physics',
            'path': '/home/grad/mppatel/Documents/DCGAN/ablation_results/loss_ablations_20251207_155318/A1_no_physics/checkpoints/best_model.pth',
        },
        'full_model': {
            'name': 'TransMAR-GAN (With Physics Loss)',
            'short_name': 'TransMAR-GAN',
            'path': '/home/grad/mppatel/Documents/DCGAN/combined_results/run_20251202_211759/checkpoints/best_model.pth',
        },
    },
    
    # Visualization
    'dpi': 300,
    'num_examples': 10,
    
    # Error map scaling
    'error_scale_sino': 10.0,  # Scale for sinogram error visualization
    'error_scale_img': 5.0,    # Scale for image error visualization
}

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    """Setup logging."""
    import logging
    log_file = os.path.join(output_dir, 'figure2_physics.log')
    
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
    
    logger = logging.getLogger('figure2_physics')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ═══════════════════════════════════════════════════════════════
# RADON TRANSFORM UTILITIES
# ═══════════════════════════════════════════════════════════════

class RadonTransform:
    """
    Wrapper for TorchRadon providing forward and back projection.
    
    Forward Projection: Image Domain → Sinogram Domain
        P(x) where x is image, P(x) is sinogram
        
    Back Projection: Sinogram Domain → Image Domain  
        P^T(y) where y is sinogram, P^T(y) is back-projected image
        Note: This is NOT filtered back projection (FBP), just adjoint
        
    Filtered Back Projection (FBP): Sinogram → Reconstructed Image
        Uses ramp filter for proper reconstruction
    """
    
    def __init__(self, img_size, num_angles=180, device='cuda'):
        if not RADON_AVAILABLE:
            raise RuntimeError("torch_radon is required")
        
        self.img_size = img_size
        self.num_angles = num_angles
        self.device = device
        
        # Create angles (0 to π)
        self.angles = np.linspace(0, np.pi, num_angles, dtype=np.float32)
        
        # Create Radon projector
        self.radon = Radon(img_size, self.angles)
        
    def forward_project(self, image):
        """
        Forward project image to sinogram domain.
        
        Args:
            image: Tensor [B, 1, H, W] or [B, H, W] in [0, 1] range
            
        Returns:
            sinogram: Tensor [B, num_angles, det_count]
        """
        # Ensure correct shape [B, H, W]
        if image.dim() == 4:
            image = image.squeeze(1)
        if image.dim() == 2:
            image = image.unsqueeze(0)
            
        image = image.to(self.device)
        sinogram = self.radon.forward(image)
        
        return sinogram
    
    def back_project(self, sinogram):
        """
        Back project sinogram to image domain (adjoint operation).
        
        Args:
            sinogram: Tensor [B, num_angles, det_count]
            
        Returns:
            image: Tensor [B, H, W]
        """
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
            
        sinogram = sinogram.to(self.device)
        image = self.radon.backprojection(sinogram)
        
        return image
    
    def fbp_reconstruct(self, sinogram):
        """
        Filtered Back Projection reconstruction.
        
        Args:
            sinogram: Tensor [B, num_angles, det_count]
            
        Returns:
            reconstructed_image: Tensor [B, H, W]
        """
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
            
        sinogram = sinogram.to(self.device)
        
        # Manual FBP implementation using newer torch.fft API
        # Apply ramp filter in frequency domain
        filtered = self._apply_ramp_filter(sinogram)
        reconstructed = self.radon.backprojection(filtered)
        
        # Ensure output matches input image size
        if reconstructed.shape[-1] != self.img_size:
            reconstructed = self._center_crop_or_pad(reconstructed, self.img_size)
        
        return reconstructed
    
    def _center_crop_or_pad(self, img, target_size):
        """Center crop or pad image to target size."""
        current_size = img.shape[-1]
        
        if current_size == target_size:
            return img
        elif current_size > target_size:
            # Center crop
            start = (current_size - target_size) // 2
            return img[:, start:start+target_size, start:start+target_size]
        else:
            # Pad
            pad_amount = (target_size - current_size) // 2
            return torch.nn.functional.pad(img, (pad_amount, pad_amount, pad_amount, pad_amount), mode='constant', value=0)
    
    def _apply_ramp_filter(self, sinogram):
        """
        Apply ramp filter to sinogram in frequency domain.
        
        The ramp filter (|ω|) is essential for FBP reconstruction.
        """
        batch_size, num_angles, det_count = sinogram.shape
        
        # Pad to next power of 2 for efficient FFT
        padded_size = 2 ** int(np.ceil(np.log2(2 * det_count)))
        pad_amount = padded_size - det_count
        
        # Pad sinogram
        padded = torch.nn.functional.pad(sinogram, (0, pad_amount), mode='constant', value=0)
        
        # Create ramp filter
        freqs = torch.fft.fftfreq(padded_size, device=self.device)
        ramp = torch.abs(freqs) * padded_size  # Scale by size
        
        # Apply filter in frequency domain
        sino_fft = torch.fft.fft(padded, dim=-1)
        filtered_fft = sino_fft * ramp.unsqueeze(0).unsqueeze(0)
        filtered = torch.fft.ifft(filtered_fft, dim=-1).real
        
        # Remove padding
        filtered = filtered[:, :, :det_count]
        
        return filtered

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array, removing batch/channel dims."""
    arr = tensor.cpu().numpy()
    # Squeeze all singleton dimensions
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.squeeze()

def to_display_range(tensor):
    """Convert from [-1, 1] to [0, 1] for display."""
    return np.clip((tensor_to_numpy(tensor) + 1) / 2, 0, 1)

def compute_metrics(pred, gt):
    """Compute PSNR, SSIM, MAE, MSE."""
    pred_np = np.clip(pred, 0, 1)
    gt_np = np.clip(gt, 0, 1)
    
    mse = np.mean((pred_np - gt_np) ** 2)
    mae = np.mean(np.abs(pred_np - gt_np))
    psnr_val = psnr_func(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim_func(gt_np, pred_np, data_range=1.0)
    
    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'MAE': mae,
        'MSE': mse,
    }

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_pipeline_figure(data, slice_idx, output_dir, logger=None):
    """
    Create comprehensive pipeline figure showing domain conversions.
    
    Layout (4 rows × 5 columns):
    
    Row 0: Image Domain Inputs/Outputs
        - Ground Truth | Corrupted Input | Baseline Output | Full Model Output | (empty)
        
    Row 1: Forward Projection (Image → Sinogram)
        - P(GT) | P(Input) | P(Baseline) | P(Full) | Arrow labels
        
    Row 2: Sinogram Residuals  
        - |P(Base)-P(GT)| | |P(Full)-P(GT)| | Difference | Per-angle error | Stats
        
    Row 3: Back Projection Verification (Sinogram → Image via FBP)
        - FBP(P(GT)) | FBP(P(Base)) | FBP(P(Full)) | Metrics | Legend
    """
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(4, 5, figure=fig, height_ratios=[1, 1, 1, 1], 
                  hspace=0.3, wspace=0.25)
    
    # Extract data
    gt = data['gt']
    ma_ct = data['ma_ct']
    out_base = data['output_baseline']
    out_full = data['output_full']
    
    sino_gt = data['sino_gt']
    sino_input = data['sino_input']
    sino_base = data['sino_baseline']
    sino_full = data['sino_full']
    
    fbp_gt = data['fbp_gt']
    fbp_base = data['fbp_baseline']
    fbp_full = data['fbp_full']
    
    # ═══════════════════════════════════════════════════════════
    # ROW 0: Image Domain
    # ═══════════════════════════════════════════════════════════
    
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(gt, cmap='gray', vmin=0, vmax=1)
    ax00.set_title('Ground Truth\n$x_{GT}$', fontsize=11, fontweight='bold')
    ax00.axis('off')
    
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(ma_ct, cmap='gray', vmin=0, vmax=1)
    ax01.set_title('Corrupted Input\n$x_{MA}$', fontsize=11, fontweight='bold')
    ax01.axis('off')
    
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(out_base, cmap='gray', vmin=0, vmax=1)
    psnr_base = data['metrics_baseline']['PSNR']
    ax02.set_title(f'No Physics Output\n$\\hat{{x}}_{{base}}$ (PSNR: {psnr_base:.2f})', fontsize=11, fontweight='bold')
    ax02.axis('off')
    
    ax03 = fig.add_subplot(gs[0, 3])
    ax03.imshow(out_full, cmap='gray', vmin=0, vmax=1)
    psnr_full = data['metrics_full']['PSNR']
    ax03.set_title(f'TransMAR-GAN Output\n$\\hat{{x}}_{{full}}$ (PSNR: {psnr_full:.2f})', fontsize=11, fontweight='bold')
    ax03.axis('off')
    
    # Row 0, Col 4: Domain label
    ax04 = fig.add_subplot(gs[0, 4])
    ax04.text(0.5, 0.5, 'IMAGE\nDOMAIN', ha='center', va='center', 
              fontsize=14, fontweight='bold', color='blue',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax04.axis('off')
    
    # ═══════════════════════════════════════════════════════════
    # ROW 1: Forward Projection (Image → Sinogram)
    # ═══════════════════════════════════════════════════════════
    
    # Normalize sinograms for display
    sino_vmin = min(sino_gt.min(), sino_base.min(), sino_full.min())
    sino_vmax = max(sino_gt.max(), sino_base.max(), sino_full.max())
    
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(sino_gt, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    ax10.set_title('$\\mathcal{P}(x_{GT})$\nClean Sinogram', fontsize=11, fontweight='bold')
    ax10.set_xlabel('Detector')
    ax10.set_ylabel('Angle')
    
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(sino_input, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    ax11.set_title('$\\mathcal{P}(x_{MA})$\nCorrupted Sinogram', fontsize=11, fontweight='bold')
    ax11.set_xlabel('Detector')
    ax11.set_ylabel('Angle')
    
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.imshow(sino_base, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    ax12.set_title('$\\mathcal{P}(\\hat{x}_{base})$\nBaseline Sinogram', fontsize=11, fontweight='bold')
    ax12.set_xlabel('Detector')
    ax12.set_ylabel('Angle')
    
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.imshow(sino_full, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    ax13.set_title('$\\mathcal{P}(\\hat{x}_{full})$\nFull Model Sinogram', fontsize=11, fontweight='bold')
    ax13.set_xlabel('Detector')
    ax13.set_ylabel('Angle')
    
    # Row 1, Col 4: Domain label with arrow
    ax14 = fig.add_subplot(gs[1, 4])
    ax14.text(0.5, 0.7, 'SINOGRAM\nDOMAIN', ha='center', va='center',
              fontsize=14, fontweight='bold', color='green',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    ax14.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.55),
                  arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax14.text(0.5, 0.15, '$\\mathcal{P}$: Forward\nProjection', ha='center', va='center', fontsize=10)
    ax14.axis('off')
    
    # ═══════════════════════════════════════════════════════════
    # ROW 2: Sinogram Residuals
    # ═══════════════════════════════════════════════════════════
    
    residual_base = np.abs(sino_base - sino_gt)
    residual_full = np.abs(sino_full - sino_gt)
    residual_vmax = max(residual_base.max(), residual_full.max())
    
    ax20 = fig.add_subplot(gs[2, 0])
    im20 = ax20.imshow(residual_base * CONFIG['error_scale_sino'], cmap='hot', aspect='auto', 
                        vmin=0, vmax=residual_vmax * CONFIG['error_scale_sino'])
    mae_base_sino = residual_base.mean()
    ax20.set_title(f'$|\\mathcal{{P}}(\\hat{{x}}_{{base}}) - \\mathcal{{P}}(x_{{GT}})|$\nMAE: {mae_base_sino:.4f}', 
                   fontsize=11, fontweight='bold')
    ax20.set_xlabel('Detector')
    ax20.set_ylabel('Angle')
    
    ax21 = fig.add_subplot(gs[2, 1])
    im21 = ax21.imshow(residual_full * CONFIG['error_scale_sino'], cmap='hot', aspect='auto',
                        vmin=0, vmax=residual_vmax * CONFIG['error_scale_sino'])
    mae_full_sino = residual_full.mean()
    ax21.set_title(f'$|\\mathcal{{P}}(\\hat{{x}}_{{full}}) - \\mathcal{{P}}(x_{{GT}})|$\nMAE: {mae_full_sino:.4f}',
                   fontsize=11, fontweight='bold')
    ax21.set_xlabel('Detector')
    ax21.set_ylabel('Angle')
    
    # Difference (baseline - full): positive = baseline worse
    ax22 = fig.add_subplot(gs[2, 2])
    diff = residual_base - residual_full
    diff_max = np.abs(diff).max()
    ax22.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-diff_max, vmax=diff_max)
    ax22.set_title('Residual Difference\n(Red = No Physics Worse)', fontsize=11, fontweight='bold')
    ax22.set_xlabel('Detector')
    ax22.set_ylabel('Angle')
    
    # Per-angle error plot
    ax23 = fig.add_subplot(gs[2, 3])
    angles = np.arange(CONFIG['num_angles'])
    per_angle_base = np.mean(residual_base, axis=1)
    per_angle_full = np.mean(residual_full, axis=1)
    ax23.plot(angles, per_angle_base, 'r-', linewidth=1.5, label='No Physics', alpha=0.8)
    ax23.plot(angles, per_angle_full, 'b-', linewidth=1.5, label='TransMAR-GAN', alpha=0.8)
    ax23.set_xlabel('Angle Index')
    ax23.set_ylabel('Mean Absolute Error')
    ax23.set_title('Per-Angle Sinogram Error', fontsize=11, fontweight='bold')
    ax23.legend(fontsize=9)
    ax23.grid(True, alpha=0.3)
    
    # Statistics box
    ax24 = fig.add_subplot(gs[2, 4])
    improvement = (mae_base_sino - mae_full_sino) / mae_base_sino * 100
    stats_text = (
        f"Sinogram Domain Metrics\n"
        f"{'='*25}\n\n"
        f"No Physics:\n"
        f"  MAE: {mae_base_sino:.6f}\n\n"
        f"TransMAR-GAN:\n"
        f"  MAE: {mae_full_sino:.6f}\n\n"
        f"Improvement:\n"
        f"  {improvement:.1f}% reduction"
    )
    ax24.text(0.1, 0.9, stats_text, transform=ax24.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    ax24.axis('off')
    
    # ═══════════════════════════════════════════════════════════
    # ROW 3: Back Projection (FBP Verification)
    # ═══════════════════════════════════════════════════════════
    
    # Normalize FBP images
    fbp_vmin = min(fbp_gt.min(), fbp_base.min(), fbp_full.min())
    fbp_vmax = max(fbp_gt.max(), fbp_base.max(), fbp_full.max())
    
    ax30 = fig.add_subplot(gs[3, 0])
    ax30.imshow(fbp_gt, cmap='gray', vmin=fbp_vmin, vmax=fbp_vmax)
    ax30.set_title('FBP$(\\mathcal{P}(x_{GT}))$\nReconstruction Check', fontsize=11, fontweight='bold')
    ax30.axis('off')
    
    ax31 = fig.add_subplot(gs[3, 1])
    ax31.imshow(fbp_base, cmap='gray', vmin=fbp_vmin, vmax=fbp_vmax)
    ax31.set_title('FBP$(\\mathcal{P}(\\hat{x}_{base}))$\nBaseline Recon', fontsize=11, fontweight='bold')
    ax31.axis('off')
    
    ax32 = fig.add_subplot(gs[3, 2])
    ax32.imshow(fbp_full, cmap='gray', vmin=fbp_vmin, vmax=fbp_vmax)
    ax32.set_title('FBP$(\\mathcal{P}(\\hat{x}_{full}))$\nFull Model Recon', fontsize=11, fontweight='bold')
    ax32.axis('off')
    
    # FBP comparison metrics
    ax33 = fig.add_subplot(gs[3, 3])
    # Compute FBP quality (how well FBP recovers original)
    fbp_psnr_gt = psnr_func(gt, np.clip(fbp_gt, 0, 1), data_range=1.0)
    fbp_psnr_base = psnr_func(out_base, np.clip(fbp_base, 0, 1), data_range=1.0)
    fbp_psnr_full = psnr_func(out_full, np.clip(fbp_full, 0, 1), data_range=1.0)
    
    labels = ['GT→FBP', 'Base→FBP', 'Full→FBP']
    psnrs = [fbp_psnr_gt, fbp_psnr_base, fbp_psnr_full]
    colors = ['green', 'red', 'blue']
    bars = ax33.bar(labels, psnrs, color=colors, alpha=0.7, edgecolor='black')
    ax33.set_ylabel('PSNR (dB)')
    ax33.set_title('FBP Reconstruction Quality\n(Higher = Better Consistency)', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, psnrs):
        ax33.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Legend/Info
    ax34 = fig.add_subplot(gs[3, 4])
    ax34.text(0.5, 0.7, 'BACK TO\nIMAGE DOMAIN', ha='center', va='center',
              fontsize=14, fontweight='bold', color='purple',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.8))
    ax34.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.55),
                  arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax34.text(0.5, 0.15, 'FBP: Filtered\nBack Projection', ha='center', va='center', fontsize=10)
    ax34.axis('off')
    
    # Main title
    fig.suptitle(f'Figure 2: Physics Consistency Pipeline - Slice {slice_idx}\n'
                 f'Forward Projection $\\mathcal{{P}}$: Image → Sinogram | '
                 f'FBP: Sinogram → Image', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    save_path = os.path.join(output_dir, f'pipeline_slice_{slice_idx}.png')
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=CONFIG['dpi'], bbox_inches='tight', format='pdf')
    plt.close()
    
    if logger:
        logger.info(f"  Saved pipeline figure: pipeline_slice_{slice_idx}.png")
    
    return save_path


def create_sinogram_comparison_figure(data, slice_idx, output_dir, logger=None):
    """
    Create detailed sinogram comparison figure.
    
    Shows sinograms side-by-side with profiles.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    sino_gt = data['sino_gt']
    sino_base = data['sino_baseline']
    sino_full = data['sino_full']
    
    num_angles, num_detectors = sino_gt.shape
    
    # Normalize
    sino_vmin = min(sino_gt.min(), sino_base.min(), sino_full.min())
    sino_vmax = max(sino_gt.max(), sino_base.max(), sino_full.max())
    
    # Row 0: Sinograms
    axes[0, 0].imshow(sino_gt, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    axes[0, 0].set_title('Ground Truth Sinogram\n$\\mathcal{P}(x_{GT})$', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Detector')
    axes[0, 0].set_ylabel('Angle')
    
    axes[0, 1].imshow(sino_base, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    axes[0, 1].set_title('No Physics Sinogram\n$\\mathcal{P}(\\hat{x}_{base})$', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Detector')
    axes[0, 1].set_ylabel('Angle')
    
    axes[0, 2].imshow(sino_full, cmap='gray', aspect='auto', vmin=sino_vmin, vmax=sino_vmax)
    axes[0, 2].set_title('TransMAR-GAN Sinogram\n$\\mathcal{P}(\\hat{x}_{full})$', fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel('Detector')
    axes[0, 2].set_ylabel('Angle')
    
    # Difference map
    diff = sino_base - sino_full
    axes[0, 3].imshow(diff, cmap='RdBu_r', aspect='auto', 
                      vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[0, 3].set_title('Sinogram Difference\n(No Physics - TransMAR-GAN)', fontsize=11, fontweight='bold')
    axes[0, 3].set_xlabel('Detector')
    axes[0, 3].set_ylabel('Angle')
    
    # Row 1: Profiles
    angle_idx = num_angles // 2  # Middle angle
    detector_idx = num_detectors // 2  # Center detector
    
    # Detector profile (fixed angle)
    axes[1, 0].plot(sino_gt[angle_idx, :], 'g-', linewidth=2, label='GT', alpha=0.8)
    axes[1, 0].plot(sino_base[angle_idx, :], 'r-', linewidth=1.5, label='No Physics', alpha=0.7)
    axes[1, 0].plot(sino_full[angle_idx, :], 'b-', linewidth=1.5, label='TransMAR-GAN', alpha=0.7)
    axes[1, 0].set_xlabel('Detector Index')
    axes[1, 0].set_ylabel('Projection Value')
    axes[1, 0].set_title(f'Detector Profile (Angle={angle_idx})', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Angle profile (fixed detector)
    axes[1, 1].plot(sino_gt[:, detector_idx], 'g-', linewidth=2, label='GT', alpha=0.8)
    axes[1, 1].plot(sino_base[:, detector_idx], 'r-', linewidth=1.5, label='No Physics', alpha=0.7)
    axes[1, 1].plot(sino_full[:, detector_idx], 'b-', linewidth=1.5, label='TransMAR-GAN', alpha=0.7)
    axes[1, 1].set_xlabel('Angle Index')
    axes[1, 1].set_ylabel('Projection Value')
    axes[1, 1].set_title(f'Angle Profile (Detector={detector_idx})', fontsize=11, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Per-angle MAE
    per_angle_base = np.mean(np.abs(sino_base - sino_gt), axis=1)
    per_angle_full = np.mean(np.abs(sino_full - sino_gt), axis=1)
    axes[1, 2].plot(per_angle_base, 'r-', linewidth=1.5, label=f'No Physics (Mean: {per_angle_base.mean():.4f})')
    axes[1, 2].plot(per_angle_full, 'b-', linewidth=1.5, label=f'TransMAR-GAN (Mean: {per_angle_full.mean():.4f})')
    axes[1, 2].set_xlabel('Angle Index')
    axes[1, 2].set_ylabel('Mean Absolute Error')
    axes[1, 2].set_title('Per-Angle MAE', fontsize=11, fontweight='bold')
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Histogram of residuals
    residual_base = (sino_base - sino_gt).flatten()
    residual_full = (sino_full - sino_gt).flatten()
    bins = np.linspace(min(residual_base.min(), residual_full.min()),
                       max(residual_base.max(), residual_full.max()), 100)
    axes[1, 3].hist(residual_base, bins=bins, alpha=0.5, color='red', density=True, label='No Physics')
    axes[1, 3].hist(residual_full, bins=bins, alpha=0.5, color='blue', density=True, label='TransMAR-GAN')
    axes[1, 3].set_xlabel('Sinogram Residual')
    axes[1, 3].set_ylabel('Density')
    axes[1, 3].set_title('Residual Distribution', fontsize=11, fontweight='bold')
    axes[1, 3].legend(fontsize=9)
    axes[1, 3].grid(True, alpha=0.3)
    
    fig.suptitle(f'Sinogram Analysis - Slice {slice_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, f'sinogram_analysis_slice_{slice_idx}.png')
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    if logger:
        logger.info(f"  Saved sinogram analysis: sinogram_analysis_slice_{slice_idx}.png")


def create_image_domain_comparison(data, slice_idx, output_dir, logger=None):
    """
    Create image domain comparison with error maps.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    gt = data['gt']
    ma_ct = data['ma_ct']
    out_base = data['output_baseline']
    out_full = data['output_full']
    
    # Row 0: Images
    axes[0, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ma_ct, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Corrupted Input', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(out_base, cmap='gray', vmin=0, vmax=1)
    psnr_base = data['metrics_baseline']['PSNR']
    ssim_base = data['metrics_baseline']['SSIM']
    axes[0, 2].set_title(f'No Physics\nPSNR: {psnr_base:.2f} | SSIM: {ssim_base:.4f}', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(out_full, cmap='gray', vmin=0, vmax=1)
    psnr_full = data['metrics_full']['PSNR']
    ssim_full = data['metrics_full']['SSIM']
    axes[0, 3].set_title(f'TransMAR-GAN\nPSNR: {psnr_full:.2f} | SSIM: {ssim_full:.4f}', fontsize=11, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 1: Error maps (5x scaled)
    error_base = np.abs(out_base - gt) * CONFIG['error_scale_img']
    error_full = np.abs(out_full - gt) * CONFIG['error_scale_img']
    error_input = np.abs(ma_ct - gt) * CONFIG['error_scale_img']
    
    axes[1, 0].imshow(np.clip(error_input, 0, 1), cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Input Error ({int(CONFIG["error_scale_img"])}x)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].axis('off')  # Empty
    
    axes[1, 2].imshow(np.clip(error_base, 0, 1), cmap='jet', vmin=0, vmax=1)
    mae_base = data['metrics_baseline']['MAE']
    axes[1, 2].set_title(f'No Physics Error\nMAE: {mae_base:.4f}', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.clip(error_full, 0, 1), cmap='jet', vmin=0, vmax=1)
    mae_full = data['metrics_full']['MAE']
    axes[1, 3].set_title(f'TransMAR-GAN Error\nMAE: {mae_full:.4f}', fontsize=11, fontweight='bold')
    axes[1, 3].axis('off')
    
    fig.suptitle(f'Image Domain Comparison - Slice {slice_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, f'image_comparison_slice_{slice_idx}.png')
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    if logger:
        logger.info(f"  Saved image comparison: image_comparison_slice_{slice_idx}.png")


def create_summary_figure(all_results, output_dir, logger=None):
    """
    Create summary figure comparing all slices.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Collect metrics
    sino_mae_base = [r['sino_metrics_baseline']['MAE'] for r in all_results]
    sino_mae_full = [r['sino_metrics_full']['MAE'] for r in all_results]
    img_psnr_base = [r['metrics_baseline']['PSNR'] for r in all_results]
    img_psnr_full = [r['metrics_full']['PSNR'] for r in all_results]
    img_ssim_base = [r['metrics_baseline']['SSIM'] for r in all_results]
    img_ssim_full = [r['metrics_full']['SSIM'] for r in all_results]
    
    slice_indices = [r['slice_idx'] for r in all_results]
    
    # Sinogram MAE comparison
    x = np.arange(len(all_results))
    width = 0.35
    axes[0, 0].bar(x - width/2, sino_mae_base, width, label='No Physics', color='red', alpha=0.7)
    axes[0, 0].bar(x + width/2, sino_mae_full, width, label='TransMAR-GAN', color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Slice')
    axes[0, 0].set_ylabel('Sinogram MAE')
    axes[0, 0].set_title('Sinogram Domain MAE', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([str(i) for i in slice_indices], fontsize=8)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Image PSNR comparison
    axes[0, 1].bar(x - width/2, img_psnr_base, width, label='No Physics', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, img_psnr_full, width, label='TransMAR-GAN', color='blue', alpha=0.7)
    axes[0, 1].set_xlabel('Slice')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('Image Domain PSNR', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(i) for i in slice_indices], fontsize=8)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Scatter: Sinogram MAE vs Image PSNR
    axes[1, 0].scatter(sino_mae_base, img_psnr_base, c='red', s=60, alpha=0.7, label='No Physics')
    axes[1, 0].scatter(sino_mae_full, img_psnr_full, c='blue', s=60, alpha=0.7, label='TransMAR-GAN')
    axes[1, 0].set_xlabel('Sinogram MAE')
    axes[1, 0].set_ylabel('Image PSNR (dB)')
    axes[1, 0].set_title('Sinogram MAE vs Image PSNR', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    ax_stats = axes[1, 1]
    ax_stats.axis('off')
    
    sino_improvement = (np.mean(sino_mae_base) - np.mean(sino_mae_full)) / np.mean(sino_mae_base) * 100
    psnr_improvement = np.mean(img_psnr_full) - np.mean(img_psnr_base)
    ssim_improvement = np.mean(img_ssim_full) - np.mean(img_ssim_base)
    
    stats_text = (
        f"Summary Statistics ({len(all_results)} slices)\n"
        f"{'='*40}\n\n"
        f"SINOGRAM DOMAIN (MAE):\n"
        f"  No Physics:    {np.mean(sino_mae_base):.6f} ± {np.std(sino_mae_base):.6f}\n"
        f"  TransMAR-GAN:  {np.mean(sino_mae_full):.6f} ± {np.std(sino_mae_full):.6f}\n"
        f"  Improvement:   {sino_improvement:.1f}%\n\n"
        f"IMAGE DOMAIN (PSNR):\n"
        f"  No Physics:    {np.mean(img_psnr_base):.2f} ± {np.std(img_psnr_base):.2f} dB\n"
        f"  TransMAR-GAN:  {np.mean(img_psnr_full):.2f} ± {np.std(img_psnr_full):.2f} dB\n"
        f"  Improvement:   +{psnr_improvement:.2f} dB\n\n"
        f"IMAGE DOMAIN (SSIM):\n"
        f"  No Physics:    {np.mean(img_ssim_base):.4f} ± {np.std(img_ssim_base):.4f}\n"
        f"  TransMAR-GAN:  {np.mean(img_ssim_full):.4f} ± {np.std(img_ssim_full):.4f}\n"
        f"  Improvement:   +{ssim_improvement:.4f}"
    )
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, fontsize=11,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    fig.suptitle('Physics Consistency: Summary Across All Slices', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, 'summary_comparison.png')
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=CONFIG['dpi'], bbox_inches='tight', format='pdf')
    plt.close()
    
    if logger:
        logger.info(f"  Saved summary: summary_comparison.png")


def save_individual_panels(data, slice_idx, output_dir, logger=None):
    """Save individual high-res panels."""
    panel_dir = os.path.join(output_dir, 'individual_panels', f'slice_{slice_idx}')
    os.makedirs(panel_dir, exist_ok=True)
    
    # Save each component
    components = {
        'gt': data['gt'],
        'ma_ct': data['ma_ct'],
        'output_baseline': data['output_baseline'],
        'output_full': data['output_full'],
        'sino_gt': data['sino_gt'],
        'sino_baseline': data['sino_baseline'],
        'sino_full': data['sino_full'],
        'fbp_gt': data['fbp_gt'],
        'fbp_baseline': data['fbp_baseline'],
        'fbp_full': data['fbp_full'],
    }
    
    for name, arr in components.items():
        # Save as numpy
        np.save(os.path.join(panel_dir, f'{name}.npy'), arr)
        
        # Save as image
        fig, ax = plt.subplots(figsize=(6, 6))
        if 'sino' in name:
            ax.imshow(arr, cmap='gray', aspect='auto')
        else:
            ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        plt.savefig(os.path.join(panel_dir, f'{name}.png'), dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    if logger:
        logger.debug(f"    Saved individual panels to: individual_panels/slice_{slice_idx}/")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'individual_panels'), exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("FIGURE 2: Physics Consistency Visualization (V2)")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    
    # Check torch_radon
    if not RADON_AVAILABLE:
        logger.error("torch_radon not available! Cannot proceed.")
        return
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load models
    logger.info("")
    logger.info("Loading models...")
    
    models = {}
    for model_id, model_info in CONFIG['models'].items():
        if not os.path.exists(model_info['path']):
            logger.warning(f"  Checkpoint not found: {model_info['path']}")
            continue
        logger.info(f"  Loading {model_info['short_name']}...")
        models[model_id] = {
            'model': load_generator(model_info['path'], device),
            'name': model_info['name'],
            'short_name': model_info['short_name'],
        }
        logger.info(f"    ✓ Loaded")
    
    if len(models) < 2:
        logger.error("Need both baseline and full_model checkpoints!")
        return
    
    # Create Radon transform
    # Use the dataset image size (416x416) not the model config size (128)
    # because TestDataset returns 416x416 images
    logger.info("")
    logger.info("Initializing Radon transform...")
    img_size = CONFIG['dataset_img_size']  # Use 416 to match TestDataset output
    radon = RadonTransform(img_size, CONFIG['num_angles'], device)
    logger.info(f"  Image size: {img_size}x{img_size}")
    logger.info(f"  Number of angles: {CONFIG['num_angles']}")
    
    # Load test dataset
    logger.info("")
    logger.info("Loading test dataset...")
    test_dataset = load_test_dataset()
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Get selected slices
    selected_indices = get_selected_slices()
    logger.info(f"  Using {len(selected_indices)} selected slices")
    
    # Process slices
    logger.info("")
    logger.info(f"Processing {CONFIG['num_examples']} slices...")
    
    all_results = []
    
    with torch.no_grad():
        for i, idx in enumerate(selected_indices[:CONFIG['num_examples']]):
            logger.info(f"")
            logger.info(f"Slice {idx} ({i+1}/{CONFIG['num_examples']})...")
            
            # Get data
            batch = test_dataset[idx]
            ma_ct = batch[0].unsqueeze(0).to(device)  # [1, 1, H, W]
            gt = batch[1].unsqueeze(0).to(device)
            
            # Run models
            out_base = models['baseline']['model'](ma_ct)
            out_full = models['full_model']['model'](ma_ct)
            
            # Convert to [0, 1] for Radon (expects non-negative)
            gt_01 = denormalize(gt).clamp(0, 1)
            ma_01 = denormalize(ma_ct).clamp(0, 1)
            base_01 = denormalize(out_base).clamp(0, 1)
            full_01 = denormalize(out_full).clamp(0, 1)
            
            # Forward project to sinogram domain
            sino_gt = radon.forward_project(gt_01)
            sino_input = radon.forward_project(ma_01)
            sino_base = radon.forward_project(base_01)
            sino_full = radon.forward_project(full_01)
            
            # Back project (FBP) to verify consistency
            fbp_gt = radon.fbp_reconstruct(sino_gt)
            fbp_base = radon.fbp_reconstruct(sino_base)
            fbp_full = radon.fbp_reconstruct(sino_full)
            
            # Normalize FBP outputs
            fbp_gt = fbp_gt / fbp_gt.max() if fbp_gt.max() > 0 else fbp_gt
            fbp_base = fbp_base / fbp_base.max() if fbp_base.max() > 0 else fbp_base
            fbp_full = fbp_full / fbp_full.max() if fbp_full.max() > 0 else fbp_full
            
            # Convert to numpy
            gt_np = tensor_to_numpy(gt_01)
            ma_np = tensor_to_numpy(ma_01)
            base_np = tensor_to_numpy(base_01)
            full_np = tensor_to_numpy(full_01)
            
            sino_gt_np = tensor_to_numpy(sino_gt)
            sino_input_np = tensor_to_numpy(sino_input)
            sino_base_np = tensor_to_numpy(sino_base)
            sino_full_np = tensor_to_numpy(sino_full)
            
            fbp_gt_np = tensor_to_numpy(fbp_gt)
            fbp_base_np = tensor_to_numpy(fbp_base)
            fbp_full_np = tensor_to_numpy(fbp_full)
            
            # Compute metrics
            metrics_base = compute_metrics(base_np, gt_np)
            metrics_full = compute_metrics(full_np, gt_np)
            
            sino_metrics_base = {
                'MAE': np.mean(np.abs(sino_base_np - sino_gt_np)),
                'MSE': np.mean((sino_base_np - sino_gt_np) ** 2),
            }
            sino_metrics_full = {
                'MAE': np.mean(np.abs(sino_full_np - sino_gt_np)),
                'MSE': np.mean((sino_full_np - sino_gt_np) ** 2),
            }
            
            logger.info(f"  Image: Base PSNR={metrics_base['PSNR']:.2f}, Full PSNR={metrics_full['PSNR']:.2f}")
            logger.info(f"  Sino:  Base MAE={sino_metrics_base['MAE']:.6f}, Full MAE={sino_metrics_full['MAE']:.6f}")
            
            # Store results
            result = {
                'slice_idx': idx,
                'gt': gt_np,
                'ma_ct': ma_np,
                'output_baseline': base_np,
                'output_full': full_np,
                'sino_gt': sino_gt_np,
                'sino_input': sino_input_np,
                'sino_baseline': sino_base_np,
                'sino_full': sino_full_np,
                'fbp_gt': fbp_gt_np,
                'fbp_baseline': fbp_base_np,
                'fbp_full': fbp_full_np,
                'metrics_baseline': metrics_base,
                'metrics_full': metrics_full,
                'sino_metrics_baseline': sino_metrics_base,
                'sino_metrics_full': sino_metrics_full,
            }
            all_results.append(result)
            
            # Create visualizations for this slice
            create_pipeline_figure(result, idx, output_dir, logger)
            create_sinogram_comparison_figure(result, idx, output_dir, logger)
            create_image_domain_comparison(result, idx, output_dir, logger)
            save_individual_panels(result, idx, output_dir, logger)
    
    # Create summary
    logger.info("")
    logger.info("Creating summary figures...")
    create_summary_figure(all_results, output_dir, logger)
    
    # Save metrics CSV
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Slice', 'Base_PSNR', 'Full_PSNR', 'Base_SSIM', 'Full_SSIM',
                        'Base_Sino_MAE', 'Full_Sino_MAE', 'Sino_Improvement%'])
        for r in all_results:
            sino_imp = (r['sino_metrics_baseline']['MAE'] - r['sino_metrics_full']['MAE']) / r['sino_metrics_baseline']['MAE'] * 100
            writer.writerow([
                r['slice_idx'],
                f"{r['metrics_baseline']['PSNR']:.4f}",
                f"{r['metrics_full']['PSNR']:.4f}",
                f"{r['metrics_baseline']['SSIM']:.4f}",
                f"{r['metrics_full']['SSIM']:.4f}",
                f"{r['sino_metrics_baseline']['MAE']:.6f}",
                f"{r['sino_metrics_full']['MAE']:.6f}",
                f"{sino_imp:.2f}",
            ])
    logger.info(f"  Saved metrics: metrics.csv")
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            'num_angles': CONFIG['num_angles'],
            'num_examples': len(all_results),
            'models': {k: v['path'] for k, v in CONFIG['models'].items()},
        }, f, indent=2)
    
    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed {len(all_results)} slices")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - pipeline_slice_*.png/pdf: Complete domain conversion pipeline")
    logger.info("  - sinogram_analysis_slice_*.png: Sinogram comparisons & profiles")
    logger.info("  - image_comparison_slice_*.png: Image domain comparisons")
    logger.info("  - summary_comparison.png/pdf: Aggregated statistics")
    logger.info("  - individual_panels/: High-res individual components")
    logger.info("  - metrics.csv: All metrics")
    logger.info("  - config.json")


if __name__ == "__main__":
    main()
