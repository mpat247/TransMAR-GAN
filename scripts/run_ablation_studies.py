"""
Loss-Term Ablation Studies Script
==================================

This script runs ablation studies sequentially:

  Loss-Term Ablations:
  A1: No Physics Loss (λ_phys = 0)
  A2: No Metal-Consistency Loss (λ_metal = 0)
  A3: No Metal-Aware Weighting (w = 1)
  A4: No Adversarial Loss (λ_adv = 0)
  A5: No Feature Matching Loss (λ_FM = 0)
  A6: No Edge Loss (λ_edge = 0)
  A7: Hinge GAN Loss (Default)
  A8: Vanilla GAN (replace hinge with BCE)

  Architecture/Design Ablations:
  B1: Single-Scale Discriminator (no multi-scale ensemble)
  B2: No Spectral Normalization (tests SN regularization)
  B3: Different Dilation Radii for Metal Band (r ∈ {0, 3, 5, 7})

Each ablation:
  - Trains for 25 epochs
  - Saves checkpoints every 5 epochs
  - Logs all metrics to CSV
  - Generates loss curves, metric plots, difference maps, histograms
  - Computes regional metrics (metal/band/non-metal)
  - Saves comprehensive data for journal paper

Usage:
  python run_loss_ablations.py
"""

import os
import sys
import json
import csv
import time
import logging
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
from torch_radon import Radon
from skimage.metrics import structural_similarity as ssim_func


# ═══════════════════════════════════════════════════════════════
# CUSTOM LOGGER CLASS
# ═══════════════════════════════════════════════════════════════
class AblationLogger:
    """Logger that writes to both console and file with timestamps."""
    
    def __init__(self, log_dir, name="ablation"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s │ %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level - logs everything)
        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s │ %(levelname)-8s │ %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
        
    def info(self, msg):
        self.logger.info(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)
    
    def section(self, title, char='═'):
        """Print a section header."""
        line = char * 70
        self.info(line)
        self.info(f"  {title}")
        self.info(line)
    
    def subsection(self, title, char='─'):
        """Print a subsection header."""
        line = char * 50
        self.info(line)
        self.info(f"  {title}")
        self.info(line)
    
    def metrics(self, metrics_dict, prefix=""):
        """Log metrics in a formatted way."""
        items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                 for k, v in metrics_dict.items()]
        msg = f"{prefix}{', '.join(items)}"
        self.info(msg)
    
    def progress(self, current, total, prefix="", metrics=None):
        """Log progress with optional metrics."""
        pct = current / total * 100
        msg = f"{prefix}[{current}/{total}] ({pct:.1f}%)"
        if metrics:
            items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                     for k, v in metrics.items()]
            msg += f" │ {', '.join(items)}"
        self.info(msg)

# ═══════════════════════════════════════════════════════════════
# IMPORTS: Models and Loss Functions
# ═══════════════════════════════════════════════════════════════
from models.generator.ngswin import NGswin
from models.discriminator.ms_patchgan import MultiScaleDiscriminator
from data.datasets import MARTrainDataset, MARValDataset, TestDataset

from losses.gan_losses import (
    hinge_d_loss,
    hinge_g_loss,
    feature_matching_loss,
    extract_metal_mask,
    dilate_mask,
    compute_metal_aware_loss,
    compute_weight_map,
    compute_metal_aware_edge_loss,
    metal_consistency_loss,
    physics_loss_syn,
)

# ═══════════════════════════════════════════════════════════════
# VANILLA GAN LOSSES (for A7 ablation)
# ═══════════════════════════════════════════════════════════════
def vanilla_d_loss(real_logits, fake_logits):
    """BCE-based discriminator loss (vanilla GAN)"""
    loss_real = 0.0
    loss_fake = 0.0
    for r_logit, f_logit in zip(real_logits, fake_logits):
        # Apply sigmoid for BCE
        loss_real += F.binary_cross_entropy_with_logits(
            r_logit, torch.ones_like(r_logit)
        )
        loss_fake += F.binary_cross_entropy_with_logits(
            f_logit, torch.zeros_like(f_logit)
        )
    return loss_real + loss_fake

def vanilla_g_loss(fake_logits):
    """BCE-based generator loss (non-saturating vanilla GAN)"""
    loss = 0.0
    for f_logit in fake_logits:
        loss += F.binary_cross_entropy_with_logits(
            f_logit, torch.ones_like(f_logit)
        )
    return loss

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset paths
SYNDEEPLESION_PATH = "/home/Drive-D/SynDeepLesion/"

# Training settings
NUM_EPOCHS = 10
BATCH_SIZE = 4
PATCH_SIZE = 128
NUM_WORKERS = 4
SAVE_EVERY = 2  # Save checkpoints/plots every N epochs (more frequent)
SAMPLE_EVERY = 200  # Save samples every N iterations (more frequent)
LOG_EVERY = 50  # Print logs every N iterations

# ─────────────────────────────────────────────────────────────────
# FIXED SAMPLE INDICES FOR CONSISTENT VISUALIZATION ACROSS ABLATIONS
# These 50 samples will be visualized for ALL ablations to ensure fair comparison
# ─────────────────────────────────────────────────────────────────
VIS_SAMPLE_INDICES = set([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,           # First 10
    40, 80, 120, 160, 200, 240, 280, 320,   # Evenly spaced
    360, 400, 440, 480, 520, 560, 600,      # More evenly spaced
    640, 680, 720, 760, 800, 840, 880,      # Continue
    920, 960, 1000, 1040, 1080, 1120,       # Continue
    1160, 1200, 1240, 1280, 1320, 1360,     # Continue
    1400, 1440, 1480, 1520, 1560, 1600,     # Continue
    1640, 1680, 1720, 1760, 1800,           # Continue to end
])
NUM_VIS_SAMPLES = len(VIS_SAMPLE_INDICES)  # Should be ~50

# Default hyperparameters (baseline)
DEFAULT_CONFIG = {
    'lambda_adv': 0.1,
    'lambda_FM': 10.0,
    'lambda_rec': 1.0,
    'lambda_edge': 0.2,
    'lambda_phys': 0.02,
    'lambda_metal': 0.5,
    'metal_threshold': 0.6,
    'dilation_radius': 5,
    'beta_weight': 1.0,
    'w_max': 3.0,
    'lrG': 1e-4,
    'lrD': 2e-4,
    'beta1': 0.5,
    'beta2': 0.999,
    'use_hinge_loss': True,
    'use_metal_weighting': True,
    'num_disc_scales': 3,       # Multi-scale discriminator (1, 1/2, 1/4)
    'use_spectral_norm': True,  # Spectral normalization in discriminator
}

# ═══════════════════════════════════════════════════════════════
# ABLATION CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# INFERENCE-ONLY ABLATIONS (already trained, just run test)
# ─────────────────────────────────────────────────────────────────
INFERENCE_ONLY_ABLATIONS = {
    'A0_baseline': {
        'name': 'Baseline (Full Model)',
        'description': 'Full NgSwinGAN with all loss terms. Pre-trained checkpoint - inference only.',
        'changes': {},  # Use all defaults
        'checkpoint_path': './ablation_results/loss_ablations_20251206_213904/A0_baseline/checkpoints/best_model.pth',
    },
}

# ─────────────────────────────────────────────────────────────────
# TRAINING ABLATIONS (will train from scratch)
# ─────────────────────────────────────────────────────────────────

ABLATIONS = {
    # ─────────────────────────────────────────────────────────────
    # COMPLETED ABLATIONS (commented out to resume from A3)
    # ─────────────────────────────────────────────────────────────
    # 'A0_mse_only': {
    #     'name': 'MSE/MAE Loss Only',
    #     'description': 'Train with only pixel-wise MSE/MAE loss. No adversarial, FM, edge, physics, or metal losses.',
    #     'changes': {
    #         'lambda_adv': 0.0,
    #         'lambda_FM': 0.0,
    #         'lambda_edge': 0.0,
    #         'lambda_phys': 0.0,
    #         'lambda_metal': 0.0,
    #         'use_metal_weighting': False,  # Just simple L1/L2 loss
    #     },
    # },
    # 'A1_no_physics': {
    #     'name': 'No Physics Loss',
    #     'description': 'λ_phys = 0. Evaluates projection-domain consistency.',
    #     'changes': {'lambda_phys': 0.0},
    # },
    # 'A2_no_metal_consistency': {
    #     'name': 'No Metal-Consistency Loss',
    #     'description': 'λ_metal = 0. Tests HU preservation inside metal.',
    #     'changes': {'lambda_metal': 0.0},
    # },
    # ─────────────────────────────────────────────────────────────
    # COMPLETED ABLATIONS
    # ─────────────────────────────────────────────────────────────
    # 'A3_no_metal_weighting': {
    #     'name': 'No Metal-Aware Weighting',
    #     'description': 'w = 1 (uniform). Tests importance of boundary emphasis.',
    #     'changes': {'use_metal_weighting': False},
    # },
    # 'A4_no_adversarial': {
    #     'name': 'No Adversarial Loss',
    #     'description': 'λ_adv = 0. Shows importance of GAN supervision.',
    #     'changes': {'lambda_adv': 0.0},
    # },
    # 'A5_no_feature_matching': {
    #     'name': 'No Feature Matching Loss',
    #     'description': 'λ_FM = 0. Tests GAN stability.',
    #     'changes': {'lambda_FM': 0.0},
    # },
    # ─────────────────────────────────────────────────────────────
    # RESUME FROM HERE
    # ─────────────────────────────────────────────────────────────
    'A6_no_edge': {
        'name': 'No Edge Loss',
        'description': 'λ_edge = 0. Evaluates anatomical boundary preservation.',
        'changes': {'lambda_edge': 0.0},
    },
    'A7_hinge_gan': {
        'name': 'Hinge GAN Loss (Default)',
        'description': '''Hinge-based adversarial formulation (our default):
        Discriminator (hinge loss): L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(x)))]
        Generator (hinge adversarial): L_G,adv = -E[D(G(x))]
        This is the default configuration used in SGAMARN.''',
        'changes': {'use_hinge_loss': True},  # Explicit hinge (default)
    },
    'A8_vanilla_gan': {
        'name': 'Vanilla GAN (BCE Loss)',
        'description': '''Classical non-saturating GAN formulation:
        Discriminator (vanilla): L_D = -E[log D(x)] - E[log(1 - D(G(x)))]
        Generator (vanilla adversarial): L_G,adv = -E[log D(G(x))]
        Replace hinge loss with BCE-based vanilla GAN loss.''',
        'changes': {'use_hinge_loss': False},
    },
    # ─────────────────────────────────────────────────────────────
    # ARCHITECTURE/DESIGN ABLATIONS (B-series)
    # ─────────────────────────────────────────────────────────────
    'B1_single_scale_disc': {
        'name': 'Single-Scale Discriminator',
        'description': '''Replaces the multi-scale PatchGAN ensemble with a single discriminator.
        Expected outcome: weaker global artifact suppression, reduced high-frequency fidelity,
        and less stable adversarial training.''',
        'changes': {'num_disc_scales': 1},  # Single scale instead of 3
    },
    'B2_no_spectral_norm': {
        'name': 'No Spectral Normalization',
        'description': '''Tests the effect of SN regularization on the discriminator.
        Expected outcome: gradient explosion, oscillatory discriminator behavior, and degraded
        convergence.''',
        'changes': {'use_spectral_norm': False},  # Disable SN in discriminator
    },
    'B3_dilation_r0': {
        'name': 'Dilation Radius r=0',
        'description': '''Evaluates sensitivity to the ring size used in metal-aware weighting.
        r=0 means no dilation (only exact metal pixels weighted).''',
        'changes': {'dilation_radius': 0},
    },
    'B3_dilation_r3': {
        'name': 'Dilation Radius r=3',
        'description': '''Evaluates sensitivity to the ring size used in metal-aware weighting.
        r=3 is a smaller dilation radius.''',
        'changes': {'dilation_radius': 3},
    },
    'B3_dilation_r5': {
        'name': 'Dilation Radius r=5 (Default)',
        'description': '''Evaluates sensitivity to the ring size used in metal-aware weighting.
        r=5 is the default dilation radius.''',
        'changes': {'dilation_radius': 5},
    },
    'B3_dilation_r7': {
        'name': 'Dilation Radius r=7',
        'description': '''Evaluates sensitivity to the ring size used in metal-aware weighting.
        r=7 is a larger dilation radius.''',
        'changes': {'dilation_radius': 7},
    },
}

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_dirs(base_dir):
    """Create all necessary directories for an ablation."""
    dirs = [
        base_dir,
        os.path.join(base_dir, 'checkpoints'),
        os.path.join(base_dir, 'samples'),
        os.path.join(base_dir, 'test_examples'),
        os.path.join(base_dir, 'loss_curves'),
        os.path.join(base_dir, 'metric_plots'),
        os.path.join(base_dir, 'difference_maps'),
        os.path.join(base_dir, 'histograms'),
        os.path.join(base_dir, 'regional_metrics'),
        os.path.join(base_dir, 'intensity_profiles'),      # Intensity profile plots
        os.path.join(base_dir, 'slice_analysis'),          # Comprehensive slice analysis
        os.path.join(base_dir, 'error_heatmaps'),          # Error heatmaps
        os.path.join(base_dir, 'intensity_segmentation'),  # Tissue segmentation
        os.path.join(base_dir, 'metal_artifact_waves'),    # Intensity waves through metal
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def weights_init(m):
    """Initialize weights for Conv and BatchNorm layers."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

def denormalize(x):
    """Convert from [-1, 1] to [0, 1]."""
    return (x + 1.0) / 2.0

def compute_psnr(pred, target):
    """Compute PSNR between two tensors in [-1, 1] range."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    # Data range is 2 for [-1, 1]
    return 10 * np.log10(4.0 / mse)

def compute_ssim(pred, target):
    """Compute SSIM between two tensors."""
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Handle batch dimension
    if pred_np.ndim == 3:
        ssim_vals = []
        for i in range(pred_np.shape[0]):
            ssim_val = ssim_func(
                target_np[i], pred_np[i],
                data_range=2.0,  # [-1, 1] range
                win_size=7
            )
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    else:
        return ssim_func(target_np, pred_np, data_range=2.0, win_size=7)

def compute_metrics(pred, target):
    """Compute all metrics between pred and target."""
    mse = F.mse_loss(pred, target).item()
    rmse = np.sqrt(mse)
    mae = F.l1_loss(pred, target).item()
    psnr = compute_psnr(pred, target)
    ssim_val = compute_ssim(pred, target)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'PSNR': psnr,
        'SSIM': ssim_val,
    }

def compute_regional_metrics(pred, target, ct, threshold=0.6, radius=5):
    """Compute metrics for metal, band, and non-metal regions."""
    # Get masks
    M = extract_metal_mask(ct, threshold=threshold)
    B = dilate_mask(M, radius=radius)
    band = B - M  # Band around metal (excluding interior)
    non_metal = 1.0 - B  # Everything outside the dilated region
    
    results = {}
    
    # Metal region metrics
    if M.sum() > 0:
        metal_pred = pred * M
        metal_target = target * M
        metal_mse = (((metal_pred - metal_target) ** 2) * M).sum() / M.sum()
        results['metal_MSE'] = metal_mse.item()
        results['metal_PSNR'] = 10 * np.log10(4.0 / (metal_mse.item() + 1e-10))
    else:
        results['metal_MSE'] = 0.0
        results['metal_PSNR'] = 0.0
    
    # Band region metrics
    if band.sum() > 0:
        band_pred = pred * band
        band_target = target * band
        band_mse = (((band_pred - band_target) ** 2) * band).sum() / band.sum()
        results['band_MSE'] = band_mse.item()
        results['band_PSNR'] = 10 * np.log10(4.0 / (band_mse.item() + 1e-10))
    else:
        results['band_MSE'] = 0.0
        results['band_PSNR'] = 0.0
    
    # Non-metal region metrics
    if non_metal.sum() > 0:
        nm_pred = pred * non_metal
        nm_target = target * non_metal
        nm_mse = (((nm_pred - nm_target) ** 2) * non_metal).sum() / non_metal.sum()
        results['non_metal_MSE'] = nm_mse.item()
        results['non_metal_PSNR'] = 10 * np.log10(4.0 / (nm_mse.item() + 1e-10))
    else:
        results['non_metal_MSE'] = 0.0
        results['non_metal_PSNR'] = 0.0
    
    return results

# ═══════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def format_axis_thousands(ax, axis='x'):
    """Format axis ticks to use 'k' for thousands (e.g., 10000 -> 10k)."""
    from matplotlib.ticker import FuncFormatter
    
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.0f}k'
        return f'{x:.0f}'
    
    if axis == 'x' or axis == 'both':
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    if axis == 'y' or axis == 'both':
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

def format_axis_scientific(ax, axis='x'):
    """Format axis to use scientific notation for very large numbers."""
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    
    if axis == 'x' or axis == 'both':
        ax.xaxis.set_major_formatter(formatter)
    if axis == 'y' or axis == 'both':
        ax.yaxis.set_major_formatter(formatter)

def auto_format_axis(ax, max_val, axis='x'):
    """Automatically format axis based on max value."""
    if max_val >= 1000:
        format_axis_thousands(ax, axis)
    # Also rotate x-tick labels if they might overlap
    if axis == 'x' or axis == 'both':
        ax.tick_params(axis='x', rotation=0)

def plot_loss_curves(loss_history, save_path, epoch):
    """Plot all loss curves."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Loss Curves - Epoch {epoch}', fontsize=16)
    
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    losses_to_plot = [
        ('D Loss', loss_history['D'], axes[0, 0]),
        ('G Total Loss', loss_history['G'], axes[0, 1]),
        ('Adversarial Loss', loss_history['adv'], axes[0, 2]),
        ('Feature Matching Loss', loss_history['FM'], axes[0, 3]),
        ('Reconstruction Loss', loss_history['rec'], axes[1, 0]),
        ('Edge Loss', loss_history['edge'], axes[1, 1]),
        ('Physics Loss', loss_history['phys'], axes[1, 2]),
        ('Metal Loss', loss_history['metal'], axes[1, 3]),
    ]
    
    for title, data, ax in losses_to_plot:
        if len(data) > 0:
            ax.plot(data, alpha=0.3, color='blue', label='Raw')
            smoothed = smooth(data)
            if len(smoothed) > 0:
                x_smooth = range(len(data) - len(smoothed), len(data))
                ax.plot(x_smooth, smoothed, color='red', linewidth=2, label='Smoothed')
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            # Format x-axis to use 'k' for thousands
            if len(data) > 0:
                auto_format_axis(ax, len(data), 'x')
            ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metric_curves(metric_history, save_dir, epoch):
    """Plot validation metric curves."""
    epochs = [m['epoch'] for m in metric_history]
    
    # PSNR plot
    fig, ax = plt.subplots(figsize=(10, 6))
    psnr_vals = [m['PSNR'] for m in metric_history]
    ax.plot(epochs, psnr_vals, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR vs Epoch', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)  # Show all epoch numbers
    ax.tick_params(axis='both', labelsize=10)
    plt.savefig(os.path.join(save_dir, f'psnr_vs_epoch.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # SSIM plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ssim_vals = [m['SSIM'] for m in metric_history]
    ax.plot(epochs, ssim_vals, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('SSIM vs Epoch', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)  # Show all epoch numbers
    ax.tick_params(axis='both', labelsize=10)
    plt.savefig(os.path.join(save_dir, f'ssim_vs_epoch.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Combined metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Validation Metrics - Epoch {epoch}', fontsize=14)
    
    axes[0, 0].plot(epochs, psnr_vals, 'b-o', markersize=6)
    axes[0, 0].set_title('PSNR (dB)', fontsize=11)
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=9)
    
    axes[0, 1].plot(epochs, ssim_vals, 'g-o', markersize=6)
    axes[0, 1].set_title('SSIM', fontsize=11)
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=9)
    
    mse_vals = [m['MSE'] for m in metric_history]
    axes[1, 0].plot(epochs, mse_vals, 'r-o', markersize=6)
    axes[1, 0].set_title('MSE', fontsize=11)
    axes[1, 0].set_xlabel('Epoch', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    axes[1, 0].tick_params(axis='both', labelsize=9)
    
    mae_vals = [m['MAE'] for m in metric_history]
    axes[1, 1].plot(epochs, mae_vals, 'm-o', markersize=6)
    axes[1, 1].set_title('MAE', fontsize=11)
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    axes[1, 1].tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'all_metrics_vs_epoch.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_difference_map(pred, target, save_path):
    """Plot difference/error map between prediction and target."""
    diff = torch.abs(pred - target)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Input (if available, use pred for now)
    axes[0].imshow(denormalize(pred).squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Prediction')
    axes[0].axis('off')
    
    # Target
    axes[1].imshow(denormalize(target).squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Difference (absolute)
    diff_np = diff.squeeze().cpu().numpy()
    im = axes[2].imshow(diff_np, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Absolute Error')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Difference (scaled for visibility)
    im2 = axes[3].imshow(diff_np * 5, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title('Error (5x scaled)')
    axes[3].axis('off')
    plt.colorbar(im2, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_histogram(pred, target, save_path, epoch):
    """Plot intensity distribution histograms."""
    pred_np = denormalize(pred).squeeze().cpu().numpy().flatten()
    target_np = denormalize(target).squeeze().cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Intensity Distributions - Epoch {epoch}', fontsize=14)
    
    # Prediction histogram
    axes[0].hist(pred_np, bins=100, alpha=0.7, color='blue', density=True)
    axes[0].set_title('Prediction Distribution', fontsize=11)
    axes[0].set_xlabel('Intensity [0-1]', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].tick_params(axis='both', labelsize=9)
    
    # Target histogram
    axes[1].hist(target_np, bins=100, alpha=0.7, color='green', density=True)
    axes[1].set_title('Ground Truth Distribution', fontsize=11)
    axes[1].set_xlabel('Intensity [0-1]', fontsize=10)
    axes[1].set_ylabel('Density', fontsize=10)
    axes[1].tick_params(axis='both', labelsize=9)
    
    # Overlay
    axes[2].hist(pred_np, bins=100, alpha=0.5, color='blue', density=True, label='Prediction')
    axes[2].hist(target_np, bins=100, alpha=0.5, color='green', density=True, label='Ground Truth')
    axes[2].set_title('Overlay Comparison', fontsize=11)
    axes[2].set_xlabel('Intensity [0-1]', fontsize=10)
    axes[2].set_ylabel('Density', fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_regional_metrics(regional_history, save_path):
    """Plot regional metrics over epochs."""
    if len(regional_history) == 0:
        return
    
    epochs = [r['epoch'] for r in regional_history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Regional PSNR Analysis', fontsize=14, fontweight='bold')
    
    # Metal region
    metal_psnr = [r.get('metal_PSNR', 0) for r in regional_history]
    axes[0].plot(epochs, metal_psnr, 'r-o', linewidth=2, markersize=6)
    axes[0].set_title('Metal Region PSNR', fontsize=11)
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('PSNR (dB)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=9)
    
    # Band region
    band_psnr = [r.get('band_PSNR', 0) for r in regional_history]
    axes[1].plot(epochs, band_psnr, 'b-o', linewidth=2)
    axes[1].set_title('Artifact Band PSNR')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].grid(True, alpha=0.3)
    
    # Non-metal region
    nm_psnr = [r.get('non_metal_PSNR', 0) for r in regional_history]
    axes[2].plot(epochs, nm_psnr, 'g-o', linewidth=2)
    axes[2].set_title('Non-Metal Region PSNR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_intensity_profile(pred, target, ct, save_path, epoch, sample_idx=0):
    """
    Plot intensity profiles across metal artifact regions.
    Shows how intensity changes through a line crossing the metal artifact.
    """
    pred_np = denormalize(pred).squeeze().cpu().numpy()
    target_np = denormalize(target).squeeze().cpu().numpy()
    ct_np = denormalize(ct).squeeze().cpu().numpy()
    
    H, W = pred_np.shape
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ─────────────────────────────────────────────────────────────
    # Top row: Images with profile lines marked
    # ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(ct_np, cmap='gray', vmin=0, vmax=1)
    ax1.axhline(y=H//2, color='red', linestyle='--', linewidth=1, label='Horizontal')
    ax1.axvline(x=W//2, color='cyan', linestyle='--', linewidth=1, label='Vertical')
    # Diagonal line
    ax1.plot([0, W-1], [0, H-1], 'yellow', linestyle='--', linewidth=1, label='Diagonal')
    ax1.set_title('Input (Metal Artifact)')
    ax1.legend(loc='upper right', fontsize=6)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    ax2.axhline(y=H//2, color='red', linestyle='--', linewidth=1)
    ax2.axvline(x=W//2, color='cyan', linestyle='--', linewidth=1)
    ax2.plot([0, W-1], [0, H-1], 'yellow', linestyle='--', linewidth=1)
    ax2.set_title('Prediction')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(target_np, cmap='gray', vmin=0, vmax=1)
    ax3.axhline(y=H//2, color='red', linestyle='--', linewidth=1)
    ax3.axvline(x=W//2, color='cyan', linestyle='--', linewidth=1)
    ax3.plot([0, W-1], [0, H-1], 'yellow', linestyle='--', linewidth=1)
    ax3.set_title('Ground Truth')
    ax3.axis('off')
    
    # ─────────────────────────────────────────────────────────────
    # Bottom row: Intensity profiles along the lines
    # ─────────────────────────────────────────────────────────────
    
    # Horizontal profile (middle row)
    ax4 = fig.add_subplot(2, 3, 4)
    h_ct = ct_np[H//2, :]
    h_pred = pred_np[H//2, :]
    h_target = target_np[H//2, :]
    x_pos = np.arange(W)
    
    ax4.plot(x_pos, h_ct, 'b-', linewidth=1.5, alpha=0.7, label='Input (Artifact)')
    ax4.plot(x_pos, h_pred, 'r-', linewidth=1.5, alpha=0.9, label='Prediction')
    ax4.plot(x_pos, h_target, 'g--', linewidth=1.5, alpha=0.7, label='Ground Truth')
    ax4.fill_between(x_pos, h_pred, h_target, alpha=0.2, color='orange', label='Error')
    ax4.set_xlabel('Pixel Position (X)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Horizontal Profile (Red Line)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Vertical profile (middle column)
    ax5 = fig.add_subplot(2, 3, 5)
    v_ct = ct_np[:, W//2]
    v_pred = pred_np[:, W//2]
    v_target = target_np[:, W//2]
    y_pos = np.arange(H)
    
    ax5.plot(y_pos, v_ct, 'b-', linewidth=1.5, alpha=0.7, label='Input (Artifact)')
    ax5.plot(y_pos, v_pred, 'r-', linewidth=1.5, alpha=0.9, label='Prediction')
    ax5.plot(y_pos, v_target, 'g--', linewidth=1.5, alpha=0.7, label='Ground Truth')
    ax5.fill_between(y_pos, v_pred, v_target, alpha=0.2, color='orange', label='Error')
    ax5.set_xlabel('Pixel Position (Y)')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Vertical Profile (Cyan Line)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # Diagonal profile
    ax6 = fig.add_subplot(2, 3, 6)
    diag_len = min(H, W)
    d_ct = np.array([ct_np[i, i] for i in range(diag_len)])
    d_pred = np.array([pred_np[i, i] for i in range(diag_len)])
    d_target = np.array([target_np[i, i] for i in range(diag_len)])
    d_pos = np.arange(diag_len)
    
    ax6.plot(d_pos, d_ct, 'b-', linewidth=1.5, alpha=0.7, label='Input (Artifact)')
    ax6.plot(d_pos, d_pred, 'r-', linewidth=1.5, alpha=0.9, label='Prediction')
    ax6.plot(d_pos, d_target, 'g--', linewidth=1.5, alpha=0.7, label='Ground Truth')
    ax6.fill_between(d_pos, d_pred, d_target, alpha=0.2, color='orange', label='Error')
    ax6.set_xlabel('Pixel Position (Diagonal)')
    ax6.set_ylabel('Intensity')
    ax6.set_title('Diagonal Profile (Yellow Line)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    
    plt.suptitle(f'Intensity Profiles - Epoch {epoch}, Sample {sample_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_slice_analysis(pred, target, ct, save_path, epoch, sample_idx=0):
    """
    Comprehensive slice analysis showing:
    - Images side by side
    - Error map
    - Intensity histogram
    - Metal region highlight
    """
    pred_np = denormalize(pred).squeeze().cpu().numpy()
    target_np = denormalize(target).squeeze().cpu().numpy()
    ct_np = denormalize(ct).squeeze().cpu().numpy()
    
    # Error map
    error_np = np.abs(pred_np - target_np)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: Images
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(ct_np, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Input (Metal Artifact)', fontsize=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Prediction', fontsize=10)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(target_np, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Ground Truth', fontsize=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.imshow(error_np, cmap='hot', vmin=0, vmax=0.3)
    ax4.set_title('Absolute Error', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: Analysis
    
    # Histogram comparison
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.hist(ct_np.flatten(), bins=50, alpha=0.5, color='blue', density=True, label='Input')
    ax5.hist(pred_np.flatten(), bins=50, alpha=0.5, color='red', density=True, label='Prediction')
    ax5.hist(target_np.flatten(), bins=50, alpha=0.5, color='green', density=True, label='GT')
    ax5.set_xlabel('Intensity')
    ax5.set_ylabel('Density')
    ax5.set_title('Intensity Distribution', fontsize=10)
    ax5.legend(fontsize=8)
    
    # Error histogram
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.hist(error_np.flatten(), bins=50, color='red', alpha=0.7, density=True)
    ax6.axvline(x=error_np.mean(), color='black', linestyle='--', label=f'Mean: {error_np.mean():.4f}')
    ax6.set_xlabel('Absolute Error')
    ax6.set_ylabel('Density')
    ax6.set_title('Error Distribution', fontsize=10)
    ax6.legend(fontsize=8)
    
    # Metal mask overlay
    ax7 = fig.add_subplot(2, 4, 7)
    # Create a simple threshold-based metal mask for visualization
    metal_threshold = 0.8  # High intensity regions in [0,1]
    metal_mask = ct_np > metal_threshold
    overlay = np.stack([ct_np, ct_np, ct_np], axis=-1)
    overlay[metal_mask, 0] = 1.0  # Red highlight for metal
    overlay[metal_mask, 1] = 0.0
    overlay[metal_mask, 2] = 0.0
    ax7.imshow(overlay)
    ax7.set_title('Metal Region (Red)', fontsize=10)
    ax7.axis('off')
    
    # Difference: Pred vs Input (artifact removal)
    ax8 = fig.add_subplot(2, 4, 8)
    artifact_removed = pred_np - ct_np
    im8 = ax8.imshow(artifact_removed, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax8.set_title('Artifact Removed\n(Pred - Input)', fontsize=10)
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    plt.suptitle(f'Slice Analysis - Epoch {epoch}, Sample {sample_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_heatmap(pred, target, ct, save_path, epoch, sample_idx=0):
    """
    Detailed error heatmap showing spatial distribution of errors.
    """
    pred_np = denormalize(pred).squeeze().cpu().numpy()
    target_np = denormalize(target).squeeze().cpu().numpy()
    ct_np = denormalize(ct).squeeze().cpu().numpy()
    
    # Different error metrics
    abs_error = np.abs(pred_np - target_np)
    squared_error = (pred_np - target_np) ** 2
    signed_error = pred_np - target_np  # Shows over/under estimation
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Error Heatmaps - Epoch {epoch}, Sample {sample_idx}', fontsize=14, fontweight='bold')
    
    # Row 1: Images
    axes[0, 0].imshow(ct_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Input (Artifact)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Prediction')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(target_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    # Row 2: Error heatmaps
    im1 = axes[1, 0].imshow(abs_error, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 0].set_title(f'Absolute Error\nMean: {abs_error.mean():.4f}')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(squared_error, cmap='hot', vmin=0, vmax=0.1)
    axes[1, 1].set_title(f'Squared Error\nMean: {squared_error.mean():.6f}')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    im3 = axes[1, 2].imshow(signed_error, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[1, 2].set_title('Signed Error\n(Red=Over, Blue=Under)')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_intensity_segmentation(pred, target, ct, save_path, epoch, sample_idx=0):
    """
    Segmentation of slice showing different intensity regions.
    Visualizes tissue types based on HU/intensity values.
    """
    pred_np = denormalize(pred).squeeze().cpu().numpy()
    target_np = denormalize(target).squeeze().cpu().numpy()
    ct_np = denormalize(ct).squeeze().cpu().numpy()
    
    # Define intensity thresholds for different tissue types (normalized [0,1])
    # Approximate mappings: Air~0, Fat~0.2-0.3, Soft tissue~0.4-0.6, Bone~0.7-0.9, Metal~0.9+
    thresholds = {
        'Air': (0, 0.15),
        'Fat/Lung': (0.15, 0.35),
        'Soft Tissue': (0.35, 0.65),
        'Bone': (0.65, 0.85),
        'Metal/High': (0.85, 1.0)
    }
    
    colors = {
        'Air': [0, 0, 0],           # Black
        'Fat/Lung': [0.2, 0.6, 0.2], # Green
        'Soft Tissue': [0.8, 0.4, 0.4], # Red
        'Bone': [0.9, 0.9, 0.5],    # Yellow
        'Metal/High': [1.0, 1.0, 1.0]  # White
    }
    
    def create_segmentation(img):
        seg = np.zeros((*img.shape, 3))
        for tissue, (low, high) in thresholds.items():
            mask = (img >= low) & (img < high)
            seg[mask] = colors[tissue]
        return seg
    
    seg_ct = create_segmentation(ct_np)
    seg_pred = create_segmentation(pred_np)
    seg_gt = create_segmentation(target_np)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Intensity Segmentation - Epoch {epoch}, Sample {sample_idx}', fontsize=14, fontweight='bold')
    
    # Row 1: Original images
    axes[0, 0].imshow(ct_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Input (Artifact)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Prediction')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(target_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    # Row 2: Segmented images
    axes[1, 0].imshow(seg_ct)
    axes[1, 0].set_title('Input Segmentation')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(seg_pred)
    axes[1, 1].set_title('Prediction Segmentation')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(seg_gt)
    axes[1, 2].set_title('GT Segmentation')
    axes[1, 2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[t], label=t) for t in thresholds.keys()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metal_artifact_wave(pred, target, ct, save_path, epoch, sample_idx=0):
    """
    Shows the intensity wave as it passes through metal artifact region.
    Finds metal regions and plots intensity profiles through them.
    """
    pred_np = denormalize(pred).squeeze().cpu().numpy()
    target_np = denormalize(target).squeeze().cpu().numpy()
    ct_np = denormalize(ct).squeeze().cpu().numpy()
    
    H, W = pred_np.shape
    
    # Find metal region (high intensity)
    metal_mask = ct_np > 0.8
    
    # Find center of metal region
    if metal_mask.sum() > 0:
        metal_coords = np.where(metal_mask)
        metal_center_y = int(np.mean(metal_coords[0]))
        metal_center_x = int(np.mean(metal_coords[1]))
    else:
        metal_center_y, metal_center_x = H // 2, W // 2
    
    fig = plt.figure(figsize=(18, 12))
    
    # ─────────────────────────────────────────────────────────────
    # Top: Images with profile lines through metal
    # ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(ct_np, cmap='gray', vmin=0, vmax=1)
    ax1.axhline(y=metal_center_y, color='red', linestyle='-', linewidth=2, label='H-line')
    ax1.axvline(x=metal_center_x, color='cyan', linestyle='-', linewidth=2, label='V-line')
    ax1.scatter([metal_center_x], [metal_center_y], color='yellow', s=100, marker='x', linewidths=3)
    ax1.set_title(f'Input (Metal Center: {metal_center_x}, {metal_center_y})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    ax2.axhline(y=metal_center_y, color='red', linestyle='-', linewidth=2)
    ax2.axvline(x=metal_center_x, color='cyan', linestyle='-', linewidth=2)
    ax2.scatter([metal_center_x], [metal_center_y], color='yellow', s=100, marker='x', linewidths=3)
    ax2.set_title('Prediction')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(target_np, cmap='gray', vmin=0, vmax=1)
    ax3.axhline(y=metal_center_y, color='red', linestyle='-', linewidth=2)
    ax3.axvline(x=metal_center_x, color='cyan', linestyle='-', linewidth=2)
    ax3.scatter([metal_center_x], [metal_center_y], color='yellow', s=100, marker='x', linewidths=3)
    ax3.set_title('Ground Truth')
    ax3.axis('off')
    
    # ─────────────────────────────────────────────────────────────
    # Bottom: Intensity waves through metal
    # ─────────────────────────────────────────────────────────────
    
    # Horizontal profile through metal center
    ax4 = fig.add_subplot(2, 3, 4)
    h_ct = ct_np[metal_center_y, :]
    h_pred = pred_np[metal_center_y, :]
    h_target = target_np[metal_center_y, :]
    x_pos = np.arange(W)
    
    ax4.plot(x_pos, h_ct, 'b-', linewidth=2, alpha=0.7, label='Input (Artifact)')
    ax4.plot(x_pos, h_pred, 'r-', linewidth=2, label='Prediction')
    ax4.plot(x_pos, h_target, 'g--', linewidth=2, alpha=0.8, label='Ground Truth')
    
    # Highlight metal region
    ax4.axvline(x=metal_center_x, color='gray', linestyle=':', alpha=0.5)
    ax4.fill_between(x_pos, 0, 1, where=(ct_np[metal_center_y, :] > 0.8), 
                     alpha=0.2, color='yellow', label='Metal Region')
    
    ax4.set_xlabel('X Position (pixels)', fontsize=11)
    ax4.set_ylabel('Intensity', fontsize=11)
    ax4.set_title('Horizontal Intensity Wave Through Metal', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Vertical profile through metal center
    ax5 = fig.add_subplot(2, 3, 5)
    v_ct = ct_np[:, metal_center_x]
    v_pred = pred_np[:, metal_center_x]
    v_target = target_np[:, metal_center_x]
    y_pos = np.arange(H)
    
    ax5.plot(y_pos, v_ct, 'b-', linewidth=2, alpha=0.7, label='Input (Artifact)')
    ax5.plot(y_pos, v_pred, 'r-', linewidth=2, label='Prediction')
    ax5.plot(y_pos, v_target, 'g--', linewidth=2, alpha=0.8, label='Ground Truth')
    
    # Highlight metal region
    ax5.axvline(x=metal_center_y, color='gray', linestyle=':', alpha=0.5)
    ax5.fill_between(y_pos, 0, 1, where=(ct_np[:, metal_center_x] > 0.8), 
                     alpha=0.2, color='yellow', label='Metal Region')
    
    ax5.set_xlabel('Y Position (pixels)', fontsize=11)
    ax5.set_ylabel('Intensity', fontsize=11)
    ax5.set_title('Vertical Intensity Wave Through Metal', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # Error along horizontal profile
    ax6 = fig.add_subplot(2, 3, 6)
    h_error = np.abs(h_pred - h_target)
    h_input_error = np.abs(h_ct - h_target)
    
    ax6.plot(x_pos, h_input_error, 'b-', linewidth=2, alpha=0.7, label='Input Error')
    ax6.plot(x_pos, h_error, 'r-', linewidth=2, label='Prediction Error')
    ax6.fill_between(x_pos, h_error, h_input_error, where=(h_input_error > h_error),
                     alpha=0.3, color='green', label='Error Reduction')
    ax6.axvline(x=metal_center_x, color='gray', linestyle=':', alpha=0.5)
    
    ax6.set_xlabel('X Position (pixels)', fontsize=11)
    ax6.set_ylabel('Absolute Error', fontsize=11)
    ax6.set_title('Error Reduction Along Horizontal Profile', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 0.5)
    
    plt.suptitle(f'Metal Artifact Intensity Analysis - Epoch {epoch}, Sample {sample_idx}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_hu_accuracy(pred, target, ct, threshold=0.6):
    """
    Compute HU accuracy metrics.
    Returns mean absolute HU error for different tissue regions.
    
    Converts normalized [0,1] values back to approximate HU:
    HU = (normalized * 4000) - 1000  (approximate)
    """
    pred_np = denormalize(pred).squeeze().cpu().numpy()
    target_np = denormalize(target).squeeze().cpu().numpy()
    ct_np = denormalize(ct).squeeze().cpu().numpy()
    
    # Convert to approximate HU (assuming normalization from [-1000, 3000] HU)
    def to_hu(x):
        return (x * 4000) - 1000
    
    pred_hu = to_hu(pred_np)
    target_hu = to_hu(target_np)
    ct_hu = to_hu(ct_np)
    
    # Define tissue regions based on HU values
    # Air: -1000 to -500, Lung: -500 to -200, Fat: -200 to -50
    # Soft tissue: -50 to 100, Bone: 100 to 1000, Metal: >1000
    
    results = {}
    
    # Overall HU error
    hu_error = np.abs(pred_hu - target_hu)
    results['overall_HU_MAE'] = hu_error.mean()
    results['overall_HU_RMSE'] = np.sqrt((hu_error ** 2).mean())
    
    # Tissue-specific errors (using normalized thresholds)
    tissue_ranges = {
        'air': (0, 0.125),           # -1000 to -500 HU
        'soft_tissue': (0.2375, 0.275),  # -50 to 100 HU  
        'bone': (0.275, 0.5),        # 100 to 1000 HU
        'metal_region': (0.5, 1.0),  # >1000 HU
    }
    
    for tissue, (low, high) in tissue_ranges.items():
        mask = (target_np >= low) & (target_np < high)
        if mask.sum() > 0:
            tissue_error = hu_error[mask]
            results[f'{tissue}_HU_MAE'] = tissue_error.mean()
            results[f'{tissue}_pixel_count'] = mask.sum()
        else:
            results[f'{tissue}_HU_MAE'] = 0.0
            results[f'{tissue}_pixel_count'] = 0
    
    return results

# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION FOR ONE ABLATION
# ═══════════════════════════════════════════════════════════════

def run_ablation(ablation_id, ablation_config, output_dir, logger):
    """Run a single ablation study."""
    
    logger.section(f"ABLATION: {ablation_id}")
    logger.info(f"Name: {ablation_config['name']}")
    logger.info(f"Description: {ablation_config['description']}")
    
    # Create directories
    create_dirs(output_dir)
    
    # Merge config with ablation changes
    config = DEFAULT_CONFIG.copy()
    config.update(ablation_config['changes'])
    config['ablation_id'] = ablation_id
    config['ablation_name'] = ablation_config['name']
    config['ablation_description'] = ablation_config['description']
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.subsection("Configuration")
    for k, v in config.items():
        logger.debug(f"  {k}: {v}")
    logger.info(f"  λ_adv={config['lambda_adv']}, λ_FM={config['lambda_FM']}, "
                f"λ_rec={config['lambda_rec']}, λ_edge={config['lambda_edge']}, "
                f"λ_phys={config['lambda_phys']}, λ_metal={config['lambda_metal']}")
    logger.info(f"  Metal weighting: {config['use_metal_weighting']}, Hinge loss: {config['use_hinge_loss']}")
    logger.info(f"  Disc scales: {config['num_disc_scales']}, Spectral norm: {config['use_spectral_norm']}, Dilation radius: {config['dilation_radius']}")
    
    # ─────────────────────────────────────────────────────────────
    # Load Dataset
    # ─────────────────────────────────────────────────────────────
    logger.subsection("Loading Datasets")
    train_mask = np.load(os.path.join(SYNDEEPLESION_PATH, 'trainmask.npy'))
    test_mask = np.load(os.path.join(SYNDEEPLESION_PATH, 'testmask.npy'))
    
    train_dataset = MARTrainDataset(
        SYNDEEPLESION_PATH,
        patchSize=PATCH_SIZE,
        length=BATCH_SIZE * 4000,  # Same as train_combined.py
        mask=train_mask
    )
    
    # Use proper test set (test_640geo folder with testmask.npy)
    val_dataset = TestDataset(
        dir=SYNDEEPLESION_PATH,
        mask=test_mask
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Test/Validation samples: {len(val_dataset)}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Iterations per epoch: {len(train_loader)}")
    
    # ─────────────────────────────────────────────────────────────
    # Build Models
    # ─────────────────────────────────────────────────────────────
    logger.subsection("Building Models")
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = NGswin()
        def forward(self, x):
            return self.main(x)
    
    netG = Generator().to(DEVICE)
    netG.apply(weights_init)
    G_params = sum(p.numel() for p in netG.parameters())
    logger.info(f"  Generator (NGswin): {G_params:,} parameters")
    
    # Build discriminator based on config (supports single-scale and no-SN ablations)
    num_disc_scales = config.get('num_disc_scales', 3)
    use_sn = config.get('use_spectral_norm', True)
    
    if num_disc_scales == 1:
        # B1 ablation: Single-scale discriminator
        from models.discriminator.ms_patchgan import SingleScaleDiscriminator
        netD = SingleScaleDiscriminator(
            in_channels=2,
            base_channels=64,
            num_layers=5,
            use_sn=use_sn
        ).to(DEVICE)
        disc_type = "Single-Scale PatchGAN"
    else:
        # Default: Multi-scale discriminator
        netD = MultiScaleDiscriminator(
            in_channels=2,
            base_channels=64,
            num_layers=5,
            num_scales=num_disc_scales,
            use_sn=use_sn
        ).to(DEVICE)
        disc_type = f"MS-PatchGAN ({num_disc_scales} scales)"
    
    netD.apply(weights_init)
    D_params = sum(p.numel() for p in netD.parameters())
    logger.info(f"  Discriminator ({disc_type}, SN={use_sn}): {D_params:,} parameters")
    logger.info(f"  Total parameters: {G_params + D_params:,}")
    
    # ─────────────────────────────────────────────────────────────
    # Optimizers
    # ─────────────────────────────────────────────────────────────
    optimizerG = optim.Adam(netG.parameters(), lr=config['lrG'], betas=(config['beta1'], config['beta2']))
    optimizerD = optim.Adam(netD.parameters(), lr=config['lrD'], betas=(config['beta1'], config['beta2']))
    logger.info(f"  Optimizer G: Adam (lr={config['lrG']}, β=({config['beta1']}, {config['beta2']}))")
    logger.info(f"  Optimizer D: Adam (lr={config['lrD']}, β=({config['beta1']}, {config['beta2']}))")
    
    # ─────────────────────────────────────────────────────────────
    # Helper for single-scale vs multi-scale discriminator output
    # ─────────────────────────────────────────────────────────────
    is_single_scale = (config.get('num_disc_scales', 3) == 1)
    
    def run_discriminator(disc, x, return_features=False):
        """Wrapper to handle both single-scale and multi-scale discriminator outputs.
        Returns logits and features in list format for consistency."""
        logits, feats = disc(x, return_features=return_features)
        if is_single_scale:
            # Wrap single-scale output into list format
            logits = [logits]
            if feats is not None:
                feats = [feats]
        return logits, feats
    
    # ─────────────────────────────────────────────────────────────
    # TorchRadon Projector (for physics loss)
    # ─────────────────────────────────────────────────────────────
    num_angles = 180
    angles = np.linspace(0, np.pi, num_angles, dtype=np.float32)
    projector = Radon(PATCH_SIZE, angles)
    logger.debug(f"  TorchRadon projector initialized with {num_angles} angles")
    
    # ─────────────────────────────────────────────────────────────
    # Fixed batch for visualization
    # ─────────────────────────────────────────────────────────────
    fixed_batch = next(iter(train_loader))
    fixed_ct = fixed_batch[0].to(DEVICE)
    fixed_gt = fixed_batch[1].to(DEVICE)
    
    # ─────────────────────────────────────────────────────────────
    # Loss and Metric Tracking
    # ─────────────────────────────────────────────────────────────
    loss_history = defaultdict(list)
    metric_history = []
    regional_history = []
    best_psnr = 0.0
    iters = 0
    
    # CSV files for logging
    training_csv = open(os.path.join(output_dir, 'training_history.csv'), 'w', newline='')
    training_writer = csv.writer(training_csv)
    training_writer.writerow(['iteration', 'epoch', 'D_loss', 'G_loss', 'adv_loss', 'FM_loss', 
                              'rec_loss', 'edge_loss', 'phys_loss', 'metal_loss'])
    
    validation_csv = open(os.path.join(output_dir, 'validation_history.csv'), 'w', newline='')
    validation_writer = csv.writer(validation_csv)
    validation_writer.writerow(['epoch', 'PSNR', 'SSIM', 'MSE', 'RMSE', 'MAE',
                                'metal_PSNR', 'band_PSNR', 'non_metal_PSNR'])
    
    # ─────────────────────────────────────────────────────────────
    # Check for existing checkpoints to resume from
    # ─────────────────────────────────────────────────────────────
    start_epoch = 0
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        existing_ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')]
        if existing_ckpts:
            # Find the latest checkpoint
            epochs_found = [int(f.split('_')[1].split('.')[0]) for f in existing_ckpts]
            latest_epoch = max(epochs_found)
            latest_ckpt = os.path.join(checkpoint_dir, f'epoch_{latest_epoch:03d}.pth')
            
            logger.info(f"  Found existing checkpoint at epoch {latest_epoch}")
            logger.info(f"  Loading: {latest_ckpt}")
            
            ckpt = torch.load(latest_ckpt, map_location=DEVICE)
            netG.load_state_dict(ckpt['netG_state_dict'])
            netD.load_state_dict(ckpt['netD_state_dict'])
            optimizerG.load_state_dict(ckpt['optimizerG_state_dict'])
            optimizerD.load_state_dict(ckpt['optimizerD_state_dict'])
            start_epoch = ckpt['epoch']  # Resume from next epoch
            
            logger.info(f"  ✓ Resumed from epoch {start_epoch}, will continue to epoch {NUM_EPOCHS}")
    
    # ─────────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────────
    logger.section(f"TRAINING ({NUM_EPOCHS} epochs, starting from epoch {start_epoch + 1})")
    start_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        logger.subsection(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        netG.train()
        netD.train()
        
        # Track epoch losses
        epoch_losses = defaultdict(list)
        
        epoch_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {epoch+1}", leave=False)
        
        for i, data in epoch_bar:
            ct = data[0].to(DEVICE)
            real = data[1].to(DEVICE)
            
            # ════════════════════════════════════════════════════
            # UPDATE DISCRIMINATOR
            # ════════════════════════════════════════════════════
            netD.zero_grad()
            
            with torch.no_grad():
                fake = netG(ct)
            
            real_pair = torch.cat([ct, real], dim=1)
            fake_pair = torch.cat([ct, fake], dim=1)
            
            real_logits, _ = run_discriminator(netD, real_pair, return_features=False)
            fake_logits, _ = run_discriminator(netD, fake_pair, return_features=False)
            
            # Choose loss function based on config
            if config['use_hinge_loss']:
                loss_D = hinge_d_loss(real_logits, fake_logits)
            else:
                loss_D = vanilla_d_loss(real_logits, fake_logits)
            
            loss_D.backward()
            optimizerD.step()
            
            # ════════════════════════════════════════════════════
            # UPDATE GENERATOR
            # ════════════════════════════════════════════════════
            netG.zero_grad()
            
            fake = netG(ct)
            fake_pair = torch.cat([ct, fake], dim=1)
            
            fake_logits, fake_feats = run_discriminator(netD, fake_pair, return_features=True)
            
            with torch.no_grad():
                _, real_feats = run_discriminator(netD, real_pair, return_features=True)
            
            # Adversarial Loss (only compute if lambda > 0)
            if config['lambda_adv'] > 0:
                if config['use_hinge_loss']:
                    loss_G_adv = hinge_g_loss(fake_logits)
                else:
                    loss_G_adv = vanilla_g_loss(fake_logits)
            else:
                loss_G_adv = torch.tensor(0.0, device=DEVICE)
            
            # Feature Matching Loss (only compute if lambda > 0)
            if config['lambda_FM'] > 0:
                loss_FM = feature_matching_loss(real_feats, fake_feats)
            else:
                loss_FM = torch.tensor(0.0, device=DEVICE)
            
            # Metal-Aware Reconstruction Loss
            if config['use_metal_weighting']:
                loss_rec = compute_metal_aware_loss(
                    fake, real, ct,
                    beta=config['beta_weight'],
                    radius=config['dilation_radius'],
                    w_max=config['w_max'],
                    threshold=config['metal_threshold']
                )
            else:
                # Simple L1 loss without metal weighting
                loss_rec = F.l1_loss(fake, real)
            
            # Metal-Aware Edge Loss (only compute if lambda > 0)
            if config['lambda_edge'] > 0:
                if config['use_metal_weighting']:
                    w = compute_weight_map(
                        ct,
                        beta=config['beta_weight'],
                        radius=config['dilation_radius'],
                        w_max=config['w_max'],
                        threshold=config['metal_threshold']
                    )
                else:
                    w = torch.ones_like(ct)
                loss_edge = compute_metal_aware_edge_loss(fake, real, w)
            else:
                loss_edge = torch.tensor(0.0, device=DEVICE)
            
            # Extract metal mask (needed for physics and metal-consistency)
            M = extract_metal_mask(ct, threshold=config['metal_threshold'])
            
            # Physics Loss (only compute if lambda > 0 - this is expensive!)
            if config['lambda_phys'] > 0:
                loss_phys = physics_loss_syn(fake, real, M, projector)
            else:
                loss_phys = torch.tensor(0.0, device=DEVICE)
            
            # Metal-Consistency Loss (only compute if lambda > 0)
            if config['lambda_metal'] > 0:
                loss_metal = metal_consistency_loss(fake, real, M)
            else:
                loss_metal = torch.tensor(0.0, device=DEVICE)
            
            # Total Generator Loss
            loss_G = (
                config['lambda_adv'] * loss_G_adv +
                config['lambda_FM'] * loss_FM +
                config['lambda_rec'] * loss_rec +
                config['lambda_edge'] * loss_edge +
                config['lambda_phys'] * loss_phys +
                config['lambda_metal'] * loss_metal
            )
            
            loss_G.backward()
            optimizerG.step()
            
            # ════════════════════════════════════════════════════
            # LOGGING
            # ════════════════════════════════════════════════════
            # Get loss values (handle tensor vs scalar)
            def get_loss_val(loss):
                return loss.item() if torch.is_tensor(loss) else loss
            
            loss_history['D'].append(get_loss_val(loss_D))
            loss_history['G'].append(get_loss_val(loss_G))
            loss_history['adv'].append(get_loss_val(loss_G_adv))
            loss_history['FM'].append(get_loss_val(loss_FM))
            loss_history['rec'].append(get_loss_val(loss_rec))
            loss_history['edge'].append(get_loss_val(loss_edge))
            loss_history['phys'].append(get_loss_val(loss_phys))
            loss_history['metal'].append(get_loss_val(loss_metal))
            
            # Track epoch losses for summary
            epoch_losses['D'].append(get_loss_val(loss_D))
            epoch_losses['G'].append(get_loss_val(loss_G))
            epoch_losses['adv'].append(get_loss_val(loss_G_adv))
            epoch_losses['FM'].append(get_loss_val(loss_FM))
            epoch_losses['rec'].append(get_loss_val(loss_rec))
            epoch_losses['edge'].append(get_loss_val(loss_edge))
            epoch_losses['phys'].append(get_loss_val(loss_phys))
            epoch_losses['metal'].append(get_loss_val(loss_metal))
            
            # Write to CSV
            training_writer.writerow([
                iters, epoch+1, get_loss_val(loss_D), get_loss_val(loss_G),
                get_loss_val(loss_G_adv), get_loss_val(loss_FM), get_loss_val(loss_rec),
                get_loss_val(loss_edge), get_loss_val(loss_phys), get_loss_val(loss_metal)
            ])
            
            # Update progress bar
            epoch_bar.set_postfix({
                'D': f"{loss_D.item():.4f}",
                'G': f"{loss_G.item():.4f}",
                'rec': f"{loss_rec.item():.4f}",
            })
            
            # Print detailed logs
            if i % LOG_EVERY == 0 and i > 0:
                logger.debug(f"  [{i}/{len(train_loader)}] D={loss_D.item():.4f} │ "
                           f"G={loss_G.item():.4f} (adv={loss_G_adv.item():.3f}, "
                           f"FM={loss_FM.item():.3f}, rec={loss_rec.item():.3f}, "
                           f"edge={loss_edge.item():.3f}, phys={loss_phys.item():.3f}, "
                           f"metal={loss_metal.item():.3f})")
            
            # Save samples
            if iters % SAMPLE_EVERY == 0:
                with torch.no_grad():
                    sample_fake = netG(fixed_ct)
                grid = vutils.make_grid(
                    torch.cat([fixed_ct[:4], sample_fake[:4], fixed_gt[:4]], dim=0),
                    nrow=4, normalize=True, padding=2
                )
                vutils.save_image(grid, os.path.join(output_dir, 'samples', f'iter_{iters:06d}.png'))
                logger.debug(f"  Saved sample: iter_{iters:06d}.png")
            
            iters += 1
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        logger.info(f"  Epoch {epoch+1} Summary:")
        logger.info(f"    Time: {epoch_time/60:.1f} min │ Iterations: {len(train_loader)}")
        logger.info(f"    Avg Losses: D={avg_losses['D']:.4f}, G={avg_losses['G']:.4f}")
        logger.info(f"    Components: adv={avg_losses['adv']:.4f}, FM={avg_losses['FM']:.4f}, "
                   f"rec={avg_losses['rec']:.4f}, edge={avg_losses['edge']:.4f}, "
                   f"phys={avg_losses['phys']:.4f}, metal={avg_losses['metal']:.4f}")
        
        # ════════════════════════════════════════════════════════
        # PERIODIC SAVING (every SAVE_EVERY epochs)
        # ════════════════════════════════════════════════════════
        if (epoch + 1) % SAVE_EVERY == 0:
            logger.subsection(f"Checkpoint & Validation (Epoch {epoch+1})")
            
            # Save checkpoint
            ckpt_path = os.path.join(output_dir, 'checkpoints', f'epoch_{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'config': config,
            }, ckpt_path)
            logger.info(f"  ✓ Checkpoint saved: {os.path.basename(ckpt_path)}")
            
            # Save loss curves
            loss_curve_path = os.path.join(output_dir, 'loss_curves', f'loss_curves_epoch_{epoch+1:03d}.png')
            plot_loss_curves(loss_history, loss_curve_path, epoch+1)
            logger.info(f"  ✓ Loss curves saved")
            
            # ────────────────────────────────────────────────────
            # VALIDATION
            # ────────────────────────────────────────────────────
            logger.info(f"  Running validation on {len(val_loader)} samples...")
            netG.eval()
            
            val_metrics_list = []
            regional_metrics_list = []
            hu_metrics_list = []
            
            with torch.no_grad():
                for val_idx, val_batch in enumerate(val_loader):
                    val_ct = val_batch[0].to(DEVICE)
                    val_gt = val_batch[1].to(DEVICE)
                    
                    val_pred = netG(val_ct)
                    
                    # Compute metrics
                    metrics = compute_metrics(val_pred, val_gt)
                    val_metrics_list.append(metrics)
                    
                    # Compute regional metrics
                    reg_metrics = compute_regional_metrics(
                        val_pred, val_gt, val_ct,
                        threshold=config['metal_threshold'],
                        radius=config['dilation_radius']
                    )
                    regional_metrics_list.append(reg_metrics)
                    
                    # Compute HU accuracy
                    hu_metrics = compute_hu_accuracy(val_pred, val_gt, val_ct)
                    hu_metrics_list.append(hu_metrics)
                    
                    # Save test examples for FIXED sample indices (consistent across all ablations)
                    if val_idx in VIS_SAMPLE_INDICES and val_idx < 50:  # Only first 50 fixed samples during training
                        # Save comparison image
                        grid = vutils.make_grid(
                            torch.cat([denormalize(val_ct), denormalize(val_pred), denormalize(val_gt)], dim=0),
                            nrow=1, padding=2, normalize=False
                        )
                        save_path = os.path.join(output_dir, 'test_examples', 
                                                 f'epoch_{epoch+1:03d}_sample_{val_idx:04d}.png')
                        vutils.save_image(grid, save_path)
                        
                        # Save difference map
                        diff_path = os.path.join(output_dir, 'difference_maps',
                                                 f'epoch_{epoch+1:03d}_diff_{val_idx:04d}.png')
                        plot_difference_map(val_pred, val_gt, diff_path)
                        
                        # Save histogram (only for first sample)
                        if val_idx == 0:
                            hist_path = os.path.join(output_dir, 'histograms',
                                                     f'epoch_{epoch+1:03d}_histogram.png')
                            plot_histogram(val_pred, val_gt, hist_path, epoch+1)
                        
                        # Save intensity profile plots (shows wave through artifact)
                        intensity_path = os.path.join(output_dir, 'intensity_profiles',
                                                      f'epoch_{epoch+1:03d}_intensity_{val_idx:04d}.png')
                        plot_intensity_profile(val_pred, val_gt, val_ct, intensity_path, epoch+1, val_idx)
                        
                        # Save comprehensive slice analysis
                        slice_path = os.path.join(output_dir, 'slice_analysis',
                                                  f'epoch_{epoch+1:03d}_slice_{val_idx:04d}.png')
                        plot_slice_analysis(val_pred, val_gt, val_ct, slice_path, epoch+1, val_idx)
                        
                        # Save error heatmaps
                        error_heatmap_path = os.path.join(output_dir, 'error_heatmaps',
                                                          f'epoch_{epoch+1:03d}_error_{val_idx:04d}.png')
                        plot_error_heatmap(val_pred, val_gt, val_ct, error_heatmap_path, epoch+1, val_idx)
                        
                        # Save intensity segmentation
                        seg_path = os.path.join(output_dir, 'intensity_segmentation',
                                                f'epoch_{epoch+1:03d}_seg_{val_idx:04d}.png')
                        plot_intensity_segmentation(val_pred, val_gt, val_ct, seg_path, epoch+1, val_idx)
                        
                        # Save metal artifact wave analysis
                        wave_path = os.path.join(output_dir, 'metal_artifact_waves',
                                                 f'epoch_{epoch+1:03d}_wave_{val_idx:04d}.png')
                        plot_metal_artifact_wave(val_pred, val_gt, val_ct, wave_path, epoch+1, val_idx)
            
            # Average metrics
            avg_metrics = {
                'epoch': epoch + 1,
                'PSNR': np.mean([m['PSNR'] for m in val_metrics_list]),
                'SSIM': np.mean([m['SSIM'] for m in val_metrics_list]),
                'MSE': np.mean([m['MSE'] for m in val_metrics_list]),
                'RMSE': np.mean([m['RMSE'] for m in val_metrics_list]),
                'MAE': np.mean([m['MAE'] for m in val_metrics_list]),
            }
            
            avg_regional = {
                'epoch': epoch + 1,
                'metal_PSNR': np.mean([m['metal_PSNR'] for m in regional_metrics_list]),
                'band_PSNR': np.mean([m['band_PSNR'] for m in regional_metrics_list]),
                'non_metal_PSNR': np.mean([m['non_metal_PSNR'] for m in regional_metrics_list]),
                'metal_MSE': np.mean([m['metal_MSE'] for m in regional_metrics_list]),
                'band_MSE': np.mean([m['band_MSE'] for m in regional_metrics_list]),
                'non_metal_MSE': np.mean([m['non_metal_MSE'] for m in regional_metrics_list]),
            }
            
            # Average HU metrics
            avg_hu = {
                'epoch': epoch + 1,
                'overall_HU_MAE': np.mean([m['overall_HU_MAE'] for m in hu_metrics_list]),
                'overall_HU_RMSE': np.mean([m['overall_HU_RMSE'] for m in hu_metrics_list]),
                'soft_tissue_HU_MAE': np.mean([m.get('soft_tissue_HU_MAE', 0) for m in hu_metrics_list]),
                'bone_HU_MAE': np.mean([m.get('bone_HU_MAE', 0) for m in hu_metrics_list]),
                'metal_region_HU_MAE': np.mean([m.get('metal_region_HU_MAE', 0) for m in hu_metrics_list]),
            }
            
            metric_history.append(avg_metrics)
            regional_history.append(avg_regional)
            
            # Write to validation CSV
            validation_writer.writerow([
                epoch+1, avg_metrics['PSNR'], avg_metrics['SSIM'], avg_metrics['MSE'],
                avg_metrics['RMSE'], avg_metrics['MAE'],
                avg_regional['metal_PSNR'], avg_regional['band_PSNR'], avg_regional['non_metal_PSNR']
            ])
            
            # Log validation results
            logger.info(f"  ╔═══════════════════════════════════════════════════════════╗")
            logger.info(f"  ║ VALIDATION RESULTS (Epoch {epoch+1:3d})                          ║")
            logger.info(f"  ╠═══════════════════════════════════════════════════════════╣")
            logger.info(f"  ║ Global Metrics:                                           ║")
            logger.info(f"  ║   PSNR  = {avg_metrics['PSNR']:7.3f} dB                             ║")
            logger.info(f"  ║   SSIM  = {avg_metrics['SSIM']:7.4f}                                ║")
            logger.info(f"  ║   MSE   = {avg_metrics['MSE']:9.6f}                              ║")
            logger.info(f"  ║   RMSE  = {avg_metrics['RMSE']:9.6f}                              ║")
            logger.info(f"  ║   MAE   = {avg_metrics['MAE']:9.6f}                              ║")
            logger.info(f"  ╠═══════════════════════════════════════════════════════════╣")
            logger.info(f"  ║ Regional PSNR:                                            ║")
            logger.info(f"  ║   Metal Region   = {avg_regional['metal_PSNR']:7.3f} dB                    ║")
            logger.info(f"  ║   Artifact Band  = {avg_regional['band_PSNR']:7.3f} dB                    ║")
            logger.info(f"  ║   Non-Metal      = {avg_regional['non_metal_PSNR']:7.3f} dB                    ║")
            logger.info(f"  ╠═══════════════════════════════════════════════════════════╣")
            logger.info(f"  ║ HU Accuracy:                                              ║")
            logger.info(f"  ║   Overall HU MAE   = {avg_hu['overall_HU_MAE']:8.2f} HU                   ║")
            logger.info(f"  ║   Overall HU RMSE  = {avg_hu['overall_HU_RMSE']:8.2f} HU                   ║")
            logger.info(f"  ║   Soft Tissue MAE  = {avg_hu['soft_tissue_HU_MAE']:8.2f} HU                   ║")
            logger.info(f"  ║   Bone HU MAE      = {avg_hu['bone_HU_MAE']:8.2f} HU                   ║")
            logger.info(f"  ║   Metal Region MAE = {avg_hu['metal_region_HU_MAE']:8.2f} HU                   ║")
            logger.info(f"  ╚═══════════════════════════════════════════════════════════╝")
            
            # Save metric plots
            plot_metric_curves(metric_history, os.path.join(output_dir, 'metric_plots'), epoch+1)
            
            # Save regional metrics plot
            regional_plot_path = os.path.join(output_dir, 'regional_metrics', 'regional_metrics_plot.png')
            plot_regional_metrics(regional_history, regional_plot_path)
            logger.info(f"  ✓ Metric plots saved")
            
            # Save best model
            if avg_metrics['PSNR'] > best_psnr:
                best_psnr = avg_metrics['PSNR']
                best_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'config': config,
                    'best_psnr': best_psnr,
                }, best_path)
                logger.info(f"  ★ NEW BEST MODEL! PSNR={best_psnr:.3f} dB")
            
            netG.train()
    
    # ─────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────
    training_csv.close()
    validation_csv.close()
    
    total_time = time.time() - start_time
    
    # Save final metrics
    final_metrics = {
        'ablation_id': ablation_id,
        'ablation_name': ablation_config['name'],
        'total_epochs': NUM_EPOCHS,
        'total_time_hours': total_time / 3600,
        'best_psnr': best_psnr,
        'final_metrics': metric_history[-1] if metric_history else {},
        'final_regional_metrics': regional_history[-1] if regional_history else {},
        'total_iterations': iters,
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.section(f"ABLATION {ablation_id} COMPLETE!")
    logger.info(f"  Total Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    logger.info(f"  Total Iterations: {iters:,}")
    logger.info(f"  Best PSNR: {best_psnr:.3f} dB")
    if metric_history:
        fm = metric_history[-1]
        logger.info(f"  Final Metrics: PSNR={fm['PSNR']:.3f}, SSIM={fm['SSIM']:.4f}")
    logger.info(f"  Results saved to: {output_dir}")
    
    return final_metrics


# ═══════════════════════════════════════════════════════════════
# INFERENCE-ONLY FUNCTION (for pre-trained checkpoints)
# ═══════════════════════════════════════════════════════════════

def run_inference_only(ablation_id, ablation_config, output_dir, logger):
    """Run inference on test set using a pre-trained checkpoint."""
    
    logger.section(f"INFERENCE-ONLY: {ablation_id}")
    logger.info(f"Name: {ablation_config['name']}")
    logger.info(f"Description: {ablation_config['description']}")
    logger.info(f"Checkpoint: {ablation_config['checkpoint_path']}")
    
    # Create directories
    create_dirs(output_dir)
    
    # Merge config
    config = DEFAULT_CONFIG.copy()
    config.update(ablation_config['changes'])
    config['ablation_id'] = ablation_id
    config['ablation_name'] = ablation_config['name']
    config['ablation_description'] = ablation_config['description']
    config['inference_only'] = True
    config['checkpoint_path'] = ablation_config['checkpoint_path']
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.subsection("Configuration")
    logger.info(f"  λ_adv={config['lambda_adv']}, λ_FM={config['lambda_FM']}, "
                f"λ_rec={config['lambda_rec']}, λ_edge={config['lambda_edge']}, "
                f"λ_phys={config['lambda_phys']}, λ_metal={config['lambda_metal']}")
    
    # ─────────────────────────────────────────────────────────────
    # Load Test Dataset
    # ─────────────────────────────────────────────────────────────
    logger.subsection("Loading Test Dataset")
    test_mask = np.load(os.path.join(SYNDEEPLESION_PATH, 'testmask.npy'))
    
    test_dataset = TestDataset(
        dir=SYNDEEPLESION_PATH,
        mask=test_mask
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # ─────────────────────────────────────────────────────────────
    # Build Generator and Load Checkpoint
    # ─────────────────────────────────────────────────────────────
    logger.subsection("Building Generator")
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = NGswin()
        def forward(self, x):
            return self.main(x)
    
    netG = Generator().to(DEVICE)
    G_params = sum(p.numel() for p in netG.parameters())
    logger.info(f"  Generator (NGswin): {G_params:,} parameters")
    
    # Load checkpoint
    checkpoint_path = ablation_config['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Handle different checkpoint formats
    if 'netG_state_dict' in checkpoint:
        # Format from run_loss_ablations.py training
        netG.load_state_dict(checkpoint['netG_state_dict'])
        logger.info(f"  ✓ Loaded generator from epoch {checkpoint.get('epoch', 'N/A')}")
        if 'best_psnr' in checkpoint:
            logger.info(f"  ✓ Training best PSNR: {checkpoint['best_psnr']:.3f} dB")
    elif 'generator_state_dict' in checkpoint:
        # Alternative format
        netG.load_state_dict(checkpoint['generator_state_dict'])
        logger.info(f"  ✓ Loaded generator from epoch {checkpoint.get('epoch', 'N/A')}")
        if 'best_psnr' in checkpoint:
            logger.info(f"  ✓ Training best PSNR: {checkpoint['best_psnr']:.3f} dB")
    else:
        # Raw state dict
        netG.load_state_dict(checkpoint)
        logger.info(f"  ✓ Loaded generator state dict")
    
    netG.eval()
    
    # ─────────────────────────────────────────────────────────────
    # Run Inference
    # ─────────────────────────────────────────────────────────────
    logger.subsection("Running Inference on Test Set")
    
    test_metrics_list = []
    regional_metrics_list = []
    hu_metrics_list = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            test_ct = batch[0].to(DEVICE)
            test_gt = batch[1].to(DEVICE)
            
            test_pred = netG(test_ct)
            
            # Compute metrics
            metrics = compute_metrics(test_pred, test_gt)
            test_metrics_list.append(metrics)
            
            # Compute regional metrics
            reg_metrics = compute_regional_metrics(
                test_pred, test_gt, test_ct,
                threshold=config['metal_threshold'],
                radius=config['dilation_radius']
            )
            regional_metrics_list.append(reg_metrics)
            
            # Compute HU accuracy
            hu_metrics = compute_hu_accuracy(test_pred, test_gt, test_ct)
            hu_metrics_list.append(hu_metrics)
            
            # Save visualizations only for FIXED sample indices (consistent across all ablations)
            if idx in VIS_SAMPLE_INDICES:
                # Save test example grid
                grid = vutils.make_grid(
                    torch.cat([denormalize(test_ct), denormalize(test_pred), denormalize(test_gt)], dim=0),
                    nrow=1, padding=2, normalize=False
                )
                save_path = os.path.join(output_dir, 'test_examples', f'test_sample_{idx:04d}.png')
                vutils.save_image(grid, save_path)
                
                # Save difference map
                diff_path = os.path.join(output_dir, 'difference_maps', f'test_diff_{idx:04d}.png')
                plot_difference_map(test_pred, test_gt, diff_path)
                
                # Save error heatmap
                heatmap_path = os.path.join(output_dir, 'error_heatmaps', f'test_heatmap_{idx:04d}.png')
                plot_error_heatmap(test_pred, test_gt, test_ct, heatmap_path, epoch='test', sample_idx=idx)
                
                # Save intensity segmentation
                seg_path = os.path.join(output_dir, 'intensity_segmentation', f'test_seg_{idx:04d}.png')
                plot_intensity_segmentation(test_pred, test_gt, test_ct, seg_path, epoch='test', sample_idx=idx)
                
                # Save metal artifact wave
                wave_path = os.path.join(output_dir, 'metal_artifact_waves', f'test_wave_{idx:04d}.png')
                plot_metal_artifact_wave(test_pred, test_gt, test_ct, wave_path, epoch='test', sample_idx=idx)
    
    inference_time = time.time() - start_time
    logger.info(f"  ✓ Inference completed in {inference_time:.2f} seconds")
    logger.info(f"  ✓ Visualizations saved for {NUM_VIS_SAMPLES} fixed samples")
    logger.info(f"  ✓ Average time per sample: {inference_time/len(test_dataset)*1000:.2f} ms")
    
    # ─────────────────────────────────────────────────────────────
    # Aggregate Metrics
    # ─────────────────────────────────────────────────────────────
    logger.subsection("Test Set Metrics")
    
    # Average test metrics
    avg_metrics = {}
    for key in test_metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in test_metrics_list])
    
    # Average regional metrics
    avg_regional = {}
    for key in regional_metrics_list[0].keys():
        values = [m[key] for m in regional_metrics_list if m[key] is not None]
        avg_regional[key] = np.mean(values) if values else 0.0
    
    # Average HU metrics
    avg_hu = {}
    for key in hu_metrics_list[0].keys():
        values = [m[key] for m in hu_metrics_list if m[key] is not None]
        avg_hu[key] = np.mean(values) if values else 0.0
    
    logger.info(f"  Overall Metrics:")
    logger.info(f"    PSNR: {avg_metrics['PSNR']:.3f} dB")
    logger.info(f"    SSIM: {avg_metrics['SSIM']:.4f}")
    logger.info(f"    MSE:  {avg_metrics['MSE']:.6f}")
    logger.info(f"    RMSE: {avg_metrics['RMSE']:.6f}")
    logger.info(f"    MAE:  {avg_metrics['MAE']:.6f}")
    
    logger.info(f"  Regional Metrics (Test Set):")
    logger.info(f"    Metal PSNR:     {avg_regional['metal_PSNR']:.3f} dB")
    logger.info(f"    Band PSNR:      {avg_regional['band_PSNR']:.3f} dB")
    logger.info(f"    Non-Metal PSNR: {avg_regional['non_metal_PSNR']:.3f} dB")
    
    logger.info(f"  HU Accuracy (Test Set):")
    logger.info(f"    HU MAE:   {avg_hu['hu_mae']:.2f} HU")
    logger.info(f"    HU RMSE:  {avg_hu['hu_rmse']:.2f} HU")
    logger.info(f"    ±10 HU:   {avg_hu['within_10hu']*100:.1f}%")
    logger.info(f"    ±20 HU:   {avg_hu['within_20hu']*100:.1f}%")
    logger.info(f"    ±50 HU:   {avg_hu['within_50hu']*100:.1f}%")
    
    # ─────────────────────────────────────────────────────────────
    # Save Results
    # ─────────────────────────────────────────────────────────────
    
    # Save test metrics CSV
    test_csv_path = os.path.join(output_dir, 'test_metrics.csv')
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'PSNR', 'SSIM', 'MSE', 'RMSE', 'MAE',
                        'metal_PSNR', 'band_PSNR', 'non_metal_PSNR',
                        'hu_mae', 'hu_rmse', 'within_10hu', 'within_20hu', 'within_50hu'])
        for i, (m, r, h) in enumerate(zip(test_metrics_list, regional_metrics_list, hu_metrics_list)):
            writer.writerow([
                i, f"{m['PSNR']:.3f}", f"{m['SSIM']:.4f}", f"{m['MSE']:.6f}",
                f"{m['RMSE']:.6f}", f"{m['MAE']:.6f}",
                f"{r['metal_PSNR']:.3f}" if r['metal_PSNR'] else 'N/A',
                f"{r['band_PSNR']:.3f}" if r['band_PSNR'] else 'N/A',
                f"{r['non_metal_PSNR']:.3f}" if r['non_metal_PSNR'] else 'N/A',
                f"{h['hu_mae']:.2f}", f"{h['hu_rmse']:.2f}",
                f"{h['within_10hu']:.4f}", f"{h['within_20hu']:.4f}", f"{h['within_50hu']:.4f}"
            ])
        # Write averages
        writer.writerow(['AVERAGE', f"{avg_metrics['PSNR']:.3f}", f"{avg_metrics['SSIM']:.4f}",
                        f"{avg_metrics['MSE']:.6f}", f"{avg_metrics['RMSE']:.6f}", f"{avg_metrics['MAE']:.6f}",
                        f"{avg_regional['metal_PSNR']:.3f}", f"{avg_regional['band_PSNR']:.3f}",
                        f"{avg_regional['non_metal_PSNR']:.3f}",
                        f"{avg_hu['hu_mae']:.2f}", f"{avg_hu['hu_rmse']:.2f}",
                        f"{avg_hu['within_10hu']:.4f}", f"{avg_hu['within_20hu']:.4f}", f"{avg_hu['within_50hu']:.4f}"])
    logger.info(f"  ✓ Test metrics CSV saved")
    
    # Return results for summary
    final_metrics = {
        'ablation_id': ablation_id,
        'ablation_name': ablation_config['name'],
        'inference_only': True,
        'checkpoint_path': checkpoint_path,
        'best_psnr': avg_metrics['PSNR'],
        'best_ssim': avg_metrics['SSIM'],
        'final_metrics': avg_metrics,
        'final_regional_metrics': avg_regional,
        'final_hu_metrics': avg_hu,
        'total_time_hours': inference_time / 3600,
        'num_test_samples': len(test_dataset),
    }
    
    # Save final results JSON
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"")
    logger.info(f"  ═══════════════════════════════════════════════")
    logger.info(f"  TEST SET RESULTS: {ablation_id}")
    logger.info(f"  ═══════════════════════════════════════════════")
    logger.info(f"  PSNR: {avg_metrics['PSNR']:.3f} dB | SSIM: {avg_metrics['SSIM']:.4f}")
    logger.info(f"  Results saved to: {output_dir}")
    
    return final_metrics


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

# SET THIS TO RESUME FROM EXISTING DIRECTORY, OR NONE FOR NEW RUN
RESUME_FROM = "./ablation_results/loss_ablations_20251207_155318"
# Set to specific ablation to start from (skip completed ones)
# A0_mse_only, A1_no_physics, A2_no_metal_consistency, A3_no_metal_weighting, A4_no_adversarial, A5_no_feature_matching are DONE
# Now resuming from A6_no_edge
START_FROM_ABLATION = "A6_no_edge"  # Set to None to run all

def main():
    # Create main results directory or use existing one for resume
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        results_root = RESUME_FROM
        timestamp = os.path.basename(RESUME_FROM).split('_')[-2] + '_' + os.path.basename(RESUME_FROM).split('_')[-1]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = f"./ablation_results/loss_ablations_{timestamp}"
    os.makedirs(results_root, exist_ok=True)
    
    # Initialize main logger
    logger = AblationLogger(results_root, name="ablation_main")
    
    logger.section("LOSS-TERM ABLATION STUDIES")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"Epochs per ablation: {NUM_EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Patch size: {PATCH_SIZE}")
    logger.info(f"Inference-only ablations: {len(INFERENCE_ONLY_ABLATIONS)}")
    logger.info(f"Training ablations: {len(ABLATIONS)}")
    logger.info(f"Total ablations: {len(INFERENCE_ONLY_ABLATIONS) + len(ABLATIONS)}")
    logger.info(f"Output directory: {results_root}")
    logger.info(f"Log file: {logger.log_file}")
    
    # Print ablation summary
    logger.subsection("Inference-Only Ablations (pre-trained checkpoints)")
    for idx, (ablation_id, ablation_config) in enumerate(INFERENCE_ONLY_ABLATIONS.items()):
        logger.info(f"  {idx+1}. {ablation_id}: {ablation_config['name']}")
        logger.info(f"     Checkpoint: {ablation_config['checkpoint_path']}")
    
    logger.subsection("Training Ablations (train from scratch)")
    for idx, (ablation_id, ablation_config) in enumerate(ABLATIONS.items()):
        logger.info(f"  {idx+1}. {ablation_id}: {ablation_config['name']}")
        logger.debug(f"     {ablation_config['description']}")
    
    # Run all ablations
    all_results = []
    ablation_start_time = time.time()
    total_ablations = len(INFERENCE_ONLY_ABLATIONS) + len(ABLATIONS)
    current_idx = 0
    
    # ─────────────────────────────────────────────────────────────
    # RUN INFERENCE-ONLY ABLATIONS FIRST (skip if resuming from training ablation)
    # ─────────────────────────────────────────────────────────────
    if START_FROM_ABLATION is None or START_FROM_ABLATION in INFERENCE_ONLY_ABLATIONS:
        for ablation_id, ablation_config in INFERENCE_ONLY_ABLATIONS.items():
            current_idx += 1
            
            # Check if already completed (has final_results.json)
            output_dir = os.path.join(results_root, ablation_id)
            if os.path.exists(os.path.join(output_dir, 'final_results.json')):
                logger.info(f"  Skipping {ablation_id} (already completed)")
                # Load existing results
                with open(os.path.join(output_dir, 'final_results.json'), 'r') as f:
                    result = json.load(f)
                all_results.append(result)
                continue
            
            logger.section(f"STARTING INFERENCE-ONLY {current_idx}/{total_ablations}: {ablation_id}")
            
            # Create ablation-specific logger
            abl_logger = AblationLogger(output_dir, name=ablation_id)
            
            try:
                result = run_inference_only(ablation_id, ablation_config, output_dir, abl_logger)
                all_results.append(result)
                logger.info(f"  ✓ Inference {ablation_id} completed successfully")
                logger.info(f"    Test PSNR: {result['best_psnr']:.3f} dB")
            except Exception as e:
                logger.error(f"  ✗ Inference {ablation_id} FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'ablation_id': ablation_id,
                    'ablation_name': ablation_config['name'],
                    'error': str(e),
                })
    else:
        logger.info(f"  Skipping inference-only ablations (resuming from {START_FROM_ABLATION})")
    
    # ─────────────────────────────────────────────────────────────
    # RUN TRAINING ABLATIONS
    # ─────────────────────────────────────────────────────────────
    skip_until_found = START_FROM_ABLATION is not None and START_FROM_ABLATION in ABLATIONS
    
    for ablation_id, ablation_config in ABLATIONS.items():
        current_idx += 1
        
        # Skip ablations until we reach the one to start from
        if skip_until_found:
            if ablation_id == START_FROM_ABLATION:
                skip_until_found = False
                logger.info(f"  Resuming from {ablation_id}")
            else:
                logger.info(f"  Skipping {ablation_id} (resuming from {START_FROM_ABLATION})")
                continue
        
        logger.section(f"STARTING ABLATION {current_idx}/{total_ablations}: {ablation_id}")
        
        output_dir = os.path.join(results_root, ablation_id)
        
        # Create ablation-specific logger
        abl_logger = AblationLogger(output_dir, name=ablation_id)
        
        try:
            result = run_ablation(ablation_id, ablation_config, output_dir, abl_logger)
            all_results.append(result)
            logger.info(f"  ✓ Ablation {ablation_id} completed successfully")
            logger.info(f"    Best PSNR: {result['best_psnr']:.3f} dB")
        except Exception as e:
            logger.error(f"  ✗ Ablation {ablation_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'ablation_id': ablation_id,
                'ablation_name': ablation_config['name'],
                'error': str(e),
            })
        
        # Progress update
        elapsed = time.time() - ablation_start_time
        remaining_ablations = total_ablations - current_idx
        if current_idx > 1:
            avg_time_per_ablation = elapsed / current_idx
            eta_hours = (avg_time_per_ablation * remaining_ablations) / 3600
            logger.info(f"  Progress: {current_idx}/{total_ablations} complete | ETA: {eta_hours:.1f} hours")
    
    # ─────────────────────────────────────────────────────────────
    # GENERATE FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────
    logger.section("GENERATING FINAL SUMMARY")
    
    summary_dir = os.path.join(results_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save all results to JSON
    with open(os.path.join(summary_dir, 'all_ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  ✓ Results JSON saved")
    
    # Create comparison CSV
    comparison_csv_path = os.path.join(summary_dir, 'ablation_comparison.csv')
    with open(comparison_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ablation', 'Name', 'PSNR', 'SSIM', 'MSE', 'RMSE', 'MAE',
                         'Metal_PSNR', 'Band_PSNR', 'NonMetal_PSNR', 'Time_Hours', 'Status'])
        
        for r in all_results:
            if 'error' not in r:
                fm = r.get('final_metrics', {})
                fr = r.get('final_regional_metrics', {})
                writer.writerow([
                    r['ablation_id'],
                    r['ablation_name'],
                    f"{fm.get('PSNR', 0):.3f}",
                    f"{fm.get('SSIM', 0):.4f}",
                    f"{fm.get('MSE', 0):.6f}",
                    f"{fm.get('RMSE', 0):.6f}",
                    f"{fm.get('MAE', 0):.6f}",
                    f"{fr.get('metal_PSNR', 0):.3f}",
                    f"{fr.get('band_PSNR', 0):.3f}",
                    f"{fr.get('non_metal_PSNR', 0):.3f}",
                    f"{r.get('total_time_hours', 0):.2f}",
                    "SUCCESS",
                ])
            else:
                writer.writerow([
                    r['ablation_id'],
                    r.get('ablation_name', 'N/A'),
                    "N/A", "N/A", "N/A", "N/A", "N/A",
                    "N/A", "N/A", "N/A", "N/A",
                    f"FAILED: {r['error'][:50]}...",
                ])
    logger.info(f"  ✓ Comparison CSV saved")
    
    # Create comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Loss-Term Ablation Study Comparison', fontsize=16, fontweight='bold')
    
    ablation_names = []
    psnr_vals = []
    ssim_vals = []
    metal_psnr_vals = []
    band_psnr_vals = []
    nm_psnr_vals = []
    
    for r in all_results:
        if 'error' not in r:
            ablation_names.append(r['ablation_id'].replace('_', '\n').replace('A', 'A'))
            fm = r.get('final_metrics', {})
            fr = r.get('final_regional_metrics', {})
            psnr_vals.append(fm.get('PSNR', 0))
            ssim_vals.append(fm.get('SSIM', 0))
            metal_psnr_vals.append(fr.get('metal_PSNR', 0))
            band_psnr_vals.append(fr.get('band_PSNR', 0))
            nm_psnr_vals.append(fr.get('non_metal_PSNR', 0))
    
    if len(ablation_names) > 0:
        x = np.arange(len(ablation_names))
        
        # PSNR comparison
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(x)))
        bars1 = axes[0, 0].bar(x, psnr_vals, color=colors, edgecolor='navy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(ablation_names, fontsize=8)
        axes[0, 0].set_ylabel('PSNR (dB)', fontsize=10)
        axes[0, 0].set_title('Overall PSNR Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars1, psnr_vals):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        
        # SSIM comparison
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(x)))
        bars2 = axes[0, 1].bar(x, ssim_vals, color=colors, edgecolor='darkgreen')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(ablation_names, fontsize=8)
        axes[0, 1].set_ylabel('SSIM', fontsize=10)
        axes[0, 1].set_title('Overall SSIM Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars2, ssim_vals):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                           f'{val:.4f}', ha='center', va='bottom', fontsize=7)
        
        # Regional PSNR comparison
        width = 0.25
        axes[1, 0].bar(x - width, metal_psnr_vals, width, label='Metal Region', color='indianred', edgecolor='darkred')
        axes[1, 0].bar(x, band_psnr_vals, width, label='Artifact Band', color='coral', edgecolor='darkorange')
        axes[1, 0].bar(x + width, nm_psnr_vals, width, label='Non-Metal', color='skyblue', edgecolor='steelblue')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(ablation_names, fontsize=8)
        axes[1, 0].set_ylabel('PSNR (dB)', fontsize=10)
        axes[1, 0].set_title('Regional PSNR Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Sorted PSNR ranking
        sorted_indices = np.argsort(psnr_vals)[::-1]
        sorted_names = [ablation_names[i] for i in sorted_indices]
        sorted_psnr = [psnr_vals[i] for i in sorted_indices]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_indices)))[::-1]
        
        bars4 = axes[1, 1].barh(range(len(sorted_names)), sorted_psnr, color=colors, edgecolor='black')
        axes[1, 1].set_yticks(range(len(sorted_names)))
        axes[1, 1].set_yticklabels(sorted_names, fontsize=8)
        axes[1, 1].set_xlabel('PSNR (dB)', fontsize=10)
        axes[1, 1].set_title('PSNR Ranking (Best to Worst)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        for i, (bar, val) in enumerate(zip(bars4, sorted_psnr)):
            axes[1, 1].text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                           f'{val:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(summary_dir, 'ablation_comparison.png')
    plt.savefig(comparison_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Comparison plot saved")
    
    # Generate LaTeX table
    latex_path = os.path.join(summary_dir, 'ablation_table.tex')
    with open(latex_path, 'w') as f:
        f.write("% Ablation Study Results Table\n")
        f.write("% Generated by run_loss_ablations.py\n")
        f.write(f"% Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Loss-Term Ablation Study Results. Bold indicates best result per metric.}\n")
        f.write("\\label{tab:loss_ablations}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Configuration} & \\textbf{PSNR (dB)} & \\textbf{SSIM} & ")
        f.write("\\textbf{Metal PSNR} & \\textbf{Band PSNR} & \\textbf{Non-Metal PSNR} & \\textbf{Time (h)} \\\\\n")
        f.write("\\midrule\n")
        
        # Find best values for bolding
        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results:
            best_psnr = max(r.get('final_metrics', {}).get('PSNR', 0) for r in valid_results)
            best_ssim = max(r.get('final_metrics', {}).get('SSIM', 0) for r in valid_results)
            best_metal = max(r.get('final_regional_metrics', {}).get('metal_PSNR', 0) for r in valid_results)
            best_band = max(r.get('final_regional_metrics', {}).get('band_PSNR', 0) for r in valid_results)
            best_nm = max(r.get('final_regional_metrics', {}).get('non_metal_PSNR', 0) for r in valid_results)
        
        for r in all_results:
            if 'error' not in r:
                fm = r.get('final_metrics', {})
                fr = r.get('final_regional_metrics', {})
                name = r['ablation_name'].replace('_', ' ')
                
                psnr = fm.get('PSNR', 0)
                ssim = fm.get('SSIM', 0)
                metal = fr.get('metal_PSNR', 0)
                band = fr.get('band_PSNR', 0)
                nm = fr.get('non_metal_PSNR', 0)
                time_h = r.get('total_time_hours', 0)
                
                # Bold best values
                psnr_str = f"\\textbf{{{psnr:.2f}}}" if abs(psnr - best_psnr) < 0.01 else f"{psnr:.2f}"
                ssim_str = f"\\textbf{{{ssim:.4f}}}" if abs(ssim - best_ssim) < 0.0001 else f"{ssim:.4f}"
                metal_str = f"\\textbf{{{metal:.2f}}}" if abs(metal - best_metal) < 0.01 else f"{metal:.2f}"
                band_str = f"\\textbf{{{band:.2f}}}" if abs(band - best_band) < 0.01 else f"{band:.2f}"
                nm_str = f"\\textbf{{{nm:.2f}}}" if abs(nm - best_nm) < 0.01 else f"{nm:.2f}"
                
                f.write(f"{name} & {psnr_str} & {ssim_str} & {metal_str} & {band_str} & {nm_str} & {time_h:.1f} \\\\\n")
            else:
                name = r.get('ablation_name', 'N/A').replace('_', ' ')
                f.write(f"{name} & \\multicolumn{{6}}{{c}}{{FAILED: {r['error'][:30]}...}} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    logger.info(f"  ✓ LaTeX table saved")
    
    # ─────────────────────────────────────────────────────────────
    # Generate Ablation × Metric Heatmap Matrix
    # ─────────────────────────────────────────────────────────────
    if len(valid_results) > 1:
        logger.info(f"  Generating ablation × metric heatmap...")
        
        # Collect all metrics for heatmap
        ablation_labels = []
        metrics_data = {
            'PSNR': [],
            'SSIM': [],
            'MSE': [],
            'Metal_PSNR': [],
            'Band_PSNR': [],
            'NonMetal_PSNR': [],
        }
        
        for r in valid_results:
            ablation_labels.append(r['ablation_id'].replace('_', '\n'))
            fm = r.get('final_metrics', {})
            fr = r.get('final_regional_metrics', {})
            metrics_data['PSNR'].append(fm.get('PSNR', 0))
            metrics_data['SSIM'].append(fm.get('SSIM', 0))
            metrics_data['MSE'].append(fm.get('MSE', 0))
            metrics_data['Metal_PSNR'].append(fr.get('metal_PSNR', 0))
            metrics_data['Band_PSNR'].append(fr.get('band_PSNR', 0))
            metrics_data['NonMetal_PSNR'].append(fr.get('non_metal_PSNR', 0))
        
        # Create heatmap matrix
        metric_names = list(metrics_data.keys())
        matrix = np.array([metrics_data[m] for m in metric_names])
        
        # Normalize each metric to [0, 1] for visualization
        matrix_normalized = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            row = matrix[i]
            if row.max() - row.min() > 1e-6:
                matrix_normalized[i] = (row - row.min()) / (row.max() - row.min())
            else:
                matrix_normalized[i] = 0.5
        
        # For MSE, lower is better, so invert
        mse_idx = metric_names.index('MSE')
        matrix_normalized[mse_idx] = 1 - matrix_normalized[mse_idx]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(matrix_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(ablation_labels)))
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_xticklabels(ablation_labels, fontsize=9)
        ax.set_yticklabels(metric_names, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add text annotations with actual values
        for i in range(len(metric_names)):
            for j in range(len(ablation_labels)):
                val = matrix[i, j]
                # Format based on metric type
                if metric_names[i] == 'SSIM':
                    text = f'{val:.4f}'
                elif metric_names[i] == 'MSE':
                    text = f'{val:.5f}'
                else:
                    text = f'{val:.2f}'
                
                # Choose text color based on background
                text_color = 'white' if matrix_normalized[i, j] < 0.4 or matrix_normalized[i, j] > 0.8 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=8)
        
        ax.set_title('Ablation Study Results Matrix\n(Green = Better, Red = Worse)', 
                     fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Score (Higher = Better)', fontsize=10)
        
        plt.tight_layout()
        heatmap_path = os.path.join(summary_dir, 'ablation_metric_heatmap.png')
        plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Ablation × Metric heatmap saved")
    
    # Generate detailed summary text file
    summary_txt_path = os.path.join(summary_dir, 'summary.txt')
    with open(summary_txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LOSS-TERM ABLATION STUDIES - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")
        f.write(f"Epochs per ablation: {NUM_EPOCHS}\n")
        f.write(f"Total ablations: {len(ABLATIONS)}\n")
        f.write(f"Successful: {len(valid_results)}\n")
        f.write(f"Failed: {len(all_results) - len(valid_results)}\n\n")
        
        total_time = time.time() - ablation_start_time
        f.write(f"Total runtime: {total_time/3600:.2f} hours\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("RESULTS RANKED BY PSNR\n")
        f.write("-" * 70 + "\n\n")
        
        sorted_results = sorted([r for r in all_results if 'error' not in r], 
                                key=lambda x: x.get('final_metrics', {}).get('PSNR', 0),
                                reverse=True)
        
        for rank, r in enumerate(sorted_results, 1):
            fm = r.get('final_metrics', {})
            fr = r.get('final_regional_metrics', {})
            f.write(f"#{rank}: {r['ablation_id']}\n")
            f.write(f"    Name: {r['ablation_name']}\n")
            f.write(f"    PSNR: {fm.get('PSNR', 0):.3f} dB\n")
            f.write(f"    SSIM: {fm.get('SSIM', 0):.4f}\n")
            f.write(f"    MSE:  {fm.get('MSE', 0):.6f}\n")
            f.write(f"    Regional PSNR: Metal={fr.get('metal_PSNR', 0):.2f}, ")
            f.write(f"Band={fr.get('band_PSNR', 0):.2f}, NonMetal={fr.get('non_metal_PSNR', 0):.2f}\n")
            f.write(f"    Training time: {r.get('total_time_hours', 0):.2f} hours\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-" * 70 + "\n\n")
        
        if len(sorted_results) >= 2:
            best = sorted_results[0]
            worst = sorted_results[-1]
            best_fm = best.get('final_metrics', {})
            worst_fm = worst.get('final_metrics', {})
            
            f.write(f"Best Configuration: {best['ablation_name']}\n")
            f.write(f"  PSNR: {best_fm.get('PSNR', 0):.3f} dB\n\n")
            
            f.write(f"Worst Configuration: {worst['ablation_name']}\n")
            f.write(f"  PSNR: {worst_fm.get('PSNR', 0):.3f} dB\n\n")
            
            delta = best_fm.get('PSNR', 0) - worst_fm.get('PSNR', 0)
            f.write(f"PSNR range: {delta:.3f} dB\n\n")
    
    logger.info(f"  ✓ Summary text saved")
    
    # Final summary
    total_time = time.time() - ablation_start_time
    
    logger.section("ALL ABLATION STUDIES COMPLETE!")
    logger.info(f"  Total Runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    logger.info(f"  Successful: {len(valid_results)}/{len(ABLATIONS)}")
    logger.info(f"  Results Directory: {results_root}")
    
    logger.subsection("Final Rankings (by PSNR)")
    sorted_results = sorted([r for r in all_results if 'error' not in r], 
                            key=lambda x: x.get('final_metrics', {}).get('PSNR', 0),
                            reverse=True)
    
    for rank, r in enumerate(sorted_results, 1):
        fm = r.get('final_metrics', {})
        logger.info(f"  #{rank}: {r['ablation_id']}: PSNR={fm.get('PSNR', 0):.3f} dB, SSIM={fm.get('SSIM', 0):.4f}")
    
    if len(all_results) - len(valid_results) > 0:
        logger.warning(f"\n  FAILED ABLATIONS:")
        for r in all_results:
            if 'error' in r:
                logger.warning(f"    - {r['ablation_id']}: {r['error']}")
    
    logger.info(f"\n  Summary files:")
    logger.info(f"    - {comparison_csv_path}")
    logger.info(f"    - {comparison_plot_path}")
    logger.info(f"    - {latex_path}")
    logger.info(f"    - {summary_txt_path}")
    
    logger.section("DONE")

if __name__ == "__main__":
    main()
