#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Fine-tune All Benchmark Models on SpineWeb Dataset
===================================================
This script fine-tunes each benchmark MAR model on the SpineWeb clinical dataset
for 15 epochs, one model at a time.

Models:
    1. DICDNet (image-domain)
    2. FIND-Net (image-domain)
    3. InDuDoNet (dual-domain)
    4. InDuDoNet+ (dual-domain with NMAR prior)
    5. MEPNet (dual-domain, sparse-view)

Usage:
    python finetune_all_benchmarks_spineweb.py
    python finetune_all_benchmarks_spineweb.py --models DICDNet InDuDoNet
    python finetune_all_benchmarks_spineweb.py --epochs 20 --batch_size 4
"""

import os
import sys
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import time
import json
import logging
import random
from datetime import datetime
from glob import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Fine-tune Benchmark Models on SpineWeb")
parser.add_argument("--data_path", type=str, default="/home/Drive-D/UWSpine_adn/",
                    help='Path to SpineWeb dataset')
parser.add_argument("--save_path", type=str, default="./finetune_results/",
                    help='Path to save results')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--epochs", type=int, default=5, help='Number of epochs per model')
parser.add_argument("--batch_size", type=int, default=1, help='Batch size')
parser.add_argument("--lr", type=float, default=1e-5, help='Learning rate for fine-tuning')
parser.add_argument("--patch_size", type=int, default=64, help='Patch size for training')
parser.add_argument("--max_train_samples", type=int, default=1000, help='Max training samples')
parser.add_argument("--max_test_samples", type=int, default=500, help='Max test samples')
parser.add_argument("--val_freq", type=int, default=5, help='Validation frequency (epochs)')
parser.add_argument("--save_freq", type=int, default=5, help='Checkpoint save frequency (epochs)')
parser.add_argument("--num_val_samples", type=int, default=50, help='Number of validation samples')
parser.add_argument("--num_test_save", type=int, default=10, help='Number of test samples to save visualizations')
parser.add_argument("--num_workers", type=int, default=4, help='Number of data loader workers')
parser.add_argument("--seed", type=int, default=42, help='Random seed')
parser.add_argument("--models", nargs='+', 
                    default=['InDuDoNet', 'InDuDoNet_plus', 'MEPNet'],
                    help='Models to fine-tune')

# Loss weights
parser.add_argument('--lambda_rec', type=float, default=1.0, help='Reconstruction loss weight')
parser.add_argument('--lambda_edge', type=float, default=0.1, help='Edge loss weight')
parser.add_argument('--lambda_sino', type=float, default=0.5, help='Sinogram loss weight (dual-domain)')

# Memory optimization
parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision (FP16)')
parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True, help='Use gradient checkpointing')
parser.add_argument('--mepnet_train_samples', type=int, default=500, help='Reduced training samples for MEPNet (memory constrained)')
parser.add_argument('--mepnet_test_samples', type=int, default=200, help='Reduced test samples for MEPNet (memory constrained)')
parser.add_argument('--mepnet_no_sino_loss', action='store_true', default=True, help='Disable sinogram loss for MEPNet to save memory')
parser.add_argument('--mepnet_freeze_stages', type=int, default=8, help='Freeze first N stages of MEPNet (0-9) to reduce memory. Higher = less memory but less fine-tuning.')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

# Benchmark directories
DCGAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BENCHMARKS_ROOT = os.path.join(DCGAN_ROOT, 'benchmarks')

# ═══════════════════════════════════════════════════════════════
# SPINEWEB DATASET
# ═══════════════════════════════════════════════════════════════

class SpineWebDataset(Dataset):
    """SpineWeb dataset for benchmark model fine-tuning"""
    
    def __init__(self, data_path, split='train', patch_size=64, normalize_range='0_255',
                 resize_mode='patch', max_samples=None, seed=42):
        """
        Args:
            data_path: Root path to SpineWeb dataset
            split: 'train' or 'test'
            patch_size: Size of patches (for patch mode) or target size (for resize mode)
            normalize_range: '0_255' for DICDNet/FIND-Net, '0_1' for InDuDoNet variants
            resize_mode: 'patch' for random patches (image-domain models), 
                        'resize' for full image resize (dual-domain models needing 416x416)
            max_samples: Maximum number of samples to use (None = use all)
            seed: Random seed for deterministic sample selection (same samples every run)
        """
        self.patch_size = patch_size
        self.split = split
        self.normalize_range = normalize_range
        self.resize_mode = resize_mode
        
        # Get image paths
        ma_dir = os.path.join(data_path, split, 'synthesized_metal_transfer')
        gt_dir = os.path.join(data_path, split, 'no_metal')
        
        self.ma_files = sorted(glob(os.path.join(ma_dir, '*.npy')))
        self.gt_files = sorted(glob(os.path.join(gt_dir, '*.npy')))
        
        # Match files by name
        self.pairs = []
        gt_names = {os.path.basename(f): f for f in self.gt_files}
        for ma_file in self.ma_files:
            name = os.path.basename(ma_file)
            if name in gt_names:
                self.pairs.append((ma_file, gt_names[name]))
        
        # Limit samples if specified (for faster training)
        # Use deterministic selection with seed for reproducibility
        total_found = len(self.pairs)
        if max_samples is not None and len(self.pairs) > max_samples:
            # Use seeded random for deterministic selection (same samples every run)
            rng = random.Random(seed)
            self.pairs = sorted(self.pairs)  # Sort first for consistency
            indices = list(range(len(self.pairs)))
            rng.shuffle(indices)
            selected_indices = sorted(indices[:max_samples])  # Keep order after selection
            self.pairs = [self.pairs[i] for i in selected_indices]
        
        mode_str = f"resize to {patch_size}x{patch_size}" if resize_mode == 'resize' else f"patches {patch_size}x{patch_size}"
        if max_samples is not None and total_found > max_samples:
            print(f"    [SpineWeb] {split}: Using {len(self.pairs)}/{total_found} samples ({mode_str})")
        else:
            print(f"    [SpineWeb] {split}: Found {len(self.pairs)} paired samples ({mode_str})")
    
    def __len__(self):
        return len(self.pairs)
    
    def normalize(self, img):
        """Normalize from HU values to target range"""
        # SpineWeb is in HU range approximately [-1000, 2000]
        img = (img + 1000) / 3000.0
        img = np.clip(img, 0, 1)
        
        if self.normalize_range == '0_255':
            img = img * 255.0
        # else '0_1' - already in [0, 1]
        
        return img.astype(np.float32)
    
    def resize_image(self, img, target_size):
        """Resize image to target size using PIL for smooth interpolation"""
        from PIL import Image
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)
        return np.array(img_resized, dtype=np.float32)
    
    def extract_random_patch(self, img, h, w):
        """Extract random patch for training"""
        half = self.patch_size // 2
        center_h = random.randint(half, h - half - 1) if h > self.patch_size else h // 2
        center_w = random.randint(half, w - half - 1) if w > self.patch_size else w // 2
        
        h_start = max(0, center_h - half)
        w_start = max(0, center_w - half)
        
        patch = img[h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
        
        # Pad if needed
        if patch.shape != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        
        return patch
    
    def create_metal_mask(self, ma_img, gt_img):
        """Create approximate metal mask from difference"""
        diff = np.abs(ma_img - gt_img)
        threshold = np.percentile(diff, 95) if diff.max() > 0 else 0.1
        mask = (diff > threshold).astype(np.float32)
        return mask
    
    def create_li_image(self, ma_img, mask):
        """Create Linear Interpolation image (simple approximation)"""
        li_img = ma_img.copy()
        if mask.sum() > 0:
            non_metal_mean = ma_img[mask < 0.5].mean() if (mask < 0.5).sum() > 0 else ma_img.mean()
            li_img[mask > 0.5] = non_metal_mean
        return li_img
    
    def __getitem__(self, idx):
        ma_path, gt_path = self.pairs[idx]
        
        # Load images
        ma_img = np.load(ma_path).astype(np.float32)
        gt_img = np.load(gt_path).astype(np.float32)
        
        # Normalize
        ma_img = self.normalize(ma_img)
        gt_img = self.normalize(gt_img)
        
        h, w = ma_img.shape
        
        if self.resize_mode == 'resize':
            # Resize full image to target size (for dual-domain models needing fixed 416x416)
            ma_patch = self.resize_image(ma_img, self.patch_size)
            gt_patch = self.resize_image(gt_img, self.patch_size)
        else:
            # Extract patches (for image-domain models)
            if self.split == 'train':
                ma_patch = self.extract_random_patch(ma_img, h, w)
                gt_patch = self.extract_random_patch(gt_img, h, w)
            else:
                # Center crop for validation
                ch, cw = h // 2, w // 2
                half = self.patch_size // 2
                ma_patch = ma_img[ch-half:ch+half, cw-half:cw+half]
                gt_patch = gt_img[ch-half:ch+half, cw-half:cw+half]
        
        # Create mask and LI image
        mask = self.create_metal_mask(ma_patch, gt_patch)
        li_patch = self.create_li_image(ma_patch, mask)
        
        # Convert to tensors with batch dim pattern expected by models
        # Shape: (1, H, W) - single channel
        Xma = torch.from_numpy(ma_patch).unsqueeze(0)
        Xgt = torch.from_numpy(gt_patch).unsqueeze(0)
        XLI = torch.from_numpy(li_patch).unsqueeze(0)
        M = torch.from_numpy(1 - mask).unsqueeze(0)  # non-metal mask
        
        return {
            'Xma': Xma,
            'Xgt': Xgt,
            'XLI': XLI,
            'M': M,
            'mask': torch.from_numpy(mask).unsqueeze(0)
        }


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def compute_image_gradients(x):
    """Compute spatial gradients"""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), mode='replicate')
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1), mode='replicate')
    return dx, dy

def reconstruction_loss(pred, gt):
    """L1 reconstruction loss"""
    return torch.mean(torch.abs(pred - gt))

def mse_loss(pred, gt):
    """MSE reconstruction loss"""
    return torch.mean((pred - gt) ** 2)

def edge_loss(pred, gt):
    """Edge preservation loss"""
    pred_dx, pred_dy = compute_image_gradients(pred)
    gt_dx, gt_dy = compute_image_gradients(gt)
    loss_x = torch.mean(torch.abs(pred_dx - gt_dx))
    loss_y = torch.mean(torch.abs(pred_dy - gt_dy))
    return loss_x + loss_y

def sinogram_loss(pred_sino, gt_sino, trace_mask=None):
    """Sinogram domain loss"""
    if trace_mask is not None:
        # Only compute loss on non-metal regions
        diff = torch.abs(pred_sino - gt_sino)
        return torch.mean(diff * trace_mask)
    else:
        return torch.mean(torch.abs(pred_sino - gt_sino))


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def calculate_metrics(gt_tensor, pred_tensor, data_range=1.0):
    """Calculate PSNR, SSIM, MAE"""
    gt_np = gt_tensor.detach().cpu().numpy().squeeze()
    pred_np = pred_tensor.detach().cpu().numpy().squeeze()
    
    # Normalize to [0, 1] if needed
    if data_range == 255.0:
        gt_np = gt_np / 255.0
        pred_np = pred_np / 255.0
        data_range = 1.0
    
    psnr_val = psnr(gt_np, pred_np, data_range=data_range)
    ssim_val = ssim(gt_np, pred_np, data_range=data_range)
    mae_val = np.mean(np.abs(gt_np - pred_np))
    
    return psnr_val, ssim_val, mae_val


def postprocess_model_output(Xout, data_range=255.0, model_type='image_domain'):
    """Apply correct post-processing to model output.
    
    The benchmark models (InDuDoNet, InDuDoNet+, MEPNet, FIND-Net) output values in
    [0, 127.5] range (i.e., [0, 0.5] when divided by 255). The post-processing
    clips to [0, 0.5] then rescales to [0, 1] by dividing by 0.5.
    
    This matches the original test scripts from the benchmark papers.
    """
    if data_range == 255.0:
        # Standard post-processing for all benchmark models:
        # 1. Divide by 255 to get [0, ~0.5] range
        # 2. Clip to [0, 0.5]
        # 3. Rescale to [0, 1] by dividing by 0.5
        out_clip = torch.clamp(Xout / 255.0, 0, 0.5)
        out_norm = out_clip / 0.5
    else:
        # For models using [0, 1] range directly
        out_norm = torch.clamp(Xout, 0, 1)
    
    return out_norm


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def save_training_curves(train_losses, val_psnrs, val_ssims, save_dir, model_name):
    """Save training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training loss
    axes[0].plot(train_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title(f'{model_name} - Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Validation PSNR
    axes[1].plot(val_psnrs, 'g-', linewidth=2, marker='o')
    axes[1].axhline(max(val_psnrs) if val_psnrs else 0, color='red', linestyle='--', 
                   label=f'Best: {max(val_psnrs):.2f} dB' if val_psnrs else '')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title(f'{model_name} - Validation PSNR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Validation SSIM
    axes[2].plot(val_ssims, 'm-', linewidth=2, marker='o')
    axes[2].axhline(max(val_ssims) if val_ssims else 0, color='red', linestyle='--',
                   label=f'Best: {max(val_ssims):.4f}' if val_ssims else '')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].set_title(f'{model_name} - Validation SSIM')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'), dpi=150)
    plt.close()


def save_sample_images(epoch, batch, output, save_dir, model_name, data_range=255.0):
    """Save sample comparison images"""
    ma = batch['Xma'][0].cpu().numpy().squeeze()
    gt = batch['Xgt'][0].cpu().numpy().squeeze()
    out = output[0].detach().cpu().numpy().squeeze()
    
    if data_range == 255.0:
        ma, gt, out = ma / 255.0, gt / 255.0, out / 255.0
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(ma, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Metal-Affected')
    axes[0].axis('off')
    
    axes[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(out, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'{model_name} Output')
    axes[2].axis('off')
    
    diff = np.abs(gt - out)
    im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
    axes[3].set_title('Abs Difference')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    plt.suptitle(f'{model_name} - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_epoch_{epoch:03d}.png'), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# TEST/INFERENCE VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_artifact_roi(ma_img, gt_img, padding=10):
    """Find ROI around metal artifact region based on difference between MA and GT images"""
    # Metal artifacts show up as differences between MA and GT
    diff = np.abs(ma_img - gt_img)
    threshold = np.percentile(diff, 90)  # Top 10% difference = artifact region
    artifact_mask = diff > threshold
    
    # Find bounding box of artifact region
    rows = np.any(artifact_mask, axis=1)
    cols = np.any(artifact_mask, axis=0)
    
    if not rows.any() or not cols.any():
        # Fallback to center if no artifact found
        h, w = ma_img.shape
        roi_size = min(h, w) // 3
        ch, cw = h // 2, w // 2
        return ch - roi_size//2, ch + roi_size//2, cw - roi_size//2, cw + roi_size//2
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add padding and ensure square-ish ROI
    h, w = ma_img.shape
    roi_h = rmax - rmin + 2 * padding
    roi_w = cmax - cmin + 2 * padding
    roi_size = max(roi_h, roi_w, min(h, w) // 4)  # At least 25% of image
    
    # Center the ROI on the artifact
    center_r = (rmin + rmax) // 2
    center_c = (cmin + cmax) // 2
    
    r1 = max(0, center_r - roi_size // 2)
    r2 = min(h, center_r + roi_size // 2)
    c1 = max(0, center_c - roi_size // 2)
    c2 = min(w, center_c + roi_size // 2)
    
    return r1, r2, c1, c2


def save_comparison_image(idx, ma_img, gt_img, out_img, save_dir, model_name, metrics=None):
    """Save side-by-side comparison image with metrics - both full view and zoomed on artifact"""
    h, w = gt_img.shape
    
    # Get ROI around metal artifact (not center)
    r1, r2, c1, c2 = get_artifact_roi(ma_img, gt_img)
    roi_h, roi_w = r2 - r1, c2 - c1
    
    fig = plt.figure(figsize=(20, 10))
    
    # Top row: Full images
    ax0 = fig.add_subplot(2, 4, 1)
    ax0.imshow(ma_img, cmap='gray', vmin=0, vmax=1)
    ax0.set_title('Metal-Affected CT', fontsize=11, fontweight='bold')
    ax0.add_patch(plt.Rectangle((c1, r1), roi_w, roi_h, fill=False, edgecolor='red', linewidth=2))
    ax0.axis('off')
    
    ax1 = fig.add_subplot(2, 4, 2)
    ax1.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Ground Truth', fontsize=11, fontweight='bold')
    ax1.add_patch(plt.Rectangle((c1, r1), roi_w, roi_h, fill=False, edgecolor='red', linewidth=2))
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 3)
    ax2.imshow(out_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'{model_name} Result', fontsize=11, fontweight='bold')
    ax2.add_patch(plt.Rectangle((c1, r1), roi_w, roi_h, fill=False, edgecolor='red', linewidth=2))
    ax2.axis('off')
    
    diff_full = np.abs(gt_img - out_img)
    ax3 = fig.add_subplot(2, 4, 4)
    im = ax3.imshow(diff_full, cmap='hot', vmin=0, vmax=0.15)
    ax3.set_title('Abs Difference', fontsize=11, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # Bottom row: Zoomed ROI (on artifact region)
    ax4 = fig.add_subplot(2, 4, 5)
    ax4.imshow(ma_img[r1:r2, c1:c2], cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Metal-Affected (Zoom)', fontsize=11)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(2, 4, 6)
    ax5.imshow(gt_img[r1:r2, c1:c2], cmap='gray', vmin=0, vmax=1)
    ax5.set_title('Ground Truth (Zoom)', fontsize=11)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 4, 7)
    ax6.imshow(out_img[r1:r2, c1:c2], cmap='gray', vmin=0, vmax=1)
    ax6.set_title(f'{model_name} (Zoom)', fontsize=11)
    ax6.axis('off')
    
    # Metrics text box
    ax7 = fig.add_subplot(2, 4, 8)
    ax7.axis('off')
    if metrics:
        metrics_text = f"Sample #{idx}\n\nPSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.4f}\nMAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}"
        ax7.text(0.1, 0.6, metrics_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'{model_name} - Sample {idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_difference_map(idx, ma_img, gt_img, out_img, save_dir, model_name):
    """Save difference map between GT and output - both full view and zoomed on artifact"""
    diff = np.abs(gt_img - out_img)
    h, w = gt_img.shape
    
    # Get ROI around metal artifact (not center)
    r1, r2, c1, c2 = get_artifact_roi(ma_img, gt_img)
    roi_h, roi_w = r2 - r1, c2 - c1
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Full images
    axes[0, 0].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth', fontweight='bold')
    axes[0, 0].add_patch(plt.Rectangle((c1, r1), roi_w, roi_h, fill=False, edgecolor='red', linewidth=2))
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(out_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'{model_name} Output', fontweight='bold')
    axes[0, 1].add_patch(plt.Rectangle((c1, r1), roi_w, roi_h, fill=False, edgecolor='red', linewidth=2))
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.15)
    axes[0, 2].set_title('Absolute Difference', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Bottom row: Zoomed ROI (on artifact region)
    axes[1, 0].imshow(gt_img[r1:r2, c1:c2], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth (Zoom)', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(out_img[r1:r2, c1:c2], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'{model_name} (Zoom)', fontweight='bold')
    axes[1, 1].axis('off')
    
    im2 = axes[1, 2].imshow(diff[r1:r2, c1:c2], cmap='hot', vmin=0, vmax=0.15)
    axes[1, 2].set_title('Difference (Zoom)', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'{model_name} - Difference Map #{idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'difference_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_test_metrics_histogram(metrics_data, save_dir, model_name):
    """Create histograms of PSNR and SSIM distributions"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR histogram
    axes[0].hist(psnr_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(psnr_values), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(psnr_values):.2f} dB')
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{model_name} - PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[1].hist(ssim_values, bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(ssim_values), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(ssim_values):.4f}')
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{model_name} - SSIM Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_test_metrics_boxplot(metrics_data, save_dir, model_name):
    """Create box plots for metrics"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bp1 = axes[0].boxplot(psnr_values, patch_artist=True)
    bp1['boxes'][0].set_facecolor('steelblue')
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title(f'{model_name} - PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    bp2 = axes[1].boxplot(ssim_values, patch_artist=True)
    bp2['boxes'][0].set_facecolor('forestgreen')
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title(f'{model_name} - SSIM Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    bp3 = axes[2].boxplot(mae_values, patch_artist=True)
    bp3['boxes'][0].set_facecolor('coral')
    axes[2].set_ylabel('MAE', fontsize=12)
    axes[2].set_title(f'{model_name} - MAE Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_test_summary_figure(metrics_data, save_dir, total_time, model_name):
    """Create a comprehensive summary figure"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Title
    fig.suptitle(f'{model_name} Fine-tuned - SpineWeb Test Set', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # PSNR histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(psnr_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(psnr_values), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('PSNR (dB)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('PSNR Distribution')
    ax1.grid(True, alpha=0.3)
    
    # SSIM histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(ssim_values, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(ssim_values), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('SSIM')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SSIM Distribution')
    ax2.grid(True, alpha=0.3)
    
    # MAE histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(mae_values, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(mae_values), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('MAE')
    ax3.set_ylabel('Frequency')
    ax3.set_title('MAE Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Box plots
    ax4 = fig.add_subplot(gs[1, :2])
    bp = ax4.boxplot([psnr_values, [s*100 for s in ssim_values], [m*100 for m in mae_values]], 
                     labels=['PSNR (dB)', 'SSIM (×100)', 'MAE (×100)'],
                     patch_artist=True)
    colors = ['steelblue', 'forestgreen', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title('Metrics Box Plot Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Summary statistics text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    stats_text = f"""
SUMMARY STATISTICS
══════════════════════

PSNR:
  Mean:  {np.mean(psnr_values):.4f} dB
  Std:   {np.std(psnr_values):.4f} dB
  Min:   {np.min(psnr_values):.4f} dB
  Max:   {np.max(psnr_values):.4f} dB

SSIM:
  Mean:  {np.mean(ssim_values):.6f}
  Std:   {np.std(ssim_values):.6f}
  Min:   {np.min(ssim_values):.6f}
  Max:   {np.max(ssim_values):.6f}

MAE:
  Mean:  {np.mean(mae_values):.6f}
  Std:   {np.std(mae_values):.6f}

══════════════════════
Total Samples: {len(metrics_data)}
Total Time: {total_time/60:.2f} min
Avg Time/Sample: {total_time/len(metrics_data):.4f} s
"""
    ax5.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
             fontfamily='monospace', transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # PSNR vs SSIM scatter
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(psnr_values, ssim_values, alpha=0.5, s=20)
    ax6.set_xlabel('PSNR (dB)')
    ax6.set_ylabel('SSIM')
    ax6.set_title('PSNR vs SSIM Correlation')
    ax6.grid(True, alpha=0.3)
    
    # Per-sample metrics line plot
    ax7 = fig.add_subplot(gs[2, 1:])
    x_vals = range(1, len(psnr_values) + 1)
    ax7.plot(x_vals, psnr_values, 'b-', linewidth=1, alpha=0.7, label='PSNR')
    ax7.axhline(np.mean(psnr_values), color='blue', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('PSNR (dB)', color='blue')
    ax7.tick_params(axis='y', labelcolor='blue')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Per-Sample Metrics')
    
    ax7_twin = ax7.twinx()
    ax7_twin.plot(x_vals, ssim_values, 'g-', linewidth=1, alpha=0.7, label='SSIM')
    ax7_twin.axhline(np.mean(ssim_values), color='green', linestyle='--', alpha=0.5)
    ax7_twin.set_ylabel('SSIM', color='green')
    ax7_twin.tick_params(axis='y', labelcolor='green')
    ax7_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_figure.png'), dpi=200, bbox_inches='tight')
    plt.close()


def save_test_metrics_csv(metrics_data, save_path):
    """Save all individual metrics to CSV"""
    csv_path = os.path.join(save_path, 'individual_metrics.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'psnr', 'ssim', 'mae', 'rmse', 'time'])
        writer.writeheader()
        writer.writerows(metrics_data)
    
    return csv_path


def save_test_summary_table(metrics_data, save_path, model_name):
    """Save summary statistics as a formatted table"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    rmse_values = [m['rmse'] for m in metrics_data]
    time_values = [m['time'] for m in metrics_data]
    
    table_path = os.path.join(save_path, 'summary_table.txt')
    
    with open(table_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{model_name} FINE-TUNED - SpineWeb Test Set\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(metrics_data)}\n\n")
        
        f.write("+" + "-"*68 + "+\n")
        f.write("| {:^20} | {:^12} | {:^12} | {:^12} |\n".format("Metric", "Mean", "Std", "Range"))
        f.write("+" + "-"*68 + "+\n")
        f.write("| {:^20} | {:>12.4f} | {:>12.4f} | {:>5.2f} - {:>5.2f} |\n".format(
            "PSNR (dB)", np.mean(psnr_values), np.std(psnr_values), 
            np.min(psnr_values), np.max(psnr_values)))
        f.write("| {:^20} | {:>12.6f} | {:>12.6f} | {:>5.4f} - {:>5.4f} |\n".format(
            "SSIM", np.mean(ssim_values), np.std(ssim_values),
            np.min(ssim_values), np.max(ssim_values)))
        f.write("| {:^20} | {:>12.6f} | {:>12.6f} | {:>5.4f} - {:>5.4f} |\n".format(
            "MAE", np.mean(mae_values), np.std(mae_values),
            np.min(mae_values), np.max(mae_values)))
        f.write("| {:^20} | {:>12.6f} | {:>12.6f} | {:>5.4f} - {:>5.4f} |\n".format(
            "RMSE", np.mean(rmse_values), np.std(rmse_values),
            np.min(rmse_values), np.max(rmse_values)))
        f.write("+" + "-"*68 + "+\n\n")
        
        f.write("Processing Time:\n")
        f.write(f"  Total: {sum(time_values)/60:.2f} minutes\n")
        f.write(f"  Average per sample: {np.mean(time_values):.4f} seconds\n")
    
    return table_path


def save_test_metrics_json(metrics_data, save_path, total_time, model_name):
    """Save comprehensive metrics report in JSON format"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    rmse_values = [m['rmse'] for m in metrics_data]
    
    report = {
        'metadata': {
            'model': model_name,
            'model_type': 'fine-tuned',
            'dataset': 'SpineWeb_test',
            'evaluation_date': datetime.now().isoformat(),
            'total_samples': len(metrics_data),
            'total_time_seconds': float(total_time),
            'avg_time_per_sample': float(total_time / len(metrics_data)) if metrics_data else 0
        },
        'aggregate_metrics': {
            'psnr': {
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values)),
                'median': float(np.median(psnr_values))
            },
            'ssim': {
                'mean': float(np.mean(ssim_values)),
                'std': float(np.std(ssim_values)),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values)),
                'median': float(np.median(ssim_values))
            },
            'mae': {
                'mean': float(np.mean(mae_values)),
                'std': float(np.std(mae_values)),
                'min': float(np.min(mae_values)),
                'max': float(np.max(mae_values))
            },
            'rmse': {
                'mean': float(np.mean(rmse_values)),
                'std': float(np.std(rmse_values)),
                'min': float(np.min(rmse_values)),
                'max': float(np.max(rmse_values))
            }
        },
        'individual_results': metrics_data
    }
    
    json_path = os.path.join(save_path, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return json_path


# ═══════════════════════════════════════════════════════════════
# FULL TEST/INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def run_full_test_image_domain(model, test_loader, save_dir, model_name, logger, 
                                data_range=255.0, num_save=25):
    """Run full inference on test set for image-domain models and save results"""
    model.eval()
    
    # Create output directories
    test_dir = os.path.join(save_dir, 'test_results')
    comparisons_dir = os.path.join(test_dir, 'comparisons')
    differences_dir = os.path.join(test_dir, 'differences')
    metrics_dir = os.path.join(test_dir, 'metrics')
    visualizations_dir = os.path.join(test_dir, 'visualizations')
    
    for d in [test_dir, comparisons_dir, differences_dir, metrics_dir, visualizations_dir]:
        os.makedirs(d, exist_ok=True)
    
    logger.info(f"\n  Running full test inference on SpineWeb test set...")
    
    metrics_data = []
    total_time = 0
    count = 0
    
    pbar = tqdm(test_loader, desc=f"Testing {model_name}")
    
    with torch.no_grad():
        for batch in pbar:
            Xma = batch['Xma'].to(device)
            Xgt = batch['Xgt'].to(device)
            XLI = batch['XLI'].to(device)
            M = batch['M'].to(device)
            
            # Inference with timing - input is (B, 1, H, W)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            _, ListX, _ = model(Xma, XLI, M)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            dur_time = end_time - start_time
            total_time += dur_time
            
            Xout = ListX[-1]
            
            # Apply proper post-processing (matches benchmark_comparison.py)
            # Models output in [0, 127.5] range, need clamp(0, 0.5)/0.5 to get [0, 1]
            out_norm = postprocess_model_output(Xout, data_range, 'image_domain')
            
            # Normalize inputs for visualization (simple divide by 255, clamp to [0,1])
            if data_range == 255.0:
                ma_norm = (Xma / 255.0).clamp(0, 1)
                gt_norm = (Xgt / 255.0).clamp(0, 1)
            else:
                ma_norm = Xma.clamp(0, 1)
                gt_norm = Xgt.clamp(0, 1)
            
            # Calculate metrics (all in [0, 1] range now)
            gt_np = gt_norm[0].cpu().numpy().squeeze()
            out_np = out_norm[0].cpu().numpy().squeeze()
            ma_np = ma_norm[0].cpu().numpy().squeeze()
            
            psnr_val = psnr(gt_np, out_np, data_range=1.0)
            ssim_val = ssim(gt_np, out_np, data_range=1.0)
            mae_val = np.mean(np.abs(gt_np - out_np))
            rmse_val = np.sqrt(np.mean((gt_np - out_np) ** 2))
            
            sample_metrics = {
                'sample_id': count + 1,
                'psnr': float(psnr_val),
                'ssim': float(ssim_val),
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'time': float(dur_time)
            }
            metrics_data.append(sample_metrics)
            
            # Save comparison and difference images for first num_save samples
            if count < num_save:
                save_comparison_image(count + 1, ma_np, gt_np, out_np, 
                                     comparisons_dir, model_name, sample_metrics)
                save_difference_map(count + 1, ma_np, gt_np, out_np, differences_dir, model_name)
            
            count += 1
            pbar.set_postfix({'PSNR': f'{psnr_val:.2f}', 'SSIM': f'{ssim_val:.4f}'})
    
    # Save all metrics and visualizations
    logger.info(f"  Saving test results...")
    
    save_test_metrics_csv(metrics_data, metrics_dir)
    save_test_summary_table(metrics_data, metrics_dir, model_name)
    save_test_metrics_json(metrics_data, metrics_dir, total_time, model_name)
    
    create_test_metrics_histogram(metrics_data, visualizations_dir, model_name)
    create_test_metrics_boxplot(metrics_data, visualizations_dir, model_name)
    create_test_summary_figure(metrics_data, visualizations_dir, total_time, model_name)
    
    # Log summary
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    
    logger.info(f"  Test Results ({count} samples):")
    logger.info(f"    PSNR: {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f} dB")
    logger.info(f"    SSIM: {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}")
    logger.info(f"    Total time: {total_time/60:.2f} minutes")
    logger.info(f"    Results saved to: {test_dir}")
    
    return np.mean(psnr_values), np.mean(ssim_values), metrics_data


def run_full_test_dual_domain(model, test_loader, save_dir, model_name, logger,
                               ray_trafo, data_range=255.0, model_type='dual_domain',
                               benchmark_dir=None, num_save=25):
    """Run full inference on test set for dual-domain models and save results"""
    model.eval()
    
    # Create output directories
    test_dir = os.path.join(save_dir, 'test_results')
    comparisons_dir = os.path.join(test_dir, 'comparisons')
    differences_dir = os.path.join(test_dir, 'differences')
    metrics_dir = os.path.join(test_dir, 'metrics')
    visualizations_dir = os.path.join(test_dir, 'visualizations')
    
    for d in [test_dir, comparisons_dir, differences_dir, metrics_dir, visualizations_dir]:
        os.makedirs(d, exist_ok=True)
    
    logger.info(f"\n  Running full test inference on SpineWeb test set...")
    
    metrics_data = []
    total_time = 0
    count = 0
    
    pbar = tqdm(test_loader, desc=f"Testing {model_name}")
    
    with torch.no_grad():
        for batch in pbar:
            Xma = batch['Xma'].to(device)
            Xgt = batch['Xgt'].to(device)
            XLI = batch['XLI'].to(device)
            M = batch['M'].to(device)
            metal_mask = batch['mask'].to(device)
            
            # Generate sinograms - input is (B, 1, H, W)
            Sma = forward_project_batch(Xma, ray_trafo)
            SLI = forward_project_batch(XLI, ray_trafo)
            Tr_metal = forward_project_batch(metal_mask, ray_trafo)
            Tr = (Tr_metal < 0.1).float()
            
            sino_max = 4.0 * 255.0
            Sma = (Sma / sino_max) * 255.0
            SLI = (SLI / sino_max) * 255.0
            
            # Inference with timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            if model_type == 'dual_domain':
                # InDuDoNet: (Xma, XLI, M, Sma, SLI, Tr)
                ListX, _, _ = model(Xma, XLI, M, Sma, SLI, Tr)
            elif model_type == 'dual_domain_nmar':
                # InDuDoNet+: (Xma, XLI, Sma, SLI, Tr, Xprior) - NO M parameter!
                Xprior = compute_nmar_prior(XLI, M, benchmark_dir)
                ListX, _, _ = model(Xma, XLI, Sma, SLI, Tr, Xprior)
            elif model_type == 'dual_domain_sparse':
                # MEPNet: (Xma, XLI, M, Sma, SLI, Tr)
                ListX, _ = model(Xma, XLI, M, Sma, SLI, Tr)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            dur_time = end_time - start_time
            total_time += dur_time
            
            Xout = ListX[-1]
            
            # Apply proper post-processing (matches benchmark_comparison.py)
            # Models output in [0, 127.5] range, need clamp(0, 0.5)/0.5 to get [0, 1]
            out_norm = postprocess_model_output(Xout, data_range, model_type)
            
            # Normalize inputs for visualization (simple divide by 255, clamp to [0,1])
            if data_range == 255.0:
                ma_norm = (Xma / 255.0).clamp(0, 1)
                gt_norm = (Xgt / 255.0).clamp(0, 1)
            else:
                ma_norm = Xma.clamp(0, 1)
                gt_norm = Xgt.clamp(0, 1)
            
            # Calculate metrics (all in [0, 1] range now)
            gt_np = gt_norm[0].cpu().numpy().squeeze()
            out_np = out_norm[0].cpu().numpy().squeeze()
            ma_np = ma_norm[0].cpu().numpy().squeeze()
            
            psnr_val = psnr(gt_np, out_np, data_range=1.0)
            ssim_val = ssim(gt_np, out_np, data_range=1.0)
            mae_val = np.mean(np.abs(gt_np - out_np))
            rmse_val = np.sqrt(np.mean((gt_np - out_np) ** 2))
            
            sample_metrics = {
                'sample_id': count + 1,
                'psnr': float(psnr_val),
                'ssim': float(ssim_val),
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'time': float(dur_time)
            }
            metrics_data.append(sample_metrics)
            
            # Save comparison and difference images for first num_save samples
            if count < num_save:
                save_comparison_image(count + 1, ma_np, gt_np, out_np, 
                                     comparisons_dir, model_name, sample_metrics)
                save_difference_map(count + 1, ma_np, gt_np, out_np, differences_dir, model_name)
            
            count += 1
            pbar.set_postfix({'PSNR': f'{psnr_val:.2f}', 'SSIM': f'{ssim_val:.4f}'})
            
            # Memory clearing after each batch (critical for MEPNet)
            del ListX, Xout, Sma, SLI, Tr, Tr_metal
            torch.cuda.empty_cache()
    
    # Save all metrics and visualizations
    logger.info(f"  Saving test results...")
    
    save_test_metrics_csv(metrics_data, metrics_dir)
    save_test_summary_table(metrics_data, metrics_dir, model_name)
    save_test_metrics_json(metrics_data, metrics_dir, total_time, model_name)
    
    create_test_metrics_histogram(metrics_data, visualizations_dir, model_name)
    create_test_metrics_boxplot(metrics_data, visualizations_dir, model_name)
    create_test_summary_figure(metrics_data, visualizations_dir, total_time, model_name)
    
    # Log summary
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    
    logger.info(f"  Test Results ({count} samples):")
    logger.info(f"    PSNR: {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f} dB")
    logger.info(f"    SSIM: {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}")
    logger.info(f"    Total time: {total_time/60:.2f} minutes")
    logger.info(f"    Results saved to: {test_dir}")
    
    return np.mean(psnr_values), np.mean(ssim_values), metrics_data


# ═══════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════

def clone_state_dict(state_dict):
    """Clone all tensors in state dict to avoid memory layout issues"""
    return {k: v.clone() for k, v in state_dict.items()}


def safe_load_state_dict(model, state_dict):
    """Safely load state dict by handling expand() memory issues in model parameters"""
    # First, clone all parameters in the model that might have shared memory
    # This handles the expand() issue where multiple elements refer to same memory
    model_state = model.state_dict()
    
    for name, param in model.named_parameters():
        if name in state_dict:
            # Get the source tensor (from checkpoint)
            src = state_dict[name]
            # Clone it and set directly to the parameter's data
            with torch.no_grad():
                param.data = src.clone()
    
    # Also handle buffers
    for name, buf in model.named_buffers():
        if name in state_dict:
            src = state_dict[name]
            buf.data = src.clone()


def load_dicdnet(benchmark_dir):
    """Load DICDNet model"""
    sys.path.insert(0, benchmark_dir)
    _cwd = os.getcwd()
    os.chdir(benchmark_dir)
    from dicdnet import DICDNet
    os.chdir(_cwd)
    
    class Args:
        num_M = 32
        num_Q = 32
        T = 3
        S = 10
        etaM = 1
        etaX = 5
    
    model = DICDNet(Args())
    checkpoint_path = os.path.join(benchmark_dir, "pretrain_model/DICDNet_latest.pt")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    safe_load_state_dict(model, state_dict)
    
    return model, 'image_domain', 255.0  # model, type, data_range


def load_findnet(benchmark_dir):
    """Load FIND-Net model"""
    sys.path.insert(0, benchmark_dir)
    _cwd = os.getcwd()
    os.chdir(benchmark_dir)
    from Model.findnet import FINDNet
    os.chdir(_cwd)
    
    class Args:
        num_M = 32
        num_Q = 32
        T = 3
        S = 10
        etaM = 1
        etaX = 5
    
    model = FINDNet(Args())
    checkpoint_path = os.path.join(benchmark_dir, "pretrained_models/FINDNet/checkpoint.pt")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    safe_load_state_dict(model, state_dict)
    
    return model, 'image_domain', 255.0


def load_indudonet(benchmark_dir):
    """Load InDuDoNet model"""
    sys.path.insert(0, benchmark_dir)
    _cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from odl.util.npy_compat import AVOID_UNNECESSARY_COPY
    globals()['AVOID_UNNECESSARY_COPY'] = AVOID_UNNECESSARY_COPY
    
    from network.indudonet import InDuDoNet
    os.chdir(_cwd)
    
    class Args:
        num_channel = 32
        T = 4
        S = 10
        eta1 = 1
        eta2 = 5
        alpha = 0.5
    
    model = InDuDoNet(Args())
    checkpoint_path = os.path.join(benchmark_dir, "pretrained_model/InDuDoNet_latest.pt")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    safe_load_state_dict(model, state_dict)
    
    return model, 'dual_domain', 255.0


def load_indudonet_plus(benchmark_dir):
    """Load InDuDoNet+ model"""
    sys.path.insert(0, benchmark_dir)
    _cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from odl.util.npy_compat import AVOID_UNNECESSARY_COPY
    globals()['AVOID_UNNECESSARY_COPY'] = AVOID_UNNECESSARY_COPY
    
    from network.indudonet_plus import InDuDoNet_plus
    os.chdir(_cwd)
    
    class Args:
        num_channel = 32
        T = 4
        S = 10
        eta1 = 1
        eta2 = 5
        alpha = 0.5
    
    model = InDuDoNet_plus(Args())
    checkpoint_path = os.path.join(benchmark_dir, "pretrained_model/InDuDoNet+_latest.pt")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    safe_load_state_dict(model, state_dict)
    
    return model, 'dual_domain_nmar', 255.0


def load_mepnet(benchmark_dir):
    """Load MEPNet model"""
    sys.path.insert(0, benchmark_dir)
    _cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from odl.util.npy_compat import AVOID_UNNECESSARY_COPY
    globals()['AVOID_UNNECESSARY_COPY'] = AVOID_UNNECESSARY_COPY
    
    from network.mepnet import MEPNet
    os.chdir(_cwd)
    
    class Args:
        num_channel = 32
        T = 4
        S = 10
        eta1 = 1
        eta2 = 5
        alpha = 0.5
        test_proj = 320
    
    model = MEPNet(Args())
    checkpoint_path = os.path.join(benchmark_dir, "pretrained_model/V320/MEPNet_latest.pt")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    safe_load_state_dict(model, state_dict)
    
    return model, 'dual_domain_sparse', 255.0


# ═══════════════════════════════════════════════════════════════
# ODL SETUP FOR DUAL-DOMAIN MODELS
# ═══════════════════════════════════════════════════════════════

def setup_odl_geometry():
    """Setup ODL ray transform matching SynDeepLesion geometry (416x416, 640 views, 641 detectors)
    
    This MUST match the geometry used by InDuDoNet/InDuDoNet+/MEPNet which have hardcoded
    ODL operators inside them expecting this exact geometry.
    """
    import odl
    
    # SynDeepLesion geometry parameters (from benchmarks/InDuDoNet/network/build_gemotry.py)
    img_size = 416
    num_angles = 640
    num_detectors = 641
    reso = 512 / 416 * 0.03  # Resolution scale
    
    # Image space
    sx = img_size * reso
    sy = img_size * reso
    
    reco_space = odl.uniform_discr(
        min_pt=[-sx / 2.0, -sy / 2.0],
        max_pt=[sx / 2.0, sy / 2.0],
        shape=[img_size, img_size],
        dtype='float32')
    
    # Angle partition (0 to 2*pi for fan beam)
    angle_partition = odl.uniform_partition(0, 2 * np.pi, num_angles)
    
    # Detector partition
    su = 2 * np.sqrt(sx**2 + sy**2)  # Detector width
    detector_partition = odl.uniform_partition(-su / 2.0, su / 2.0, num_detectors)
    
    # Fan beam geometry (matching SynDeepLesion)
    dso = 1075 * reso  # Source to origin distance
    dde = 1075 * reso  # Origin to detector distance
    
    geometry = odl.tomo.FanBeamGeometry(
        angle_partition, detector_partition,
        src_radius=dso, det_radius=dde)
    
    # Create ray transform (using astra_cuda if available)
    try:
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
    except:
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')
    
    # Create FBP operator
    fbp_op = odl.tomo.fbp_op(ray_trafo)
    
    return ray_trafo, fbp_op


def forward_project_batch(images, ray_trafo):
    """Forward project a batch of images to sinogram domain"""
    batch_size = images.shape[0]
    sinograms = []
    
    for i in range(batch_size):
        img_np = images[i, 0].detach().cpu().numpy()
        sino = np.asarray(ray_trafo(img_np))
        sinograms.append(sino)
    
    sinograms = np.stack(sinograms, axis=0)
    sinograms = torch.from_numpy(sinograms).unsqueeze(1).to(images.device)
    return sinograms


# ═══════════════════════════════════════════════════════════════
# NMAR PRIOR COMPUTATION (for InDuDoNet+)
# ═══════════════════════════════════════════════════════════════

def compute_nmar_prior(XLI, M, benchmark_dir):
    """Compute NMAR prior image for InDuDoNet+"""
    import scipy.ndimage
    import scipy.io as sio
    from sklearn.cluster import k_means
    
    # Load Gaussian filter
    filter_path = os.path.join(benchmark_dir, 'deeplesion/gaussianfilter.mat')
    smFilter = sio.loadmat(filter_path)['smFilter']
    
    miuAir = 0
    miuWater = 0.192 * 255  # Scaled for [0, 255] range
    
    def nmarprior(im, threshWater, threshBone, miuAir, miuWater, smFilter):
        imSm = scipy.ndimage.filters.convolve(im, smFilter, mode='nearest')
        priorimgHU = imSm.copy()
        priorimgHU[imSm <= threshWater] = miuAir
        h, w = imSm.shape
        priorimgHUvector = priorimgHU.flatten()
        region1_1d = np.where(priorimgHUvector > threshWater)
        region2_1d = np.where(priorimgHUvector < threshBone)
        region_1d = np.intersect1d(region1_1d, region2_1d)
        priorimgHUvector[region_1d] = miuWater
        return priorimgHU.reshape(h, w)
    
    batch_size = XLI.shape[0]
    priors = []
    
    for i in range(batch_size):
        xli = XLI[i, 0].detach().cpu().numpy()
        m = M[i, 0].detach().cpu().numpy()
        
        xli_copy = xli.copy()
        xli_copy[m < 0.5] = miuWater  # Fill metal regions
        
        h, w = xli_copy.shape
        im1d = xli_copy.flatten().reshape(-1, 1)
        
        starpoint = np.array([[miuAir], [miuWater], [2 * miuWater]])
        try:
            best_centers, labels, _ = k_means(im1d, n_clusters=3, init=starpoint, n_init=1, max_iter=300)
            threshBone = np.max([np.min(im1d[labels == 2]), 1.2 * miuWater])
            threshWater = np.min(im1d[labels == 1])
        except:
            threshBone = 1.5 * miuWater
            threshWater = 0.5 * miuWater
        
        prior = nmarprior(xli_copy, threshWater, threshBone, miuAir, miuWater, smFilter)
        priors.append(prior)
    
    priors = np.stack(priors, axis=0)
    priors = torch.from_numpy(priors).unsqueeze(1).float().to(XLI.device)
    return priors


# ═══════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS FOR EACH MODEL TYPE
# ═══════════════════════════════════════════════════════════════

def train_image_domain_model(model, train_loader, val_loader, optimizer, 
                              num_epochs, save_dir, model_name, logger, data_range=255.0):
    """Train image-domain models (DICDNet, FIND-Net)"""
    model.train()
    train_losses = []
    val_psnrs = []
    val_ssims = []
    best_psnr = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            Xma = batch['Xma'].to(device)
            Xgt = batch['Xgt'].to(device)
            XLI = batch['XLI'].to(device)
            M = batch['M'].to(device)
            
            # Shape should be (B, 1, H, W) - models expect 4D input
            # No need to add extra dimension
            
            optimizer.zero_grad()
            
            # Forward pass
            X0, ListX, ListA = model(Xma, XLI, M)
            Xout = ListX[-1]
            
            # Compute losses
            loss_rec = reconstruction_loss(Xout, Xgt)
            loss_edge = edge_loss(Xout, Xgt)
            
            loss = opt.lambda_rec * loss_rec + opt.lambda_edge * loss_edge
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        if (epoch + 1) % opt.val_freq == 0:
            val_psnr, val_ssim = validate_image_domain(model, val_loader, data_range)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)
            
            logger.info(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pt'))
                logger.info(f"  → New best model saved! PSNR={val_psnr:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % opt.save_freq == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), 
                      os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pt'))
            
            # Save sample images
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                Xma = sample_batch['Xma'].to(device)
                XLI = sample_batch['XLI'].to(device)
                M = sample_batch['M'].to(device)
                _, ListX, _ = model(Xma, XLI, M)
                save_sample_images(epoch+1, sample_batch, ListX[-1], 
                                 save_dir, model_name, data_range)
    
    # Save training curves
    save_training_curves(train_losses, val_psnrs, val_ssims, save_dir, model_name)
    
    return best_psnr, val_ssims[-1] if val_ssims else 0.0


def validate_image_domain(model, val_loader, data_range=255.0):
    """Validate image-domain model"""
    model.eval()
    psnr_list, ssim_list = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            Xma = batch['Xma'].to(device)
            Xgt = batch['Xgt'].to(device)
            XLI = batch['XLI'].to(device)
            M = batch['M'].to(device)
            
            _, ListX, _ = model(Xma, XLI, M)
            Xout = ListX[-1]
            
            # Apply proper post-processing for output
            out_norm = postprocess_model_output(Xout, data_range, 'image_domain')
            # Normalize GT for comparison
            if data_range == 255.0:
                gt_norm = (Xgt / 255.0).clamp(0, 1)
            else:
                gt_norm = Xgt.clamp(0, 1)
            
            for i in range(gt_norm.shape[0]):
                gt_np = gt_norm[i].cpu().numpy().squeeze()
                out_np = out_norm[i].cpu().numpy().squeeze()
                p = psnr(gt_np, out_np, data_range=1.0)
                s = ssim(gt_np, out_np, data_range=1.0)
                psnr_list.append(p)
                ssim_list.append(s)
    
    return np.mean(psnr_list), np.mean(ssim_list)


def train_dual_domain_model(model, train_loader, val_loader, optimizer,
                            num_epochs, save_dir, model_name, logger, 
                            ray_trafo, data_range=255.0, model_type='dual_domain'):
    """Train dual-domain models (InDuDoNet, InDuDoNet+, MEPNet) with AMP and gradient checkpointing"""
    model.train()
    train_losses = []
    val_psnrs = []
    val_ssims = []
    best_psnr = 0.0
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler() if opt.use_amp else None
    
    # Enable gradient checkpointing if requested
    if opt.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info(f"  Gradient checkpointing enabled for {model_name}")
    
    # Get benchmark dir for NMAR prior if needed
    if model_type == 'dual_domain_nmar':
        benchmark_dir = os.path.join(BENCHMARKS_ROOT, 'InDuDoNet_plus')
    
    logger.info(f"  Using mixed precision (FP16): {opt.use_amp}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            Xma = batch['Xma'].to(device)
            Xgt = batch['Xgt'].to(device)
            XLI = batch['XLI'].to(device)
            M = batch['M'].to(device)
            metal_mask = batch['mask'].to(device)
            
            # Input shape is (B, 1, H, W) - models expect this
            
            optimizer.zero_grad()
            
            # For MEPNet, use memory-optimized sinogram generation
            # Delete intermediate tensors as soon as possible
            if model_type == 'dual_domain_sparse':
                # MEPNet memory optimization: compute sinograms one at a time
                # Convert inputs to FP16 to save memory
                Xma = Xma.half()
                Xgt_fp32 = Xgt.clone()  # Keep FP32 copy for loss
                Xgt = Xgt.half()
                XLI = XLI.half()
                M = M.half()
                
                with torch.cuda.amp.autocast(enabled=True):
                    Sma = forward_project_batch(Xma.float(), ray_trafo).half()
                    sino_max = 4.0 * 255.0
                    Sma = (Sma / sino_max) * 255.0
                    
                    SLI = forward_project_batch(XLI.float(), ray_trafo).half()
                    SLI = (SLI / sino_max) * 255.0
                    
                    Tr_metal = forward_project_batch(metal_mask.float(), ray_trafo)
                    Tr = (Tr_metal < 0.1).half()
                    del Tr_metal
                    torch.cuda.empty_cache()
                    
                    # Forward pass - model will use FP16 via autocast
                    ListX, ListYS = model(Xma.float(), XLI.float(), M.float(), Sma.float(), SLI.float(), Tr.float())
                    Xout = ListX[-1]
                    
                    # Compute image reconstruction loss
                    loss_rec = reconstruction_loss(Xout, Xgt_fp32)
                    loss_edge = edge_loss(Xout, Xgt_fp32)
                    loss = opt.lambda_rec * loss_rec + opt.lambda_edge * loss_edge
                
                # Clean up before backward
                del ListX, ListYS, Sma, SLI, Tr, Xgt
                torch.cuda.empty_cache()
            else:
                # Standard processing for other models
                # Generate sinograms via forward projection (keep in FP32 for ODL)
                Sma = forward_project_batch(Xma, ray_trafo)
                Sgt = forward_project_batch(Xgt, ray_trafo)
                SLI = forward_project_batch(XLI, ray_trafo)
                
                # Create trace mask (metal regions in sinogram)
                Tr_metal = forward_project_batch(metal_mask, ray_trafo)
                Tr = (Tr_metal < 0.1).float()  # Non-metal regions = 1
                
                # Normalize sinograms
                sino_max = 4.0 * 255.0  # Approximate max for [0, 255] image range
                Sma = (Sma / sino_max) * 255.0
                Sgt = (Sgt / sino_max) * 255.0
                SLI = (SLI / sino_max) * 255.0
                
                # Forward pass with mixed precision
                with autocast(enabled=opt.use_amp):
                    # Forward pass based on model type
                    if model_type == 'dual_domain':
                        # InDuDoNet: (Xma, XLI, M, Sma, SLI, Tr)
                        ListX, ListS, ListYS = model(Xma, XLI, M, Sma, SLI, Tr)
                    elif model_type == 'dual_domain_nmar':
                        # InDuDoNet+: (Xma, XLI, Sma, SLI, Tr, Xprior) - NO M parameter!
                        Xprior = compute_nmar_prior(XLI, M, benchmark_dir)
                        ListX, ListS, ListYS = model(Xma, XLI, Sma, SLI, Tr, Xprior)
                    
                    Xout = ListX[-1]
                    Sout = ListYS[-1] if 'ListYS' in dir() else None
                    
                    # Compute losses
                    loss_rec = reconstruction_loss(Xout, Xgt)
                    loss_edge = edge_loss(Xout, Xgt)
                    
                    loss = opt.lambda_rec * loss_rec + opt.lambda_edge * loss_edge
                    
                    # Add sinogram loss if available
                    if Sout is not None:
                        loss_sino = sinogram_loss(Sout, Sgt, Tr)
                        loss = loss + opt.lambda_sino * loss_sino
            
            # Backward pass with gradient scaling for AMP
            if opt.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Memory clearing after each batch
            # For MEPNet, most cleanup already done above
            if model_type != 'dual_domain_sparse':
                del ListX, ListYS, Xout, Sout, loss, loss_rec, loss_edge
                del Sma, Sgt, SLI, Tr, Tr_metal
                if model_type == 'dual_domain':
                    del ListS
                elif model_type == 'dual_domain_nmar':
                    del ListS, Xprior
            else:
                # MEPNet - only clean what's left
                del Xout, loss, loss_rec, loss_edge
            
            # Clear batch tensors
            del Xma, Xgt, XLI, M, metal_mask
            torch.cuda.empty_cache()
            gc.collect()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation (image domain only for simplicity)
        if (epoch + 1) % opt.val_freq == 0:
            val_psnr, val_ssim = validate_dual_domain(
                model, val_loader, ray_trafo, data_range, model_type,
                benchmark_dir if model_type == 'dual_domain_nmar' else None)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)
            
            logger.info(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pt'))
                logger.info(f"  → New best model saved! PSNR={val_psnr:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % opt.save_freq == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), 
                      os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pt'))
    
    # Save training curves
    save_training_curves(train_losses, val_psnrs, val_ssims, save_dir, model_name)
    
    return best_psnr, val_ssims[-1] if val_ssims else 0.0


def validate_dual_domain(model, val_loader, ray_trafo, data_range=255.0, 
                         model_type='dual_domain', benchmark_dir=None):
    """Validate dual-domain model"""
    model.eval()
    psnr_list, ssim_list = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            Xma = batch['Xma'].to(device)
            Xgt = batch['Xgt'].to(device)
            XLI = batch['XLI'].to(device)
            M = batch['M'].to(device)
            metal_mask = batch['mask'].to(device)
            
            # Generate sinograms - input is (B, 1, H, W)
            Sma = forward_project_batch(Xma, ray_trafo)
            SLI = forward_project_batch(XLI, ray_trafo)
            Tr_metal = forward_project_batch(metal_mask, ray_trafo)
            Tr = (Tr_metal < 0.1).float()
            
            sino_max = 4.0 * 255.0
            Sma = (Sma / sino_max) * 255.0
            SLI = (SLI / sino_max) * 255.0
            
            # Forward pass
            if model_type == 'dual_domain':
                # InDuDoNet: (Xma, XLI, M, Sma, SLI, Tr)
                ListX, _, _ = model(Xma, XLI, M, Sma, SLI, Tr)
            elif model_type == 'dual_domain_nmar':
                # InDuDoNet+: (Xma, XLI, Sma, SLI, Tr, Xprior) - NO M parameter!
                Xprior = compute_nmar_prior(XLI, M, benchmark_dir)
                ListX, _, _ = model(Xma, XLI, Sma, SLI, Tr, Xprior)
            elif model_type == 'dual_domain_sparse':
                # MEPNet: (Xma, XLI, M, Sma, SLI, Tr)
                ListX, _ = model(Xma, XLI, M, Sma, SLI, Tr)
            
            Xout = ListX[-1]
            
            # Apply proper post-processing for output
            out_norm = postprocess_model_output(Xout, data_range, model_type)
            # Normalize GT for comparison
            if data_range == 255.0:
                gt_norm = (Xgt / 255.0).clamp(0, 1)
            else:
                gt_norm = Xgt.clamp(0, 1)
            
            for i in range(gt_norm.shape[0]):
                gt_np = gt_norm[i].cpu().numpy().squeeze()
                out_np = out_norm[i].cpu().numpy().squeeze()
                p = psnr(gt_np, out_np, data_range=1.0)
                s = ssim(gt_np, out_np, data_range=1.0)
                psnr_list.append(p)
                ssim_list.append(s)
            
            # Memory clearing after each batch (important for MEPNet)
            del ListX, Xout, Sma, SLI, Tr, Tr_metal
            torch.cuda.empty_cache()
    
    return np.mean(psnr_list), np.mean(ssim_list)


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def main():
    # Setup directories - create both the parent and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(opt.save_path, exist_ok=True)  # Create parent directory first
    run_dir = os.path.join(opt.save_path, f'finetune_all_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(run_dir, 'finetune.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("BENCHMARK MODELS FINE-TUNING ON SPINEWEB")
    logger.info("=" * 70)
    logger.info(f"Models to fine-tune: {opt.models}")
    logger.info(f"Epochs per model: {opt.epochs}")
    logger.info(f"Learning rate: {opt.lr}")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Save path: {run_dir}")
    logger.info("=" * 70)
    
    # Model configurations
    MODEL_CONFIGS = {
        'DICDNet': {
            'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'DICDNet'),
            'loader': load_dicdnet,
            'normalize_range': '0_255',
        },
        'FIND-Net': {
            'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'FIND-Net'),
            'loader': load_findnet,
            'normalize_range': '0_255',
        },
        'InDuDoNet': {
            'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet'),
            'loader': load_indudonet,
            'normalize_range': '0_255',
        },
        'InDuDoNet_plus': {
            'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet_plus'),
            'loader': load_indudonet_plus,
            'normalize_range': '0_255',
        },
        'MEPNet': {
            'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'MEPNet'),
            'loader': load_mepnet,
            'normalize_range': '0_255',
        },
    }
    
    # Results summary
    results_summary = {}
    
    # Fine-tune each model
    for model_name in opt.models:
        if model_name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_name}, skipping...")
            continue
        
        logger.info("\n" + "═" * 70)
        logger.info(f"FINE-TUNING: {model_name}")
        logger.info("═" * 70)
        
        # AGGRESSIVE MEMORY CLEARING before loading new model
        # This is critical for MEPNet which has the highest memory footprint
        logger.info("  Clearing GPU memory...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # For MEPNet, do extra aggressive cleanup
        if model_name == 'MEPNet':
            for _ in range(3):  # Multiple passes of gc
                gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Log current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU Memory before MEPNet: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        
        # Clear network module from cache before loading new model
        # This prevents conflicts between different benchmark implementations
        if 'network' in sys.modules:
            del sys.modules['network']
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('network'):
                del sys.modules[mod_name]
        
        config = MODEL_CONFIGS[model_name]
        model_save_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        try:
            # Load model
            logger.info(f"Loading pretrained {model_name}...")
            model, model_type, data_range = config['loader'](config['benchmark_dir'])
            model = model.to(device)
            logger.info(f"  Model type: {model_type}")
            logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # For MEPNet: freeze early stages to reduce memory usage
            if model_name == 'MEPNet' and opt.mepnet_freeze_stages > 0:
                freeze_count = opt.mepnet_freeze_stages
                logger.info(f"  Freezing first {freeze_count} stages of MEPNet to reduce memory...")
                
                # Freeze priornet
                for param in model.priornet.parameters():
                    param.requires_grad = False
                
                # Freeze early proxNet stages
                for i in range(min(freeze_count, len(model.proxNet_u_S))):
                    for param in model.proxNet_u_S[i].parameters():
                        param.requires_grad = False
                for i in range(min(freeze_count, len(model.proxNet_f_S))):
                    for param in model.proxNet_f_S[i].parameters():
                        param.requires_grad = False
                
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
            
            # Create datasets - different settings for image-domain vs dual-domain models
            logger.info("Loading SpineWeb dataset...")
            
            # Dual-domain models (InDuDoNet, InDuDoNet+, MEPNet) require fixed 416x416 images
            # Image-domain models (DICDNet, FIND-Net) can use flexible patch sizes
            if model_type == 'image_domain':
                # Use random patches for image-domain models
                img_size = opt.patch_size  # Default 64x64
                resize_mode = 'patch'
            else:
                # Use full image resize for dual-domain models (must be 416x416)
                img_size = 416
                resize_mode = 'resize'
            
            logger.info(f"  Mode: {resize_mode}, Size: {img_size}x{img_size}")
            
            # Use reduced samples for MEPNet (memory constrained)
            if model_name == 'MEPNet':
                train_samples = opt.mepnet_train_samples
                test_samples = opt.mepnet_test_samples
                logger.info(f"  MEPNet: Using reduced samples (train={train_samples}, test={test_samples})")
            else:
                train_samples = opt.max_train_samples
                test_samples = opt.max_test_samples
            
            train_dataset = SpineWebDataset(
                opt.data_path, split='train', 
                patch_size=img_size,
                normalize_range=config['normalize_range'],
                resize_mode=resize_mode,
                max_samples=train_samples,
                seed=opt.seed  # Deterministic sample selection
            )
            val_dataset = SpineWebDataset(
                opt.data_path, split='test',
                patch_size=img_size,
                normalize_range=config['normalize_range'],
                resize_mode=resize_mode,
                max_samples=test_samples,
                seed=opt.seed  # Deterministic sample selection
            )
            
            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, 
                                     shuffle=True, num_workers=opt.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=1, 
                                   shuffle=False, num_workers=opt.num_workers)
            
            # Create optimizer
            optimizer = optim.Adam(model.parameters(), lr=opt.lr)
            
            # Train based on model type
            start_time = time.time()
            
            if model_type == 'image_domain':
                best_psnr, final_ssim = train_image_domain_model(
                    model, train_loader, val_loader, optimizer,
                    opt.epochs, model_save_dir, model_name, logger, data_range
                )
                
                # Load best model for testing
                logger.info(f"\n  Loading best checkpoint for final testing...")
                best_ckpt_path = os.path.join(model_save_dir, f'{model_name}_best.pt')
                if os.path.exists(best_ckpt_path):
                    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
                
                # Run full test inference
                test_psnr, test_ssim, test_metrics = run_full_test_image_domain(
                    model, val_loader, model_save_dir, model_name, logger,
                    data_range, num_save=opt.num_test_save
                )
            else:
                # Setup ODL for dual-domain models (uses fixed SynDeepLesion geometry)
                logger.info("Setting up ODL geometry for dual-domain training (416x416, 640 views)...")
                ray_trafo, fbp_op = setup_odl_geometry()
                
                best_psnr, final_ssim = train_dual_domain_model(
                    model, train_loader, val_loader, optimizer,
                    opt.epochs, model_save_dir, model_name, logger,
                    ray_trafo, data_range, model_type
                )
                
                # Load best model for testing
                logger.info(f"\n  Loading best checkpoint for final testing...")
                best_ckpt_path = os.path.join(model_save_dir, f'{model_name}_best.pt')
                if os.path.exists(best_ckpt_path):
                    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
                
                # Run full test inference
                test_psnr, test_ssim, test_metrics = run_full_test_dual_domain(
                    model, val_loader, model_save_dir, model_name, logger,
                    ray_trafo, data_range, model_type,
                    config['benchmark_dir'] if model_type == 'dual_domain_nmar' else None,
                    num_save=opt.num_test_save
                )
            
            elapsed_time = time.time() - start_time
            
            # Save results
            results_summary[model_name] = {
                'best_val_psnr': float(best_psnr),
                'best_val_ssim': float(final_ssim),
                'test_psnr': float(test_psnr),
                'test_ssim': float(test_ssim),
                'training_time_min': float(elapsed_time / 60),
                'checkpoint_dir': model_save_dir,
                'num_test_samples': len(test_metrics)
            }
            
            logger.info(f"\n{model_name} COMPLETE!")
            logger.info(f"  Best Val PSNR: {best_psnr:.2f} dB")
            logger.info(f"  Test PSNR: {test_psnr:.2f} dB")
            logger.info(f"  Test SSIM: {test_ssim:.4f}")
            logger.info(f"  Training time: {elapsed_time/60:.2f} minutes")
            logger.info(f"  Checkpoints saved to: {model_save_dir}")
            
        except Exception as e:
            logger.error(f"Error fine-tuning {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[model_name] = {'error': str(e)}
            continue
    
    # Save final summary
    summary_path = os.path.join(run_dir, 'finetune_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINE-TUNING COMPLETE - SUMMARY")
    logger.info("=" * 70)
    for model_name, results in results_summary.items():
        if 'error' in results:
            logger.info(f"  {model_name}: FAILED - {results['error']}")
        else:
            logger.info(f"  {model_name}:")
            logger.info(f"    Val PSNR={results['best_val_psnr']:.2f} dB, Val SSIM={results['best_val_ssim']:.4f}")
            logger.info(f"    Test PSNR={results['test_psnr']:.2f} dB, Test SSIM={results['test_ssim']:.4f}")
            logger.info(f"    Time={results['training_time_min']:.1f} min, Samples={results['num_test_samples']}")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
