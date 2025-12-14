#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
DICDNet Benchmark Test Script - SynDeepLesion Dataset
=====================================================
Tests the pretrained DICDNet model on the SynDeepLesion test set.
Calculates PSNR and SSIM metrics and saves comprehensive results.

Usage:
    python test_dicdnet_syndeeplesion.py
    python test_dicdnet_syndeeplesion.py --num_images 50 --save_images
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import h5py
import json
import csv
import logging
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Add benchmark DICDNet path to import the model
BENCHMARK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../benchmarks/DICDNet'))
sys.path.insert(0, BENCHMARK_DIR)

# Change to benchmark directory before importing (dicdnet.py uses relative paths for init_kernel.mat)
_original_cwd = os.getcwd()
os.chdir(BENCHMARK_DIR)
from dicdnet import DICDNet
os.chdir(_original_cwd)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="DICDNet Test on SynDeepLesion")
parser.add_argument("--data_path", type=str, default="/home/Drive-D/SynDeepLesion/", 
                    help='Path to SynDeepLesion dataset')
parser.add_argument("--model_path", type=str, 
                    default=os.path.join(BENCHMARK_DIR, "pretrain_model/DICDNet_latest.pt"),
                    help='Path to pretrained model')
parser.add_argument("--save_path", type=str, 
                    default="./results/syndeeplesion_test/",
                    help='Path to save results')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--num_images", type=int, default=200, help='Number of test images (max 200)')
parser.add_argument("--num_masks", type=int, default=10, help='Number of masks per image (max 10)')
parser.add_argument("--save_images", action="store_true", help='Save output images')
parser.add_argument("--save_all_images", action="store_true", help='Save ALL output images (not just samples)')
parser.add_argument("--use_selected_slices", action="store_true", 
                    help='Use the same selected slices as inference figures (25 best slices)')
parser.add_argument("--selected_slices_path", type=str, 
                    default="/home/grad/mppatel/Documents/DCGAN/inference_figure_outputs/selected_slice_indices.npy",
                    help='Path to selected slice indices file')

# DICDNet model parameters (must match pretrained model)
parser.add_argument('--num_M', type=int, default=32, help='Number of feature maps')
parser.add_argument('--num_Q', type=int, default=32, help='Number of channel concatenation')
parser.add_argument('--T', type=int, default=3, help='Number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='Stage number')
parser.add_argument('--etaM', type=float, default=1, help='Stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='Stepsize for updating X')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════
# SETUP DIRECTORIES AND LOGGING
# ═══════════════════════════════════════════════════════════════

def setup_directories(base_path):
    """Create all necessary directories"""
    # Shared benchmark figures directory (outside of individual model results)
    BENCHMARK_FIGURES_ROOT = '/home/grad/mppatel/Documents/DCGAN/benchmark_outputs/benchmark_figures_v2'
    
    dirs = {
        'root': base_path,
        'metrics': os.path.join(base_path, 'metrics'),
        'logs': os.path.join(base_path, 'logs'),
        'images': os.path.join(base_path, 'images'),
        'comparisons': os.path.join(base_path, 'comparisons'),
        'visualizations': os.path.join(base_path, 'visualizations'),
        'tables': os.path.join(base_path, 'tables'),
        # Shared benchmark output directories (absolute path so all models use same location)
        'benchmark_individual': os.path.join(BENCHMARK_FIGURES_ROOT, 'individual_images'),
        'benchmark_composite': os.path.join(BENCHMARK_FIGURES_ROOT, 'composite_figures'),
        'benchmark_numpy': os.path.join(BENCHMARK_FIGURES_ROOT, 'numpy_arrays'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def setup_logging(log_dir):
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'test_log_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('DICDNet_Test')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    """Normalize data to [0, 255] range for DICDNet input"""
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data

def tohu(X):
    """Convert to HU display window [-175HU, 275HU]"""
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-175, 275)
    CT_winnorm = (CT_win + 175) / (275 + 175)
    return CT_winnorm

def calculate_metrics(gt_tensor, pred_tensor):
    """Calculate PSNR, SSIM, and MAE metrics"""
    gt_np = gt_tensor.data.cpu().numpy().squeeze()
    pred_np = pred_tensor.data.cpu().numpy().squeeze()
    
    psnr_val = psnr(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim(gt_np, pred_np, data_range=1.0)
    mae_val = np.mean(np.abs(gt_np - pred_np))
    rmse_val = np.sqrt(np.mean((gt_np - pred_np) ** 2))
    
    return psnr_val, ssim_val, mae_val, rmse_val

def load_test_sample(data_path, test_mask, imag_idx, mask_idx):
    """Load a single test sample from SynDeepLesion"""
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    
    abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
    gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
    
    # Load ground truth
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    
    # Load metal-affected and LI images
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    XLI = file['LI_CT'][()]
    file.close()
    
    # Get mask and resize to 416x416
    M512 = test_mask[:, :, mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    
    # Normalize
    Xma = normalize(Xma, image_get_minmax())
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    non_mask = 1 - Mask
    
    return (torch.Tensor(Xma).to(device), 
            torch.Tensor(Xgt).to(device), 
            torch.Tensor(XLI).to(device), 
            torch.Tensor(non_mask).to(device))

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def save_comparison_image(idx, ma_img, gt_img, out_img, save_dir, metrics=None):
    """Save side-by-side comparison image with metrics"""
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.3])
    
    # Metal-affected
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(ma_img, cmap='gray', vmin=0, vmax=1)
    ax0.set_title('Metal-Affected CT', fontsize=12, fontweight='bold')
    ax0.axis('off')
    
    # Ground truth
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # DICDNet result
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(out_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('DICDNet Result', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Metrics text box
    ax3 = fig.add_subplot(gs[3])
    ax3.axis('off')
    if metrics:
        metrics_text = f"PSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.4f}\nMAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}"
        ax3.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_difference_map(idx, gt_img, out_img, save_dir):
    """Save difference map between GT and output"""
    diff = np.abs(gt_img - out_img)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(out_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('DICDNet Output', fontweight='bold')
    axes[1].axis('off')
    
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
    axes[2].set_title('Absolute Difference', fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'difference_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════
# BENCHMARK FIGURE SAVING FUNCTIONS (CONSISTENT GRAYSCALE)
# ═══════════════════════════════════════════════════════════════

def get_consistent_grayscale_window(gt_np):
    """
    Get consistent vmin/vmax for grayscale display based on ground truth.
    This ensures all images (corrupted, GT, model outputs) appear with same intensity.
    """
    vmin = 0.0
    vmax = 0.9  # Fixed vmax for proper CT contrast
    return vmin, vmax


def find_metal_center(img_np):
    """Find the center of metal artifacts for zooming."""
    threshold = np.percentile(img_np, 99)
    metal_mask = img_np > threshold
    if metal_mask.sum() > 0:
        coords = np.where(metal_mask)
        center_y = int(np.mean(coords[0]))
        center_x = int(np.mean(coords[1]))
    else:
        center_y, center_x = img_np.shape[0] // 2, img_np.shape[1] // 2
    return center_y, center_x


def get_zoom_region(img_shape, center_y, center_x, zoom_size=80):
    """Get zoom region coordinates."""
    half = zoom_size // 2
    h, w = img_shape[:2]
    y1, y2 = max(0, center_y - half), min(h, center_y + half)
    x1, x2 = max(0, center_x - half), min(w, center_x + half)
    return y1, y2, x1, x2


def save_benchmark_individual_image(img_np, save_path, vmin=0.0, vmax=1.0, zoom_coords=None, box_color=None):
    """Save individual image with consistent grayscale window and optional zoom region box."""
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(img_np, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    
    # Add zoom region rectangle if coordinates provided
    if zoom_coords is not None and box_color is not None:
        y1, y2, x1, x2 = zoom_coords
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
    
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_benchmark_zoomed_image(img_np, save_path, vmin, vmax, y1, y2, x1, x2, box_color='lime'):
    """Save zoomed ROI image with colored border."""
    from matplotlib.patches import Rectangle
    
    cropped = img_np[y1:y2, x1:x2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.clip(cropped, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    
    # Add border around the zoomed image
    rect = Rectangle((0, 0), cropped.shape[1]-1, cropped.shape[0]-1, 
                      linewidth=3, edgecolor=box_color, facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_benchmark_outputs(sample_idx, ma_np, gt_np, out_np, dirs, model_name='DICDNet'):
    """
    Save benchmark outputs for building composite figures later.
    
    Saves:
    1. Individual PNG images with consistent grayscale (GT-based windowing)
    2. Zoomed ROI crops with colored borders
    3. Numpy arrays for later composite figure generation
    
    Directory structure:
    benchmark_figures/
    ├── individual_images/
    │   └── sample_XXXX/
    │       ├── corrupted.png
    │       ├── corrupted_zoomed.png
    │       ├── ground_truth.png
    │       ├── ground_truth_zoomed.png
    │       ├── DICDNet.png
    │       └── DICDNet_zoomed.png
    └── numpy_arrays/
        └── sample_XXXX/
            ├── corrupted.npy
            ├── ground_truth.npy
            └── DICDNet.npy
    """
    # Get consistent grayscale window from GT
    vmin, vmax = get_consistent_grayscale_window(gt_np)
    
    # Find metal center and zoom region (use corrupted image to find metal)
    center_y, center_x = find_metal_center(ma_np)
    y1, y2, x1, x2 = get_zoom_region(ma_np.shape, center_y, center_x, zoom_size=80)
    
    # Create sample directories
    sample_img_dir = os.path.join(dirs['benchmark_individual'], f'sample_{sample_idx:04d}')
    sample_npy_dir = os.path.join(dirs['benchmark_numpy'], f'sample_{sample_idx:04d}')
    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(sample_npy_dir, exist_ok=True)
    
    # Save numpy arrays (always - needed for composite figures)
    np.save(os.path.join(sample_npy_dir, 'corrupted.npy'), ma_np)
    np.save(os.path.join(sample_npy_dir, 'ground_truth.npy'), gt_np)
    np.save(os.path.join(sample_npy_dir, f'{model_name}.npy'), out_np)
    
    # Save grayscale window info and zoom coordinates
    window_info = {
        'vmin': float(vmin), 
        'vmax': float(vmax),
        'zoom_y1': int(y1), 'zoom_y2': int(y2),
        'zoom_x1': int(x1), 'zoom_x2': int(x2)
    }
    with open(os.path.join(sample_npy_dir, 'grayscale_window.json'), 'w') as f:
        json.dump(window_info, f)
    
    # Save individual PNG images with consistent grayscale
    # Only save corrupted and GT if they don't exist (avoid overwriting from other models)
    corrupted_path = os.path.join(sample_img_dir, 'corrupted.png')
    corrupted_zoomed_path = os.path.join(sample_img_dir, 'corrupted_zoomed.png')
    gt_path = os.path.join(sample_img_dir, 'ground_truth.png')
    gt_zoomed_path = os.path.join(sample_img_dir, 'ground_truth_zoomed.png')
    
    # Zoom coordinates tuple for drawing boxes on full images
    zoom_coords = (y1, y2, x1, x2)
    
    if not os.path.exists(corrupted_path):
        save_benchmark_individual_image(ma_np, corrupted_path, vmin, vmax, zoom_coords=zoom_coords, box_color='red')
        save_benchmark_zoomed_image(ma_np, corrupted_zoomed_path, vmin, vmax, y1, y2, x1, x2, box_color='red')
    
    if not os.path.exists(gt_path):
        save_benchmark_individual_image(gt_np, gt_path, vmin, vmax, zoom_coords=zoom_coords, box_color='lime')
        save_benchmark_zoomed_image(gt_np, gt_zoomed_path, vmin, vmax, y1, y2, x1, x2, box_color='lime')
    
    # Always save model output (full and zoomed)
    model_path = os.path.join(sample_img_dir, f'{model_name}.png')
    model_zoomed_path = os.path.join(sample_img_dir, f'{model_name}_zoomed.png')
    save_benchmark_individual_image(out_np, model_path, vmin, vmax, zoom_coords=zoom_coords, box_color='lime')
    save_benchmark_zoomed_image(out_np, model_zoomed_path, vmin, vmax, y1, y2, x1, x2, box_color='lime')
    
    return sample_img_dir, sample_npy_dir


def update_benchmark_composite(sample_idx, dirs, model_order=None):
    """
    Create/update composite figure with all available model outputs for a sample.
    
    Layout: Horizontal comparison
    Row 1: Full images (Corrupted, GT, Model1, Model2, ...)
    Row 2: Zoomed crops (Corrupted, GT, Model1, Model2, ...)
    
    This function is called after each model test, and it will include
    all models that have been tested so far.
    """
    from matplotlib.patches import Rectangle
    
    sample_npy_dir = os.path.join(dirs['benchmark_numpy'], f'sample_{sample_idx:04d}')
    
    if not os.path.exists(sample_npy_dir):
        return None
    
    # Load base images
    corrupted_path = os.path.join(sample_npy_dir, 'corrupted.npy')
    gt_path = os.path.join(sample_npy_dir, 'ground_truth.npy')
    
    if not os.path.exists(corrupted_path) or not os.path.exists(gt_path):
        return None
    
    corrupted = np.load(corrupted_path)
    gt = np.load(gt_path)
    
    # Load grayscale window and zoom coordinates
    window_path = os.path.join(sample_npy_dir, 'grayscale_window.json')
    if os.path.exists(window_path):
        with open(window_path, 'r') as f:
            window = json.load(f)
        vmin, vmax = window['vmin'], window['vmax']
        # Use saved zoom coordinates if available
        if 'zoom_y1' in window:
            y1, y2 = window['zoom_y1'], window['zoom_y2']
            x1, x2 = window['zoom_x1'], window['zoom_x2']
        else:
            center_y, center_x = find_metal_center(corrupted)
            y1, y2, x1, x2 = get_zoom_region(corrupted.shape, center_y, center_x)
    else:
        vmin, vmax = get_consistent_grayscale_window(gt)
        center_y, center_x = find_metal_center(corrupted)
        y1, y2, x1, x2 = get_zoom_region(corrupted.shape, center_y, center_x)
    
    # Find all available model outputs
    if model_order is None:
        model_order = ['DICDNet', 'FIND-Net', 'InDuDoNet', 'InDuDoNet+', 'MEPNet', 'SGA-MARN', 'TransMAR-GAN']
    
    available_models = []
    model_outputs = {}
    for model_name in model_order:
        model_path = os.path.join(sample_npy_dir, f'{model_name}.npy')
        if os.path.exists(model_path):
            available_models.append(model_name)
            model_outputs[model_name] = np.load(model_path)
    
    if len(available_models) == 0:
        return None
    
    # Create figure: 2 rows (full, zoomed), N columns (Corrupted + GT + models)
    n_cols = 2 + len(available_models)  # Corrupted + GT + models
    fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 5))
    
    # Column labels
    col_labels = ['Corrupted', 'Ground Truth'] + available_models
    
    # Row 0: Full images
    # Corrupted (RED box)
    axes[0, 0].imshow(np.clip(corrupted, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none'))
    axes[0, 0].set_title('Corrupted', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # GT (GREEN box)
    axes[0, 1].imshow(np.clip(gt, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none'))
    axes[0, 1].set_title('Ground Truth', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Model outputs (GREEN boxes)
    for i, model_name in enumerate(available_models):
        col = i + 2
        output = model_outputs[model_name]
        axes[0, col].imshow(np.clip(output, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, col].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none'))
        axes[0, col].set_title(model_name, fontsize=10, fontweight='bold')
        axes[0, col].axis('off')
    
    # Row 1: Zoomed crops
    zoom_h, zoom_w = y2 - y1, x2 - x1
    
    # Corrupted zoomed (RED border)
    axes[1, 0].imshow(np.clip(corrupted[y1:y2, x1:x2], 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].add_patch(Rectangle((0, 0), zoom_w-1, zoom_h-1, linewidth=3, edgecolor='red', facecolor='none'))
    axes[1, 0].axis('off')
    
    # GT zoomed (GREEN border)
    axes[1, 1].imshow(np.clip(gt[y1:y2, x1:x2], 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].add_patch(Rectangle((0, 0), zoom_w-1, zoom_h-1, linewidth=3, edgecolor='lime', facecolor='none'))
    axes[1, 1].axis('off')
    
    # Model outputs zoomed (GREEN borders)
    for i, model_name in enumerate(available_models):
        col = i + 2
        output = model_outputs[model_name]
        axes[1, col].imshow(np.clip(output[y1:y2, x1:x2], 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, col].add_patch(Rectangle((0, 0), zoom_w-1, zoom_h-1, linewidth=3, edgecolor='lime', facecolor='none'))
        axes[1, col].axis('off')
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.02, hspace=0.05)
    
    # Save composite
    composite_path = os.path.join(dirs['benchmark_composite'], f'sample_{sample_idx:04d}_composite.png')
    plt.savefig(composite_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    return composite_path


def create_metrics_histogram(metrics_data, save_dir):
    """Create histograms of PSNR and SSIM distributions"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR histogram
    axes[0].hist(psnr_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(psnr_values), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(psnr_values):.2f} dB')
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[1].hist(ssim_values, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(ssim_values), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(ssim_values):.4f}')
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('SSIM Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_metrics_boxplot(metrics_data, save_dir):
    """Create box plots for metrics"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bp1 = axes[0].boxplot(psnr_values, patch_artist=True)
    bp1['boxes'][0].set_facecolor('steelblue')
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    bp2 = axes[1].boxplot(ssim_values, patch_artist=True)
    bp2['boxes'][0].set_facecolor('forestgreen')
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    bp3 = axes[2].boxplot(mae_values, patch_artist=True)
    bp3['boxes'][0].set_facecolor('coral')
    axes[2].set_ylabel('MAE', fontsize=12)
    axes[2].set_title('MAE Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_per_image_metrics_plot(metrics_data, save_dir):
    """Create line plot showing metrics per image"""
    # Group by image
    image_metrics = {}
    for m in metrics_data:
        img_idx = m['image_idx']
        if img_idx not in image_metrics:
            image_metrics[img_idx] = {'psnr': [], 'ssim': []}
        image_metrics[img_idx]['psnr'].append(m['psnr'])
        image_metrics[img_idx]['ssim'].append(m['ssim'])
    
    # Average per image
    img_indices = sorted(image_metrics.keys())
    avg_psnr = [np.mean(image_metrics[i]['psnr']) for i in img_indices]
    avg_ssim = [np.mean(image_metrics[i]['ssim']) for i in img_indices]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(img_indices, avg_psnr, 'b-', linewidth=1, alpha=0.7)
    axes[0].axhline(np.mean(avg_psnr), color='red', linestyle='--', label=f'Mean: {np.mean(avg_psnr):.2f} dB')
    axes[0].fill_between(img_indices, np.min(avg_psnr), avg_psnr, alpha=0.3)
    axes[0].set_xlabel('Image Index', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('Average PSNR per Image', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(img_indices, avg_ssim, 'g-', linewidth=1, alpha=0.7)
    axes[1].axhline(np.mean(avg_ssim), color='red', linestyle='--', label=f'Mean: {np.mean(avg_ssim):.4f}')
    axes[1].fill_between(img_indices, np.min(avg_ssim), avg_ssim, alpha=0.3)
    axes[1].set_xlabel('Image Index', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('Average SSIM per Image', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_image_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_figure(metrics_data, save_dir, total_time, model_name='DICDNet'):
    """Create a comprehensive summary figure"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Title
    fig.suptitle(f'{model_name} Benchmark Results - SynDeepLesion Test Set', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # PSNR histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(psnr_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(psnr_values), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('PSNR (dB)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('PSNR Distribution')
    ax1.grid(True, alpha=0.3)
    
    # SSIM histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(ssim_values, bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(ssim_values), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('SSIM')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SSIM Distribution')
    ax2.grid(True, alpha=0.3)
    
    # MAE histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(mae_values, bins=30, color='coral', edgecolor='black', alpha=0.7)
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
    ax6.scatter(psnr_values, ssim_values, alpha=0.5, s=10)
    ax6.set_xlabel('PSNR (dB)')
    ax6.set_ylabel('SSIM')
    ax6.set_title('PSNR vs SSIM Correlation')
    ax6.grid(True, alpha=0.3)
    
    # Per-mask average
    ax7 = fig.add_subplot(gs[2, 1:])
    mask_psnr = {i: [] for i in range(10)}
    for m in metrics_data:
        mask_psnr[m['mask_idx']].append(m['psnr'])
    mask_avg = [np.mean(mask_psnr[i]) if mask_psnr[i] else 0 for i in range(10)]
    ax7.bar(range(10), mask_avg, color='steelblue', edgecolor='black', alpha=0.7)
    ax7.axhline(np.mean(psnr_values), color='red', linestyle='--', label='Overall Mean')
    ax7.set_xlabel('Mask Index')
    ax7.set_ylabel('Average PSNR (dB)')
    ax7.set_title('Average PSNR by Mask Type')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_figure.png'), dpi=200, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════
# SAVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def save_metrics_csv(metrics_data, save_path):
    """Save all individual metrics to CSV"""
    csv_path = os.path.join(save_path, 'individual_metrics.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'image_idx', 'mask_idx', 
                                                'psnr', 'ssim', 'mae', 'rmse', 'time'])
        writer.writeheader()
        writer.writerows(metrics_data)
    
    return csv_path

def save_summary_table(metrics_data, save_path):
    """Save summary statistics as a formatted table"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    rmse_values = [m['rmse'] for m in metrics_data]
    time_values = [m['time'] for m in metrics_data]
    
    table_path = os.path.join(save_path, 'summary_table.txt')
    
    with open(table_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DICDNet BENCHMARK RESULTS - SynDeepLesion Test Set\n")
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

def save_metrics_json(metrics_data, save_path, total_time):
    """Save comprehensive metrics report in JSON format"""
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    mae_values = [m['mae'] for m in metrics_data]
    rmse_values = [m['rmse'] for m in metrics_data]
    
    report = {
        'metadata': {
            'model': 'DICDNet',
            'dataset': 'SynDeepLesion_test',
            'evaluation_date': datetime.now().isoformat(),
            'total_samples': len(metrics_data),
            'total_time_seconds': float(total_time),
            'avg_time_per_sample': float(total_time / len(metrics_data))
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
        'per_mask_metrics': {},
        'individual_results': metrics_data
    }
    
    # Per-mask statistics
    for mask_idx in range(10):
        mask_data = [m for m in metrics_data if m['mask_idx'] == mask_idx]
        if mask_data:
            report['per_mask_metrics'][f'mask_{mask_idx}'] = {
                'count': len(mask_data),
                'psnr_mean': float(np.mean([m['psnr'] for m in mask_data])),
                'ssim_mean': float(np.mean([m['ssim'] for m in mask_data]))
            }
    
    json_path = os.path.join(save_path, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return json_path

# ═══════════════════════════════════════════════════════════════
# MAIN TEST FUNCTION
# ═══════════════════════════════════════════════════════════════

def main():
    # Setup
    dirs = setup_directories(opt.save_path)
    logger, log_file = setup_logging(dirs['logs'])
    
    logger.info("=" * 70)
    logger.info("DICDNet BENCHMARK TEST - SynDeepLesion Dataset")
    logger.info("=" * 70)
    logger.info(f"Save path: {opt.save_path}")
    logger.info(f"Log file: {log_file}")
    
    # Load model
    logger.info(f"\n[1/4] Loading DICDNet model from: {opt.model_path}")
    
    model = DICDNet(opt).to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    Model loaded successfully! Parameters: {num_params:,}")
    
    # Load test mask
    logger.info(f"\n[2/4] Loading test data from: {opt.data_path}")
    test_mask = np.load(os.path.join(opt.data_path, 'testmask.npy'))
    
    # Determine which samples to test
    if opt.use_selected_slices:
        # Use the same selected slices as inference figures
        if os.path.exists(opt.selected_slices_path):
            selected_indices = np.load(opt.selected_slices_path)
            # Convert absolute indices to (image_idx, mask_idx) pairs
            # Each image has 10 masks, so: image_idx = idx // 10, mask_idx = idx % 10
            test_samples = [(int(idx // 10), int(idx % 10)) for idx in selected_indices]
            total_samples = len(test_samples)
            logger.info(f"    Using {total_samples} selected slices from: {opt.selected_slices_path}")
            logger.info(f"    Selected indices: {list(selected_indices)}")
        else:
            logger.error(f"    Selected slices file not found: {opt.selected_slices_path}")
            logger.info(f"    Falling back to sequential testing...")
            test_samples = [(i, j) for i in range(opt.num_images) for j in range(opt.num_masks)]
            total_samples = len(test_samples)
    else:
        # Sequential testing
        test_samples = [(i, j) for i in range(opt.num_images) for j in range(opt.num_masks)]
        total_samples = len(test_samples)
        logger.info(f"    Testing {opt.num_images} images × {opt.num_masks} masks = {total_samples} samples")
    
    # Run evaluation
    logger.info(f"\n[3/4] Running evaluation...")
    logger.info("-" * 70)
    
    metrics_data = []
    total_time = 0
    count = 0
    
    pbar = tqdm(total=total_samples, desc="Testing")
    
    with torch.no_grad():
        for imag_idx, mask_idx in test_samples:
            try:
                # Load sample
                Xma, Xgt, XLI, M = load_test_sample(opt.data_path, test_mask, imag_idx, mask_idx)
                
                # Inference
                torch.cuda.synchronize()
                start_time = time.time()
                X0, ListX, ListA = model(Xma, XLI, M)
                torch.cuda.synchronize()
                end_time = time.time()
                
                dur_time = end_time - start_time
                total_time += dur_time
                
                # Post-process outputs
                Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
                Xgtclip = torch.clamp(Xgt / 255.0, 0, 0.5)
                Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
                
                Xoutnorm = Xoutclip / 0.5
                Xgtnorm = Xgtclip / 0.5
                Xmanorm = Xmaclip / 0.5
                
                # Calculate metrics
                psnr_val, ssim_val, mae_val, rmse_val = calculate_metrics(Xgtnorm, Xoutnorm)
                
                sample_metrics = {
                    'sample_id': count + 1,
                    'image_idx': imag_idx,
                    'mask_idx': mask_idx,
                    'psnr': float(psnr_val),
                    'ssim': float(ssim_val),
                    'mae': float(mae_val),
                    'rmse': float(rmse_val),
                    'time': float(dur_time)
                }
                metrics_data.append(sample_metrics)
                
                # Always save sample images (first 20 comparisons, first 10 difference maps)
                # Save more if --save_images or --save_all_images is passed
                ma_np = Xmanorm.cpu().numpy().squeeze()
                gt_np = Xgtnorm.cpu().numpy().squeeze()
                out_np = Xoutnorm.cpu().numpy().squeeze()
                
                # Default: save first 20 comparisons (or all if using selected slices)
                save_comparison = count < 20 or opt.save_images or opt.save_all_images or opt.use_selected_slices
                if save_comparison:
                    if opt.save_all_images or opt.use_selected_slices or count < 50:
                        save_comparison_image(count + 1, ma_np, gt_np, out_np, 
                                            dirs['comparisons'], sample_metrics)
                
                # Default: save first 10 difference maps (or all if using selected slices)
                save_diff = count < 10 or (opt.save_images and count < 20) or opt.use_selected_slices
                if save_diff:
                    save_difference_map(count + 1, gt_np, out_np, dirs['images'])
                
                # ═══════════════════════════════════════════════════════════════
                # SAVE BENCHMARK OUTPUTS (consistent grayscale for composites)
                # ═══════════════════════════════════════════════════════════════
                # Save individual images and numpy arrays for benchmark comparison
                save_benchmark_outputs(count, ma_np, gt_np, out_np, dirs, model_name='DICDNet')
                
                # Update composite figure (includes all models tested so far)
                update_benchmark_composite(count, dirs)
                
                count += 1
                pbar.update(1)
                pbar.set_postfix({'PSNR': f'{psnr_val:.2f}', 'SSIM': f'{ssim_val:.4f}'})
                
            except Exception as e:
                logger.error(f"Error on image {imag_idx}, mask {mask_idx}: {e}")
                continue
    
    pbar.close()
    
    # Save all results
    logger.info(f"\n[4/4] Saving results...")
    
    # Save CSV
    csv_path = save_metrics_csv(metrics_data, dirs['tables'])
    logger.info(f"    CSV saved: {csv_path}")
    
    # Save summary table
    table_path = save_summary_table(metrics_data, dirs['tables'])
    logger.info(f"    Summary table saved: {table_path}")
    
    # Save JSON
    json_path = save_metrics_json(metrics_data, dirs['metrics'], total_time)
    logger.info(f"    JSON saved: {json_path}")
    
    # Create visualizations
    logger.info("    Creating visualizations...")
    create_metrics_histogram(metrics_data, dirs['visualizations'])
    create_metrics_boxplot(metrics_data, dirs['visualizations'])
    create_per_image_metrics_plot(metrics_data, dirs['visualizations'])
    create_summary_figure(metrics_data, dirs['visualizations'], total_time)
    logger.info(f"    Visualizations saved to: {dirs['visualizations']}")
    
    # Log benchmark figure outputs
    logger.info(f"\n    BENCHMARK FIGURES (consistent grayscale):")
    logger.info(f"    Individual images: {dirs['benchmark_individual']}")
    logger.info(f"    Composite figures: {dirs['benchmark_composite']}")
    logger.info(f"    Numpy arrays: {dirs['benchmark_numpy']}")
    
    # Final summary
    psnr_values = [m['psnr'] for m in metrics_data]
    ssim_values = [m['ssim'] for m in metrics_data]
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Model:          DICDNet (pretrained)")
    logger.info(f"  Dataset:        SynDeepLesion Test ({count} samples)")
    logger.info(f"  PSNR:           {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f} dB")
    logger.info(f"  SSIM:           {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}")
    logger.info(f"  Total Time:     {total_time/60:.2f} minutes")
    logger.info(f"  Avg per Sample: {total_time/count:.4f} seconds")
    logger.info(f"  Results saved:  {opt.save_path}")
    logger.info("=" * 70)
    
    # Print to console as well
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  PSNR: {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f} dB")
    print(f"  SSIM: {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}")
    print(f"  Results saved to: {opt.save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
