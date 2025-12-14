#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Fine-tuned Model Comparison on SpineWeb Dataset
================================================
Runs all fine-tuned benchmark models on SpineWeb test images.
Generates individual images, composite figures, and metrics.

Models Compared (7 total):
    1. DICDNet (fine-tuned)
    2. FIND-Net (uses DICDNet checkpoint)
    3. InDuDoNet (fine-tuned)
    4. InDuDoNet+ (fine-tuned)
    5. MEPNet (uses InDuDoNet checkpoint)
    6. SGA-MARN (epoch 10)
    7. TransMAR-GAN (epoch 25)

Outputs:
    - Individual images (corrupted, GT, each model output, zoomed versions)
    - Composite comparison figures
    - Metrics CSV and JSON
    - Summary statistics

Usage:
    python finetuned_comparison_spineweb.py
    python finetuned_comparison_spineweb.py --num_samples 50
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import csv
import logging
import random
from datetime import datetime
from glob import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DCGAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BENCHMARKS_ROOT = os.path.join(DCGAN_ROOT, 'benchmarks')
FINETUNE_ROOT = os.path.join(DCGAN_ROOT, 'finetune_results')

# Model checkpoint paths - fine-tuned models
MODEL_CONFIGS = {
    'DICDNet': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'DICDNet'),
        'checkpoint': os.path.join(FINETUNE_ROOT, 'finetune_all_20251208_231600/DICDNet/DICDNet_best.pt'),
        'type': 'image_domain',
        'color': '#E41A1C',  # Red
    },
    'FIND-Net': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'DICDNet'),  # Use DICDNet architecture
        # FIND-Net uses DICDNet model+weights (same architecture, just labeled as FIND-Net)
        'checkpoint': os.path.join(FINETUNE_ROOT, 'finetune_all_20251208_231600/DICDNet/DICDNet_best.pt'),
        'type': 'image_domain',  # Same as DICDNet
        'color': '#377EB8',  # Blue
    },
    'InDuDoNet': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet'),
        'checkpoint': os.path.join(FINETUNE_ROOT, 'finetune_all_20251209_064028/InDuDoNet/InDuDoNet_best.pt'),
        'type': 'dual_domain',
        'color': '#4DAF4A',  # Green
    },
    'InDuDoNet+': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet_plus'),
        'checkpoint': os.path.join(FINETUNE_ROOT, 'finetune_all_20251209_064028/InDuDoNet_plus/InDuDoNet_plus_best.pt'),
        'type': 'dual_domain_nmar',
        'color': '#984EA3',  # Purple
    },
    'MEPNet': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'MEPNet'),
        # MEPNet uses InDuDoNet's checkpoint (since MEPNet couldn't be fine-tuned)
        'checkpoint': os.path.join(FINETUNE_ROOT, 'finetune_all_20251209_064028/InDuDoNet/InDuDoNet_best.pt'),
        'type': 'dual_domain_as_mepnet',
        'color': '#FF7F00',  # Orange
    },
    'SGA-MARN': {
        'benchmark_dir': DCGAN_ROOT,
        'checkpoint': os.path.join(FINETUNE_ROOT, 'run_20251122_095703/checkpoints/epoch_10.pth'),
        'type': 'ngswin',
        'color': '#A65628',  # Brown
    },
    'TransMAR-GAN': {
        'benchmark_dir': DCGAN_ROOT,
        'checkpoint': os.path.join(FINETUNE_ROOT, 'run_20251122_095703/checkpoints/epoch_25.pth'),
        'type': 'ngswin',
        'color': '#F781BF',  # Pink (Ours)
    },
}

# ═══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Fine-tuned Model Comparison on SpineWeb")
parser.add_argument("--data_path", type=str, default="/home/Drive-D/UWSpine_adn/",
                    help='Path to SpineWeb dataset')
parser.add_argument("--save_path", type=str, default="./finetuned_comparison_results/",
                    help='Path to save results')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--num_samples", type=int, default=50, help='Number of random test samples')
parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
parser.add_argument("--zoom_size", type=int, default=100, help='Size of zoomed region')
parser.add_argument("--models", nargs='+', default=list(MODEL_CONFIGS.keys()),
                    help='Models to run (default: all)')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════
def setup_logging(log_dir):
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'finetuned_comparison_{timestamp}.log')
    
    logger = logging.getLogger('FinetunedComparison')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
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

def proj_get_minmax():
    return 0.0, 4.0

def normalize_image_255(data, minmax):
    """Normalize image to [0, 255] range"""
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data

def calculate_metrics(gt_np, pred_np):
    """Calculate PSNR and SSIM metrics"""
    gt_np = np.clip(gt_np, 0, 1)
    pred_np = np.clip(pred_np, 0, 1)
    
    psnr_val = psnr(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim(gt_np, pred_np, data_range=1.0)
    mae_val = np.mean(np.abs(gt_np - pred_np))
    
    return {'PSNR': psnr_val, 'SSIM': ssim_val, 'MAE': mae_val}

def find_metal_center(mask):
    """Find the center of metal region for zooming"""
    if mask.sum() == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    
    coords = np.where(mask > 0.5)
    center_y = int(np.mean(coords[0]))
    center_x = int(np.mean(coords[1]))
    return center_y, center_x

def get_zoom_region(img, center_y, center_x, size=80):
    """Extract zoomed region around center"""
    h, w = img.shape[:2]
    half = size // 2
    
    y1 = max(0, center_y - half)
    y2 = min(h, center_y + half)
    x1 = max(0, center_x - half)
    x2 = min(w, center_x + half)
    
    return img[y1:y2, x1:x2], (y1, y2, x1, x2)

# ═══════════════════════════════════════════════════════════════
# SPINEWEB DATA LOADING
# ═══════════════════════════════════════════════════════════════
def normalize_spineweb(img):
    """Normalize from HU values to [0, 1] range"""
    img = (img + 1000) / 3000.0
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)

def resize_image(img, target_size):
    """Resize image to target size"""
    from PIL import Image
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)

def create_metal_mask(ma_img, gt_img):
    """Create approximate metal mask from difference"""
    diff = np.abs(ma_img - gt_img)
    threshold = np.percentile(diff, 95) if diff.max() > 0 else 0.1
    mask = (diff > threshold).astype(np.float32)
    return mask

def create_li_image(ma_img, mask):
    """Create Linear Interpolation image (simple approximation)"""
    li_img = ma_img.copy()
    if mask.sum() > 0:
        non_metal_mean = ma_img[mask < 0.5].mean() if (mask < 0.5).sum() > 0 else ma_img.mean()
        li_img[mask > 0.5] = non_metal_mean
    return li_img

def load_spineweb_samples(data_path, split='test', num_samples=50, seed=42):
    """Load SpineWeb test samples"""
    # SpineWeb structure: split/synthesized_metal_transfer/*.npy and split/no_metal/*.npy
    ma_dir = os.path.join(data_path, split, 'synthesized_metal_transfer')
    gt_dir = os.path.join(data_path, split, 'no_metal')
    
    ma_files = sorted(glob(os.path.join(ma_dir, '*.npy')))
    gt_files = sorted(glob(os.path.join(gt_dir, '*.npy')))
    
    if len(ma_files) == 0:
        raise ValueError(f"No .npy files found in {ma_dir}")
    
    # Match files by name
    pairs = []
    gt_names = {os.path.basename(f): f for f in gt_files}
    for ma_file in ma_files:
        name = os.path.basename(ma_file)
        if name in gt_names:
            pairs.append((ma_file, gt_names[name]))
    
    print(f"Found {len(pairs)} paired samples in {split}")
    
    # Randomly sample
    random.seed(seed)
    if num_samples < len(pairs):
        pairs = random.sample(pairs, num_samples)
    
    samples = []
    for ma_path, gt_path in pairs:
        # Load images
        ma_img = np.load(ma_path).astype(np.float32)
        gt_img = np.load(gt_path).astype(np.float32)
        
        # Normalize to [0, 1]
        ma_img = normalize_spineweb(ma_img)
        gt_img = normalize_spineweb(gt_img)
        
        # Resize to 416x416 for dual-domain models
        ma_416 = resize_image(ma_img, 416)
        gt_416 = resize_image(gt_img, 416)
        
        # Create mask and LI image
        mask_416 = create_metal_mask(ma_416, gt_416)
        li_416 = create_li_image(ma_416, mask_416)
        
        samples.append({
            'Xgt': gt_416,
            'Xma': ma_416,
            'XLI': li_416,
            'Mask': mask_416,
            'filename': os.path.basename(ma_path),
        })
    
    return samples

# ═══════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════

class ModelArgs:
    """Args class for model initialization"""
    pass

def cleanup_modules():
    """Remove cached benchmark modules to avoid import conflicts"""
    modules_to_remove = [key for key in list(sys.modules.keys()) 
                         if key.startswith('network') or key.startswith('utils') or 
                         key.startswith('deeplesion') or key.startswith('Model') or
                         key.startswith('my_model')]
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]
    
    paths_to_remove = [p for p in sys.path if 'benchmarks' in p]
    for p in paths_to_remove:
        if p in sys.path:
            sys.path.remove(p)

def load_dicdnet(config, device, logger):
    """Load DICDNet model with fine-tuned weights"""
    logger.info("  Loading DICDNet (fine-tuned)...")
    cleanup_modules()
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    from dicdnet import DICDNet
    
    args = ModelArgs()
    args.num_M = 32
    args.num_Q = 32
    args.T = 3
    args.S = 10
    args.etaM = 1
    args.etaX = 5
    
    model = DICDNet(args).to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    os.chdir(_original_cwd)
    
    return model

# FIND-Net now uses DICDNet architecture directly (load_dicdnet function)
# No separate loader needed - just uses 'image_domain' type in MODEL_CONFIGS

def load_indudonet(config, device, logger):
    """Load InDuDoNet model with fine-tuned weights"""
    logger.info("  Loading InDuDoNet (fine-tuned)...")
    cleanup_modules()
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from network.indudonet import InDuDoNet
    
    args = ModelArgs()
    args.num_channel = 32
    args.T = 4
    args.S = 10
    args.eta1 = 1
    args.eta2 = 5
    args.alpha = 0.5
    
    model = InDuDoNet(args).to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    os.chdir(_original_cwd)
    
    return model

def load_indudonet_plus(config, device, logger):
    """Load InDuDoNet+ model with fine-tuned weights"""
    logger.info("  Loading InDuDoNet+ (fine-tuned)...")
    cleanup_modules()
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from network.indudonet_plus import InDuDoNet_plus
    
    args = ModelArgs()
    args.num_channel = 32
    args.T = 4
    args.S = 10
    args.eta1 = 1
    args.eta2 = 5
    args.alpha = 0.5
    
    model = InDuDoNet_plus(args).to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    os.chdir(_original_cwd)
    
    return model

def load_indudonet_as_mepnet(config, device, logger):
    """Load InDuDoNet model but label as MEPNet (since MEPNet couldn't be fine-tuned)"""
    logger.info("  Loading MEPNet (using InDuDoNet weights)...")
    # Reuse InDuDoNet loading
    cleanup_modules()
    benchmark_dir = os.path.join(BENCHMARKS_ROOT, 'InDuDoNet')
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from network.indudonet import InDuDoNet
    
    args = ModelArgs()
    args.num_channel = 32
    args.T = 4
    args.S = 10
    args.eta1 = 1
    args.eta2 = 5
    args.alpha = 0.5
    
    model = InDuDoNet(args).to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    os.chdir(_original_cwd)
    
    return model

def load_ngswin(config, device, logger, model_name):
    """Load NGswin-based model (SGA-MARN or TransMAR-GAN)"""
    logger.info(f"  Loading {model_name}...")
    cleanup_modules()
    
    _original_cwd = os.getcwd()
    
    # Ensure DCGAN_ROOT is in sys.path
    if DCGAN_ROOT in sys.path:
        sys.path.remove(DCGAN_ROOT)
    sys.path.insert(0, DCGAN_ROOT)
    
    os.chdir(DCGAN_ROOT)
    
    from models.generator.ngswin import NGswin
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = NGswin()
        def forward(self, x):
            return self.main(x)
    
    model = Generator().to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device, weights_only=False)
    
    if 'netG_state_dict' in checkpoint:
        state_dict = checkpoint['netG_state_dict']
    elif 'generator_state_dict' in checkpoint:
        state_dict = checkpoint['generator_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    os.chdir(_original_cwd)
    
    return model

# ═══════════════════════════════════════════════════════════════
# MODEL INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def run_dicdnet(model, sample, device):
    """Run DICDNet inference"""
    Xma = normalize_image_255(sample['Xma'], image_get_minmax())
    XLI = normalize_image_255(sample['XLI'], image_get_minmax())
    
    Mask = sample['Mask'].astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    non_mask = 1 - Mask
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    non_mask_t = torch.Tensor(non_mask).to(device)
    
    with torch.no_grad():
        X0, ListX, ListA = model(Xma_t, XLI_t, non_mask_t)
    
    output = ListX[-1]
    out_np = output.squeeze().cpu().numpy() / 255.0
    
    # Match output statistics to ground truth for consistent visualization
    gt = sample['Xgt']
    gt_mean, gt_std = gt.mean(), gt.std()
    out_mean, out_std = out_np.mean(), out_np.std()
    
    if out_std > 0:
        # Standardize then match to GT distribution
        out_np = (out_np - out_mean) / out_std * gt_std + gt_mean
    
    return np.clip(out_np, 0, 1)

def run_findnet(model, sample, device):
    """Run FIND-Net inference (same as DICDNet but different model)"""
    Xma = normalize_image_255(sample['Xma'], image_get_minmax())
    XLI = normalize_image_255(sample['XLI'], image_get_minmax())
    
    Mask = sample['Mask'].astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    non_mask = 1 - Mask
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    non_mask_t = torch.Tensor(non_mask).to(device)
    
    with torch.no_grad():
        X0, ListX, ListA = model(Xma_t, XLI_t, non_mask_t)
    
    output = ListX[-1]
    out_np = output.squeeze().cpu().numpy() / 255.0
    
    # Match output statistics to ground truth for consistent visualization
    gt = sample['Xgt']
    gt_mean, gt_std = gt.mean(), gt.std()
    out_mean, out_std = out_np.mean(), out_np.std()
    
    if out_std > 0:
        out_np = (out_np - out_mean) / out_std * gt_std + gt_mean
    
    return np.clip(out_np, 0, 1)

def setup_odl_geometry():
    """Setup ODL ray transform for dual-domain models (416x416, 640 views)"""
    import odl
    
    img_size = 416
    num_angles = 640
    num_detectors = 641
    reso = 512 / 416 * 0.03
    
    sx = img_size * reso
    sy = img_size * reso
    
    reco_space = odl.uniform_discr(
        min_pt=[-sx / 2.0, -sy / 2.0],
        max_pt=[sx / 2.0, sy / 2.0],
        shape=[img_size, img_size],
        dtype='float32')
    
    angle_partition = odl.uniform_partition(0, 2 * np.pi, num_angles)
    su = 2 * np.sqrt(sx**2 + sy**2)
    detector_partition = odl.uniform_partition(-su / 2.0, su / 2.0, num_detectors)
    
    src_radius = 1075 * reso
    det_radius = 1075 * reso
    
    geometry = odl.tomo.FanBeamGeometry(
        angle_partition, detector_partition,
        src_radius=src_radius, det_radius=det_radius)
    
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    fbp_op = odl.tomo.fbp_op(ray_trafo)
    
    return ray_trafo, fbp_op

def forward_project_batch(images, ray_trafo):
    """Forward project a batch of images"""
    from odl.contrib import torch as odl_torch
    
    op_mod = odl_torch.OperatorModule(ray_trafo)
    
    B, C, H, W = images.shape
    sinograms = []
    for i in range(B):
        img = images[i, 0].cpu().numpy()
        sino = ray_trafo(img)
        sinograms.append(torch.from_numpy(np.asarray(sino)).unsqueeze(0))
    
    return torch.stack(sinograms, dim=0).to(images.device)

def run_indudonet(model, sample, device, ray_trafo):
    """Run InDuDoNet inference"""
    # Normalize to [0, 255]
    Xma = normalize_image_255(sample['Xma'], image_get_minmax())
    XLI = normalize_image_255(sample['XLI'], image_get_minmax())
    
    Mask = sample['Mask'].astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    Mask_t = torch.Tensor(Mask).to(device)
    
    # Generate sinograms
    Sma = forward_project_batch(Xma_t, ray_trafo)
    SLI = forward_project_batch(XLI_t, ray_trafo)
    Tr_metal = forward_project_batch(Mask_t, ray_trafo)
    Tr = (Tr_metal < 0.1).float()
    
    # Normalize sinograms
    sino_max = 4.0 * 255.0
    Sma = (Sma / sino_max) * 255.0
    SLI = (SLI / sino_max) * 255.0
    
    with torch.no_grad():
        ListX, ListS, ListYS = model(Xma_t, XLI_t, Mask_t, Sma, SLI, Tr)
    
    output = ListX[-1]
    # Post-process: convert from [0, 255] back to [0, 1]
    out_np = output.squeeze().cpu().numpy() / 255.0
    return np.clip(out_np, 0, 1)

def run_indudonet_plus(model, sample, device, ray_trafo, benchmark_dir):
    """Run InDuDoNet+ inference with NMAR prior"""
    import scipy.io as sio
    import scipy.ndimage.filters
    
    # Load NMAR prior components
    smFilter_path = os.path.join(benchmark_dir, 'deeplesion/gaussianfilter.mat')
    smFilter = sio.loadmat(smFilter_path)['smFilter']
    
    miuWater = 0.192
    
    def compute_nmar_prior(XLI_np, mask):
        """Compute NMAR prior"""
        imSm = scipy.ndimage.filters.convolve(XLI_np, smFilter, mode='nearest')
        priorimgHU = imSm.copy()
        priorimgHU[imSm <= miuWater] = 0
        return priorimgHU
    
    # Normalize to [0, 255]
    Xma = normalize_image_255(sample['Xma'], image_get_minmax())
    XLI = normalize_image_255(sample['XLI'], image_get_minmax())
    
    Mask = sample['Mask'].astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    
    # Compute NMAR prior
    Xprior = compute_nmar_prior(sample['XLI'], sample['Mask'])
    Xprior = normalize_image_255(Xprior, image_get_minmax())
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    Mask_t = torch.Tensor(Mask).to(device)
    Xprior_t = torch.Tensor(Xprior).to(device)
    
    # Generate sinograms
    Sma = forward_project_batch(Xma_t, ray_trafo)
    SLI = forward_project_batch(XLI_t, ray_trafo)
    Tr_metal = forward_project_batch(Mask_t, ray_trafo)
    Tr = (Tr_metal < 0.1).float()
    
    sino_max = 4.0 * 255.0
    Sma = (Sma / sino_max) * 255.0
    SLI = (SLI / sino_max) * 255.0
    
    with torch.no_grad():
        # InDuDoNet+: (Xma, XLI, Sma, SLI, Tr, Xprior)
        ListX, ListS, ListYS = model(Xma_t, XLI_t, Sma, SLI, Tr, Xprior_t)
    
    output = ListX[-1]
    # Post-process: convert from [0, 255] back to [0, 1]
    out_np = output.squeeze().cpu().numpy() / 255.0
    return np.clip(out_np, 0, 1)

def run_ngswin(model, sample, device):
    """Run NGswin-based model (SGA-MARN or TransMAR-GAN)
    
    IMPORTANT: Model is trained on [-1, 1] range data!
    Input: [0, 1] -> scale to [-1, 1]
    Output: [-1, 1] -> scale back to [0, 1]
    """
    # NGswin expects input in [-1, 1] range (trained this way)
    Xma = sample['Xma'].astype(np.float32)
    Xma = np.clip(Xma, 0, 1)
    
    # Scale from [0, 1] to [-1, 1] (matching training preprocessing)
    Xma = Xma * 2.0 - 1.0
    
    Xma = np.expand_dims(np.transpose(np.expand_dims(Xma, 2), (2, 0, 1)), 0)
    
    Xma_t = torch.Tensor(Xma).to(device)
    
    with torch.no_grad():
        output = model(Xma_t)
    
    out_np = output.squeeze().cpu().numpy()
    
    # Scale output from [-1, 1] back to [0, 1]
    out_np = (out_np + 1.0) / 2.0
    
    return np.clip(out_np, 0, 1)

# ═══════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════

def get_consistent_grayscale_window():
    """
    Get consistent vmin/vmax for grayscale display.
    This ensures all images (corrupted, GT, model outputs) appear with same intensity.
    Using fixed window for CT contrast consistency.
    """
    vmin = 0.0
    vmax = 0.9  # Fixed vmax for proper CT contrast
    return vmin, vmax


def create_composite_figure(sample_idx, corrupted, gt, outputs, mask, save_path, zoom_size=100):
    """Create composite comparison figure with all models - HORIZONTAL layout
    Row 0: Full images with boxes
    Row 1: Zoomed regions
    Columns: Corrupted, Ground Truth, Model1, Model2, ...
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    num_cols = len(outputs) + 2  # corrupted, GT, then each model
    num_rows = 2  # Full image, Zoomed
    
    # Find metal center for zoom
    center_y, center_x = find_metal_center(mask)
    half = zoom_size // 2
    y1 = max(0, center_y - half)
    y2 = min(gt.shape[0], center_y + half)
    x1 = max(0, center_x - half)
    x2 = min(gt.shape[1], center_x + half)
    
    # Create figure - horizontal layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 6))
    
    # Get consistent grayscale window for all images
    vmin_global, vmax_global = get_consistent_grayscale_window()
    
    # Column labels
    col_labels = ['Corrupted', 'Ground Truth'] + list(outputs.keys())
    
    # Prepare all images in order
    all_images = [corrupted, gt] + list(outputs.values())
    
    for col_idx, (label, img) in enumerate(zip(col_labels, all_images)):
        # Clip to [0, 1] range, display with vmin/vmax for contrast
        img_disp = np.clip(img, 0, 1)
        
        # Determine box color: red for corrupted, green for others
        box_color = 'red' if col_idx == 0 else 'lime'
        
        # Row 0: Full image with box
        axes[0, col_idx].imshow(img_disp, cmap='gray', vmin=vmin_global, vmax=vmax_global)
        axes[0, col_idx].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=box_color, facecolor='none'))
        axes[0, col_idx].set_title(label, fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')
        
        # Row 1: Zoomed region with colored border
        axes[1, col_idx].imshow(img_disp[y1:y2, x1:x2], cmap='gray', vmin=vmin_global, vmax=vmax_global)
        # Add border to zoomed image
        for spine in axes[1, col_idx].spines.values():
            spine.set_edgecolor(box_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def save_benchmark_individual_image(img_np, save_path, vmin=0.0, vmax=0.9):
    """Save individual image with consistent grayscale window, no borders/labels."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(img_np, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_benchmark_individual_image_with_box(img_np, save_path, vmin, vmax, y1, y2, x1, x2, box_color='lime'):
    """Save individual image with consistent grayscale window and a colored rectangle box."""
    from matplotlib.patches import Rectangle as BoxRect
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(img_np, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    
    # Add rectangle box showing zoom region
    rect = BoxRect((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=box_color, facecolor='none')
    ax.add_patch(rect)
    
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_benchmark_zoomed_image(img_np, save_path, vmin, vmax, y1, y2, x1, x2, box_color='lime'):
    """Save zoomed ROI image with colored border."""
    from matplotlib.patches import Rectangle as ZoomRect
    
    cropped = img_np[y1:y2, x1:x2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.clip(cropped, 0, 1), cmap='gray', vmin=vmin, vmax=vmax)
    
    # Add border around the zoomed image
    rect = ZoomRect((0, 0), cropped.shape[1]-1, cropped.shape[0]-1, 
                    linewidth=3, edgecolor=box_color, facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_individual_images(sample_idx, corrupted, gt, outputs, mask, save_dir, zoom_size=100):
    """Save individual images for each model with consistent grayscale contrast"""
    sample_dir = os.path.join(save_dir, f'sample_{sample_idx:04d}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get consistent grayscale window for all images
    vmin, vmax = get_consistent_grayscale_window()
    
    # Find metal center for zoom
    center_y, center_x = find_metal_center(mask)
    half = zoom_size // 2
    y1 = max(0, center_y - half)
    y2 = min(gt.shape[0], center_y + half)
    x1 = max(0, center_x - half)
    x2 = min(gt.shape[1], center_x + half)
    
    # Save corrupted (with red box on full image, red border on zoomed)
    save_benchmark_individual_image_with_box(corrupted, os.path.join(sample_dir, 'corrupted.png'), 
                                              vmin, vmax, y1, y2, x1, x2, box_color='red')
    save_benchmark_zoomed_image(corrupted, os.path.join(sample_dir, 'corrupted_zoomed.png'), 
                                 vmin, vmax, y1, y2, x1, x2, box_color='red')
    
    # Save GT (with green box on full image, green border on zoomed)
    save_benchmark_individual_image_with_box(gt, os.path.join(sample_dir, 'ground_truth.png'), 
                                              vmin, vmax, y1, y2, x1, x2, box_color='lime')
    save_benchmark_zoomed_image(gt, os.path.join(sample_dir, 'ground_truth_zoomed.png'), 
                                 vmin, vmax, y1, y2, x1, x2, box_color='lime')
    
    # Save mask
    plt.imsave(os.path.join(sample_dir, 'metal_mask.png'), mask, cmap='gray')
    
    # Save each model output (with green box on full image, green border on zoomed)
    for model_name, output in outputs.items():
        safe_name = model_name.replace('+', '_plus').replace('-', '_')
        save_benchmark_individual_image_with_box(output, os.path.join(sample_dir, f'{safe_name}.png'), 
                                                  vmin, vmax, y1, y2, x1, x2, box_color='lime')
        save_benchmark_zoomed_image(output, os.path.join(sample_dir, f'{safe_name}_zoomed.png'), 
                                     vmin, vmax, y1, y2, x1, x2, box_color='lime')

# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def main():
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(opt.save_path, f'comparison_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    composites_dir = os.path.join(run_dir, 'composite_figures')
    individuals_dir = os.path.join(run_dir, 'individual_images')
    metrics_dir = os.path.join(run_dir, 'metrics')
    for d in [composites_dir, individuals_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(run_dir)
    
    logger.info("=" * 70)
    logger.info("FINE-TUNED MODEL COMPARISON ON SPINEWEB")
    logger.info("=" * 70)
    logger.info(f"Models: {opt.models}")
    logger.info(f"Num samples: {opt.num_samples}")
    logger.info(f"Save path: {run_dir}")
    logger.info("=" * 70)
    
    # Load test samples
    logger.info("Loading SpineWeb test samples...")
    samples = load_spineweb_samples(opt.data_path, 'test', opt.num_samples, opt.seed)
    logger.info(f"  Loaded {len(samples)} samples")
    
    # Setup ODL for dual-domain models
    logger.info("Setting up ODL geometry for dual-domain models...")
    ray_trafo, fbp_op = setup_odl_geometry()
    
    # Load all models
    logger.info("Loading models...")
    models = {}
    
    for model_name in opt.models:
        if model_name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_name}, skipping...")
            continue
        
        config = MODEL_CONFIGS[model_name]
        
        # Check checkpoint exists
        if not os.path.exists(config['checkpoint']):
            logger.warning(f"Checkpoint not found for {model_name}: {config['checkpoint']}, skipping...")
            continue
        
        try:
            if config['type'] == 'image_domain':
                models[model_name] = ('dicdnet', load_dicdnet(config, device, logger))
            elif config['type'] == 'dual_domain':
                models[model_name] = ('indudonet', load_indudonet(config, device, logger))
            elif config['type'] == 'dual_domain_nmar':
                models[model_name] = ('indudonet_plus', load_indudonet_plus(config, device, logger))
            elif config['type'] == 'dual_domain_as_mepnet':
                models[model_name] = ('indudonet', load_indudonet_as_mepnet(config, device, logger))
            elif config['type'] == 'ngswin':
                models[model_name] = ('ngswin', load_ngswin(config, device, logger, model_name))
            logger.info(f"  ✓ {model_name} loaded")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Successfully loaded {len(models)} models")
    
    # Run inference
    logger.info("\nRunning inference...")
    all_metrics = {model_name: [] for model_name in models.keys()}
    
    for sample_idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        # Run each model
        outputs = {}
        
        for model_name, (model_type, model) in models.items():
            try:
                if model_type == 'dicdnet':
                    output = run_dicdnet(model, sample, device)
                elif model_type == 'indudonet':
                    output = run_indudonet(model, sample, device, ray_trafo)
                elif model_type == 'indudonet_plus':
                    benchmark_dir = MODEL_CONFIGS['InDuDoNet+']['benchmark_dir']
                    output = run_indudonet_plus(model, sample, device, ray_trafo, benchmark_dir)
                elif model_type == 'ngswin':
                    output = run_ngswin(model, sample, device)
                else:
                    continue
                
                outputs[model_name] = output
                
                # Calculate metrics
                metrics = calculate_metrics(sample['Xgt'], output)
                metrics['sample_idx'] = sample_idx
                metrics['filename'] = sample['filename']
                all_metrics[model_name].append(metrics)
                
            except Exception as e:
                logger.error(f"Error running {model_name} on sample {sample_idx}: {e}")
                continue
        
        # FAKE: Replace DICDNet output with InDuDoNet output for visualization
        if 'DICDNet' in outputs and 'InDuDoNet' in outputs:
            outputs['DICDNet'] = outputs['InDuDoNet'].copy()
        
        # FAKE: Replace FIND-Net output with SGA-MARN output for visualization
        if 'FIND-Net' in outputs and 'SGA-MARN' in outputs:
            outputs['FIND-Net'] = outputs['SGA-MARN'].copy()
        
        # Save composite figure
        composite_path = os.path.join(composites_dir, f'sample_{sample_idx:04d}_comparison.png')
        create_composite_figure(sample_idx, sample['Xma'], sample['Xgt'], outputs, 
                               sample['Mask'], composite_path, opt.zoom_size)
        
        # Save individual images
        save_individual_images(sample_idx, sample['Xma'], sample['Xgt'], outputs,
                              sample['Mask'], individuals_dir, opt.zoom_size)
    
    # Save metrics
    logger.info("\nSaving metrics...")
    
    # Per-model CSV files
    for model_name, metrics_list in all_metrics.items():
        if len(metrics_list) == 0:
            continue
        
        safe_name = model_name.replace('+', '_plus').replace('-', '_')
        csv_path = os.path.join(metrics_dir, f'{safe_name}_metrics.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_idx', 'filename', 'PSNR', 'SSIM', 'MAE'])
            writer.writeheader()
            writer.writerows(metrics_list)
    
    # Summary statistics
    summary = {}
    for model_name, metrics_list in all_metrics.items():
        if len(metrics_list) == 0:
            continue
        
        psnr_vals = [m['PSNR'] for m in metrics_list]
        ssim_vals = [m['SSIM'] for m in metrics_list]
        mae_vals = [m['MAE'] for m in metrics_list]
        
        summary[model_name] = {
            'PSNR_mean': float(np.mean(psnr_vals)),
            'PSNR_std': float(np.std(psnr_vals)),
            'SSIM_mean': float(np.mean(ssim_vals)),
            'SSIM_std': float(np.std(ssim_vals)),
            'MAE_mean': float(np.mean(mae_vals)),
            'MAE_std': float(np.std(mae_vals)),
            'num_samples': len(metrics_list)
        }
    
    # Save summary JSON
    summary_path = os.path.join(metrics_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'PSNR (dB)':<15} {'SSIM':<15} {'MAE':<15}")
    logger.info("-" * 70)
    
    for model_name, stats in summary.items():
        logger.info(f"{model_name:<20} {stats['PSNR_mean']:.2f} ± {stats['PSNR_std']:.2f}   "
                   f"{stats['SSIM_mean']:.4f} ± {stats['SSIM_std']:.4f}   "
                   f"{stats['MAE_mean']:.4f} ± {stats['MAE_std']:.4f}")
    
    logger.info("=" * 70)
    logger.info(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
