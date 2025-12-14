#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Comprehensive Benchmark Comparison on SynDeepLesion Dataset
============================================================
Runs ALL benchmark models + our models on the same test images.
Generates individual images, composite figures, line profiles, and metrics.

Models Compared (7 total):
    1. DICDNet
    2. FIND-Net  
    3. InDuDoNet
    4. InDuDoNet+
    5. MEPNet
    6. SGA-MARN (our previous model)
    7. TransMAR-GAN (Ours - full model)

Outputs:
    - Individual images (corrupted, GT, each model output, zoomed versions)
    - Composite comparison figures
    - Line profile plots (sinogram + intensity)
    - Metrics CSV tables
    - Comprehensive logging

For IEEE TMI Paper.

Usage:
    python benchmark_comparison_syndeeplesion.py
    python benchmark_comparison_syndeeplesion.py --num_samples 25
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import h5py
import json
import csv
import logging
import random
import scipy.io as sio
from datetime import datetime
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

try:
    from torch_radon import Radon
    HAS_TORCH_RADON = True
except ImportError:
    HAS_TORCH_RADON = False
    print("Warning: torch_radon not available, sinogram profiles will be skipped")

# ═══════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DCGAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BENCHMARKS_ROOT = os.path.join(DCGAN_ROOT, 'benchmarks')

# Model checkpoint paths
MODEL_CONFIGS = {
    'DICDNet': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'DICDNet'),
        'checkpoint': os.path.join(BENCHMARKS_ROOT, 'DICDNet/pretrain_model/DICDNet_latest.pt'),
        'type': 'image_domain',
        'color': '#E41A1C',  # Red
    },
    'FIND-Net': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'FIND-Net'),
        'checkpoint': os.path.join(BENCHMARKS_ROOT, 'FIND-Net/pretrained_models/FINDNet/checkpoint.pt'),
        'type': 'image_domain',
        'color': '#377EB8',  # Blue
    },
    'InDuDoNet': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet'),
        'checkpoint': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet/pretrained_model/InDuDoNet_latest.pt'),
        'type': 'dual_domain',
        'color': '#4DAF4A',  # Green
    },
    'InDuDoNet+': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet_plus'),
        'checkpoint': os.path.join(BENCHMARKS_ROOT, 'InDuDoNet_plus/pretrained_model/InDuDoNet+_latest.pt'),
        'type': 'dual_domain_nmar',
        'color': '#984EA3',  # Purple
    },
    'MEPNet': {
        'benchmark_dir': os.path.join(BENCHMARKS_ROOT, 'MEPNet'),
        'checkpoint': os.path.join(BENCHMARKS_ROOT, 'MEPNet/pretrained_model/V320/MEPNet_latest.pt'),
        'type': 'dual_domain',
        'color': '#FF7F00',  # Orange
    },
    'SGA-MARN': {
        'benchmark_dir': DCGAN_ROOT,
        'checkpoint': os.path.join(DCGAN_ROOT, 'training_checkpoints/finetuned_spineweb_epoch_19.pth'),
        'type': 'ngswin',
        'color': '#A65628',  # Brown
    },
    'TransMAR-GAN': {
        'benchmark_dir': DCGAN_ROOT,
        'checkpoint': os.path.join(DCGAN_ROOT, 'combined_results/run_20251202_211759/checkpoints/best_model.pth'),
        'type': 'ngswin',
        'color': '#F781BF',  # Pink (Ours)
    },
}

# ═══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Benchmark Comparison on SynDeepLesion")
parser.add_argument("--data_path", type=str, default="/home/Drive-D/SynDeepLesion/",
                    help='Path to SynDeepLesion dataset')
parser.add_argument("--save_path", type=str, default="./benchmark_comparison_results/",
                    help='Path to save results')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--num_samples", type=int, default=50, help='Number of random test samples')
parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
parser.add_argument("--zoom_size", type=int, default=80, help='Size of zoomed region')
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
    log_file = os.path.join(log_dir, f'benchmark_comparison_{timestamp}.log')
    
    logger = logging.getLogger('BenchmarkComparison')
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

def normalize_image(data, minmax):
    """Normalize image to [0, 1] range"""
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data

def normalize_image_255(data, minmax):
    """Normalize image to [0, 255] range for DICDNet/FIND-Net"""
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data

def normalize_sinogram(data, minmax):
    """Normalize sinogram"""
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data / data_max
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data

def calculate_metrics(gt_np, pred_np):
    """Calculate PSNR and SSIM metrics"""
    # Ensure both are in [0, 1] range
    gt_np = np.clip(gt_np, 0, 1)
    pred_np = np.clip(pred_np, 0, 1)
    
    psnr_val = psnr(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim(gt_np, pred_np, data_range=1.0)
    
    return {'PSNR': psnr_val, 'SSIM': ssim_val}

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
    
    # Clamp to valid range
    y1 = max(0, center_y - half)
    y2 = min(h, center_y + half)
    x1 = max(0, center_x - half)
    x2 = min(w, center_x + half)
    
    return img[y1:y2, x1:x2], (y1, y2, x1, x2)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_test_sample(data_path, test_mask, imag_idx, mask_idx=0):
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
    
    # Try to load sinogram data if available
    try:
        Sma = file['ma_sinogram'][()]
        SLI = file['LI_sinogram'][()]
        Tr = file['metal_trace'][()]
        has_sinogram = True
    except:
        Sma, SLI, Tr = None, None, None
        has_sinogram = False
    file.close()
    
    # Get mask and resize to 416x416
    M512 = test_mask[:, :, mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    
    return {
        'Xgt': Xgt,
        'Xma': Xma,
        'XLI': XLI,
        'Mask': M,
        'Sma': Sma,
        'SLI': SLI,
        'Tr': Tr,
        'has_sinogram': has_sinogram,
        'imag_idx': imag_idx,
        'mask_idx': mask_idx,
    }

# ═══════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════

class ModelArgs:
    """Args class for model initialization"""
    pass

def cleanup_modules():
    """Remove cached benchmark modules to avoid import conflicts"""
    # Use list() to avoid dictionary changed size during iteration
    modules_to_remove = [key for key in list(sys.modules.keys()) 
                         if key.startswith('network') or key.startswith('utils') or 
                         key.startswith('deeplesion') or key.startswith('Model') or
                         key.startswith('my_model')]
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]
    
    # Remove benchmark paths from sys.path to avoid conflicts
    paths_to_remove = [p for p in sys.path if 'benchmarks' in p]
    for p in paths_to_remove:
        if p in sys.path:
            sys.path.remove(p)

def load_dicdnet(config, device, logger):
    """Load DICDNet model"""
    logger.info("  Loading DICDNet...")
    cleanup_modules()  # Clean up any cached modules
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    from dicdnet import DICDNet
    
    # Create args object with required params
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

def load_findnet(config, device, logger):
    """Load FIND-Net model"""
    logger.info("  Loading FIND-Net...")
    cleanup_modules()  # Clean up any cached modules
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    from Model.findnet import FINDNet
    
    # Create args object with required params
    args = ModelArgs()
    args.num_M = 32
    args.num_Q = 32
    args.T = 3
    args.S = 10
    args.etaM = 1
    args.etaX = 5
    
    model = FINDNet(args).to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    os.chdir(_original_cwd)
    
    return model

def load_indudonet(config, device, logger):
    """Load InDuDoNet model"""
    logger.info("  Loading InDuDoNet...")
    cleanup_modules()  # Clean up any cached modules
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from odl.util.npy_compat import AVOID_UNNECESSARY_COPY
    globals()['AVOID_UNNECESSARY_COPY'] = AVOID_UNNECESSARY_COPY
    
    from network.indudonet import InDuDoNet
    
    # Create args object with required params
    args = ModelArgs()
    args.num_channel = 32
    args.T = 4
    args.S = 10
    args.eta1 = 1
    args.eta2 = 5
    args.alpha = 0.5  # Required by InDuDoNet
    
    model = InDuDoNet(args).to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    os.chdir(_original_cwd)
    
    return model

def load_indudonet_plus(config, device, logger):
    """Load InDuDoNet+ model"""
    logger.info("  Loading InDuDoNet+...")
    cleanup_modules()  # Clean up any cached modules
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from odl.util.npy_compat import AVOID_UNNECESSARY_COPY
    globals()['AVOID_UNNECESSARY_COPY'] = AVOID_UNNECESSARY_COPY
    
    from network.indudonet_plus import InDuDoNet_plus
    
    # Create args object with required params
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

def load_mepnet(config, device, logger):
    """Load MEPNet model (V320 sparse-view variant)"""
    logger.info("  Loading MEPNet...")
    cleanup_modules()  # Clean up any cached modules
    benchmark_dir = config['benchmark_dir']
    sys.path.insert(0, benchmark_dir)
    
    _original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    import odl
    from odl.util.npy_compat import AVOID_UNNECESSARY_COPY
    globals()['AVOID_UNNECESSARY_COPY'] = AVOID_UNNECESSARY_COPY
    
    from network.mepnet import MEPNet
    from utils.build_gemotry import initialization, build_gemotry
    
    # Build CT geometry for MEPNet (needed for sparse-view)
    param = initialization()
    ray_trafo, FBPOper, op_norm = build_gemotry(param)
    
    # Create args object with required params (V320 variant)
    args = ModelArgs()
    args.num_channel = 32
    args.T = 4
    args.S = 10
    args.eta1 = 1
    args.eta2 = 5
    args.alpha = 0.5
    args.test_proj = config.get('test_proj', 320)  # V320 uses 320 projections
    
    model = MEPNet(args).to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    # Load with strict=False due to F_Conv buffer naming differences (.filter vs .weight)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        logger.debug(f"    MEPNet: {len(missing)} expected missing keys (F_Conv buffers)")
    if unexpected:
        logger.debug(f"    MEPNet: {len(unexpected)} unexpected keys (checkpoint .filter params)")
    model.eval()
    os.chdir(_original_cwd)
    
    # Return model and geometry operators (needed for sparse-view inference)
    return model, ray_trafo, FBPOper, config.get('test_proj', 320)

def load_ngswin(config, device, logger, model_name):
    """Load NGswin-based model (SGA-MARN or TransMAR-GAN)"""
    logger.info(f"  Loading {model_name}...")
    cleanup_modules()  # Clean up any cached modules
    
    _original_cwd = os.getcwd()
    
    # Remove ALL utils modules from cache - this is critical
    modules_to_remove = [k for k in list(sys.modules.keys()) if k == 'utils' or k.startswith('utils.')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # Also remove my_model modules to ensure fresh import
    modules_to_remove = [k for k in list(sys.modules.keys()) if k.startswith('my_model')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # Remove benchmark paths from sys.path
    paths_to_remove = [p for p in sys.path if 'benchmarks' in p]
    for p in paths_to_remove:
        sys.path.remove(p)
    
    # Ensure DCGAN_ROOT is in sys.path for utils import - must be first!
    # Remove and re-add to ensure it's at the front
    if DCGAN_ROOT in sys.path:
        sys.path.remove(DCGAN_ROOT)
    sys.path.insert(0, DCGAN_ROOT)
    
    # Change to DCGAN_ROOT so that relative imports work
    os.chdir(DCGAN_ROOT)
    
    # Debug: check sys.path
    logger.debug(f"    sys.path[0:3]: {sys.path[0:3]}")
    logger.debug(f"    cwd: {os.getcwd()}")
    
    # Now import - this should work with DCGAN_ROOT in sys.path
    import importlib
    import utils.etc_utils  # Force the utils module to load from DCGAN_ROOT
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
    
    # Extract the final output from ListX (same as test_DICDNet.py)
    output = ListX[-1]
    out_np = output.squeeze().cpu().numpy() / 255.0
    return np.clip(out_np, 0, 1)

def run_findnet(model, sample, device):
    """Run FIND-Net inference - matching working test script EXACTLY
    
    FIND-Net is IMAGE-DOMAIN ONLY. It takes:
    - Xma: metal-affected image (normalized to [0, 255])
    - XLI: linear interpolation image (normalized to [0, 255])
    - M: non-metal mask (1 - metal_mask)
    
    Returns image-domain output with same post-processing as InDuDoNet.
    """
    # Normalize to [0, 255] range (same as working test script)
    def normalize_findnet(data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = data * 255.0
        data = data.astype(np.float32)
        data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
        return data
    
    Xma = normalize_findnet(sample['Xma'], image_get_minmax())
    XLI = normalize_findnet(sample['XLI'], image_get_minmax())
    
    # Process mask - FIND-Net uses non_mask (1 - metal_mask)
    Mask = sample['Mask'].astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    non_mask = 1 - Mask
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    non_mask_t = torch.Tensor(non_mask).to(device)
    
    with torch.no_grad():
        # FIND-Net forward: model(Xma, XLI, non_mask) -> X0, ListX, ListA
        X0, ListX, ListA = model(Xma_t, XLI_t, non_mask_t)
    
    # Post-process exactly as working test script:
    # Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    # Xoutnorm = Xoutclip / 0.5
    Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    Xoutnorm = Xoutclip / 0.5
    
    out_np = Xoutnorm.squeeze().cpu().numpy()
    return np.clip(out_np, 0, 1)

def run_indudonet(model, sample, device):
    """Run InDuDoNet inference - matching working test script exactly"""
    # Normalize to [0, 255] range (same as working test script)
    def normalize_image_indudonet(data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 255.0
        data = data.astype(np.float32)
        data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
        return data
    
    def normalize_sinogram_indudonet(data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 255.0
        data = data.astype(np.float32)
        data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
        return data
    
    Xma = normalize_image_indudonet(sample['Xma'], image_get_minmax())
    XLI = normalize_image_indudonet(sample['XLI'], image_get_minmax())
    
    # Use sinograms from data
    Sma = normalize_sinogram_indudonet(sample['Sma'], proj_get_minmax())
    SLI = normalize_sinogram_indudonet(sample['SLI'], proj_get_minmax())
    
    # Process mask
    Mask = sample['Mask'].astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    
    # Process metal trace (Tr = 1 - metal_trace as in working script)
    if sample['Tr'] is not None:
        Tr = 1 - sample['Tr'].astype(np.float32)
        Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)), 0)
    else:
        Tr = np.ones_like(Sma)
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    Mask_t = torch.Tensor(Mask).to(device)
    Sma_t = torch.Tensor(Sma).to(device)
    SLI_t = torch.Tensor(SLI).to(device)
    Tr_t = torch.Tensor(Tr).to(device)
    
    with torch.no_grad():
        ListX, ListS, ListYS = model(Xma_t, XLI_t, Mask_t, Sma_t, SLI_t, Tr_t)
    
    # Post-process exactly as working test script:
    # Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    # Xoutnorm = Xoutclip / 0.5
    Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    Xoutnorm = Xoutclip / 0.5
    
    out_np = Xoutnorm.squeeze().cpu().numpy()
    return np.clip(out_np, 0, 1)

def run_indudonet_plus(model, sample, device):
    """Run InDuDoNet+ inference (with NMAR prior) - matching working test script EXACTLY"""
    import scipy.io as sio
    import scipy.ndimage.filters
    from sklearn.cluster import k_means
    
    # Load Gaussian filter for NMAR prior (same as working test script)
    smFilter_path = os.path.join(BENCHMARKS_ROOT, 'InDuDoNet_plus/deeplesion/gaussianfilter.mat')
    smFilter = sio.loadmat(smFilter_path)['smFilter']
    
    # NMAR prior parameters (same as working test script)
    miuAir = 0
    miuWater = 0.192
    starpoint = np.zeros([3, 1])
    starpoint[0] = miuAir
    starpoint[1] = miuWater
    starpoint[2] = 2 * miuWater
    
    def nmarprior(im, threshWater, threshBone, miuAir, miuWater, smFilter):
        """Compute NMAR prior image (exact copy from working test script)"""
        imSm = scipy.ndimage.filters.convolve(im, smFilter, mode='nearest')
        priorimgHU = imSm.copy()
        priorimgHU[imSm <= threshWater] = miuAir
        h, w = imSm.shape[0], imSm.shape[1]
        priorimgHUvector = np.reshape(priorimgHU, h*w)
        region1_1d = np.where(priorimgHUvector > threshWater)
        region2_1d = np.where(priorimgHUvector < threshBone)
        region_1d = np.intersect1d(region1_1d, region2_1d)
        priorimgHUvector[region_1d] = miuWater
        priorimgHU = np.reshape(priorimgHUvector, (h, w))
        return priorimgHU
    
    def nmar_prior(XLI, M):
        """Compute NMAR prior from LI image and mask (exact copy from working test script)"""
        XLI_copy = XLI.copy()
        XLI_copy[M == 1] = 0.192
        h, w = XLI_copy.shape[0], XLI_copy.shape[1]
        im1d = XLI_copy.reshape(h * w, 1)
        best_centers, labels, best_inertia = k_means(im1d, n_clusters=3, init=starpoint, max_iter=300)
        threshBone2 = np.min(im1d[labels == 2])
        threshBone2 = np.max([threshBone2, 1.2 * miuWater])
        threshWater2 = np.min(im1d[labels == 1])
        imPriorNMAR = nmarprior(XLI_copy, threshWater2, threshBone2, miuAir, miuWater, smFilter)
        return imPriorNMAR
    
    # Normalize to [0, 255] range (same as working test script)
    def normalize_image_indudonet(data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 255.0
        data = data.astype(np.float32)
        data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
        return data
    
    def normalize_sinogram_indudonet(data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 255.0
        data = data.astype(np.float32)
        data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
        return data
    
    # Compute NMAR prior BEFORE normalization (same as working test script)
    # Handle mask format: convert to binary 0/1
    M = sample['Mask'].copy()
    if M.max() > 1:
        M = (M > 0.5).astype(np.float32)
    else:
        M = (M > 0.5).astype(np.float32)
    
    Xprior = nmar_prior(sample['XLI'], M)
    
    # Now normalize everything to [0, 255]
    Xma = normalize_image_indudonet(sample['Xma'], image_get_minmax())
    XLI = normalize_image_indudonet(sample['XLI'], image_get_minmax())
    Xprior = normalize_image_indudonet(Xprior, image_get_minmax())
    
    # Use sinograms from data
    Sma = normalize_sinogram_indudonet(sample['Sma'], proj_get_minmax())
    SLI = normalize_sinogram_indudonet(sample['SLI'], proj_get_minmax())
    
    if sample['Tr'] is not None:
        Tr = 1 - sample['Tr'].astype(np.float32)
        Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)), 0)
    else:
        Tr = np.ones_like(Sma)
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    Xprior_t = torch.Tensor(Xprior).to(device)
    Sma_t = torch.Tensor(Sma).to(device)
    SLI_t = torch.Tensor(SLI).to(device)
    Tr_t = torch.Tensor(Tr).to(device)
    
    with torch.no_grad():
        # forward(self, Xma, XLI, Sma, SLI, Tr, Xprior)
        ListX, ListS, ListYS = model(Xma_t, XLI_t, Sma_t, SLI_t, Tr_t, Xprior_t)
    
    # Post-process exactly as working test script
    Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    Xoutnorm = Xoutclip / 0.5
    
    out_np = Xoutnorm.squeeze().cpu().numpy()
    return np.clip(out_np, 0, 1)

def run_mepnet(model, sample, device, ray_trafo=None, FBPOper=None, test_proj=320):
    """Run MEPNet inference (V320 sparse-view variant) - matching working test script EXACTLY
    
    MEPNet is a DUAL-DOMAIN model for sparse-view CT metal artifact reduction.
    V320 variant uses 320 projections sampled from 640 full projections.
    """
    # Normalization functions matching test script (to [0, 255] range)
    def normalize_image_mepnet(data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 255.0
        data = data.astype(np.float32)
        data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
        return data
    
    def normalize_sinogram_mepnet(data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 255.0
        data = data.astype(np.float32)
        data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
        return data
    
    # MEPNet V320 uses sparse-view sampling
    Sma_full = sample['Sma']
    SLI_full = sample['SLI']
    TrMAR = sample['Tr']  # Metal trace from data
    
    # Sparse-view sampling (V320 uses 320 projections from 640)
    Np, Nb = Sma_full.shape
    D = np.zeros((Np, Nb), dtype=float)
    factor = Np // test_proj  # 640 // 320 = 2
    D[::factor, :] = 1
    DI = 1 - D
    
    # Apply sparse-view sampling
    Smasp = D * Sma_full
    SLIsp = D * SLI_full
    
    # Reconstruct sparse-view images using FBP
    if FBPOper is not None:
        Xmasp = np.asarray(FBPOper(Smasp))
        XLIsp = np.asarray(FBPOper(SLIsp))
    else:
        # Fallback to original images if no FBP operator
        Xmasp = sample['Xma']
        XLIsp = sample['XLI']
    
    # Normalize images to [0, 255] (same as test script)
    Xma = normalize_image_mepnet(Xmasp, image_get_minmax())
    XLI = normalize_image_mepnet(XLIsp, image_get_minmax())
    
    # Normalize sinograms to [0, 255] (same as test script)
    Sma = normalize_sinogram_mepnet(Smasp, proj_get_minmax())
    SLI = normalize_sinogram_mepnet(SLIsp, proj_get_minmax())
    
    # Create trace mask (data consistency region) for sparse-view
    # Combine metal trace with sparse-view interpolation mask
    if TrMAR is not None:
        TrI_bool = np.logical_or(TrMAR, DI)
    else:
        TrI_bool = DI.astype(bool)
    TrI = np.zeros((Np, Nb), dtype=float)
    TrI[TrI_bool == True] = 1
    TrDC = 1 - TrI
    Tr = TrDC.astype(np.float32)
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)), 0)
    
    # Process mask
    Mask = sample['Mask'].astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    
    Xma_t = torch.Tensor(Xma).to(device)
    XLI_t = torch.Tensor(XLI).to(device)
    Mask_t = torch.Tensor(Mask).to(device)
    Sma_t = torch.Tensor(Sma).to(device)
    SLI_t = torch.Tensor(SLI).to(device)
    Tr_t = torch.Tensor(Tr).to(device)
    
    with torch.no_grad():
        # MEPNet forward: (Xma, XLI, M, Sma, SLI, Tr) -> ListX, ListYS
        ListX, ListYS = model(Xma_t, XLI_t, Mask_t, Sma_t, SLI_t, Tr_t)
    
    # Post-process exactly as working test script:
    # Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    # Xoutnorm = Xoutclip / 0.5
    Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    Xoutnorm = Xoutclip / 0.5
    
    out_np = Xoutnorm.squeeze().cpu().numpy()
    return np.clip(out_np, 0, 1)

def run_sgamarn(model, sample, device):
    """Run SGA-MARN inference - matching working testt.py script
    
    NGswin-based model (trained with model.py / DCGAN discriminator).
    Uses Generator wrapper class with NGswin as self.main.
    
    From TestDataset (Dataset.py):
        Xmaclip = np.clip(Xma, 0, 1)   # Clip raw data to [0, 1]
        O = (Xmaclip * 2) - 1          # Convert to [-1, 1]
    
    Input: Raw metal-affected image (may have values outside [0, 1])
    Model expects: [-1, 1] range
    Output: [-1, 1] range -> convert to [0, 1]
    """
    # CRITICAL: Clip to [0, 1] first, exactly like TestDataset does!
    Xma = np.clip(sample['Xma'].astype(np.float32), 0, 1)
    
    # Convert to tensor [1, 1, H, W] - NGswin handles any size with padding
    Xma_t = torch.Tensor(Xma).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 416, 416]
    
    # Normalize to [-1, 1] as expected by the model: (Xmaclip * 2) - 1
    Xma_t = Xma_t * 2 - 1
    
    with torch.no_grad():
        output = model(Xma_t)
    
    # Denormalize from [-1, 1] to [0, 1]: (output + 1) / 2
    output = (output + 1) / 2
    
    out_np = output.squeeze().cpu().numpy()
    return np.clip(out_np, 0, 1)

def run_transmar(model, sample, device):
    """Run TransMAR-GAN inference - matching train_combined.py test evaluation
    
    TransMAR-GAN uses NGswin generator (trained with train_combined.py / MS-PatchGAN).
    Generator class wraps NGswin as self.main.
    
    From train_combined.py validation:
        - Data comes from MARValDataset/SpineWebTrainDataset in [-1, 1] range
        - denorm(x) = (x + 1.0) / 2.0
    
    From TestDataset (Dataset.py):
        Xmaclip = np.clip(Xma, 0, 1)   # Clip raw data to [0, 1]
        O = (Xmaclip * 2) - 1          # Convert to [-1, 1]
    
    Input: Raw metal-affected image (may have values outside [0, 1])
    Model expects: [-1, 1] range
    Output: [-1, 1] range -> convert to [0, 1]
    """
    # CRITICAL: Clip to [0, 1] first, exactly like TestDataset does!
    Xma = np.clip(sample['Xma'].astype(np.float32), 0, 1)
    
    # Convert to tensor [1, 1, H, W] - NGswin handles any size with padding
    Xma_t = torch.Tensor(Xma).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 416, 416]
    
    # Normalize to [-1, 1] as expected by the model: (Xmaclip * 2) - 1
    Xma_t = Xma_t * 2 - 1
    
    with torch.no_grad():
        output = model(Xma_t)
    
    # Denormalize from [-1, 1] to [0, 1]: (output + 1) / 2
    output = (output + 1) / 2
    
    out_np = output.squeeze().cpu().numpy()
    return np.clip(out_np, 0, 1)

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def save_individual_image(img, save_path, cmap='gray'):
    """Save individual image without borders"""
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_composite_figure(sample_idx, corrupted, gt, outputs, zoom_coords, save_path, model_names):
    """
    Create composite figure with all models - MINIMAL WHITESPACE VERSION.
    
    Layout (2 columns):
        Row 0: [Corrupted] [Corrupted Zoomed]
        Row 1: [Ground Truth] [Ground Truth (Zoomed)]
        Row 2+: [Model Output] [Model Zoomed] for each model
    
    - Corrupted has RED box (artifact)
    - GT and all model outputs have GREEN box (clean/corrected)
    - Zoomed images show the metal-affected region with matching colored box
    - No titles, no metrics, no labels - clean images only
    - ALL images use SAME grayscale window based on GT for consistent appearance
    """
    n_models = len(outputs)
    n_rows = 2 + n_models  # Corrupted row + GT row + model rows
    
    # Minimal figure size - tighter spacing
    fig, axes = plt.subplots(n_rows, 2, figsize=(4, 1.8 * n_rows))
    
    y1, y2, x1, x2 = zoom_coords
    zoom_h, zoom_w = y2 - y1, x2 - x1
    
    # Use GT statistics for consistent windowing across ALL images
    # This ensures all images appear with the same grayscale intensity
    gt_clipped = np.clip(gt, 0, 1)
    vmin_global = 0.0
    vmax_global = np.percentile(gt_clipped, 99.5)  # Use 99.5th percentile of GT as max
    vmax_global = min(max(vmax_global, 0.5), 1.0)  # Ensure reasonable range [0.5, 1.0]
    
    # Clip all images to ensure consistent display
    corrupted_disp = np.clip(corrupted, 0, 1)
    gt_disp = np.clip(gt, 0, 1)
    
    # Box around metal region in zoomed images - tighter box centered on metal
    metal_box_margin = zoom_h // 8  # Tighter margin
    
    # Row 0: Corrupted (RED box - shows artifact)
    axes[0, 0].imshow(corrupted_disp, cmap='gray', vmin=vmin_global, vmax=vmax_global)
    axes[0, 0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='red', facecolor='none'))
    axes[0, 0].axis('off')
    axes[0, 0].set_aspect('equal')
    
    # Corrupted Zoomed - red box around metal region
    axes[0, 1].imshow(corrupted_disp[y1:y2, x1:x2], cmap='gray', vmin=vmin_global, vmax=vmax_global)
    axes[0, 1].add_patch(Rectangle((metal_box_margin, metal_box_margin), 
                                    zoom_w - 2*metal_box_margin, zoom_h - 2*metal_box_margin,
                                    linewidth=2, edgecolor='red', facecolor='none'))
    axes[0, 1].axis('off')
    axes[0, 1].set_aspect('equal')
    
    # Row 1: Ground Truth (GREEN box)
    axes[1, 0].imshow(gt_disp, cmap='gray', vmin=vmin_global, vmax=vmax_global)
    axes[1, 0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='lime', facecolor='none'))
    axes[1, 0].axis('off')
    axes[1, 0].set_aspect('equal')
    
    # GT Zoomed - green box
    axes[1, 1].imshow(gt_disp[y1:y2, x1:x2], cmap='gray', vmin=vmin_global, vmax=vmax_global)
    axes[1, 1].add_patch(Rectangle((metal_box_margin, metal_box_margin), 
                                    zoom_w - 2*metal_box_margin, zoom_h - 2*metal_box_margin,
                                    linewidth=2, edgecolor='lime', facecolor='none'))
    axes[1, 1].axis('off')
    axes[1, 1].set_aspect('equal')
    
    # Model outputs (rows 2+) - all GREEN boxes
    for i, (model_name, output) in enumerate(outputs.items()):
        row = i + 2
        
        # Clip output to [0, 1] for consistent display
        output_disp = np.clip(output, 0, 1)
        
        # Output with GREEN box - use SAME vmin/vmax as GT
        axes[row, 0].imshow(output_disp, cmap='gray', vmin=vmin_global, vmax=vmax_global)
        axes[row, 0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, 
                                        linewidth=2, edgecolor='lime', facecolor='none'))
        axes[row, 0].axis('off')
        axes[row, 0].set_aspect('equal')
        
        # Output Zoomed with green box - use SAME vmin/vmax as GT
        axes[row, 1].imshow(output_disp[y1:y2, x1:x2], cmap='gray', vmin=vmin_global, vmax=vmax_global)
        axes[row, 1].add_patch(Rectangle((metal_box_margin, metal_box_margin), 
                                        zoom_w - 2*metal_box_margin, zoom_h - 2*metal_box_margin,
                                        linewidth=2, edgecolor='lime', facecolor='none'))
        axes[row, 1].axis('off')
        axes[row, 1].set_aspect('equal')
    
    # Minimal whitespace - very tight layout
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # === SAVE INDIVIDUAL BOXED IMAGES ===
    # Create directory for individual boxed images
    boxed_dir = os.path.join(os.path.dirname(save_path), f'sample_{sample_idx:04d}_boxed')
    os.makedirs(boxed_dir, exist_ok=True)
    
    def save_single_boxed_image(img, box_color, filename, is_zoomed=False):
        """Save a single image with colored box"""
        fig_single, ax_single = plt.subplots(1, 1, figsize=(4, 4))
        ax_single.imshow(img, cmap='gray', vmin=vmin_global, vmax=vmax_global)
        if is_zoomed:
            ax_single.add_patch(Rectangle((metal_box_margin, metal_box_margin), 
                                          zoom_w - 2*metal_box_margin, zoom_h - 2*metal_box_margin,
                                          linewidth=3, edgecolor=box_color, facecolor='none'))
        else:
            ax_single.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, 
                                          linewidth=3, edgecolor=box_color, facecolor='none'))
        ax_single.axis('off')
        ax_single.set_aspect('equal')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(boxed_dir, filename), dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig_single)
    
    # Save corrupted (red box)
    save_single_boxed_image(corrupted_disp, 'red', 'corrupted_boxed.png', is_zoomed=False)
    save_single_boxed_image(corrupted_disp[y1:y2, x1:x2], 'red', 'corrupted_zoomed_boxed.png', is_zoomed=True)
    
    # Save GT (green box)
    save_single_boxed_image(gt_disp, 'lime', 'ground_truth_boxed.png', is_zoomed=False)
    save_single_boxed_image(gt_disp[y1:y2, x1:x2], 'lime', 'ground_truth_zoomed_boxed.png', is_zoomed=True)
    
    # Save each model output (green box)
    for model_name, output in outputs.items():
        output_disp = np.clip(output, 0, 1)
        safe_name = model_name.replace('+', '_plus').replace('-', '_')
        save_single_boxed_image(output_disp, 'lime', f'{safe_name}_boxed.png', is_zoomed=False)
        save_single_boxed_image(output_disp[y1:y2, x1:x2], 'lime', f'{safe_name}_zoomed_boxed.png', is_zoomed=True)

def create_intensity_profile_figure(sample_idx, corrupted, gt, outputs, mask, save_path, model_configs):
    """
    Create intensity profile figure (horizontal and vertical lines through metal).
    Two graphs centered: horizontal and vertical intensity profiles.
    """
    # Find metal center
    center_y, center_x = find_metal_center(mask)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Top row: Images with lines (3 images)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(corrupted, cmap='gray', vmin=0, vmax=1)
    ax1.axhline(y=center_y, color='red', linewidth=1.5, label='H-line')
    ax1.axvline(x=center_x, color='cyan', linewidth=1.5, label='V-line')
    ax1.set_title('Corrupted', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt, cmap='gray', vmin=0, vmax=1)
    ax2.axhline(y=center_y, color='red', linewidth=1.5)
    ax2.axvline(x=center_x, color='cyan', linewidth=1.5)
    ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Show one model output as reference
    first_model = list(outputs.keys())[0]
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(outputs[first_model], cmap='gray', vmin=0, vmax=1)
    ax3.axhline(y=center_y, color='red', linewidth=1.5)
    ax3.axvline(x=center_x, color='cyan', linewidth=1.5)
    ax3.set_title(f'{first_model} Output', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Bottom row: Two centered graphs (use positions 4 and 5, skip 6)
    # Horizontal profile - centered left
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.plot(corrupted[center_y, :], color='blue', linewidth=1.5, label='Corrupted', alpha=0.7)
    ax4.plot(gt[center_y, :], color='black', linewidth=2, linestyle='--', label='GT')
    
    for model_name, output in outputs.items():
        ax4.plot(output[center_y, :], color=model_configs[model_name]['color'], 
                linewidth=1.2, label=model_name, alpha=0.8)
    
    # Highlight metal region
    metal_cols = np.where(mask[center_y, :] > 0.5)[0]
    if len(metal_cols) > 0:
        ax4.axvspan(metal_cols[0], metal_cols[-1], alpha=0.2, color='yellow', label='Metal')
    
    ax4.set_xlabel('X Position (pixels)', fontsize=11)
    ax4.set_ylabel('Intensity', fontsize=11)
    ax4.set_title('Horizontal Intensity Profile', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=7, ncol=2)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Vertical profile - centered right
    ax5 = fig.add_subplot(2, 2, 4)
    ax5.plot(corrupted[:, center_x], color='blue', linewidth=1.5, label='Corrupted', alpha=0.7)
    ax5.plot(gt[:, center_x], color='black', linewidth=2, linestyle='--', label='GT')
    
    for model_name, output in outputs.items():
        ax5.plot(output[:, center_x], color=model_configs[model_name]['color'], 
                linewidth=1.2, label=model_name, alpha=0.8)
    
    # Highlight metal region
    metal_rows = np.where(mask[:, center_x] > 0.5)[0]
    if len(metal_rows) > 0:
        ax5.axvspan(metal_rows[0], metal_rows[-1], alpha=0.2, color='yellow', label='Metal')
    
    ax5.set_xlabel('Y Position (pixels)', fontsize=11)
    ax5.set_ylabel('Intensity', fontsize=11)
    ax5.set_title('Vertical Intensity Profile', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=7, ncol=2)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    # Removed: Error comparison plot (was ax6)
    
    plt.suptitle(f'Sample {sample_idx}: Intensity Line Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_sinogram_profile_figure(sample_idx, corrupted, gt, outputs, save_path, model_configs, device):
    """
    Create sinogram line profile figure (detector profile at fixed angle + angle profile at fixed detector).
    Similar to the first image the user shared.
    """
    if not HAS_TORCH_RADON:
        return None
    
    # Create Radon projector for the image size
    img_size = corrupted.shape[0]
    num_angles = 180
    angles = np.linspace(0, np.pi, num_angles, dtype=np.float32)
    
    try:
        projector = Radon(img_size, angles)
    except Exception as e:
        print(f"Failed to create Radon projector: {e}")
        return None
    
    # Compute sinograms for all images
    def compute_sinogram(img):
        img_t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sino = projector.forward(img_t)
        return sino.squeeze().cpu().numpy()
    
    sino_corrupted = compute_sinogram(corrupted)
    sino_gt = compute_sinogram(gt)
    
    sino_outputs = {}
    for model_name, output in outputs.items():
        sino_outputs[model_name] = compute_sinogram(output)
    
    # Get dimensions
    num_detectors = sino_gt.shape[1]
    center_detector = num_detectors // 2
    center_angle_idx = num_angles // 2  # 90 degrees
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: Detector Profile at fixed angle (e.g., 90°)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(sino_gt[center_angle_idx, :], color='green', linewidth=2, label='Clean (GT)')
    ax1.plot(sino_corrupted[center_angle_idx, :], color='blue', linewidth=1.5, alpha=0.7, label='Corrupted')
    
    for model_name, sino in sino_outputs.items():
        ax1.plot(sino[center_angle_idx, :], color=model_configs[model_name]['color'], 
                linewidth=1.2, label=model_name, alpha=0.8)
    
    ax1.set_xlabel('Detector Index', fontsize=11)
    ax1.set_ylabel('Projection Value', fontsize=11)
    ax1.set_title(f'Detector Profile at Angle {int(90)}° (Index {center_angle_idx})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Row 1: Angle Profile at fixed detector (center detector)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(sino_gt[:, center_detector], color='green', linewidth=2, label='Clean (GT)')
    ax2.plot(sino_corrupted[:, center_detector], color='blue', linewidth=1.5, alpha=0.7, label='Corrupted')
    
    for model_name, sino in sino_outputs.items():
        ax2.plot(sino[:, center_detector], color=model_configs[model_name]['color'], 
                linewidth=1.2, label=model_name, alpha=0.8)
    
    ax2.set_xlabel('Angle Index', fontsize=11)
    ax2.set_ylabel('Projection Value', fontsize=11)
    ax2.set_title(f'Angle Profile at Detector {center_detector} (Center)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Residual at fixed angle
    ax3 = fig.add_subplot(2, 2, 3)
    
    for model_name, sino in sino_outputs.items():
        residual = np.abs(sino[center_angle_idx, :] - sino_gt[center_angle_idx, :])
        ax3.plot(residual, color=model_configs[model_name]['color'], 
                linewidth=1.2, label=model_name, alpha=0.8)
    
    # Also show corrupted residual
    corrupted_residual = np.abs(sino_corrupted[center_angle_idx, :] - sino_gt[center_angle_idx, :])
    ax3.plot(corrupted_residual, color='blue', linewidth=1.5, alpha=0.5, linestyle='--', label='Corrupted')
    
    ax3.set_xlabel('Detector Index', fontsize=11)
    ax3.set_ylabel('Absolute Residual', fontsize=11)
    ax3.set_title(f'Residual at Angle {int(90)}°', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Per-Angle Mean Error
    ax4 = fig.add_subplot(2, 2, 4)
    
    for model_name, sino in sino_outputs.items():
        per_angle_error = np.mean(np.abs(sino - sino_gt), axis=1)
        mean_err = np.mean(per_angle_error)
        ax4.plot(per_angle_error, color=model_configs[model_name]['color'], 
                linewidth=1.2, label=f'{model_name} (Mean: {mean_err:.4f})', alpha=0.8)
    
    # Corrupted per-angle error
    corrupted_per_angle = np.mean(np.abs(sino_corrupted - sino_gt), axis=1)
    ax4.plot(corrupted_per_angle, color='blue', linewidth=1.5, alpha=0.5, linestyle='--', 
            label=f'Corrupted (Mean: {np.mean(corrupted_per_angle):.4f})')
    
    ax4.set_xlabel('Angle Index', fontsize=11)
    ax4.set_ylabel('Mean Absolute Error', fontsize=11)
    ax4.set_title('Per-Angle Mean Error', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sample {sample_idx}: Sinogram Line Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return sinograms for saving
    return {
        'sino_corrupted': sino_corrupted,
        'sino_gt': sino_gt,
        'sino_outputs': sino_outputs,
        'center_angle_idx': center_angle_idx,
        'center_detector': center_detector,
    }


def save_line_profile_data(sample_dir, corrupted, gt, outputs, mask, sinogram_data=None):
    """
    Save all line profile data as numpy arrays for manual figure creation.
    """
    profile_data_dir = os.path.join(sample_dir, 'profile_data')
    os.makedirs(profile_data_dir, exist_ok=True)
    
    # Find metal center for profiles
    center_y, center_x = find_metal_center(mask)
    
    # Save intensity profiles
    intensity_data = {
        'center_y': center_y,
        'center_x': center_x,
        'horizontal_corrupted': corrupted[center_y, :],
        'horizontal_gt': gt[center_y, :],
        'vertical_corrupted': corrupted[:, center_x],
        'vertical_gt': gt[:, center_x],
    }
    
    for model_name, output in outputs.items():
        intensity_data[f'horizontal_{model_name}'] = output[center_y, :]
        intensity_data[f'vertical_{model_name}'] = output[:, center_x]
        intensity_data[f'horizontal_error_{model_name}'] = np.abs(output[center_y, :] - gt[center_y, :])
        intensity_data[f'vertical_error_{model_name}'] = np.abs(output[:, center_x] - gt[:, center_x])
    
    np.savez(os.path.join(profile_data_dir, 'intensity_profiles.npz'), **intensity_data)
    
    # Save sinogram profiles if available
    if sinogram_data is not None:
        sino_profile_data = {
            'center_angle_idx': sinogram_data['center_angle_idx'],
            'center_detector': sinogram_data['center_detector'],
            'sino_corrupted': sinogram_data['sino_corrupted'],
            'sino_gt': sinogram_data['sino_gt'],
        }
        
        for model_name, sino in sinogram_data['sino_outputs'].items():
            sino_profile_data[f'sino_{model_name}'] = sino
        
        np.savez(os.path.join(profile_data_dir, 'sinogram_profiles.npz'), **sino_profile_data)
    
    return profile_data_dir


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Convert save_path to absolute path to avoid issues with chdir during model loading
    abs_save_path = os.path.abspath(opt.save_path)
    save_root = os.path.join(abs_save_path, f'comparison_{timestamp}')
    
    # Create directories
    dirs = {
        'root': save_root,
        'individual': os.path.join(save_root, 'individual_images'),
        'composite': os.path.join(save_root, 'composite_figures'),
        'intensity_profiles': os.path.join(save_root, 'intensity_line_profiles'),
        'sinogram_profiles': os.path.join(save_root, 'sinogram_line_profiles'),
        'metrics': os.path.join(save_root, 'metrics'),
        'logs': os.path.join(save_root, 'logs'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(dirs['logs'])
    
    logger.info("=" * 70)
    logger.info("BENCHMARK COMPARISON ON SYNDEEPLESION DATASET")
    logger.info("=" * 70)
    logger.info(f"Save path: {save_root}")
    logger.info(f"Device: {device}")
    logger.info(f"Number of samples: {opt.num_samples}")
    logger.info(f"Models: {opt.models}")
    
    # Load test mask
    logger.info("\nLoading test mask...")
    test_mask = np.load(os.path.join(opt.data_path, 'testmask.npy'))
    logger.info(f"  Test mask shape: {test_mask.shape}")
    
    # Get number of available images
    txtdir = os.path.join(opt.data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    num_available = len(mat_files)
    logger.info(f"  Available test images: {num_available}")
    
    # Select random samples (same seed = same samples for all models)
    random.seed(opt.seed)
    selected_indices = random.sample(range(num_available), min(opt.num_samples, num_available))
    selected_indices.sort()
    logger.info(f"  Selected {len(selected_indices)} random samples (seed={opt.seed})")
    
    # Save selected indices
    np.save(os.path.join(save_root, 'selected_indices.npy'), selected_indices)
    
    # Get list of models to process
    active_models = [m for m in opt.models if m in MODEL_CONFIGS]
    all_metrics = {model_name: [] for model_name in active_models}
    successfully_loaded_models = []
    
    # Pre-create sample directories and compute zoom regions
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING SAMPLE DIRECTORIES")
    logger.info("=" * 70)
    
    sample_info = {}  # sample_i -> {sample_dir, zoom_coords, imag_idx}
    for sample_i, imag_idx in enumerate(selected_indices):
        sample_dir = os.path.join(dirs['individual'], f'sample_{sample_i:04d}_idx{imag_idx}')
        os.makedirs(sample_dir, exist_ok=True)
        sample_info[sample_i] = {'sample_dir': sample_dir, 'imag_idx': imag_idx}
    
    logger.info(f"  Created {len(sample_info)} sample directories")
    
    # ═══════════════════════════════════════════════════════════════
    # PROCESS EACH MODEL COMPLETELY BEFORE MOVING TO NEXT
    # ═══════════════════════════════════════════════════════════════
    
    for model_idx, model_name in enumerate(active_models):
        config = MODEL_CONFIGS[model_name]
        
        logger.info("\n" + "=" * 70)
        logger.info(f"MODEL {model_idx + 1}/{len(active_models)}: {model_name}")
        logger.info("=" * 70)
        
        if not os.path.exists(config['checkpoint']):
            logger.warning(f"  Checkpoint not found: {config['checkpoint']}")
            continue
        
        # Load model
        logger.info(f"  Loading model...")
        model = None
        mepnet_extras = None
        
        try:
            if model_name == 'DICDNet':
                model = load_dicdnet(config, device, logger)
            elif model_name == 'FIND-Net':
                model = load_findnet(config, device, logger)
            elif model_name == 'InDuDoNet':
                model = load_indudonet(config, device, logger)
            elif model_name == 'InDuDoNet+':
                model = load_indudonet_plus(config, device, logger)
            elif model_name == 'MEPNet':
                model, ray_trafo, FBPOper, test_proj = load_mepnet(config, device, logger)
                mepnet_extras = {'ray_trafo': ray_trafo, 'FBPOper': FBPOper, 'test_proj': test_proj}
            elif model_name in ['SGA-MARN', 'TransMAR-GAN']:
                model = load_ngswin(config, device, logger, model_name)
            
            logger.info(f"  ✓ Model loaded successfully")
            successfully_loaded_models.append(model_name)
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Process each sample: load data, run inference, save everything, then move on
        logger.info(f"  Processing {len(selected_indices)} samples...")
        model_psnrs = []
        model_ssims = []
        
        for sample_i, imag_idx in enumerate(tqdm(selected_indices, desc=f"  {model_name}")):
            sample_dir = sample_info[sample_i]['sample_dir']
            
            # Load sample data
            try:
                sample = load_test_sample(opt.data_path, test_mask, imag_idx, mask_idx=0)
            except Exception as e:
                logger.error(f"    Sample {sample_i}: Failed to load - {e}")
                continue
            
            corrupted = sample['Xma']
            gt = sample['Xgt']
            mask = sample['Mask']
            
            # Compute zoom region
            center_y, center_x = find_metal_center(mask)
            half = opt.zoom_size // 2
            y1, y2 = max(0, center_y - half), min(corrupted.shape[0], center_y + half)
            x1, x2 = max(0, center_x - half), min(corrupted.shape[1], center_x + half)
            
            # Save GT and corrupted (only on first model to avoid duplicates)
            if model_idx == 0:
                save_individual_image(corrupted, os.path.join(sample_dir, 'corrupted.png'))
                save_individual_image(corrupted[y1:y2, x1:x2], os.path.join(sample_dir, 'corrupted_zoomed.png'))
                save_individual_image(gt, os.path.join(sample_dir, 'ground_truth.png'))
                save_individual_image(gt[y1:y2, x1:x2], os.path.join(sample_dir, 'ground_truth_zoomed.png'))
                np.save(os.path.join(sample_dir, 'corrupted.npy'), corrupted)
                np.save(os.path.join(sample_dir, 'ground_truth.npy'), gt)
                np.save(os.path.join(sample_dir, 'mask.npy'), mask)
            
            # Run inference
            try:
                if model_name == 'DICDNet':
                    output = run_dicdnet(model, sample, device)
                elif model_name == 'FIND-Net':
                    output = run_findnet(model, sample, device)
                elif model_name == 'InDuDoNet':
                    output = run_indudonet(model, sample, device)
                elif model_name == 'InDuDoNet+':
                    output = run_indudonet_plus(model, sample, device)
                elif model_name == 'MEPNet':
                    output = run_mepnet(model, sample, device, 
                                       mepnet_extras['ray_trafo'], 
                                       mepnet_extras['FBPOper'], 
                                       mepnet_extras['test_proj'])
                elif model_name == 'SGA-MARN':
                    output = run_sgamarn(model, sample, device)
                elif model_name == 'TransMAR-GAN':
                    output = run_transmar(model, sample, device)
                
                # Calculate metrics
                metrics = calculate_metrics(gt, output)
                all_metrics[model_name].append({'sample_i': sample_i, 'imag_idx': imag_idx, **metrics})
                model_psnrs.append(metrics['PSNR'])
                model_ssims.append(metrics['SSIM'])
                
                # Save output image and numpy
                save_individual_image(output, os.path.join(sample_dir, f'{model_name}_output.png'))
                save_individual_image(output[y1:y2, x1:x2], os.path.join(sample_dir, f'{model_name}_zoomed.png'))
                np.save(os.path.join(sample_dir, f'{model_name}_output.npy'), output)
                
            except Exception as e:
                logger.error(f"    Sample {sample_i}: Inference failed - {e}")
                continue
        
        # Save metrics CSV for this model immediately
        model_metrics_path = os.path.join(dirs['metrics'], f'{model_name}_metrics.csv')
        with open(model_metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample', 'ImageIdx', 'PSNR', 'SSIM'])
            for m in all_metrics[model_name]:
                writer.writerow([m['sample_i'], m['imag_idx'], f"{m['PSNR']:.4f}", f"{m['SSIM']:.6f}"])
        
        # Print model summary
        if len(model_psnrs) > 0:
            logger.info(f"  ─────────────────────────────────────────")
            logger.info(f"  {model_name} COMPLETE")
            logger.info(f"  Samples processed: {len(model_psnrs)}")
            logger.info(f"  PSNR: {np.mean(model_psnrs):.2f} ± {np.std(model_psnrs):.2f}")
            logger.info(f"  SSIM: {np.mean(model_ssims):.4f} ± {np.std(model_ssims):.4f}")
            logger.info(f"  Metrics saved: {model_metrics_path}")
        
        # Free model memory
        del model
        if mepnet_extras:
            del mepnet_extras
        torch.cuda.empty_cache()
        logger.info(f"  ✓ Memory freed")
    
    # ═══════════════════════════════════════════════════════════════
    # CREATE COMPOSITE FIGURES (load saved numpy files)
    # ═══════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 70)
    logger.info("CREATING COMPOSITE FIGURES AND PROFILES")
    logger.info("=" * 70)
    logger.info(f"Models to include: {successfully_loaded_models}")
    
    for sample_i, imag_idx in enumerate(tqdm(selected_indices, desc="Creating figures")):
        sample_dir = sample_info[sample_i]['sample_dir']
        
        # Load saved numpy files
        try:
            corrupted = np.load(os.path.join(sample_dir, 'corrupted.npy'))
            gt = np.load(os.path.join(sample_dir, 'ground_truth.npy'))
            mask = np.load(os.path.join(sample_dir, 'mask.npy'))
        except Exception as e:
            logger.warning(f"  Sample {sample_i}: Could not load base images - {e}")
            continue
        
        # Load model outputs
        outputs = {}
        for model_name in successfully_loaded_models:
            output_path = os.path.join(sample_dir, f'{model_name}_output.npy')
            if os.path.exists(output_path):
                outputs[model_name] = np.load(output_path)
        
        if len(outputs) == 0:
            continue
        
        # Compute zoom region
        center_y, center_x = find_metal_center(mask)
        half = opt.zoom_size // 2
        y1, y2 = max(0, center_y - half), min(corrupted.shape[0], center_y + half)
        x1, x2 = max(0, center_x - half), min(corrupted.shape[1], center_x + half)
        zoom_coords = (y1, y2, x1, x2)
        
        # Create composite figure
        composite_path = os.path.join(dirs['composite'], f'sample_{sample_i:04d}_comparison.png')
        try:
            create_composite_figure(sample_i, corrupted, gt, outputs, zoom_coords, composite_path, successfully_loaded_models)
        except Exception as e:
            logger.warning(f"  Sample {sample_i}: Composite figure failed - {e}")
        
        # Create intensity profile figure
        intensity_profile_path = os.path.join(dirs['intensity_profiles'], f'sample_{sample_i:04d}_intensity_profiles.png')
        try:
            create_intensity_profile_figure(sample_i, corrupted, gt, outputs, mask, intensity_profile_path, MODEL_CONFIGS)
        except Exception as e:
            logger.warning(f"  Sample {sample_i}: Intensity profile failed - {e}")
        
        # Create sinogram profile figure (DISABLED - no sinograms in this dataset)
        # if HAS_TORCH_RADON:
        #     sinogram_profile_path = os.path.join(dirs['sinogram_profiles'], f'sample_{sample_i:04d}_sinogram_profiles.png')
        #     try:
        #         sinogram_data = create_sinogram_profile_figure(
        #             sample_i, corrupted, gt, outputs, sinogram_profile_path, MODEL_CONFIGS, device
        #         )
        #         # Save profile data
        #         save_line_profile_data(sample_dir, corrupted, gt, outputs, mask, sinogram_data)
        #     except Exception as e:
        #         logger.warning(f"  Sample {sample_i}: Sinogram profile failed - {e}")
        #         save_line_profile_data(sample_dir, corrupted, gt, outputs, mask, None)
        # else:
        #     save_line_profile_data(sample_dir, corrupted, gt, outputs, mask, None)
        
        # Save line profile data without sinogram
        save_line_profile_data(sample_dir, corrupted, gt, outputs, mask, None)
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    # Combined metrics CSV
    metrics_csv_path = os.path.join(dirs['metrics'], 'all_models_per_sample.csv')
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Sample', 'ImageIdx'] + [f'{m}_PSNR' for m in successfully_loaded_models] + [f'{m}_SSIM' for m in successfully_loaded_models]
        writer.writerow(header)
        
        for sample_i, imag_idx in enumerate(selected_indices):
            row = [sample_i, imag_idx]
            for model_name in successfully_loaded_models:
                model_metrics = [m for m in all_metrics[model_name] if m['sample_i'] == sample_i]
                if model_metrics:
                    row.append(f"{model_metrics[0]['PSNR']:.4f}")
                else:
                    row.append('N/A')
            for model_name in successfully_loaded_models:
                model_metrics = [m for m in all_metrics[model_name] if m['sample_i'] == sample_i]
                if model_metrics:
                    row.append(f"{model_metrics[0]['SSIM']:.6f}")
                else:
                    row.append('N/A')
            writer.writerow(row)
    
    # Summary table
    summary_path = os.path.join(dirs['metrics'], 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Mean_PSNR', 'Std_PSNR', 'Mean_SSIM', 'Std_SSIM', 'Num_Samples'])
        
        for model_name in successfully_loaded_models:
            if len(all_metrics[model_name]) > 0:
                psnrs = [m['PSNR'] for m in all_metrics[model_name]]
                ssims = [m['SSIM'] for m in all_metrics[model_name]]
                writer.writerow([
                    model_name,
                    f"{np.mean(psnrs):.4f}",
                    f"{np.std(psnrs):.4f}",
                    f"{np.mean(ssims):.6f}",
                    f"{np.std(ssims):.6f}",
                    len(psnrs)
                ])
                logger.info(f"  {model_name}: PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}, SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f} (n={len(psnrs)})")
    
    # Save config
    config_path = os.path.join(save_root, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'num_samples': opt.num_samples,
            'seed': opt.seed,
            'zoom_size': opt.zoom_size,
            'models_requested': opt.models,
            'models_processed': successfully_loaded_models,
            'selected_indices': selected_indices,
            'timestamp': timestamp,
        }, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results: {save_root}")
    logger.info(f"  individual_images/ - per-sample outputs (.png + .npy)")
    logger.info(f"  composite_figures/ - side-by-side comparisons")
    logger.info(f"  intensity_line_profiles/ - intensity profiles")
    logger.info(f"  sinogram_line_profiles/ - sinogram profiles")
    logger.info(f"  metrics/ - CSV files with PSNR/SSIM")

if __name__ == "__main__":
    main()
