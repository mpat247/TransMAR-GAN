#!/usr/bin/env python
"""
Quick contrast test - generate sample images with different vmax settings
"""
import os
import sys
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import PIL

# Add benchmark DICDNet path
BENCHMARK_DIR = '/home/grad/mppatel/Documents/DCGAN/benchmarks/DICDNet'
sys.path.insert(0, BENCHMARK_DIR)
_original_cwd = os.getcwd()
os.chdir(BENCHMARK_DIR)
from dicdnet import DICDNet
os.chdir(_original_cwd)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data

def load_test_sample(data_path, test_mask, imag_idx, mask_idx):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    
    abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
    gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
    
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    XLI = file['LI_CT'][()]
    file.close()
    
    M512 = test_mask[:, :, mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    
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

# Create output directory
output_dir = '/home/grad/mppatel/Documents/DCGAN/benchmark_outputs/contrast_test'
os.makedirs(output_dir, exist_ok=True)

# Load model
class Args:
    num_M = 32
    num_Q = 32
    T = 3
    S = 10
    etaM = 1
    etaX = 5

opt = Args()
model = DICDNet(opt).to(device)
model.load_state_dict(torch.load(os.path.join(BENCHMARK_DIR, "pretrain_model/DICDNet_latest.pt"), map_location=device))
model.eval()

# Load data
data_path = "/home/Drive-D/SynDeepLesion/"
test_mask = np.load(os.path.join(data_path, 'testmask.npy'))

# Test on a few samples
selected_indices = np.load('/home/grad/mppatel/Documents/DCGAN/inference_figure_outputs/selected_slice_indices.npy')
test_samples = [(int(idx // 10), int(idx % 10)) for idx in selected_indices[:3]]  # First 3 samples

# Different vmax settings to test
vmax_settings = [0.5, 0.7, 0.8, 0.9, 1.0]

print("Generating contrast comparison images...")

for sample_idx, (imag_idx, mask_idx) in enumerate(test_samples):
    print(f"\nProcessing sample {sample_idx + 1}...")
    
    # Load and process
    Xma, Xgt, XLI, M = load_test_sample(data_path, test_mask, imag_idx, mask_idx)
    
    with torch.no_grad():
        X0, ListX, ListA = model(Xma, XLI, M)
    
    # Post-process
    Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
    Xgtclip = torch.clamp(Xgt / 255.0, 0, 0.5)
    Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
    
    out_np = (Xoutclip / 0.5).cpu().numpy().squeeze()
    gt_np = (Xgtclip / 0.5).cpu().numpy().squeeze()
    ma_np = (Xmaclip / 0.5).cpu().numpy().squeeze()
    
    # Create comparison figure for different vmax values
    fig, axes = plt.subplots(3, len(vmax_settings), figsize=(4*len(vmax_settings), 12))
    
    for col, vmax in enumerate(vmax_settings):
        # Corrupted
        axes[0, col].imshow(ma_np, cmap='gray', vmin=0, vmax=vmax)
        axes[0, col].set_title(f'Corrupted\nvmax={vmax}', fontsize=10)
        axes[0, col].axis('off')
        
        # Ground Truth
        axes[1, col].imshow(gt_np, cmap='gray', vmin=0, vmax=vmax)
        axes[1, col].set_title(f'Ground Truth\nvmax={vmax}', fontsize=10)
        axes[1, col].axis('off')
        
        # DICDNet Output
        axes[2, col].imshow(out_np, cmap='gray', vmin=0, vmax=vmax)
        axes[2, col].set_title(f'DICDNet\nvmax={vmax}', fontsize=10)
        axes[2, col].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'contrast_comparison_sample{sample_idx+1}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

print(f"\n✓ Done! Check images in: {output_dir}")
print("Pick the vmax value that looks best to you!")
