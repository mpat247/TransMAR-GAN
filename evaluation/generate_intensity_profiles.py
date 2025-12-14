#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Standalone Intensity Line Profile Generator
============================================
Generates intensity line profile figures from saved benchmark comparison results.
Uses the EXACT same approach as benchmark_comparison_syndeeplesion.py.

This script loads the pre-saved .npy files from a benchmark comparison run
and generates publication-quality intensity profile figures.

Models Compared (7 total):
    1. DICDNet
    2. FIND-Net  
    3. InDuDoNet
    4. InDuDoNet+
    5. MEPNet
    6. SGA-MARN (our previous model)
    7. TransMAR-GAN (Ours - full model)

Usage:
    python generate_intensity_profiles.py --results_dir ./benchmark_comparison_results/comparison_YYYYMMDD_HHMMSS
    python generate_intensity_profiles.py --results_dir ./benchmark_comparison_results/comparison_YYYYMMDD_HHMMSS --samples 0 1 2
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from tqdm import tqdm
import json

# ═══════════════════════════════════════════════════════════════
# MODEL COLORS (same as benchmark_comparison_syndeeplesion.py)
# ═══════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    'DICDNet': {
        'color': '#E41A1C',  # Red
        'linestyle': '-',
    },
    'FIND-Net': {
        'color': '#377EB8',  # Blue
        'linestyle': '-',
    },
    'InDuDoNet': {
        'color': '#4DAF4A',  # Green
        'linestyle': '-',
    },
    'InDuDoNet+': {
        'color': '#984EA3',  # Purple
        'linestyle': '-',
    },
    'MEPNet': {
        'color': '#FF7F00',  # Orange
        'linestyle': '-',
    },
    'SGA-MARN': {
        'color': '#A65628',  # Brown
        'linestyle': '-',
    },
    'TransMAR-GAN': {
        'color': '#F781BF',  # Pink (Ours)
        'linestyle': '-',
    },
}

# ═══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Generate Intensity Line Profiles from Benchmark Results")
parser.add_argument("--results_dir", type=str, required=True,
                    help='Path to benchmark comparison results directory (e.g., ./benchmark_comparison_results/comparison_YYYYMMDD)')
parser.add_argument("--output_dir", type=str, default=None,
                    help='Path to save profile figures (default: results_dir/intensity_profiles_standalone)')
parser.add_argument("--samples", nargs='+', type=int, default=None,
                    help='Specific sample indices to process (default: all)')
parser.add_argument("--zoom_size", type=int, default=80, help='Size of zoomed region')
parser.add_argument("--dpi", type=int, default=200, help='DPI for saved figures')

opt = parser.parse_args()

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (same as benchmark_comparison_syndeeplesion.py)
# ═══════════════════════════════════════════════════════════════

def find_metal_center(mask):
    """Find the center of metal region for zooming"""
    if mask.sum() == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    
    coords = np.where(mask > 0.5)
    center_y = int(np.mean(coords[0]))
    center_x = int(np.mean(coords[1]))
    return center_y, center_x


def create_intensity_profile_figure(sample_idx, corrupted, gt, outputs, mask, save_path, model_configs, zoom_coords=None):
    """
    Create intensity profile figure (horizontal and vertical lines through metal).
    EXACT same format as benchmark_comparison_syndeeplesion.py.
    
    Layout:
        Top row: 3 images - Corrupted, Ground Truth, TransMAR-GAN (Ours)
        Bottom row: 2 graphs - Horizontal profile, Vertical profile
    
    All models included in the line plots.
    """
    # Find metal center
    center_y, center_x = find_metal_center(mask)
    
    # Use GT statistics for consistent windowing (same as benchmark script)
    gt_clipped = np.clip(gt, 0, 1)
    vmin_global = 0.0
    vmax_global = np.percentile(gt_clipped, 99.5)
    vmax_global = min(max(vmax_global, 0.5), 1.0)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Top row: Images with lines (3 images)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.clip(corrupted, 0, 1), cmap='gray', vmin=vmin_global, vmax=vmax_global)
    ax1.axhline(y=center_y, color='red', linewidth=1.5, label='H-line')
    ax1.axvline(x=center_x, color='cyan', linewidth=1.5, label='V-line')
    ax1.set_title('Corrupted', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(np.clip(gt, 0, 1), cmap='gray', vmin=vmin_global, vmax=vmax_global)
    ax2.axhline(y=center_y, color='red', linewidth=1.5)
    ax2.axvline(x=center_x, color='cyan', linewidth=1.5)
    ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Show TransMAR-GAN (Ours) as reference output in 3rd image
    if 'TransMAR-GAN' in outputs:
        ref_output = outputs['TransMAR-GAN']
        ref_name = 'TransMAR-GAN (Ours)'
    else:
        # Fall back to first available model
        ref_name = list(outputs.keys())[0]
        ref_output = outputs[ref_name]
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(np.clip(ref_output, 0, 1), cmap='gray', vmin=vmin_global, vmax=vmax_global)
    ax3.axhline(y=center_y, color='red', linewidth=1.5)
    ax3.axvline(x=center_x, color='cyan', linewidth=1.5)
    ax3.set_title(f'{ref_name}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Bottom row: Two centered graphs (same as benchmark script)
    # Horizontal profile
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.plot(corrupted[center_y, :], color='blue', linewidth=1.5, label='Corrupted', alpha=0.7)
    ax4.plot(gt[center_y, :], color='black', linewidth=2, linestyle='--', label='GT')
    
    for model_name, output in outputs.items():
        color = model_configs.get(model_name, {}).get('color', 'gray')
        label = f'{model_name}' if model_name != 'TransMAR-GAN' else f'{model_name} (Ours)'
        ax4.plot(output[center_y, :], color=color, linewidth=1.2, label=label, alpha=0.8)
    
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
    
    # Vertical profile
    ax5 = fig.add_subplot(2, 2, 4)
    ax5.plot(corrupted[:, center_x], color='blue', linewidth=1.5, label='Corrupted', alpha=0.7)
    ax5.plot(gt[:, center_x], color='black', linewidth=2, linestyle='--', label='GT')
    
    for model_name, output in outputs.items():
        color = model_configs.get(model_name, {}).get('color', 'gray')
        label = f'{model_name}' if model_name != 'TransMAR-GAN' else f'{model_name} (Ours)'
        ax5.plot(output[:, center_x], color=color, linewidth=1.2, label=label, alpha=0.8)
    
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
    
    plt.suptitle(f'Sample {sample_idx}: Intensity Line Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=opt.dpi, bbox_inches='tight')
    plt.close()


def create_zoomed_intensity_profile_figure(sample_idx, corrupted, gt, outputs, mask, save_path, model_configs, zoom_size=80):
    """
    Create zoomed intensity profile figure focusing on the metal region.
    
    Layout:
        Top row: 3 zoomed images - Corrupted, Ground Truth, TransMAR-GAN (Ours)
        Bottom row: 2 graphs - Horizontal profile (zoomed), Vertical profile (zoomed)
    """
    # Find metal center and zoom region
    center_y, center_x = find_metal_center(mask)
    half = zoom_size // 2
    h, w = corrupted.shape[:2]
    
    y1, y2 = max(0, center_y - half), min(h, center_y + half)
    x1, x2 = max(0, center_x - half), min(w, center_x + half)
    
    # Zoomed images
    corrupted_zoom = corrupted[y1:y2, x1:x2]
    gt_zoom = gt[y1:y2, x1:x2]
    mask_zoom = mask[y1:y2, x1:x2]
    
    outputs_zoom = {name: out[y1:y2, x1:x2] for name, out in outputs.items()}
    
    # Relative center in zoomed region
    rel_center_y = center_y - y1
    rel_center_x = center_x - x1
    
    # Use GT statistics for consistent windowing
    gt_clipped = np.clip(gt, 0, 1)
    vmin_global = 0.0
    vmax_global = np.percentile(gt_clipped, 99.5)
    vmax_global = min(max(vmax_global, 0.5), 1.0)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Top row: Zoomed images with lines
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.clip(corrupted_zoom, 0, 1), cmap='gray', vmin=vmin_global, vmax=vmax_global)
    ax1.axhline(y=rel_center_y, color='red', linewidth=1.5, label='H-line')
    ax1.axvline(x=rel_center_x, color='cyan', linewidth=1.5, label='V-line')
    ax1.set_title('Corrupted (Zoomed)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(np.clip(gt_zoom, 0, 1), cmap='gray', vmin=vmin_global, vmax=vmax_global)
    ax2.axhline(y=rel_center_y, color='red', linewidth=1.5)
    ax2.axvline(x=rel_center_x, color='cyan', linewidth=1.5)
    ax2.set_title('Ground Truth (Zoomed)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Show TransMAR-GAN (Ours) as reference
    if 'TransMAR-GAN' in outputs_zoom:
        ref_output = outputs_zoom['TransMAR-GAN']
        ref_name = 'TransMAR-GAN (Ours)'
    else:
        ref_name = list(outputs_zoom.keys())[0]
        ref_output = outputs_zoom[ref_name]
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(np.clip(ref_output, 0, 1), cmap='gray', vmin=vmin_global, vmax=vmax_global)
    ax3.axhline(y=rel_center_y, color='red', linewidth=1.5)
    ax3.axvline(x=rel_center_x, color='cyan', linewidth=1.5)
    ax3.set_title(f'{ref_name} (Zoomed)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Bottom row: Zoomed profiles
    # Horizontal profile (zoomed)
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.plot(corrupted_zoom[rel_center_y, :], color='blue', linewidth=1.5, label='Corrupted', alpha=0.7)
    ax4.plot(gt_zoom[rel_center_y, :], color='black', linewidth=2, linestyle='--', label='GT')
    
    for model_name, output in outputs_zoom.items():
        color = model_configs.get(model_name, {}).get('color', 'gray')
        label = f'{model_name}' if model_name != 'TransMAR-GAN' else f'{model_name} (Ours)'
        ax4.plot(output[rel_center_y, :], color=color, linewidth=1.2, label=label, alpha=0.8)
    
    # Highlight metal region
    metal_cols = np.where(mask_zoom[rel_center_y, :] > 0.5)[0]
    if len(metal_cols) > 0:
        ax4.axvspan(metal_cols[0], metal_cols[-1], alpha=0.2, color='yellow', label='Metal')
    
    ax4.set_xlabel('X Position (pixels)', fontsize=11)
    ax4.set_ylabel('Intensity', fontsize=11)
    ax4.set_title('Horizontal Intensity Profile (Zoomed)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=7, ncol=2)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Vertical profile (zoomed)
    ax5 = fig.add_subplot(2, 2, 4)
    ax5.plot(corrupted_zoom[:, rel_center_x], color='blue', linewidth=1.5, label='Corrupted', alpha=0.7)
    ax5.plot(gt_zoom[:, rel_center_x], color='black', linewidth=2, linestyle='--', label='GT')
    
    for model_name, output in outputs_zoom.items():
        color = model_configs.get(model_name, {}).get('color', 'gray')
        label = f'{model_name}' if model_name != 'TransMAR-GAN' else f'{model_name} (Ours)'
        ax5.plot(output[:, rel_center_x], color=color, linewidth=1.2, label=label, alpha=0.8)
    
    # Highlight metal region
    metal_rows = np.where(mask_zoom[:, rel_center_x] > 0.5)[0]
    if len(metal_rows) > 0:
        ax5.axvspan(metal_rows[0], metal_rows[-1], alpha=0.2, color='yellow', label='Metal')
    
    ax5.set_xlabel('Y Position (pixels)', fontsize=11)
    ax5.set_ylabel('Intensity', fontsize=11)
    ax5.set_title('Vertical Intensity Profile (Zoomed)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=7, ncol=2)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sample {sample_idx}: Zoomed Intensity Profiles (Metal Region)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=opt.dpi, bbox_inches='tight')
    plt.close()


def save_profile_data_csv(sample_idx, corrupted, gt, outputs, mask, save_path):
    """
    Save line profile data as CSV for easy import into other tools.
    """
    center_y, center_x = find_metal_center(mask)
    
    # Horizontal profile
    h_data = {
        'position': np.arange(corrupted.shape[1]),
        'corrupted': corrupted[center_y, :],
        'ground_truth': gt[center_y, :],
    }
    for model_name, output in outputs.items():
        safe_name = model_name.replace('+', '_plus').replace('-', '_')
        h_data[safe_name] = output[center_y, :]
    
    # Write horizontal CSV
    h_csv_path = save_path.replace('.csv', '_horizontal.csv')
    with open(h_csv_path, 'w') as f:
        headers = list(h_data.keys())
        f.write(','.join(headers) + '\n')
        for i in range(len(h_data['position'])):
            row = [str(h_data[h][i]) for h in headers]
            f.write(','.join(row) + '\n')
    
    # Vertical profile
    v_data = {
        'position': np.arange(corrupted.shape[0]),
        'corrupted': corrupted[:, center_x],
        'ground_truth': gt[:, center_x],
    }
    for model_name, output in outputs.items():
        safe_name = model_name.replace('+', '_plus').replace('-', '_')
        v_data[safe_name] = output[:, center_x]
    
    # Write vertical CSV
    v_csv_path = save_path.replace('.csv', '_vertical.csv')
    with open(v_csv_path, 'w') as f:
        headers = list(v_data.keys())
        f.write(','.join(headers) + '\n')
        for i in range(len(v_data['position'])):
            row = [str(v_data[h][i]) for h in headers]
            f.write(','.join(row) + '\n')
    
    return h_csv_path, v_csv_path


def save_line_profile_data_npz(sample_dir, corrupted, gt, outputs, mask):
    """
    Save all line profile data as numpy arrays (same as benchmark script).
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
        safe_name = model_name.replace('+', '_plus').replace('-', '_')
        intensity_data[f'horizontal_{safe_name}'] = output[center_y, :]
        intensity_data[f'vertical_{safe_name}'] = output[:, center_x]
        intensity_data[f'horizontal_error_{safe_name}'] = np.abs(output[center_y, :] - gt[center_y, :])
        intensity_data[f'vertical_error_{safe_name}'] = np.abs(output[:, center_x] - gt[:, center_x])
    
    np.savez(os.path.join(profile_data_dir, 'intensity_profiles.npz'), **intensity_data)
    
    return profile_data_dir


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def main():
    results_dir = os.path.abspath(opt.results_dir)
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Setup output directory
    if opt.output_dir:
        output_dir = os.path.abspath(opt.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(results_dir, f'intensity_profiles_standalone_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'full_profiles'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'zoomed_profiles'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'csv_data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'npz_data'), exist_ok=True)
    
    print("=" * 70)
    print("INTENSITY LINE PROFILE GENERATOR")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find individual_images directory
    individual_dir = os.path.join(results_dir, 'individual_images')
    if not os.path.exists(individual_dir):
        print(f"ERROR: individual_images directory not found in {results_dir}")
        sys.exit(1)
    
    # Get list of sample directories
    sample_dirs = sorted([d for d in os.listdir(individual_dir) if d.startswith('sample_')])
    print(f"Found {len(sample_dirs)} sample directories")
    
    # Filter by specific samples if requested
    if opt.samples:
        sample_dirs = [d for d in sample_dirs if any(f'sample_{s:04d}' in d for s in opt.samples)]
        print(f"Processing {len(sample_dirs)} requested samples")
    
    # Detect available models from first sample
    first_sample_dir = os.path.join(individual_dir, sample_dirs[0])
    available_models = []
    for model_name in MODEL_CONFIGS.keys():
        output_path = os.path.join(first_sample_dir, f'{model_name}_output.npy')
        if os.path.exists(output_path):
            available_models.append(model_name)
    
    print(f"Available models: {available_models}")
    print()
    
    # Process each sample
    for sample_dir_name in tqdm(sample_dirs, desc="Generating profiles"):
        sample_dir = os.path.join(individual_dir, sample_dir_name)
        
        # Extract sample index from directory name (e.g., 'sample_0001_idx123')
        try:
            sample_idx = int(sample_dir_name.split('_')[1])
        except:
            sample_idx = 0
        
        # Load base images
        corrupted_path = os.path.join(sample_dir, 'corrupted.npy')
        gt_path = os.path.join(sample_dir, 'ground_truth.npy')
        mask_path = os.path.join(sample_dir, 'mask.npy')
        
        if not all(os.path.exists(p) for p in [corrupted_path, gt_path, mask_path]):
            print(f"  Skipping {sample_dir_name}: Missing base files")
            continue
        
        corrupted = np.load(corrupted_path)
        gt = np.load(gt_path)
        mask = np.load(mask_path)
        
        # Load model outputs
        outputs = {}
        for model_name in available_models:
            output_path = os.path.join(sample_dir, f'{model_name}_output.npy')
            if os.path.exists(output_path):
                outputs[model_name] = np.load(output_path)
        
        if len(outputs) == 0:
            print(f"  Skipping {sample_dir_name}: No model outputs found")
            continue
        
        # Create full intensity profile figure
        full_profile_path = os.path.join(output_dir, 'full_profiles', f'{sample_dir_name}_intensity_profiles.png')
        try:
            create_intensity_profile_figure(sample_idx, corrupted, gt, outputs, mask, full_profile_path, MODEL_CONFIGS)
        except Exception as e:
            print(f"  {sample_dir_name}: Full profile failed - {e}")
        
        # Create zoomed intensity profile figure
        zoomed_profile_path = os.path.join(output_dir, 'zoomed_profiles', f'{sample_dir_name}_intensity_profiles_zoomed.png')
        try:
            create_zoomed_intensity_profile_figure(sample_idx, corrupted, gt, outputs, mask, zoomed_profile_path, MODEL_CONFIGS, opt.zoom_size)
        except Exception as e:
            print(f"  {sample_dir_name}: Zoomed profile failed - {e}")
        
        # Save CSV data
        csv_path = os.path.join(output_dir, 'csv_data', f'{sample_dir_name}_profiles.csv')
        try:
            save_profile_data_csv(sample_idx, corrupted, gt, outputs, mask, csv_path)
        except Exception as e:
            print(f"  {sample_dir_name}: CSV save failed - {e}")
        
        # Save NPZ data (same format as benchmark script)
        npz_dir = os.path.join(output_dir, 'npz_data', sample_dir_name)
        try:
            save_line_profile_data_npz(npz_dir, corrupted, gt, outputs, mask)
        except Exception as e:
            print(f"  {sample_dir_name}: NPZ save failed - {e}")
    
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output saved to: {output_dir}")
    print(f"  full_profiles/    - Full image intensity profiles")
    print(f"  zoomed_profiles/  - Zoomed metal region profiles")
    print(f"  csv_data/         - Raw profile data as CSV")
    print(f"  npz_data/         - Raw profile data as NPZ")


if __name__ == "__main__":
    main()
