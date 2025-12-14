#!/usr/bin/env python3
"""
Figure 3b: Metal-Aware Mask Radius Comparison

Layout:
    6 columns × 3 rows
    
    Column 1: Corrupted image (centered vertically)
    Columns 2-6: r=0, r=3, r=5, r=7, r=9
    
    Row 1: Dilated mask B overlay
    Row 2: Ring band B' = B - M overlay  
    Row 3: Weight map w' = 1 + β·B'

Minimal labels - just "r=X" for radius columns
Saves composite figure AND individual panels

Author: NgSwinGAN Paper
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared config
from shared_config import (
    PATHS,
    get_device,
    load_test_dataset,
    get_selected_slices,
)

# Import metal mask functions from training code
from losses.gan_losses import extract_metal_mask, dilate_mask

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    'output_dir': 'figure3_outputs',
    
    # Radii to compare
    'radii': [0, 3, 5, 7, 9],
    
    # Metal-aware weighting parameters
    'metal_threshold': 0.6,
    'beta': 1.0,
    'w_max': 3.0,
    
    # Visualization
    'dpi': 300,
    'figsize': (16, 9),  # Tighter spacing, no colorbar
    
    # Number of examples
    'num_examples': 10,
    
    # Colors
    'B_color': 'blue',
    'B_prime_color': 'cyan',
    'overlay_alpha': 0.5,
}

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, 'figure3_radius_comparison.log')
    
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
    
    logger = logging.getLogger('figure3_radius')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ═══════════════════════════════════════════════════════════════
# MASK COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_masks_for_radius(ct_tensor, radius, threshold=0.6, beta=1.0, w_max=3.0):
    """
    Compute B, B', and w' for a given radius.
    
    Args:
        ct_tensor: [1, 1, H, W] corrupted CT in [-1, 1]
        radius: Dilation radius
        threshold: Metal detection threshold
        beta: Weight factor
        w_max: Maximum weight
        
    Returns:
        dict with M, B, B_prime, w_prime
    """
    # Metal mask M
    M = extract_metal_mask(ct_tensor, threshold=threshold)
    
    # Dilated mask B
    if radius == 0:
        B = M.clone()
    else:
        B = dilate_mask(M, radius=radius)
    
    # Ring band B' = B - M
    B_prime = B - M
    B_prime = torch.clamp(B_prime, min=0)
    
    # Weight map w' = 1 + β·B'
    w_prime = 1.0 + beta * B_prime
    w_prime = torch.clamp(w_prime, max=w_max)
    
    return {
        'M': M,
        'B': B,
        'B_prime': B_prime,
        'w_prime': w_prime,
    }

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def create_weight_colormap():
    """Create colormap: dark (w=1) to bright (w=w_max)."""
    colors = [
        (0.1, 0.1, 0.3),    # Dark for w=1
        (0.2, 0.4, 0.8),    # Blue
        (0.4, 0.8, 0.9),    # Cyan
        (1.0, 1.0, 0.4),    # Yellow
        (1.0, 0.5, 0.2),    # Orange
        (0.9, 0.2, 0.2),    # Red for w=w_max
    ]
    return LinearSegmentedColormap.from_list('weight_map', colors, N=256)


def tensor_to_display(tensor):
    """Convert tensor from [-1,1] to [0,1] for display."""
    img = tensor.squeeze().cpu().numpy()
    return np.clip((img + 1) / 2, 0, 1)


def plot_ct_image(ax, ct_np):
    """Plot CT image with no title/axis."""
    ax.imshow(ct_np, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')


def plot_mask_overlay(ax, ct_np, mask_np, color='blue', alpha=0.5):
    """Plot CT with mask overlay, no title."""
    # Create RGB from grayscale
    rgb = np.stack([ct_np] * 3, axis=-1)
    
    # Color mapping
    color_map = {
        'blue': np.array([0.2, 0.4, 1.0]),
        'cyan': np.array([0.2, 1.0, 1.0]),
        'red': np.array([1.0, 0.2, 0.2]),
    }
    overlay_color = color_map.get(color, np.array([0.2, 0.4, 1.0]))
    
    # Apply overlay
    mask_3d = np.stack([mask_np] * 3, axis=-1)
    overlay = rgb * (1 - alpha * mask_3d) + overlay_color * alpha * mask_3d
    overlay = np.clip(overlay, 0, 1)
    
    ax.imshow(overlay)
    ax.axis('off')


def plot_weight_map(ax, w_np, w_min=1.0, w_max=3.0, cmap=None):
    """Plot weight map heatmap, no title."""
    if cmap is None:
        cmap = create_weight_colormap()
    
    im = ax.imshow(w_np, cmap=cmap, vmin=w_min, vmax=w_max)
    ax.axis('off')
    return im


def create_radius_comparison_figure(ct_tensor, slice_idx, output_dir, logger=None):
    """
    Create the main radius comparison figure.
    
    Layout:
        Col 0: Corrupted image (centered)
        Cols 1-5: r=0, 3, 5, 7, 9
        
        Row 0: B (dilated mask)
        Row 1: B' (ring band)
        Row 2: w' (weight map)
    """
    radii = CONFIG['radii']
    n_radii = len(radii)
    n_cols = n_radii + 1  # +1 for corrupted image column
    n_rows = 3
    
    # Create figure with tight layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=CONFIG['figsize'])
    
    # Get numpy image
    ct_np = tensor_to_display(ct_tensor)
    
    if logger:
        logger.debug(f"  Creating figure with {n_cols} columns, {n_rows} rows")
    
    # Row labels
    row_labels = ['Dilated Mask $B$', "Ring Band $B'$", "Weight Map $w'$"]
    
    # ─────────────────────────────────────────────────────────────
    # Column 0: Corrupted image (centered in middle row)
    # ─────────────────────────────────────────────────────────────
    
    # Hide top and bottom cells
    axes[0, 0].axis('off')
    axes[2, 0].axis('off')
    
    # Center: Show corrupted image
    plot_ct_image(axes[1, 0], ct_np)
    axes[1, 0].set_title('Corrupted', fontsize=12, fontweight='bold', pad=5)
    
    # ─────────────────────────────────────────────────────────────
    # Columns 1-5: Each radius
    # ─────────────────────────────────────────────────────────────
    
    cmap = create_weight_colormap()
    
    for col_idx, r in enumerate(radii):
        col = col_idx + 1  # Shift by 1 for corrupted image column
        
        if logger:
            logger.debug(f"    Processing r={r}")
        
        # Compute masks
        masks = compute_masks_for_radius(
            ct_tensor,
            radius=r,
            threshold=CONFIG['metal_threshold'],
            beta=CONFIG['beta'],
            w_max=CONFIG['w_max']
        )
        
        B_np = masks['B'].squeeze().cpu().numpy()
        B_prime_np = masks['B_prime'].squeeze().cpu().numpy()
        w_prime_np = masks['w_prime'].squeeze().cpu().numpy()
        
        # Row 0: Dilated mask B
        plot_mask_overlay(axes[0, col], ct_np, B_np, 
                         color=CONFIG['B_color'], alpha=CONFIG['overlay_alpha'])
        
        # Row 1: Ring band B'
        plot_mask_overlay(axes[1, col], ct_np, B_prime_np,
                         color=CONFIG['B_prime_color'], alpha=CONFIG['overlay_alpha'])
        
        # Row 2: Weight map w'
        plot_weight_map(axes[2, col], w_prime_np, 
                        w_min=1.0, w_max=CONFIG['w_max'], cmap=cmap)
        
        # Add radius label at top (column header)
        axes[0, col].set_title(f'r = {r}', fontsize=12, fontweight='bold', pad=5)
    
    # ─────────────────────────────────────────────────────────────
    # Add row labels on the left side
    # ─────────────────────────────────────────────────────────────
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(-0.1, 0.5, label, transform=axes[row_idx, 0].transAxes,
                              ha='right', va='center', fontsize=11, fontweight='bold',
                              rotation=90)
    
    # Adjust layout with minimal spacing (no colorbar)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.02, 
                        wspace=0.02, hspace=0.05)
    
    # Save composite figure
    composite_path = os.path.join(output_dir, f'radius_comparison_slice_{slice_idx}.png')
    plt.savefig(composite_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    pdf_path = composite_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    
    plt.close()
    
    if logger:
        logger.info(f"  Saved: radius_comparison_slice_{slice_idx}.png/pdf")
    
    return composite_path


def save_individual_panels(ct_tensor, slice_idx, output_dir, logger=None):
    """Save each panel as individual high-res image."""
    
    panel_dir = os.path.join(output_dir, 'individual_panels', f'slice_{slice_idx}')
    os.makedirs(panel_dir, exist_ok=True)
    
    ct_np = tensor_to_display(ct_tensor)
    radii = CONFIG['radii']
    cmap = create_weight_colormap()
    
    if logger:
        logger.info(f"  Saving individual panels...")
    
    # Save corrupted image
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_ct_image(ax, ct_np)
    plt.savefig(os.path.join(panel_dir, 'corrupted_image.png'),
               dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(panel_dir, 'corrupted_image.pdf'),
               dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    
    # Save numpy
    np.save(os.path.join(panel_dir, 'corrupted_image.npy'), ct_np)
    
    # For each radius, save B, B', w'
    for r in radii:
        masks = compute_masks_for_radius(
            ct_tensor,
            radius=r,
            threshold=CONFIG['metal_threshold'],
            beta=CONFIG['beta'],
            w_max=CONFIG['w_max']
        )
        
        B_np = masks['B'].squeeze().cpu().numpy()
        B_prime_np = masks['B_prime'].squeeze().cpu().numpy()
        w_prime_np = masks['w_prime'].squeeze().cpu().numpy()
        
        # Save B overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_mask_overlay(ax, ct_np, B_np, color=CONFIG['B_color'], alpha=CONFIG['overlay_alpha'])
        plt.savefig(os.path.join(panel_dir, f'B_r{r}.png'),
                   dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(panel_dir, f'B_r{r}.pdf'),
                   dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
        plt.close()
        
        # Save B' overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_mask_overlay(ax, ct_np, B_prime_np, color=CONFIG['B_prime_color'], alpha=CONFIG['overlay_alpha'])
        plt.savefig(os.path.join(panel_dir, f'B_prime_r{r}.png'),
                   dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(panel_dir, f'B_prime_r{r}.pdf'),
                   dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
        plt.close()
        
        # Save w' heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        im = plot_weight_map(ax, w_prime_np, w_min=1.0, w_max=CONFIG['w_max'], cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, label="$w'$")
        plt.savefig(os.path.join(panel_dir, f'w_prime_r{r}.png'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(panel_dir, f'w_prime_r{r}.pdf'),
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        # Save numpy arrays
        np.save(os.path.join(panel_dir, f'B_r{r}.npy'), B_np)
        np.save(os.path.join(panel_dir, f'B_prime_r{r}.npy'), B_prime_np)
        np.save(os.path.join(panel_dir, f'w_prime_r{r}.npy'), w_prime_np)
    
    if logger:
        logger.debug(f"    Saved to: individual_panels/slice_{slice_idx}/")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output_dir'], f'radius_comparison_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("FIGURE 3b: Metal-Aware Mask Radius Comparison")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    
    # Log configuration
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Radii to compare: {CONFIG['radii']}")
    logger.info(f"  Metal threshold: {CONFIG['metal_threshold']}")
    logger.info(f"  Beta: {CONFIG['beta']}")
    logger.info(f"  w_max: {CONFIG['w_max']}")
    logger.info(f"  Number of examples: {CONFIG['num_examples']}")
    
    logger.info("")
    logger.info("Figure Layout:")
    logger.info("  Column 0: Corrupted image (centered)")
    logger.info(f"  Columns 1-{len(CONFIG['radii'])}: r={CONFIG['radii']}")
    logger.info("  Row 0: Dilated mask B")
    logger.info("  Row 1: Ring band B' = B - M")
    logger.info("  Row 2: Weight map w' = 1 + β·B'")
    
    # Get device
    device = get_device()
    logger.info(f"\nUsing device: {device}")
    
    # Load test dataset
    logger.info("")
    logger.info("Loading test dataset...")
    test_dataset = load_test_dataset()
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Get selected slices
    selected_indices = get_selected_slices()
    logger.info(f"  Using {len(selected_indices)} selected slices")
    
    # Statistics
    processed = 0
    skipped = 0
    
    # Process slices
    logger.info("")
    logger.info(f"Processing {CONFIG['num_examples']} example slices...")
    
    for i, idx in enumerate(selected_indices[:CONFIG['num_examples']]):
        logger.info("")
        logger.info(f"Processing slice {idx} ({i+1}/{CONFIG['num_examples']})...")
        
        # Get data
        batch = test_dataset[idx]
        ct_tensor = batch[0].unsqueeze(0)  # [1, 1, H, W]
        
        # Check for metal
        M = extract_metal_mask(ct_tensor, threshold=CONFIG['metal_threshold'])
        metal_pixels = M.sum().item()
        
        if metal_pixels < 100:
            logger.warning(f"  Only {metal_pixels} metal pixels, skipping...")
            skipped += 1
            continue
        
        logger.info(f"  Metal pixels: {int(metal_pixels)}")
        
        # Create composite figure
        create_radius_comparison_figure(ct_tensor, idx, output_dir, logger)
        
        # Save individual panels
        save_individual_panels(ct_tensor, idx, output_dir, logger)
        
        processed += 1
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'figure': 'Figure 3b: Metal-Aware Mask Radius Comparison',
            'layout': {
                'columns': ['Corrupted'] + [f'r={r}' for r in CONFIG['radii']],
                'rows': ['B (dilated)', "B' (ring)", "w' (weight)"],
            },
            'parameters': {
                'radii': CONFIG['radii'],
                'metal_threshold': CONFIG['metal_threshold'],
                'beta': CONFIG['beta'],
                'w_max': CONFIG['w_max'],
            },
            'num_processed': processed,
            'num_skipped': skipped,
            'selected_indices': [int(x) for x in selected_indices[:CONFIG['num_examples']]],
        }, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("FIGURE 3b GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {processed} slices")
    logger.info(f"Skipped: {skipped} slices")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - radius_comparison_slice_*.png/pdf: Composite figures")
    logger.info("  - individual_panels/slice_*/: Individual high-res panels")
    logger.info("    - corrupted_image.png/pdf/npy")
    logger.info("    - B_r*.png/pdf/npy (dilated mask)")
    logger.info("    - B_prime_r*.png/pdf/npy (ring band)")
    logger.info("    - w_prime_r*.png/pdf/npy (weight map)")
    logger.info("  - config.json")
    logger.info("  - figure3_radius_comparison.log")


if __name__ == "__main__":
    main()
