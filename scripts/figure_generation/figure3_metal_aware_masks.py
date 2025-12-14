#!/usr/bin/env python3
"""
Figure 3: Metal-Aware Mask Construction

Purpose:
    Visualize how M, B, B' and the weight map w are constructed.

Data:
    - A single 2D slice with metal
    - Binary metal mask M (thresholded from x_corr)

Panel Layout:
    (A) Corrupted image x_corr
    (B) Metal mask M overlaid on x_corr
    (C) Dilated mask B = dilate(M, r)
    (D) Ring band B' = B - M
    (E) Weight map w = 1 + β·B (shown as heatmap)
    (F) Weight map w' = 1 + β·B' (shown as heatmap)

Implementation Notes:
    - Choose r (e.g., 35 pixels) and β (e.g., 1.0) for demonstration
    - Overlay masks as semitransparent color on grayscale CT
    - For w and w', use a colormap where w = 1 is dark and w > 1 is bright
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared config
from shared_config import (
    PATHS, VIS_CONFIG,
    denormalize, get_device,
    load_test_dataset, get_selected_slices,
    print_checkpoint_status
)

# Import the actual loss functions used in training
from losses.gan_losses import extract_metal_mask, dilate_mask

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Output settings
    'output_dir': os.path.join(PATHS['output_root'], 'figure3_metal_aware_masks'),
    
    # Metal-aware weighting parameters
    # As per Figure 3 spec: "Choose r (e.g., 35 pixels) and β (e.g., 1.0)"
    'metal_threshold': 0.6,      # Threshold for metal detection (data in [-1,1] range)
    'dilation_radius': 15,       # r = 15 pixels for better visualization
    'beta': 1.0,                 # β = 1.0 as suggested in spec
    'w_max': 3.0,                # Maximum weight clamp
    
    # Radius comparison range (3-7 pixels as requested)
    'radius_comparison_range': [3, 4, 5, 6, 7],
    
    # Visualization
    'dpi': 300,                  # High DPI for paper quality
    'figsize_main': (18, 12),    # Main 6-panel figure
    'figsize_individual': (8, 8),
    
    # Number of example slices to generate
    'num_examples': 10,          # More slices for comprehensive testing
}

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, 'figure3_generation.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('figure3')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ═══════════════════════════════════════════════════════════════
# MASK COMPUTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def compute_all_masks(ct_tensor, threshold=0.6, radius=15, beta=1.0, w_max=3.0):
    """
    Compute all masks and weight maps for visualization.
    
    Args:
        ct_tensor: Corrupted CT image [1, 1, H, W] in [-1, 1] range
        threshold: Metal detection threshold
        radius: Dilation radius
        beta: Weight boost factor
        w_max: Maximum weight clamp
    
    Returns:
        dict with M, B, B_prime, w, w_prime tensors
    """
    # (B) Metal mask M - thresholded from corrupted CT
    M = extract_metal_mask(ct_tensor, threshold=threshold)
    
    # (C) Dilated mask B = dilate(M, r)
    B = dilate_mask(M, radius=radius)
    
    # (D) Ring band B' = B - M
    B_prime = B - M
    B_prime = torch.clamp(B_prime, min=0)  # Ensure non-negative
    
    # (E) Weight map w = 1 + β·B (clamped)
    w = 1.0 + beta * B
    w = torch.clamp(w, max=w_max)
    
    # (F) Weight map w' = 1 + β·B' (clamped)
    w_prime = 1.0 + beta * B_prime
    w_prime = torch.clamp(w_prime, max=w_max)
    
    return {
        'M': M,
        'B': B,
        'B_prime': B_prime,
        'w': w,
        'w_prime': w_prime,
    }

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_weight_colormap():
    """
    Create a colormap where w=1 is dark and w>1 is bright.
    Dark blue -> Light blue -> Yellow -> Red
    """
    colors = [
        (0.1, 0.1, 0.3),    # Dark blue-gray for w=1
        (0.2, 0.4, 0.8),    # Blue
        (0.4, 0.8, 0.9),    # Cyan
        (1.0, 1.0, 0.4),    # Yellow
        (1.0, 0.5, 0.2),    # Orange
        (0.9, 0.2, 0.2),    # Red for w=w_max
    ]
    return LinearSegmentedColormap.from_list('weight_map', colors, N=256)


def plot_ct_image(ax, ct_np, title, cmap='gray'):
    """Plot CT image with proper normalization."""
    # Convert from [-1, 1] to [0, 1] for display
    ct_display = (ct_np + 1) / 2
    ct_display = np.clip(ct_display, 0, 1)
    
    ax.imshow(ct_display, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')


def plot_mask_overlay(ax, ct_np, mask_np, title, mask_color='red', alpha=0.5):
    """Plot CT image with mask overlay."""
    # Convert CT from [-1, 1] to [0, 1]
    ct_display = (ct_np + 1) / 2
    ct_display = np.clip(ct_display, 0, 1)
    
    # Create RGB image from grayscale
    rgb = np.stack([ct_display] * 3, axis=-1)
    
    # Create colored overlay
    if mask_color == 'red':
        overlay_color = np.array([1.0, 0.2, 0.2])
    elif mask_color == 'green':
        overlay_color = np.array([0.2, 1.0, 0.2])
    elif mask_color == 'blue':
        overlay_color = np.array([0.2, 0.4, 1.0])
    elif mask_color == 'yellow':
        overlay_color = np.array([1.0, 1.0, 0.2])
    elif mask_color == 'cyan':
        overlay_color = np.array([0.2, 1.0, 1.0])
    elif mask_color == 'magenta':
        overlay_color = np.array([1.0, 0.2, 1.0])
    else:
        overlay_color = np.array([1.0, 0.2, 0.2])
    
    # Apply overlay where mask is active
    mask_3d = np.stack([mask_np] * 3, axis=-1)
    overlay = rgb * (1 - alpha * mask_3d) + overlay_color * alpha * mask_3d
    overlay = np.clip(overlay, 0, 1)
    
    ax.imshow(overlay)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')


def plot_weight_map(ax, w_np, title, w_min=1.0, w_max=3.0, cmap=None):
    """Plot weight map with custom colormap."""
    if cmap is None:
        cmap = create_weight_colormap()
    
    im = ax.imshow(w_np, cmap=cmap, vmin=w_min, vmax=w_max)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return im


def create_main_figure(ct_tensor, masks, output_path, slice_idx, params):
    """
    Create the main 6-panel figure (A-F).
    
    Layout:
        (A) x_corr          (B) M overlay       (C) B overlay
        (D) B' overlay      (E) w heatmap       (F) w' heatmap
    """
    fig, axes = plt.subplots(2, 3, figsize=CONFIG['figsize_main'])
    
    # Convert tensors to numpy
    ct_np = ct_tensor.squeeze().cpu().numpy()
    M_np = masks['M'].squeeze().cpu().numpy()
    B_np = masks['B'].squeeze().cpu().numpy()
    B_prime_np = masks['B_prime'].squeeze().cpu().numpy()
    w_np = masks['w'].squeeze().cpu().numpy()
    w_prime_np = masks['w_prime'].squeeze().cpu().numpy()
    
    # (A) Corrupted image x_corr
    plot_ct_image(axes[0, 0], ct_np, '(A) Corrupted Image $x_{corr}$')
    
    # (B) Metal mask M overlaid on x_corr
    plot_mask_overlay(axes[0, 1], ct_np, M_np, 
                      '(B) Metal Mask $M$ (red overlay)', 
                      mask_color='red', alpha=0.6)
    
    # (C) Dilated mask B = dilate(M, r)
    plot_mask_overlay(axes[0, 2], ct_np, B_np,
                      f'(C) Dilated Mask $B$ = dilate($M$, r={params["radius"]})',
                      mask_color='blue', alpha=0.5)
    
    # (D) Ring band B' = B - M
    plot_mask_overlay(axes[1, 0], ct_np, B_prime_np,
                      "(D) Ring Band $B'$ = $B$ - $M$",
                      mask_color='cyan', alpha=0.6)
    
    # (E) Weight map w = 1 + β·B
    im_w = plot_weight_map(axes[1, 1], w_np,
                           f'(E) Weight Map $w$ = 1 + {params["beta"]}·$B$',
                           w_min=1.0, w_max=params['w_max'])
    
    # (F) Weight map w' = 1 + β·B'
    im_w_prime = plot_weight_map(axes[1, 2], w_prime_np,
                                  f"(F) Weight Map $w'$ = 1 + {params['beta']}·$B'$",
                                  w_min=1.0, w_max=params['w_max'])
    
    # Add colorbars for weight maps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar = fig.colorbar(im_w_prime, cax=cbar_ax)
    cbar.set_label('Weight Value', fontsize=12)
    
    # Add legend for masks
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.6, label='Metal Mask $M$'),
        mpatches.Patch(facecolor='blue', alpha=0.5, label='Dilated Mask $B$'),
        mpatches.Patch(facecolor='cyan', alpha=0.6, label="Ring Band $B'$"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    
    # Add parameter info
    param_text = (f"Parameters: threshold={params['threshold']}, "
                  f"r={params['radius']}, β={params['beta']}, w_max={params['w_max']}")
    fig.suptitle(f'Figure 3: Metal-Aware Mask Construction (Slice {slice_idx})\n{param_text}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()


def create_mask_comparison_figure(ct_tensor, slice_idx, output_path, logger=None):
    """
    Create comparison showing different dilation radii.
    Uses radius_comparison_range from CONFIG.
    No colorbar, minimal whitespace, clear headings.
    """
    radii = CONFIG['radius_comparison_range']
    n_radii = len(radii)
    
    fig, axes = plt.subplots(2, n_radii, figsize=(4*n_radii, 8))
    
    ct_np = ct_tensor.squeeze().cpu().numpy()
    
    if logger:
        logger.debug(f"Creating radius comparison with radii: {radii}")
    
    # Top row: Dilated masks B for different radii
    for i, r in enumerate(radii):
        masks = compute_all_masks(ct_tensor, 
                                   threshold=CONFIG['metal_threshold'],
                                   radius=r,
                                   beta=CONFIG['beta'],
                                   w_max=CONFIG['w_max'])
        B_np = masks['B'].squeeze().cpu().numpy()
        plot_mask_overlay(axes[0, i], ct_np, B_np, 
                          f'$B$ (r={r})', mask_color='blue', alpha=0.5)
    
    # Bottom row: Weight maps w for different radii
    cmap = create_weight_colormap()
    for i, r in enumerate(radii):
        masks = compute_all_masks(ct_tensor,
                                   threshold=CONFIG['metal_threshold'],
                                   radius=r,
                                   beta=CONFIG['beta'],
                                   w_max=CONFIG['w_max'])
        w_np = masks['w'].squeeze().cpu().numpy()
        axes[1, i].imshow(w_np, cmap=cmap, vmin=1.0, vmax=CONFIG['w_max'])
        axes[1, i].set_title(f'$w$ (r={r})', fontsize=11, fontweight='bold')
        axes[1, i].axis('off')
    
    # Main title
    fig.suptitle(f'Dilation Radius Comparison - Slice {slice_idx}', 
                 fontsize=14, fontweight='bold')
    
    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_weight_comparison_figure(ct_tensor, slice_idx, output_path, logger=None):
    """
    Create comparison between w (full dilated) and w' (ring only).
    Shows why ring-only weighting might be preferred.
    No colorbar, minimal whitespace, clear headings.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    ct_np = ct_tensor.squeeze().cpu().numpy()
    
    # Compute masks with the main dilation radius
    masks = compute_all_masks(ct_tensor,
                               threshold=CONFIG['metal_threshold'],
                               radius=CONFIG['dilation_radius'],
                               beta=CONFIG['beta'],
                               w_max=CONFIG['w_max'])
    
    if logger:
        logger.debug(f"Creating weight comparison with r={CONFIG['dilation_radius']}")
    
    M_np = masks['M'].squeeze().cpu().numpy()
    B_np = masks['B'].squeeze().cpu().numpy()
    B_prime_np = masks['B_prime'].squeeze().cpu().numpy()
    w_np = masks['w'].squeeze().cpu().numpy()
    w_prime_np = masks['w_prime'].squeeze().cpu().numpy()
    
    cmap = create_weight_colormap()
    
    # Top row: Masks with clear headings
    plot_mask_overlay(axes[0, 0], ct_np, M_np, 'Metal Mask $M$', 
                      mask_color='red', alpha=0.6)
    plot_mask_overlay(axes[0, 1], ct_np, B_np, 'Dilated $B$ (includes $M$)', 
                      mask_color='blue', alpha=0.5)
    plot_mask_overlay(axes[0, 2], ct_np, B_prime_np, "Ring Band $B'$ = $B$ - $M$", 
                      mask_color='cyan', alpha=0.6)
    
    # Bottom row: Weight maps (no individual colorbars)
    axes[1, 0].imshow(w_np, cmap=cmap, vmin=1.0, vmax=CONFIG['w_max'])
    axes[1, 0].set_title('$w$ = 1 + β·$B$\n(Metal + Ring weighted)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(w_prime_np, cmap=cmap, vmin=1.0, vmax=CONFIG['w_max'])
    axes[1, 1].set_title("$w'$ = 1 + β·$B'$\n(Ring only weighted)", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Difference: w - w' shows metal interior (no colorbar)
    diff = w_np - w_prime_np
    axes[1, 2].imshow(diff, cmap='Reds', vmin=0, vmax=CONFIG['beta'])
    axes[1, 2].set_title("Difference $w$ - $w'$\n(Metal interior only)", fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Main title with parameters
    fig.suptitle(f'Weight Map Comparison (r={CONFIG["dilation_radius"]}, β={CONFIG["beta"]}) - Slice {slice_idx}',
                 fontsize=13, fontweight='bold')
    
    # Tight layout with minimal whitespace
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_individual_panels(ct_tensor, masks, output_dir, slice_idx, params):
    """Save each panel as a separate high-resolution image."""
    panel_dir = os.path.join(output_dir, 'individual_panels', f'slice_{slice_idx}')
    os.makedirs(panel_dir, exist_ok=True)
    
    ct_np = ct_tensor.squeeze().cpu().numpy()
    M_np = masks['M'].squeeze().cpu().numpy()
    B_np = masks['B'].squeeze().cpu().numpy()
    B_prime_np = masks['B_prime'].squeeze().cpu().numpy()
    w_np = masks['w'].squeeze().cpu().numpy()
    w_prime_np = masks['w_prime'].squeeze().cpu().numpy()
    
    cmap = create_weight_colormap()
    
    # (A) Corrupted image
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    plot_ct_image(ax, ct_np, '')
    plt.savefig(os.path.join(panel_dir, 'A_corrupted_image.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # (B) Metal mask overlay
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    plot_mask_overlay(ax, ct_np, M_np, '', mask_color='red', alpha=0.6)
    plt.savefig(os.path.join(panel_dir, 'B_metal_mask_overlay.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # (C) Dilated mask overlay
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    plot_mask_overlay(ax, ct_np, B_np, '', mask_color='blue', alpha=0.5)
    plt.savefig(os.path.join(panel_dir, 'C_dilated_mask_overlay.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # (D) Ring band overlay
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    plot_mask_overlay(ax, ct_np, B_prime_np, '', mask_color='cyan', alpha=0.6)
    plt.savefig(os.path.join(panel_dir, 'D_ring_band_overlay.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # (E) Weight map w
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    im = plot_weight_map(ax, w_np, '', w_min=1.0, w_max=params['w_max'], cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, label='Weight')
    plt.savefig(os.path.join(panel_dir, 'E_weight_map_w.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # (F) Weight map w'
    fig, ax = plt.subplots(figsize=CONFIG['figsize_individual'])
    im = plot_weight_map(ax, w_prime_np, '', w_min=1.0, w_max=params['w_max'], cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, label='Weight')
    plt.savefig(os.path.join(panel_dir, 'F_weight_map_w_prime.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also save raw masks as numpy
    np.save(os.path.join(panel_dir, 'M_metal_mask.npy'), M_np)
    np.save(os.path.join(panel_dir, 'B_dilated_mask.npy'), B_np)
    np.save(os.path.join(panel_dir, 'B_prime_ring_band.npy'), B_prime_np)
    np.save(os.path.join(panel_dir, 'w_weight_map.npy'), w_np)
    np.save(os.path.join(panel_dir, 'w_prime_weight_map.npy'), w_prime_np)


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    # Create output directory first (needed for logging)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("FIGURE 3: Metal-Aware Mask Construction")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    
    # Parameters for main visualization (using the spec's r=35)
    params = {
        'threshold': CONFIG['metal_threshold'],
        'radius': CONFIG['dilation_radius'],  # r=35 as per spec
        'beta': CONFIG['beta'],
        'w_max': CONFIG['w_max'],
    }
    
    logger.info("")
    logger.info("Configuration Parameters:")
    logger.info(f"  Metal threshold: {params['threshold']} (in [-1,1] range)")
    logger.info(f"  Main dilation radius: {params['radius']} pixels (for main figure)")
    logger.info(f"  Radius comparison range: {CONFIG['radius_comparison_range']} pixels")
    logger.info(f"  Beta (weight factor): {params['beta']}")
    logger.info(f"  Max weight: {params['w_max']}")
    logger.info(f"  Number of example slices: {CONFIG['num_examples']}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    logger.info("")
    logger.info("Loading test dataset...")
    test_dataset = load_test_dataset()
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Get selected slices (with high artifact visibility)
    logger.info("")
    logger.info("Getting selected slices...")
    selected_indices = get_selected_slices()
    logger.info(f"  Using {len(selected_indices)} selected slices")
    logger.debug(f"  Selected indices: {selected_indices[:CONFIG['num_examples']]}")
    
    # Statistics tracking
    processed_count = 0
    skipped_count = 0
    
    # Process slices
    logger.info("")
    logger.info(f"Generating figures for {CONFIG['num_examples']} example slices...")
    
    for i, idx in enumerate(selected_indices[:CONFIG['num_examples']]):
        logger.info("")
        logger.info(f"Processing slice {idx} ({i+1}/{CONFIG['num_examples']})...")
        
        # Get data
        batch = test_dataset[idx]
        ct_tensor = batch[0].unsqueeze(0)  # [1, 1, H, W] - corrupted CT
        
        # Compute all masks
        masks = compute_all_masks(
            ct_tensor,
            threshold=params['threshold'],
            radius=params['radius'],
            beta=params['beta'],
            w_max=params['w_max']
        )
        
        # Check if there's actually metal in this slice
        metal_pixels = masks['M'].sum().item()
        if metal_pixels < 100:
            logger.warning(f"  Only {metal_pixels} metal pixels detected, skipping...")
            skipped_count += 1
            continue
        
        # Log mask statistics
        dilated_pixels = int(masks['B'].sum().item())
        ring_pixels = int(masks['B_prime'].sum().item())
        logger.info(f"  Metal pixels: {int(metal_pixels)}")
        logger.info(f"  Dilated region pixels (B): {dilated_pixels}")
        logger.info(f"  Ring band pixels (B'): {ring_pixels}")
        logger.debug(f"  Expansion factor: {dilated_pixels / max(metal_pixels, 1):.2f}x")
        
        # Create main 6-panel figure
        main_path = os.path.join(output_dir, f'figure3_slice_{idx}.png')
        create_main_figure(ct_tensor, masks, main_path, idx, params)
        logger.info(f"  Saved main figure: figure3_slice_{idx}.png")
        
        # Create radius comparison figure
        comparison_path = os.path.join(output_dir, f'radius_comparison_slice_{idx}.png')
        create_mask_comparison_figure(ct_tensor, idx, comparison_path, logger)
        logger.info(f"  Saved radius comparison: radius_comparison_slice_{idx}.png")
        
        # Create w vs w' comparison
        weight_path = os.path.join(output_dir, f'weight_comparison_slice_{idx}.png')
        create_weight_comparison_figure(ct_tensor, idx, weight_path, logger)
        logger.info(f"  Saved weight comparison: weight_comparison_slice_{idx}.png")
        
        # Save individual panels
        create_individual_panels(ct_tensor, masks, output_dir, idx, params)
        logger.info(f"  Saved individual panels to: individual_panels/slice_{idx}/")
        
        processed_count += 1
    
    # Save configuration
    import json
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'figure': 'Figure 3: Metal-Aware Mask Construction',
            'parameters': params,
            'training_config': {
                'metal_threshold': CONFIG['metal_threshold'],
                'dilation_radius': CONFIG['dilation_radius'],
                'radius_comparison_range': CONFIG['radius_comparison_range'],
                'beta': CONFIG['beta'],
                'w_max': CONFIG['w_max'],
            },
            'num_examples_requested': CONFIG['num_examples'],
            'num_examples_processed': processed_count,
            'num_examples_skipped': skipped_count,
            'selected_indices': [int(x) for x in selected_indices[:CONFIG['num_examples']]],
        }, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("FIGURE 3 GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {processed_count} slices")
    logger.info(f"Skipped: {skipped_count} slices (insufficient metal)")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - figure3_slice_*.png: Main 6-panel figures (A-F)")
    logger.info("  - radius_comparison_*.png: Different dilation radii comparison")
    logger.info("  - weight_comparison_*.png: w vs w' comparison")
    logger.info("  - individual_panels/: High-res individual panels + numpy arrays")
    logger.info("  - config.json: Configuration used")
    logger.info("  - figure3_generation.log: Detailed log file")


if __name__ == "__main__":
    main()
