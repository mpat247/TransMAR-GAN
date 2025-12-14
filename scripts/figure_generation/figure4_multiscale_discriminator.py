#!/usr/bin/env python3
"""
Figure 4: Multi-Scale Discriminator Illustration

Purpose:
    Illustrate how the multi-scale PatchGAN discriminators operate on 
    different resolutions and local patches.

Content:
    - Show input image at full, half, and quarter resolution
    - For each scale, depict a grid representing patch outputs
    - Visualize receptive fields with overlaid boxes
    - Label scales: 1×, 1/2×, 1/4×
    - Label discriminators D⁽¹⁾, D⁽¹/²⁾, D⁽¹/⁴⁾

Outputs:
    - Main composite figure (PNG + PDF)
    - Individual scale images (full, half, quarter)
    - PatchGAN output maps for each scale
    - Receptive field overlay visualizations
    - Architecture conceptual diagram
    - Feature maps from intermediate layers
    - Real vs Fake discriminator response comparison

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
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared config and utilities
from shared_config import (
    PATHS,
    MODEL_CONFIG,
    get_device,
    load_test_dataset,
    load_generator,
    get_selected_slices,
)

# Import the multi-scale discriminator
from models.discriminator.ms_patchgan import MultiScaleDiscriminator, SingleScaleDiscriminator

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Output settings
    'output_dir': 'figure4_outputs',
    
    # Discriminator architecture (must match training)
    'disc_in_channels': 2,       # concat(x_corr, x_pred/x_gt)
    'disc_base_channels': 64,
    'disc_num_layers': 5,
    'disc_num_scales': 3,
    'disc_use_sn': True,
    
    # Image settings
    'image_size': 128,           # Input image size
    
    # Scale labels
    'scale_labels': ['1×', '1/2×', '1/4×'],
    'scale_names': ['full', 'half', 'quarter'],
    'disc_labels': ['D⁽¹⁾', 'D⁽¹/²⁾', 'D⁽¹/⁴⁾'],
    
    # Visualization
    'dpi': 300,
    'pdf_dpi': 300,
    'figsize_main': (18, 14),
    'figsize_individual': (8, 8),
    'figsize_architecture': (16, 10),
    
    # Receptive field colors
    'rf_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1'],  # Red, Teal, Blue
    'rf_alpha': 0.3,
    
    # Number of example slices
    'num_examples': 5,
    
    # Colormap for discriminator outputs
    'disc_cmap': 'RdYlGn',  # Red (fake) to Green (real)
}

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, 'figure4_generation.log')
    
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
    
    logger = logging.getLogger('figure4')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ═══════════════════════════════════════════════════════════════
# DISCRIMINATOR UTILITIES
# ═══════════════════════════════════════════════════════════════

def load_discriminator(device, logger=None):
    """
    Load multi-scale discriminator.
    Note: For visualization, we can use a randomly initialized discriminator
    since we're showing the architecture concept, not learned features.
    """
    if logger:
        logger.info("Initializing Multi-Scale Discriminator...")
        logger.info(f"  In channels: {CONFIG['disc_in_channels']}")
        logger.info(f"  Base channels: {CONFIG['disc_base_channels']}")
        logger.info(f"  Num layers: {CONFIG['disc_num_layers']}")
        logger.info(f"  Num scales: {CONFIG['disc_num_scales']}")
        logger.info(f"  Spectral norm: {CONFIG['disc_use_sn']}")
    
    disc = MultiScaleDiscriminator(
        in_channels=CONFIG['disc_in_channels'],
        base_channels=CONFIG['disc_base_channels'],
        num_layers=CONFIG['disc_num_layers'],
        num_scales=CONFIG['disc_num_scales'],
        use_sn=CONFIG['disc_use_sn']
    )
    
    disc = disc.to(device)
    disc.eval()
    
    if logger:
        # Count parameters
        total_params = sum(p.numel() for p in disc.parameters())
        logger.info(f"  Total parameters: {total_params:,}")
    
    return disc


def compute_output_sizes(input_size, num_layers=5):
    """
    Compute the output grid size for SingleScaleDiscriminator.
    
    Each layer (except last) has stride=2, last has stride=1.
    Kernel size = 4, padding = 1
    
    Formula: out = floor((in + 2*pad - kernel) / stride + 1)
    For stride=2, k=4, p=1: out = floor((in + 2 - 4) / 2 + 1) = floor(in/2)
    For stride=1, k=4, p=1: out = in - 2
    """
    size = input_size
    for i in range(num_layers):
        stride = 1 if i == num_layers - 1 else 2
        if stride == 2:
            size = size // 2
        else:
            size = size - 2
    return max(1, size)


def compute_receptive_field(num_layers=5):
    """
    Compute the receptive field size for the discriminator.
    
    For conv with kernel k, stride s:
    RF_new = RF_old + (k - 1) * product_of_previous_strides
    
    Returns receptive field in pixels.
    """
    rf = 1  # Start with 1 pixel
    stride_product = 1
    
    for i in range(num_layers):
        k = 4  # kernel size
        stride = 1 if i == num_layers - 1 else 2
        
        rf = rf + (k - 1) * stride_product
        stride_product *= stride
    
    # Add one more for the final 1x1 conv
    rf = rf + (1 - 1) * stride_product  # 1x1 conv doesn't change RF
    
    return rf


def get_multiscale_inputs(x, num_scales=3):
    """
    Get input images at multiple scales.
    
    Args:
        x: Input tensor [B, C, H, W]
        num_scales: Number of scales
        
    Returns:
        List of tensors at each scale
    """
    scales = [x]
    x_scale = x
    for _ in range(num_scales - 1):
        x_scale = F.avg_pool2d(x_scale, kernel_size=2, stride=2, padding=0)
        scales.append(x_scale)
    return scales


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def tensor_to_display(tensor):
    """Convert tensor from [-1,1] to [0,1] for display."""
    img = tensor.squeeze().cpu().numpy()
    img = (img + 1) / 2
    return np.clip(img, 0, 1)


def create_disc_colormap():
    """Create colormap for discriminator outputs: Red (fake) -> Yellow -> Green (real)."""
    colors = [
        (0.8, 0.2, 0.2),   # Red (fake/low)
        (1.0, 0.6, 0.2),   # Orange
        (1.0, 1.0, 0.4),   # Yellow
        (0.6, 0.9, 0.4),   # Light green
        (0.2, 0.7, 0.3),   # Green (real/high)
    ]
    return LinearSegmentedColormap.from_list('disc_output', colors, N=256)


def draw_receptive_field_boxes(ax, img_size, grid_size, rf_size, color, alpha=0.3, 
                                num_boxes=3, logger=None):
    """
    Draw receptive field boxes on the image.
    
    Args:
        ax: Matplotlib axis
        img_size: Size of the image
        grid_size: Size of the output grid
        rf_size: Receptive field size in pixels
        color: Box color
        alpha: Box transparency
        num_boxes: Number of sample boxes to draw
    """
    if grid_size <= 0:
        return
    
    # Calculate stride between patches in the original image
    # The output grid covers the input, so stride = input_size / grid_size
    stride = img_size / grid_size
    
    # Draw a few sample receptive fields
    positions = []
    if grid_size >= 3:
        # Corner and center positions
        positions = [(0, 0), (grid_size//2, grid_size//2), (grid_size-1, grid_size-1)]
    elif grid_size == 2:
        positions = [(0, 0), (1, 1)]
    else:
        positions = [(0, 0)]
    
    for i, (gx, gy) in enumerate(positions[:num_boxes]):
        # Center of this patch in the original image
        center_x = (gx + 0.5) * stride
        center_y = (gy + 0.5) * stride
        
        # Top-left corner of receptive field
        x = center_x - rf_size / 2
        y = center_y - rf_size / 2
        
        # Clip to image bounds
        x = max(0, min(x, img_size - rf_size))
        y = max(0, min(y, img_size - rf_size))
        
        # Draw rectangle
        rect = Rectangle(
            (x, y), rf_size, rf_size,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            linestyle='-' if i == 0 else '--'
        )
        ax.add_patch(rect)
    
    if logger:
        logger.debug(f"    Drew {len(positions)} RF boxes, size={rf_size}px")


def create_grid_overlay(ax, grid_size, img_size, color='white', alpha=0.5):
    """Draw a grid overlay showing patch boundaries."""
    if grid_size <= 1:
        return
    
    step = img_size / grid_size
    
    # Vertical lines
    for i in range(1, grid_size):
        ax.axvline(x=i * step, color=color, alpha=alpha, linewidth=0.5)
    
    # Horizontal lines
    for i in range(1, grid_size):
        ax.axhline(y=i * step, color=color, alpha=alpha, linewidth=0.5)


def plot_discriminator_output(ax, output_grid, title, cmap=None, show_values=True):
    """
    Plot discriminator output grid as heatmap.
    
    Args:
        ax: Matplotlib axis
        output_grid: 2D numpy array of discriminator outputs
        title: Plot title
        cmap: Colormap
        show_values: Whether to annotate cells with values
    """
    if cmap is None:
        cmap = create_disc_colormap()
    
    # Apply sigmoid to convert logits to probabilities
    probs = 1 / (1 + np.exp(-output_grid))
    
    im = ax.imshow(probs, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add value annotations if grid is small enough
    h, w = probs.shape
    if show_values and h <= 8 and w <= 8:
        for i in range(h):
            for j in range(w):
                val = probs[i, j]
                text_color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color=text_color, fontweight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    return im


def create_main_figure(ct_corr, ct_output, disc, device, output_path, slice_idx, logger=None):
    """
    Create the main composite figure showing multi-scale discriminator operation.
    
    Layout (4 rows × 3 columns):
        Row 1: Input images at each scale
        Row 2: Discriminator output grids
        Row 3: Receptive field visualization
        Row 4: Feature maps from intermediate layers
    """
    if logger:
        logger.info("  Creating main composite figure...")
    
    fig = plt.figure(figsize=CONFIG['figsize_main'])
    
    # Create grid spec for flexible layout
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25,
                          height_ratios=[1, 1, 1, 0.8])
    
    # Prepare inputs
    # Discriminator input: concat(x_corr, x_output)
    disc_input = torch.cat([ct_corr, ct_output], dim=1)  # [1, 2, H, W]
    
    # Get multi-scale inputs
    scale_inputs = get_multiscale_inputs(disc_input, CONFIG['disc_num_scales'])
    ct_corr_scales = get_multiscale_inputs(ct_corr, CONFIG['disc_num_scales'])
    
    # Run discriminator
    with torch.no_grad():
        logits_all, features_all = disc(disc_input, return_features=True)
    
    if logger:
        logger.debug(f"    Discriminator output shapes:")
        for i, logits in enumerate(logits_all):
            logger.debug(f"      Scale {i}: {logits.shape}")
    
    # Compute sizes and receptive fields
    img_size = CONFIG['image_size']
    rf_size = compute_receptive_field(CONFIG['disc_num_layers'])
    
    if logger:
        logger.info(f"  Receptive field size: {rf_size} pixels")
    
    # ─────────────────────────────────────────────────────────────
    # ROW 1: Input images at each scale
    # ─────────────────────────────────────────────────────────────
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        img = tensor_to_display(ct_corr_scales[i])
        scale_size = img.shape[0]
        
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Input at {CONFIG["scale_labels"][i]}\n({scale_size}×{scale_size})',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add scale label
        ax.text(0.02, 0.98, CONFIG['disc_labels'][i], transform=ax.transAxes,
               fontsize=14, fontweight='bold', color='yellow',
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='black', alpha=0.7))
    
    # ─────────────────────────────────────────────────────────────
    # ROW 2: Discriminator output grids (PatchGAN outputs)
    # ─────────────────────────────────────────────────────────────
    cmap = create_disc_colormap()
    
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        output_grid = logits_all[i].squeeze().cpu().numpy()
        grid_h, grid_w = output_grid.shape
        
        im = plot_discriminator_output(
            ax, output_grid,
            f'{CONFIG["disc_labels"][i]} Output\n({grid_h}×{grid_w} patches)',
            cmap=cmap
        )
        
        if logger:
            probs = 1 / (1 + np.exp(-output_grid))
            logger.debug(f"    Scale {i} output: {grid_h}×{grid_w}, "
                        f"prob range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Add colorbar for discriminator outputs
    cbar_ax = fig.add_axes([0.92, 0.52, 0.015, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('P(Real)', fontsize=10)
    
    # ─────────────────────────────────────────────────────────────
    # ROW 3: Receptive field visualization
    # ─────────────────────────────────────────────────────────────
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        
        # Show full-resolution image
        img = tensor_to_display(ct_corr)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        
        # Compute effective RF at this scale
        # RF is defined for the single-scale disc, but input is smaller
        scale_factor = 2 ** i
        effective_rf = rf_size * scale_factor  # RF in original image coordinates
        
        # Get output grid size for this scale
        input_size_at_scale = img_size // scale_factor
        grid_size = compute_output_sizes(input_size_at_scale, CONFIG['disc_num_layers'])
        
        # Draw receptive field boxes (on full-res image)
        draw_receptive_field_boxes(
            ax, img_size, grid_size, 
            min(effective_rf, img_size),  # Clamp to image size
            CONFIG['rf_colors'][i],
            alpha=CONFIG['rf_alpha'],
            num_boxes=3,
            logger=logger
        )
        
        ax.set_title(f'Receptive Fields for {CONFIG["disc_labels"][i]}\n'
                    f'(RF ≈ {min(effective_rf, img_size)}px, {grid_size}×{grid_size} grid)',
                    fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # ─────────────────────────────────────────────────────────────
    # ROW 4: Feature maps from intermediate layers (Scale 1 only)
    # ─────────────────────────────────────────────────────────────
    if features_all and len(features_all[0]) > 0:
        # Show first 3 intermediate feature maps from D⁽¹⁾
        feats = features_all[0]  # Features from first scale
        num_feat_maps = min(3, len(feats))
        
        for i in range(num_feat_maps):
            ax = fig.add_subplot(gs[3, i])
            feat = feats[i]  # [B, C, H, W]
            
            # Average across channels for visualization
            feat_avg = feat[0].mean(dim=0).cpu().numpy()
            feat_avg = (feat_avg - feat_avg.min()) / (feat_avg.max() - feat_avg.min() + 1e-8)
            
            ax.imshow(feat_avg, cmap='viridis')
            ax.set_title(f'{CONFIG["disc_labels"][0]} Layer {i+1} Features\n'
                        f'({feat.shape[1]}ch, {feat.shape[2]}×{feat.shape[3]})',
                        fontsize=10, fontweight='bold')
            ax.axis('off')
    
    # ─────────────────────────────────────────────────────────────
    # Add legend and title
    # ─────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=CONFIG['rf_colors'][i], alpha=0.5,
                      label=f'{CONFIG["disc_labels"][i]} Receptive Field')
        for i in range(3)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=11, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle(f'Figure 4: Multi-Scale PatchGAN Discriminator (Slice {slice_idx})\n'
                f'Architecture: {CONFIG["disc_num_scales"]} scales × '
                f'{CONFIG["disc_num_layers"]} layers, '
                f'base channels = {CONFIG["disc_base_channels"]}',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    
    # Save PNG
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # Save PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=CONFIG['pdf_dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    
    plt.close()
    
    if logger:
        logger.info(f"  Saved: {os.path.basename(output_path)}")
        logger.info(f"  Saved: {os.path.basename(pdf_path)}")


def create_scale_comparison_figure(ct_corr, ct_output, disc, device, output_path, 
                                   slice_idx, logger=None):
    """
    Create a figure comparing discriminator operation across scales.
    Shows input → discriminator → output grid for each scale side by side.
    """
    if logger:
        logger.info("  Creating scale comparison figure...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    disc_input = torch.cat([ct_corr, ct_output], dim=1)
    scale_inputs = get_multiscale_inputs(disc_input, 3)
    ct_corr_scales = get_multiscale_inputs(ct_corr, 3)
    
    with torch.no_grad():
        logits_all, features_all = disc(disc_input, return_features=True)
    
    cmap = create_disc_colormap()
    
    for i in range(3):
        # Column 0: Input at this scale
        img = tensor_to_display(ct_corr_scales[i])
        axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Input {CONFIG["scale_labels"][i]}\n{img.shape[0]}×{img.shape[1]}',
                            fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Column 1: Architecture diagram (simplified)
        axes[i, 1].text(0.5, 0.5, f'{CONFIG["disc_labels"][i]}\n\n'
                       f'{CONFIG["disc_num_layers"]} Conv Layers\n'
                       f'Stride: 2→2→2→2→1\n'
                       f'Kernel: 4×4',
                       ha='center', va='center', fontsize=11,
                       transform=axes[i, 1].transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Discriminator', fontsize=12, fontweight='bold')
        
        # Column 2: Output grid
        output_grid = logits_all[i].squeeze().cpu().numpy()
        im = plot_discriminator_output(axes[i, 2], output_grid, 
                                       f'Output Grid\n{output_grid.shape[0]}×{output_grid.shape[1]}',
                                       cmap=cmap)
        
        # Column 3: Upsampled output overlaid on input
        output_probs = 1 / (1 + np.exp(-output_grid))
        output_upsampled = F.interpolate(
            torch.from_numpy(output_probs).unsqueeze(0).unsqueeze(0).float(),
            size=(img.shape[0], img.shape[1]),
            mode='nearest'
        ).squeeze().numpy()
        
        axes[i, 3].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].imshow(output_upsampled, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
        axes[i, 3].set_title('Output Overlay', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
    
    # Row labels
    for i in range(3):
        axes[i, 0].set_ylabel(CONFIG['disc_labels'][i], fontsize=14, fontweight='bold',
                             rotation=0, labelpad=40, va='center')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('P(Real)', fontsize=12)
    
    fig.suptitle(f'Multi-Scale Discriminator: Scale-by-Scale Comparison (Slice {slice_idx})',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=CONFIG['pdf_dpi'],
                bbox_inches='tight', format='pdf')
    plt.close()
    
    if logger:
        logger.info(f"  Saved: {os.path.basename(output_path)}")


def create_receptive_field_figure(ct_corr, output_path, slice_idx, logger=None):
    """
    Create dedicated receptive field visualization figure.
    Shows how each scale's receptive field covers different portions of the image.
    """
    if logger:
        logger.info("  Creating receptive field visualization...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    img = tensor_to_display(ct_corr)
    img_size = img.shape[0]
    rf_size = compute_receptive_field(CONFIG['disc_num_layers'])
    
    # Panel 0: Original image
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Panels 1-3: RF visualization for each scale
    for i in range(3):
        axes[i+1].imshow(img, cmap='gray', vmin=0, vmax=1)
        
        scale_factor = 2 ** i
        effective_rf = min(rf_size * scale_factor, img_size)
        input_size_at_scale = img_size // scale_factor
        grid_size = compute_output_sizes(input_size_at_scale, CONFIG['disc_num_layers'])
        
        # Draw grid lines
        create_grid_overlay(axes[i+1], grid_size, img_size, 
                           color=CONFIG['rf_colors'][i], alpha=0.3)
        
        # Draw sample receptive fields
        draw_receptive_field_boxes(
            axes[i+1], img_size, grid_size, effective_rf,
            CONFIG['rf_colors'][i], alpha=0.4, num_boxes=4
        )
        
        axes[i+1].set_title(f'{CONFIG["disc_labels"][i]}\n'
                           f'RF = {effective_rf}px, Grid = {grid_size}×{grid_size}',
                           fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
    
    fig.suptitle(f'Receptive Field Analysis (Slice {slice_idx})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=CONFIG['pdf_dpi'],
                bbox_inches='tight', format='pdf')
    plt.close()
    
    if logger:
        logger.info(f"  Saved: {os.path.basename(output_path)}")


def create_real_vs_fake_comparison(ct_corr, ct_gt, ct_output, disc, device, 
                                   output_path, slice_idx, logger=None):
    """
    Create comparison showing discriminator response to real vs fake images.
    """
    if logger:
        logger.info("  Creating real vs fake comparison...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Prepare inputs
    disc_input_real = torch.cat([ct_corr, ct_gt], dim=1)      # Real pair
    disc_input_fake = torch.cat([ct_corr, ct_output], dim=1)  # Fake pair
    
    with torch.no_grad():
        logits_real, _ = disc(disc_input_real, return_features=False)
        logits_fake, _ = disc(disc_input_fake, return_features=False)
    
    cmap = create_disc_colormap()
    
    # Row labels
    row_labels = ['Input', 'D(Real)', 'D(Fake)']
    
    for scale_idx in range(3):
        # Get images at this scale
        scale_factor = 2 ** scale_idx
        ct_corr_scale = F.avg_pool2d(ct_corr, scale_factor) if scale_factor > 1 else ct_corr
        ct_gt_scale = F.avg_pool2d(ct_gt, scale_factor) if scale_factor > 1 else ct_gt
        ct_output_scale = F.avg_pool2d(ct_output, scale_factor) if scale_factor > 1 else ct_output
        
        # Column header
        col_idx = scale_idx + 1
        
        # Row 0: Input corrupted image
        img = tensor_to_display(ct_corr_scale)
        axes[0, col_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f'{CONFIG["disc_labels"][scale_idx]}\nInput {CONFIG["scale_labels"][scale_idx]}',
                                  fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')
        
        # Row 1: Discriminator output for REAL pair
        real_grid = logits_real[scale_idx].squeeze().cpu().numpy()
        real_probs = 1 / (1 + np.exp(-real_grid))
        im = axes[1, col_idx].imshow(real_probs, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        axes[1, col_idx].set_title(f'Real: μ={real_probs.mean():.3f}', fontsize=11, fontweight='bold')
        axes[1, col_idx].axis('off')
        
        # Annotate values
        h, w = real_probs.shape
        if h <= 8 and w <= 8:
            for i in range(h):
                for j in range(w):
                    val = real_probs[i, j]
                    axes[1, col_idx].text(j, i, f'{val:.2f}', ha='center', va='center',
                                         fontsize=7, color='white' if val < 0.5 else 'black')
        
        # Row 2: Discriminator output for FAKE pair
        fake_grid = logits_fake[scale_idx].squeeze().cpu().numpy()
        fake_probs = 1 / (1 + np.exp(-fake_grid))
        axes[2, col_idx].imshow(fake_probs, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        axes[2, col_idx].set_title(f'Fake: μ={fake_probs.mean():.3f}', fontsize=11, fontweight='bold')
        axes[2, col_idx].axis('off')
        
        if h <= 8 and w <= 8:
            for i in range(h):
                for j in range(w):
                    val = fake_probs[i, j]
                    axes[2, col_idx].text(j, i, f'{val:.2f}', ha='center', va='center',
                                         fontsize=7, color='white' if val < 0.5 else 'black')
        
        if logger:
            logger.debug(f"    Scale {scale_idx}: Real μ={real_probs.mean():.3f}, "
                        f"Fake μ={fake_probs.mean():.3f}")
    
    # Column 0: Labels and legend images
    axes[0, 0].imshow(tensor_to_display(ct_corr), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Corrupted\n(Input)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(tensor_to_display(ct_gt), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth\n(Real)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[2, 0].imshow(tensor_to_display(ct_output), cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('Generator Output\n(Fake)', fontsize=11, fontweight='bold')
    axes[2, 0].axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('P(Real)', fontsize=12)
    
    fig.suptitle(f'Real vs Fake Discriminator Response (Slice {slice_idx})',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=CONFIG['pdf_dpi'],
                bbox_inches='tight', format='pdf')
    plt.close()
    
    if logger:
        logger.info(f"  Saved: {os.path.basename(output_path)}")


def create_architecture_diagram(output_path, logger=None):
    """
    Create a conceptual architecture diagram of the multi-scale discriminator.
    """
    if logger:
        logger.info("  Creating architecture diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=CONFIG['figsize_architecture'])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E8F4FD',
        'conv': '#B8D4E8',
        'output': '#FFE4B5',
        'arrow': '#666666',
        'text': '#333333'
    }
    
    # Title
    ax.text(8, 9.5, 'Multi-Scale PatchGAN Discriminator Architecture',
           ha='center', fontsize=16, fontweight='bold')
    
    # Input box
    input_box = FancyBboxPatch((0.5, 4), 2, 2, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 5, 'Input\n(x_corr, x)\n128×128×2', ha='center', va='center', fontsize=10)
    
    # Draw three discriminator branches
    y_positions = [7, 5, 3]
    scale_labels = ['1×', '1/2×', '1/4×']
    disc_labels = ['D⁽¹⁾', 'D⁽¹/²⁾', 'D⁽¹/⁴⁾']
    input_sizes = ['128×128', '64×64', '32×32']
    output_sizes = ['8×8', '4×4', '2×2']
    
    for i, (y, scale, disc, in_size, out_size) in enumerate(
            zip(y_positions, scale_labels, disc_labels, input_sizes, output_sizes)):
        
        # Downsampling arrow (except for first scale)
        if i > 0:
            ax.annotate('', xy=(3.5, y + 0.5), xytext=(2.5, 5),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
            ax.text(2.8, (y + 5.5) / 2, f'↓{2**i}×', fontsize=9, ha='center')
        else:
            ax.annotate('', xy=(3.5, y + 0.5), xytext=(2.5, 5),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
        
        # Scale input box
        scale_box = FancyBboxPatch((3.5, y - 0.3), 2, 1.6, boxstyle="round,pad=0.05",
                                   facecolor=colors['input'], edgecolor='black', linewidth=1.5)
        ax.add_patch(scale_box)
        ax.text(4.5, y + 0.5, f'{scale}\n{in_size}', ha='center', va='center', fontsize=9)
        
        # Arrow to discriminator
        ax.annotate('', xy=(6.5, y + 0.5), xytext=(5.5, y + 0.5),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
        
        # Discriminator box (conv layers)
        disc_box = FancyBboxPatch((6.5, y - 0.5), 4, 2, boxstyle="round,pad=0.1",
                                  facecolor=colors['conv'], edgecolor='black', linewidth=2)
        ax.add_patch(disc_box)
        ax.text(8.5, y + 0.5, f'{disc}\n5 Conv Layers\n(stride 2,2,2,2,1)',
               ha='center', va='center', fontsize=9)
        
        # Arrow to output
        ax.annotate('', xy=(11.5, y + 0.5), xytext=(10.5, y + 0.5),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
        
        # Output box
        out_box = FancyBboxPatch((11.5, y - 0.3), 2.5, 1.6, boxstyle="round,pad=0.05",
                                 facecolor=colors['output'], edgecolor='black', linewidth=1.5)
        ax.add_patch(out_box)
        ax.text(12.75, y + 0.5, f'Output\n{out_size}×1', ha='center', va='center', fontsize=9)
    
    # Loss aggregation
    ax.annotate('', xy=(15, 5), xytext=(14, 7.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(15, 5), xytext=(14, 5.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(15, 5), xytext=(14, 3.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    
    loss_box = FancyBboxPatch((14.5, 4), 1.2, 2, boxstyle="round,pad=0.1",
                              facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(15.1, 5, 'Σ\nLoss', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Add legend
    legend_y = 1
    for color, label in [(colors['input'], 'Input/Scale'),
                         (colors['conv'], 'Discriminator'),
                         (colors['output'], 'Patch Output')]:
        legend_box = FancyBboxPatch((10, legend_y), 0.8, 0.5, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(legend_box)
        ax.text(11, legend_y + 0.25, label, va='center', fontsize=9)
        legend_y -= 0.7
    
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=CONFIG['pdf_dpi'],
                bbox_inches='tight', format='pdf')
    plt.close()
    
    if logger:
        logger.info(f"  Saved: {os.path.basename(output_path)}")


def save_individual_outputs(ct_corr, ct_gt, ct_output, disc, device, output_dir, 
                           slice_idx, logger=None):
    """
    Save individual components for publication.
    """
    if logger:
        logger.info("  Saving individual outputs...")
    
    panel_dir = os.path.join(output_dir, 'individual_panels', f'slice_{slice_idx}')
    os.makedirs(panel_dir, exist_ok=True)
    
    # Prepare discriminator input
    disc_input = torch.cat([ct_corr, ct_output], dim=1)
    
    with torch.no_grad():
        logits_all, features_all = disc(disc_input, return_features=True)
    
    # Get multi-scale inputs
    ct_corr_scales = get_multiscale_inputs(ct_corr, 3)
    
    cmap = create_disc_colormap()
    
    for i in range(3):
        scale_name = CONFIG['scale_names'][i]
        
        # Save input image at this scale
        fig, ax = plt.subplots(figsize=(8, 8))
        img = tensor_to_display(ct_corr_scales[i])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        plt.savefig(os.path.join(panel_dir, f'input_{scale_name}.png'),
                   dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(panel_dir, f'input_{scale_name}.pdf'),
                   dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
        plt.close()
        
        # Save discriminator output grid
        fig, ax = plt.subplots(figsize=(8, 8))
        output_grid = logits_all[i].squeeze().cpu().numpy()
        probs = 1 / (1 + np.exp(-output_grid))
        im = ax.imshow(probs, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.savefig(os.path.join(panel_dir, f'disc_output_{scale_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(panel_dir, f'disc_output_{scale_name}.pdf'),
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        # Save raw numpy arrays
        np.save(os.path.join(panel_dir, f'input_{scale_name}.npy'), img)
        np.save(os.path.join(panel_dir, f'disc_logits_{scale_name}.npy'), output_grid)
        np.save(os.path.join(panel_dir, f'disc_probs_{scale_name}.npy'), probs)
        
        if logger:
            logger.debug(f"    Saved {scale_name} scale outputs")
    
    # Save feature maps
    if features_all:
        feat_dir = os.path.join(panel_dir, 'feature_maps')
        os.makedirs(feat_dir, exist_ok=True)
        
        for scale_idx, feats in enumerate(features_all):
            for layer_idx, feat in enumerate(feats):
                feat_avg = feat[0].mean(dim=0).cpu().numpy()
                np.save(os.path.join(feat_dir, 
                       f'scale{scale_idx}_layer{layer_idx}_features.npy'), 
                       feat[0].cpu().numpy())
                
                fig, ax = plt.subplots(figsize=(6, 6))
                feat_norm = (feat_avg - feat_avg.min()) / (feat_avg.max() - feat_avg.min() + 1e-8)
                ax.imshow(feat_norm, cmap='viridis')
                ax.axis('off')
                plt.savefig(os.path.join(feat_dir, 
                           f'scale{scale_idx}_layer{layer_idx}_avg.png'),
                           dpi=200, bbox_inches='tight', pad_inches=0)
                plt.close()
    
    if logger:
        logger.info(f"    Saved to: individual_panels/slice_{slice_idx}/")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("FIGURE 4: Multi-Scale Discriminator Illustration")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    
    # Log configuration
    logger.info("")
    logger.info("Discriminator Configuration:")
    logger.info(f"  Input channels: {CONFIG['disc_in_channels']}")
    logger.info(f"  Base channels: {CONFIG['disc_base_channels']}")
    logger.info(f"  Number of layers: {CONFIG['disc_num_layers']}")
    logger.info(f"  Number of scales: {CONFIG['disc_num_scales']}")
    logger.info(f"  Spectral normalization: {CONFIG['disc_use_sn']}")
    logger.info(f"  Input image size: {CONFIG['image_size']}×{CONFIG['image_size']}")
    
    # Compute and log output sizes
    logger.info("")
    logger.info("Expected output grid sizes:")
    for i in range(CONFIG['disc_num_scales']):
        scale_factor = 2 ** i
        input_size = CONFIG['image_size'] // scale_factor
        output_size = compute_output_sizes(input_size, CONFIG['disc_num_layers'])
        logger.info(f"  {CONFIG['disc_labels'][i]}: {input_size}×{input_size} → {output_size}×{output_size}")
    
    rf_size = compute_receptive_field(CONFIG['disc_num_layers'])
    logger.info(f"  Single-scale receptive field: {rf_size} pixels")
    
    # Get device
    device = get_device()
    logger.info(f"\nUsing device: {device}")
    
    # Load discriminator
    logger.info("")
    disc = load_discriminator(device, logger)
    
    # Load generator for getting realistic outputs
    logger.info("")
    logger.info("Loading generator for realistic fake samples...")
    generator_path = PATHS['checkpoints']['A0_baseline']
    logger.info(f"  Checkpoint: {generator_path}")
    generator = load_generator(generator_path, device)
    logger.info("  Generator loaded successfully")
    
    # Load test dataset
    logger.info("")
    logger.info("Loading test dataset...")
    test_dataset = load_test_dataset()
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Get selected slices
    selected_indices = get_selected_slices()
    logger.info(f"  Using {len(selected_indices)} selected slices")
    
    # Create architecture diagram (only once)
    logger.info("")
    logger.info("Creating architecture diagram...")
    arch_path = os.path.join(output_dir, 'architecture_diagram.png')
    create_architecture_diagram(arch_path, logger)
    
    # Statistics tracking
    processed_count = 0
    
    # Process slices
    logger.info("")
    logger.info(f"Processing {CONFIG['num_examples']} example slices...")
    
    for i, idx in enumerate(selected_indices[:CONFIG['num_examples']]):
        logger.info("")
        logger.info(f"Processing slice {idx} ({i+1}/{CONFIG['num_examples']})...")
        
        # Get data
        batch = test_dataset[idx]
        ct_corr = batch[0].unsqueeze(0).to(device)   # [1, 1, H, W] corrupted
        ct_gt = batch[1].unsqueeze(0).to(device)     # [1, 1, H, W] ground truth
        
        # Generate output using generator
        with torch.no_grad():
            ct_output = generator(ct_corr)
        
        logger.info(f"  Input shape: {ct_corr.shape}")
        logger.info(f"  Output shape: {ct_output.shape}")
        
        # Create main composite figure
        main_path = os.path.join(output_dir, f'figure4_slice_{idx}.png')
        create_main_figure(ct_corr, ct_output, disc, device, main_path, idx, logger)
        
        # Create scale comparison figure
        scale_path = os.path.join(output_dir, f'scale_comparison_slice_{idx}.png')
        create_scale_comparison_figure(ct_corr, ct_output, disc, device, 
                                       scale_path, idx, logger)
        
        # Create receptive field figure
        rf_path = os.path.join(output_dir, f'receptive_fields_slice_{idx}.png')
        create_receptive_field_figure(ct_corr, rf_path, idx, logger)
        
        # Create real vs fake comparison
        rvf_path = os.path.join(output_dir, f'real_vs_fake_slice_{idx}.png')
        create_real_vs_fake_comparison(ct_corr, ct_gt, ct_output, disc, device,
                                       rvf_path, idx, logger)
        
        # Save individual outputs
        save_individual_outputs(ct_corr, ct_gt, ct_output, disc, device,
                               output_dir, idx, logger)
        
        processed_count += 1
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'figure': 'Figure 4: Multi-Scale Discriminator Illustration',
            'discriminator_config': {
                'in_channels': CONFIG['disc_in_channels'],
                'base_channels': CONFIG['disc_base_channels'],
                'num_layers': CONFIG['disc_num_layers'],
                'num_scales': CONFIG['disc_num_scales'],
                'use_spectral_norm': CONFIG['disc_use_sn'],
            },
            'image_size': CONFIG['image_size'],
            'receptive_field': rf_size,
            'output_grid_sizes': {
                CONFIG['scale_names'][i]: compute_output_sizes(
                    CONFIG['image_size'] // (2**i), CONFIG['disc_num_layers']
                ) for i in range(3)
            },
            'num_examples_processed': processed_count,
            'selected_indices': [int(x) for x in selected_indices[:CONFIG['num_examples']]],
        }, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("FIGURE 4 GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {processed_count} slices")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - architecture_diagram.png/pdf: Conceptual architecture diagram")
    logger.info("  - figure4_slice_*.png/pdf: Main composite figures")
    logger.info("  - scale_comparison_*.png/pdf: Scale-by-scale comparison")
    logger.info("  - receptive_fields_*.png/pdf: Receptive field visualization")
    logger.info("  - real_vs_fake_*.png/pdf: Discriminator response comparison")
    logger.info("  - individual_panels/: High-res individual components")
    logger.info("    - input_*.png/pdf/npy: Input images at each scale")
    logger.info("    - disc_output_*.png/pdf/npy: Discriminator outputs")
    logger.info("    - feature_maps/: Intermediate feature maps")
    logger.info("  - config.json: Configuration and parameters")
    logger.info("  - figure4_generation.log: Detailed log file")


if __name__ == "__main__":
    main()
