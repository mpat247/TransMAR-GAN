"""
Shared Configuration for All Figure Inference Scripts

This module provides:
1. Common paths and settings used across all figure generation scripts
2. Shared slice selection logic for consistent comparisons
3. Utility functions for loading models and data

All figure scripts should import from here to ensure consistency.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator.ngswin import NGswin
from data.datasets import TestDataset

# ═══════════════════════════════════════════════════════════════
# PATHS CONFIGURATION
# ═══════════════════════════════════════════════════════════════

PATHS = {
    # Dataset
    'data_root': '/home/Drive-D/SynDeepLesion/',
    'test_mask_path': '/home/Drive-D/SynDeepLesion/testmask.npy',
    
    # Ablation results folder (completed run)
    'ablation_root': '/home/grad/mppatel/Documents/DCGAN/ablation_results/loss_ablations_20251206_213904',
    
    # Key checkpoints
    'checkpoints': {
        # Full model with all losses (baseline)
        'A0_baseline': '/home/grad/mppatel/Documents/DCGAN/combined_results/run_20251202_211759/checkpoints/best_model.pth',
        
        # No physics loss
        'A1_no_physics': '/home/grad/mppatel/Documents/DCGAN/ablation_results/loss_ablations_20251206_213904/A1_no_physics/checkpoints/best_model.pth',
        
        # Original best checkpoint (before ablations)
        'original_best': '/home/grad/mppatel/Documents/DCGAN/training_checkpoints/bestcheckpoint.pth',
        
        # MSE-only (will be available after current ablation run)
        'A0_mse_only': None,  # UPDATE when available
    },
    
    # Output root
    'output_root': '/home/grad/mppatel/Documents/DCGAN/inference_figure_outputs/',
}

# ═══════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    'img_size': 128,
    'in_chans': 1,
    'embed_dim': 60,
    'depths': [6, 6, 6, 6],
    'num_heads': [6, 6, 6, 6],
    'mlp_ratio': 2,
    'upscale': 1,
    'resi_connection': '1conv',
}

# TorchRadon configuration (for physics consistency)
RADON_CONFIG = {
    'num_angles': 180,
    'det_count': None,  # Will be set to image size
}

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION SETTINGS
# ═══════════════════════════════════════════════════════════════

VIS_CONFIG = {
    'cmap': 'gray',
    'vmin': 0.0,
    'vmax': 1.0,
    'dpi': 200,
    'figsize_grid': (15, 50),
    'figsize_individual': (15, 5),
    'metal_threshold': 0.9,
}

# ═══════════════════════════════════════════════════════════════
# SELECTED SLICES - SHARED ACROSS ALL FIGURES
# ═══════════════════════════════════════════════════════════════

# Number of slices to use for detailed analysis
NUM_SELECTED_SLICES = 25

# Cache file for selected slice indices (computed once, reused)
SELECTED_SLICES_CACHE = os.path.join(PATHS['output_root'], 'selected_slice_indices.npy')

# These will be populated by select_best_slices() function
SELECTED_SLICE_INDICES = None

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2

def normalize_to_model(tensor):
    """Convert from [0, 1] to [-1, 1] for model input."""
    return tensor * 2 - 1

def get_device():
    """Get the best available device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Generator(torch.nn.Module):
    """
    Generator wrapper class matching the training setup.
    The checkpoints save state_dict with 'main.' prefix because
    training uses this wrapper: self.main = NGswin()
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.main = NGswin()
    
    def forward(self, x):
        return self.main(x)


def load_generator(checkpoint_path, device=None):
    """
    Load NGswin generator from checkpoint.
    
    The checkpoints were saved with a Generator wrapper class that
    contains NGswin as self.main, so state_dict keys have 'main.' prefix.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (default: auto-detect)
    
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = get_device()
    
    # Use Generator wrapper to match checkpoint structure
    model = Generator().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
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
    return model

def load_test_dataset():
    """Load the test dataset."""
    # TestDataset expects the loaded mask array, not the path
    test_mask = np.load(PATHS['test_mask_path'])
    return TestDataset(PATHS['data_root'], test_mask)

def compute_artifact_score(ma_ct, gt):
    """
    Compute an artifact score to rank slices by artifact visibility.
    Higher score = more visible artifacts = better for demonstration.
    
    Args:
        ma_ct: Metal artifact CT image (numpy array or tensor, [0,1] range)
        gt: Ground truth clean image (numpy array or tensor, [0,1] range)
    
    Returns:
        float: Artifact visibility score
    """
    if torch.is_tensor(ma_ct):
        ma_np = ma_ct.squeeze().cpu().numpy()
    else:
        ma_np = ma_ct.squeeze()
    
    if torch.is_tensor(gt):
        gt_np = gt.squeeze().cpu().numpy()
    else:
        gt_np = gt.squeeze()
    
    # Artifact score based on difference and metal presence
    diff = np.abs(ma_np - gt_np)
    
    # Metal presence (high intensity regions)
    metal_mask = ma_np > VIS_CONFIG['metal_threshold']
    metal_area = np.sum(metal_mask)
    
    # Artifact intensity in non-metal regions
    artifact_mask = diff > 0.1  # Significant differences
    artifact_intensity = np.mean(diff[artifact_mask]) if np.sum(artifact_mask) > 0 else 0
    
    # Combined score: prioritize slices with metal AND visible artifacts
    score = metal_area * artifact_intensity * np.sum(artifact_mask)
    
    return float(score)

def select_best_slices(force_recompute=False):
    """
    Select the best slices for visualization based on artifact visibility.
    Results are cached to ensure consistency across all figure scripts.
    
    Args:
        force_recompute: If True, recompute even if cache exists
    
    Returns:
        list: Indices of selected slices (sorted by artifact score, descending)
    """
    global SELECTED_SLICE_INDICES
    
    # Check cache first
    if not force_recompute and os.path.exists(SELECTED_SLICES_CACHE):
        print(f"Loading cached slice indices from: {SELECTED_SLICES_CACHE}")
        SELECTED_SLICE_INDICES = list(np.load(SELECTED_SLICES_CACHE))
        print(f"  Loaded {len(SELECTED_SLICE_INDICES)} slice indices")
        return SELECTED_SLICE_INDICES
    
    print("Computing best slices based on artifact visibility...")
    print("  (This will be cached for future runs)")
    
    # Load test dataset
    test_dataset = load_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Score all slices
    slice_scores = []
    
    for idx, batch in enumerate(tqdm(test_loader, desc="Scoring slices")):
        ma_ct = batch[0]  # Input with artifacts
        gt = batch[1]     # Ground truth
        
        score = compute_artifact_score(denormalize(ma_ct), denormalize(gt))
        slice_scores.append((idx, score))
    
    # Sort by score (descending) and select top N
    slice_scores.sort(key=lambda x: x[1], reverse=True)
    SELECTED_SLICE_INDICES = [idx for idx, score in slice_scores[:NUM_SELECTED_SLICES]]
    
    # Save cache
    os.makedirs(os.path.dirname(SELECTED_SLICES_CACHE), exist_ok=True)
    np.save(SELECTED_SLICES_CACHE, np.array(SELECTED_SLICE_INDICES))
    print(f"  Cached {len(SELECTED_SLICE_INDICES)} slice indices to: {SELECTED_SLICES_CACHE}")
    
    # Print selected indices
    print(f"\nSelected slice indices (top {NUM_SELECTED_SLICES} by artifact visibility):")
    print(f"  {SELECTED_SLICE_INDICES}")
    
    return SELECTED_SLICE_INDICES

def get_selected_slices():
    """
    Get the selected slice indices, computing if necessary.
    
    Returns:
        list: Indices of selected slices
    """
    global SELECTED_SLICE_INDICES
    
    if SELECTED_SLICE_INDICES is None:
        SELECTED_SLICE_INDICES = select_best_slices()
    
    return SELECTED_SLICE_INDICES

def check_checkpoint_exists(checkpoint_key):
    """
    Check if a checkpoint exists and is accessible.
    
    Args:
        checkpoint_key: Key in PATHS['checkpoints'] dict
    
    Returns:
        tuple: (exists: bool, path: str or None)
    """
    path = PATHS['checkpoints'].get(checkpoint_key)
    
    if path is None:
        return False, None
    
    if os.path.exists(path):
        return True, path
    
    return False, path

def print_checkpoint_status():
    """Print the status of all checkpoints."""
    print("\nCheckpoint Status:")
    print("-" * 60)
    
    for key, path in PATHS['checkpoints'].items():
        if path is None:
            status = "❌ NOT SET"
        elif os.path.exists(path):
            status = "✓ EXISTS"
        else:
            status = "⚠️  NOT FOUND"
        
        print(f"  {key}: {status}")
        if path:
            print(f"      {path}")
    
    print("-" * 60)

# ═══════════════════════════════════════════════════════════════
# MAIN - Run this to pre-compute slice selection
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SHARED CONFIG - Inference Figure Outputs")
    print("=" * 70)
    
    # Print checkpoint status
    print_checkpoint_status()
    
    # Compute and cache slice selection
    print("\n" + "=" * 70)
    print("SLICE SELECTION")
    print("=" * 70)
    
    indices = select_best_slices(force_recompute=False)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nSelected {len(indices)} slices for visualization:")
    print(f"  Indices: {indices}")
    print(f"\nCache file: {SELECTED_SLICES_CACHE}")
    print("\nAll figure scripts will use these same slices for consistency.")
