"""
Combined Training Script: NGswin Generator + MS-PatchGAN Discriminator
=======================================================================

This script combines:
- NGswin Generator
- Multi-Scale PatchGAN Discriminator (3 scales, spectral norm, hinge loss)
- Metal-Aware Reconstruction Loss (Eq 3)
- Metal-Aware Edge Loss (Eq 4)
- Feature Matching Loss (Section 2.2)
- Physics-Consistency Loss (Eq 6)
- TTUR optimizer setup (lrD = 2 * lrG)

Based on: "Implementation Notes: Robust Adversarial Head and Metal-Aware 
Physics-Consistent Generator" - NgSwinGAN Project
"""

import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
from torch_radon import Radon

# ═══════════════════════════════════════════════════════════════
# IMPORTS: Models and Loss Functions
# ═══════════════════════════════════════════════════════════════

# Generator
from models.generator.ngswin import NGswin

# Discriminator (Multi-Scale PatchGAN)
from models.discriminator.ms_patchgan import MultiScaleDiscriminator

# Dataset
from data.datasets import MARTrainDataset, MARValDataset, SpineWebTrainDataset

# Loss functions
from losses.gan_losses import (
    # Adversarial losses (hinge)
    hinge_d_loss,
    hinge_g_loss,
    # Feature matching
    feature_matching_loss,
    # Metal-aware losses
    extract_metal_mask,
    dilate_mask,
    compute_metal_aware_loss,
    compute_weight_map,
    compute_metal_aware_edge_loss,
    metal_consistency_loss,
    # Physics loss
    physics_loss_syn,
)

# ═══════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Using: {device}")

# ═══════════════════════════════════════════════════════════════
# DATASET SELECTION
# ═══════════════════════════════════════════════════════════════
USE_SPINEWEB = False  # False = SynDeepLesion from scratch, True = SpineWeb fine-tuning

# Dataset paths
SYNDEEPLESION_PATH = "/home/Drive-D/SynDeepLesion/"
SPINEWEB_TRAIN_ARTIFACT = "/home/Drive-D/UWSpine_adn/train/synthesized_metal_transfer/"
SPINEWEB_TRAIN_CLEAN = "/home/Drive-D/UWSpine_adn/train/no_metal/"
SPINEWEB_TEST_ARTIFACT = "/home/Drive-D/UWSpine_adn/test/synthesized_metal_transfer/"
SPINEWEB_TEST_CLEAN = "/home/Drive-D/UWSpine_adn/test/no_metal/"

# ═══════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════

# Data
PATCH_SIZE = 128  # Increased from 64 to support multi-scale discriminator
BATCH_SIZE = 4
workers = 2

# Training
num_epochs = 25 if USE_SPINEWEB else 100
ngpu = 1
test_every_n_epochs = 5  # Run validation/test every N epochs

# TTUR: Two Time-Scale Update Rule (lrD = 2 * lrG)
lrG = 1e-4 if not USE_SPINEWEB else 1e-5
lrD = 2 * lrG

# Optimizer
beta1 = 0.5
beta2 = 0.999

# ═══════════════════════════════════════════════════════════════
# LOSS WEIGHTS (from Section 2.6 of implementation notes)
# ═══════════════════════════════════════════════════════════════
lambda_adv = 0.1      # Adversarial loss weight
lambda_FM = 10.0      # Feature matching loss weight
lambda_rec = 1.0      # Metal-aware reconstruction loss weight
lambda_edge = 0.2     # Metal-aware edge loss weight
lambda_phys = 0.02    # Physics-consistency loss weight (reduced from 0.5 to balance with rec loss)
lambda_metal = 0.5    # Metal-consistency loss weight

# Metal mask parameters
metal_threshold = 0.6     # Threshold for normalized data in [-1,1] range (metal regions > 0.6)
dilation_radius = 5       # Dilation radius for metal band
beta_weight = 1.0         # Weight factor for metal-aware weighting
w_max = 3.0               # Maximum weight clipping

# ═══════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════
manualSeed = 999
print(f"[Seed] Random Seed: {manualSeed}")
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT / RESULTS SETUP
# ═══════════════════════════════════════════════════════════════
run_root = "./combined_results"
os.makedirs(run_root, exist_ok=True)
run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = os.path.join(run_root, run_id)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "test_examples"), exist_ok=True)

print(f"[Run] Results will be saved under {run_dir}")

# TensorBoard
writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

# ═══════════════════════════════════════════════════════════════
# LOAD DATASET
# ═══════════════════════════════════════════════════════════════
if USE_SPINEWEB:
    print("═" * 60)
    print("LOADING SPINEWEB DATASET")
    print("═" * 60)
    
    train_dataset = SpineWebTrainDataset(
        artifact_dir=SPINEWEB_TRAIN_ARTIFACT,
        clean_dir=SPINEWEB_TRAIN_CLEAN,
        patchSize=PATCH_SIZE,
        paired=True,
        hu_range=(-1000, 2000)
    )
    
    # Test dataset (if exists)
    if os.path.isdir(SPINEWEB_TEST_ARTIFACT) and os.path.isdir(SPINEWEB_TEST_CLEAN):
        test_dataset = SpineWebTrainDataset(
            artifact_dir=SPINEWEB_TEST_ARTIFACT,
            clean_dir=SPINEWEB_TEST_CLEAN,
            patchSize=PATCH_SIZE,
            paired=True,
            hu_range=(-1000, 2000)
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers)
    else:
        test_loader = None
else:
    print("═" * 60)
    print("LOADING SYNDEEPLESION DATASET")
    print("═" * 60)
    
    train_mask = np.load(os.path.join(SYNDEEPLESION_PATH, 'trainmask.npy'))
    train_dataset = MARTrainDataset(
        SYNDEEPLESION_PATH,
        patchSize=PATCH_SIZE,
        length=BATCH_SIZE * 4000,
        mask=train_mask
    )
    
    # Validation dataset (10% of training data)
    val_dataset = MARValDataset(
        dir=SYNDEEPLESION_PATH,
        mask=train_mask
    )
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers)
    print(f"[Data] Validation samples: {len(val_dataset)}")

dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
print(f"[Data] Train samples: {len(train_dataset)}, Batches: {len(dataloader)}")

# Fixed batch for visualization
fixed_batch = next(iter(dataloader))
fixed_ct = fixed_batch[0].to(device)
fixed_gt = fixed_batch[1].to(device)

# ═══════════════════════════════════════════════════════════════
# WEIGHT INITIALIZATION
# ═══════════════════════════════════════════════════════════════
def weights_init(m):
    """Initialize weights for Conv and BatchNorm layers."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

# ═══════════════════════════════════════════════════════════════
# BUILD GENERATOR (NGswin)
# ═══════════════════════════════════════════════════════════════
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = NGswin()
        
    def forward(self, x):
        return self.main(x)

print("[Init] Building Generator (NGswin)...")
netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG)

netG.apply(weights_init)
print(f"[Init] Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")

# ═══════════════════════════════════════════════════════════════
# BUILD DISCRIMINATOR (Multi-Scale PatchGAN)
# ═══════════════════════════════════════════════════════════════
print("[Init] Building Discriminator (MS-PatchGAN)...")
netD = MultiScaleDiscriminator(
    in_channels=2,      # concat(ct, real/fake) -> 2 channels
    base_channels=64,
    num_layers=5,
    num_scales=3,       # D(1), D(1/2), D(1/4)
    use_sn=True         # Spectral Normalization
).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD)

netD.apply(weights_init)
print(f"[Init] Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

# ═══════════════════════════════════════════════════════════════
# OPTIMIZERS (TTUR: lrD = 2 * lrG)
# ═══════════════════════════════════════════════════════════════
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, beta2))
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, beta2))

print(f"[Optim] Generator LR: {lrG}, Discriminator LR: {lrD} (TTUR)")

# ═══════════════════════════════════════════════════════════════
# INITIALIZE TORCHRADON PROJECTOR (for physics loss)
# ═══════════════════════════════════════════════════════════════
num_angles = 180
angles = np.linspace(0, np.pi, num_angles, dtype=np.float32)
projector = Radon(PATCH_SIZE, angles)
print(f"[Init] TorchRadon projector: {PATCH_SIZE}x{PATCH_SIZE}, {num_angles} angles")

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT LOADING - Resume Training
# ═══════════════════════════════════════════════════════════════
start_epoch = 0
resume_training = input("\n[Resume] Start new training or continue from checkpoint? (new/continue): ").strip().lower()

if resume_training == 'continue':
    # List available runs
    if os.path.exists(run_root):
        runs = [d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))]
        if runs:
            print("\n[Resume] Available runs:")
            for i, r in enumerate(runs):
                print(f"  {i+1}. {r}")
            run_idx = int(input(f"[Resume] Select run (1-{len(runs)}): ")) - 1
            selected_run = os.path.join(run_root, runs[run_idx])
            
            # List checkpoints in selected run
            ckpt_dir = os.path.join(selected_run, 'checkpoints')
            if os.path.exists(ckpt_dir):
                ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
                if ckpts:
                    print("\n[Resume] Available checkpoints:")
                    for i, c in enumerate(ckpts):
                        print(f"  {i+1}. {c}")
                    ckpt_idx = int(input(f"[Resume] Select checkpoint (1-{len(ckpts)}): ")) - 1
                    resume_path = os.path.join(ckpt_dir, ckpts[ckpt_idx])
                    
                    print(f"[Resume] Loading from {resume_path}")
                    checkpoint = torch.load(resume_path, map_location=device)
                    netG.load_state_dict(checkpoint['netG_state_dict'])
                    netD.load_state_dict(checkpoint['netD_state_dict'])
                    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
                    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
                    start_epoch = checkpoint['epoch']
                    
                    # Use the same run directory
                    run_dir = selected_run
                    print(f"[Resume] Continuing from epoch {start_epoch}")
                else:
                    print("[Resume] No checkpoints found, starting fresh")
            else:
                print("[Resume] No checkpoint directory found, starting fresh")
        else:
            print("[Resume] No previous runs found, starting fresh")
    else:
        print("[Resume] No results directory found, starting fresh")
else:
    print("[Checkpoint] Starting new training")

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STARTING TRAINING")
print(f"Epochs: {num_epochs}, Batch size: {BATCH_SIZE}")
print(f"Dataset: {'SpineWeb' if USE_SPINEWEB else 'SynDeepLesion'}")
print(f"Validation every {test_every_n_epochs} epochs")
print("═" * 60 + "\n")

# Loss tracking - individual losses for plotting
G_losses = []
D_losses = []
adv_losses = []
FM_losses = []
rec_losses = []
edge_losses = []
phys_losses = []
metal_losses = []
val_psnr_history = []
best_psnr = 0.0
iters = 0

def plot_loss_curves(save_dir, epoch):
    """Plot and save loss curves for all losses."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Loss Curves - Epoch {epoch}', fontsize=16)
    
    # Smooth the curves with moving average for better visualization
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    losses_to_plot = [
        ('D Loss', D_losses, axes[0, 0]),
        ('G Total Loss', G_losses, axes[0, 1]),
        ('Adversarial Loss', adv_losses, axes[0, 2]),
        ('Feature Matching Loss', FM_losses, axes[0, 3]),
        ('Reconstruction Loss', rec_losses, axes[1, 0]),
        ('Edge Loss', edge_losses, axes[1, 1]),
        ('Physics Loss', phys_losses, axes[1, 2]),
        ('Metal Loss', metal_losses, axes[1, 3]),
    ]
    
    for title, data, ax in losses_to_plot:
        if len(data) > 0:
            ax.plot(data, alpha=0.3, color='blue', label='Raw')
            smoothed = smooth(data)
            if len(smoothed) > 0:
                ax.plot(range(len(data) - len(smoothed), len(data)), smoothed, 
                       color='red', linewidth=2, label='Smoothed')
            ax.set_title(title)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'loss_curves_epoch_{epoch:03d}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  [Plot] Loss curves saved: {plot_path}")

for epoch in range(start_epoch, num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    
    epoch_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch+1}",
        leave=False,
        dynamic_ncols=True
    )
    
    for i, data in epoch_bar:
        # Get data
        ct = data[0].to(device)      # Corrupted input (xcorr)
        real = data[1].to(device)    # Clean ground truth (x)
        
        # ════════════════════════════════════════════════════════
        # (1) UPDATE DISCRIMINATOR
        # ════════════════════════════════════════════════════════
        netD.zero_grad()
        
        # Generate fake (no grad for D update)
        with torch.no_grad():
            fake = netG(ct)
        
        # Concatenate input with real/fake for discriminator
        real_pair = torch.cat([ct, real], dim=1)  # (B, 2, H, W)
        fake_pair = torch.cat([ct, fake], dim=1)  # (B, 2, H, W)
        
        # Get discriminator outputs (multi-scale logits)
        real_logits, _ = netD(real_pair, return_features=False)
        fake_logits, _ = netD(fake_pair, return_features=False)
        
        # Hinge loss for discriminator (Eq 1)
        loss_D = hinge_d_loss(real_logits, fake_logits)
        
        loss_D.backward()
        optimizerD.step()
        
        # ════════════════════════════════════════════════════════
        # (2) UPDATE GENERATOR
        # ════════════════════════════════════════════════════════
        netG.zero_grad()
        
        # Generate fake (with grad for G update)
        fake = netG(ct)
        fake_pair = torch.cat([ct, fake], dim=1)
        
        # Get discriminator outputs with features (for FM loss)
        fake_logits, fake_feats = netD(fake_pair, return_features=True)
        
        # Get real features (no grad needed)
        with torch.no_grad():
            _, real_feats = netD(real_pair, return_features=True)
        
        # ─────────────────────────────────────────────────────────
        # Adversarial Loss (Eq 2): fool discriminator
        # ─────────────────────────────────────────────────────────
        loss_G_adv = hinge_g_loss(fake_logits)
        
        # ─────────────────────────────────────────────────────────
        # Feature Matching Loss (Section 2.2)
        # ─────────────────────────────────────────────────────────
        loss_FM = feature_matching_loss(real_feats, fake_feats)
        
        # ─────────────────────────────────────────────────────────
        # Metal-Aware Reconstruction Loss (Eq 3)
        # ─────────────────────────────────────────────────────────
        loss_rec_mw = compute_metal_aware_loss(
            fake, real, ct,
            beta=beta_weight,
            radius=dilation_radius,
            w_max=w_max,
            threshold=metal_threshold
        )
        
        # ─────────────────────────────────────────────────────────
        # Metal-Aware Edge Loss (Eq 4)
        # ─────────────────────────────────────────────────────────
        w = compute_weight_map(
            ct,
            beta=beta_weight,
            radius=dilation_radius,
            w_max=w_max,
            threshold=metal_threshold
        )
        loss_edge_mw = compute_metal_aware_edge_loss(fake, real, w)
        
        # ─────────────────────────────────────────────────────────
        # Physics-Consistency Loss (Eq 6)
        # ─────────────────────────────────────────────────────────
        M = extract_metal_mask(ct, threshold=metal_threshold)
        loss_phys = physics_loss_syn(fake, real, M, projector)
        
        # ─────────────────────────────────────────────────────────
        # Metal-Consistency Loss (L_metal)
        # ─────────────────────────────────────────────────────────
        loss_metal = metal_consistency_loss(fake, real, M)
        
        # ─────────────────────────────────────────────────────────
        # Total Generator Loss (Eq 9, partial)
        # ─────────────────────────────────────────────────────────
        loss_G = (
            lambda_adv   * loss_G_adv +
            lambda_FM    * loss_FM +
            lambda_rec   * loss_rec_mw +
            lambda_edge  * loss_edge_mw +
            lambda_phys  * loss_phys +
            lambda_metal * loss_metal
        )
        
        loss_G.backward()
        optimizerG.step()
        
        # ════════════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════════════
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())
        adv_losses.append(loss_G_adv.item())
        FM_losses.append(loss_FM.item())
        rec_losses.append(loss_rec_mw.item())
        edge_losses.append(loss_edge_mw.item())
        phys_losses.append(loss_phys.item())
        metal_losses.append(loss_metal.item())
        
        # Update progress bar
        epoch_bar.set_postfix({
            'D': f"{loss_D.item():.4f}",
            'G': f"{loss_G.item():.4f}",
            'adv': f"{loss_G_adv.item():.3f}",
            'FM': f"{loss_FM.item():.3f}",
            'rec': f"{loss_rec_mw.item():.3f}",
        })
        
        # Print detailed stats every 50 batches
        if i % 50 == 0:
            print(f"  [{i}/{len(dataloader)}] "
                  f"D: {loss_D.item():.4f} | "
                  f"G: {loss_G.item():.4f} "
                  f"(adv: {loss_G_adv.item():.3f}, "
                  f"FM: {loss_FM.item():.3f}, "
                  f"rec: {loss_rec_mw.item():.3f}, "
                  f"edge: {loss_edge_mw.item():.3f}, "
                  f"phys: {loss_phys.item():.3f}, "
                  f"metal: {loss_metal.item():.3f})")
        
        # TensorBoard logging
        if iters % 100 == 0:
            writer.add_scalar('Loss/D', loss_D.item(), iters)
            writer.add_scalar('Loss/G_total', loss_G.item(), iters)
            writer.add_scalar('Loss/G_adv', loss_G_adv.item(), iters)
            writer.add_scalar('Loss/G_FM', loss_FM.item(), iters)
            writer.add_scalar('Loss/G_rec_mw', loss_rec_mw.item(), iters)
            writer.add_scalar('Loss/G_edge_mw', loss_edge_mw.item(), iters)
            writer.add_scalar('Loss/G_phys', loss_phys.item(), iters)
            writer.add_scalar('Loss/G_metal', loss_metal.item(), iters)
        
        # Save sample images every 500 iterations
        if iters % 500 == 0:
            with torch.no_grad():
                sample_fake = netG(fixed_ct)
            
            # Save grid: input | output | ground truth
            grid = vutils.make_grid(
                torch.cat([fixed_ct[:4], sample_fake[:4], fixed_gt[:4]], dim=0),
                nrow=4,
                normalize=True,
                padding=2
            )
            vutils.save_image(grid, os.path.join(run_dir, "samples", f"iter_{iters:06d}.png"))
        
        iters += 1
    
    # ════════════════════════════════════════════════════════════
    # SAVE CHECKPOINT (end of epoch)
    # ════════════════════════════════════════════════════════════
    ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch_{epoch+1:03d}.pth")
    torch.save({
        'epoch': epoch + 1,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'loss_G': loss_G.item(),
        'loss_D': loss_D.item(),
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")
    
    # ════════════════════════════════════════════════════════════
    # PLOT LOSS CURVES (every 10 epochs)
    # ════════════════════════════════════════════════════════════
    if (epoch + 1) % 10 == 0:
        plot_loss_curves(run_dir, epoch + 1)
    
    # ════════════════════════════════════════════════════════════
    # PERIODIC VALIDATION/TEST (every N epochs)
    # ════════════════════════════════════════════════════════════
    if test_loader is not None and (epoch + 1) % test_every_n_epochs == 0:
        print(f"\n  [Validation] Running test at epoch {epoch+1}...")
        netG.eval()
        
        val_mse_list = []
        val_psnr_list = []
        
        with torch.no_grad():
            for val_idx, val_batch in enumerate(test_loader):
                # MARValDataset returns (artifact, clean, LI, mask)
                # SpineWebTrainDataset returns (artifact, clean)
                val_ct = val_batch[0].to(device)
                val_gt = val_batch[1].to(device)
                
                val_pred = netG(val_ct)
                
                # MSE in normalized space
                mse_val = F.mse_loss(val_pred, val_gt, reduction='mean').item()
                val_mse_list.append(mse_val)
                
                # PSNR: data range is 2 for [-1, 1] normalized images
                max_val = 1.0
                psnr_val = 10 * np.log10((2 * max_val) ** 2 / (mse_val + 1e-8))
                val_psnr_list.append(psnr_val)
                
                # Save a few qualitative examples
                if val_idx < 5:
                    def denorm(x):
                        return (x + 1.0) / 2.0
                    
                    grid = vutils.make_grid(
                        torch.cat([denorm(val_ct), denorm(val_pred), denorm(val_gt)], dim=0),
                        nrow=1,
                        padding=2,
                        normalize=False
                    )
                    save_path = os.path.join(run_dir, "test_examples", f"epoch_{epoch+1:03d}_sample_{val_idx:03d}.png")
                    vutils.save_image(grid, save_path)
        
        mean_mse = float(np.mean(val_mse_list))
        mean_psnr = float(np.mean(val_psnr_list))
        val_psnr_history.append((epoch + 1, mean_psnr))
        
        print(f"  [Validation] Epoch {epoch+1}: MSE={mean_mse:.6f}, PSNR={mean_psnr:.2f} dB ({len(val_mse_list)} samples)")
        
        # Log to TensorBoard
        writer.add_scalar('Val/MSE', mean_mse, epoch + 1)
        writer.add_scalar('Val/PSNR', mean_psnr, epoch + 1)
        
        # Save best model
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'best_psnr': best_psnr,
            }, best_ckpt_path)
            print(f"  [Validation] New best PSNR! Saved to {best_ckpt_path}")
        
        # Back to training mode
        netG.train()

# ═══════════════════════════════════════════════════════════════
# FINAL TEST EVALUATION
# ═══════════════════════════════════════════════════════════════
if test_loader is not None:
    print("\n" + "═" * 60)
    print("FINAL TEST EVALUATION")
    print("═" * 60)
    
    netG.eval()
    final_mse_list = []
    final_psnr_list = []
    
    with torch.no_grad():
        for test_idx, test_batch in enumerate(test_loader):
            test_ct = test_batch[0].to(device)
            test_gt = test_batch[1].to(device)
            
            test_pred = netG(test_ct)
            
            mse_test = F.mse_loss(test_pred, test_gt, reduction='mean').item()
            final_mse_list.append(mse_test)
            
            psnr_test = 10 * np.log10((2 * 1.0) ** 2 / (mse_test + 1e-8))
            final_psnr_list.append(psnr_test)
            
            # Save more examples for final test
            if test_idx < 20:
                def denorm(x):
                    return (x + 1.0) / 2.0
                
                grid = vutils.make_grid(
                    torch.cat([denorm(test_ct), denorm(test_pred), denorm(test_gt)], dim=0),
                    nrow=1,
                    padding=2,
                    normalize=False
                )
                save_path = os.path.join(run_dir, "test_examples", f"final_sample_{test_idx:03d}.png")
                vutils.save_image(grid, save_path)
    
    final_mse = float(np.mean(final_mse_list))
    final_psnr = float(np.mean(final_psnr_list))
    
    print(f"[Final Test] MSE: {final_mse:.6f}, PSNR: {final_psnr:.2f} dB ({len(final_mse_list)} samples)")
    print(f"[Final Test] Best validation PSNR during training: {best_psnr:.2f} dB")
    
    # Save final test metrics
    test_metrics = {
        'final_mse': final_mse,
        'final_psnr': final_psnr,
        'best_val_psnr': best_psnr,
        'num_samples': len(final_mse_list),
        'val_psnr_history': val_psnr_history,
    }
    with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# SAVE FINAL OUTPUTS
# ═══════════════════════════════════════════════════════════════

# Save training config
config = {
    'USE_SPINEWEB': USE_SPINEWEB,
    'num_epochs': num_epochs,
    'batch_size': BATCH_SIZE,
    'patch_size': PATCH_SIZE,
    'lrG': lrG,
    'lrD': lrD,
    'lambda_adv': lambda_adv,
    'lambda_FM': lambda_FM,
    'lambda_rec': lambda_rec,
    'lambda_edge': lambda_edge,
    'lambda_phys': lambda_phys,
    'lambda_metal': lambda_metal,
}
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Plot loss curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(G_losses, label='Generator', alpha=0.7)
ax.plot(D_losses, label='Discriminator', alpha=0.7)
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curves')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(run_dir, 'loss_curves.png'), dpi=150)
plt.close(fig)

# Close TensorBoard
writer.close()

print("\n" + "═" * 60)
print("TRAINING COMPLETE")
print(f"Results saved to: {run_dir}")
print("═" * 60)
