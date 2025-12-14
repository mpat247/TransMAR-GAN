import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from data.datasets import SpineWebTestDataset
from models.generator.ngswin import NGswin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    """Wrapper to match the Generator used in model.py (NGswin-based)."""

    def __init__(self, ngpu: int = 1):
        super().__init__()
        self.ngpu = ngpu
        self.main = NGswin()

    def forward(self, x):  # x: [B, 1, H, W]
        return self.main(x)


def load_generator(checkpoint_path: str, ngpu: int = 1) -> nn.Module:
    """Load Generator weights from a fine-tuned checkpoint."""
    netG = Generator(ngpu).to(device)
    if device.type == "cuda" and ngpu > 1:
        netG = nn.DataParallel(netG, list(range(ngpu)))

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("netG_state_dict", ckpt)
    netG.load_state_dict(state_dict)
    netG.eval()
    print(f"[Load] Loaded Generator from {checkpoint_path}")
    return netG


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    # Inputs are in [-1,1]; dynamic range is 2*max_val
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10((2 * max_val) ** 2 / (mse + 1e-8))


def denorm_to_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] tensor [1,H,W] or [B,1,H,W] to HxW or HxWx3 numpy in [0,1]."""
    if img.dim() == 3:  # [C,H,W]
        t = img
    elif img.dim() == 4:  # [B,C,H,W] -> take first
        t = img[0]
    else:
        raise ValueError(f"Unexpected tensor shape for image: {img.shape}")

    t = (t + 1.0) / 2.0  # [-1,1] -> [0,1]
    t = t.clamp(0.0, 1.0).cpu().numpy()
    if t.shape[0] == 1:
        return t[0]
    else:
        # CxHxW -> HxWxC
        return np.transpose(t, (1, 2, 0))


def save_triplet_figure(artifact, pred, clean, out_path: str, title: str = "") -> None:
    """Save a 3-row (artifact, pred, clean) figure for one slice."""
    art_np = denorm_to_numpy(artifact)
    pred_np = denorm_to_numpy(pred)
    clean_np = denorm_to_numpy(clean)

    fig, axes = plt.subplots(3, 1, figsize=(4, 8))
    if title:
        fig.suptitle(title)

    axes[0].imshow(art_np, cmap="gray")
    axes[0].set_title("Artifact")
    axes[0].axis("off")

    axes[1].imshow(pred_np, cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(clean_np, cmap="gray")
    axes[2].set_title("Clean")
    axes[2].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_full_slice_eval(netG: nn.Module,
                        test_dataset: SpineWebTestDataset,
                        out_dir: str) -> dict:
    """Run full-slice evaluation: feed entire slice through netG and save triplets."""
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    full_dir = os.path.join(out_dir, "full_slice")
    os.makedirs(full_dir, exist_ok=True)

    mse_list, psnr_list = [], []

    print("[Eval] Full-slice evaluation...")
    for idx, (artifact, clean, _) in enumerate(loader):
        artifact = artifact.to(device)  # [1,1,H,W]
        clean = clean.to(device)

        with torch.no_grad():
            pred = netG(artifact)

        mse = F.mse_loss(pred, clean, reduction="mean").item()
        mse_list.append(mse)
        psnr = psnr_from_mse(mse)
        psnr_list.append(psnr)

        if idx < 10:
            title = f"Slice {idx:03d} | MSE={mse:.6f}, PSNR={psnr:.2f} dB"
            out_path = os.path.join(full_dir, f"slice_{idx:03d}.png")
            save_triplet_figure(artifact[0], pred[0], clean[0], out_path, title=title)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(loader)} slices...")

    mean_mse = float(np.mean(mse_list)) if mse_list else None
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else None

    metrics = {
        "mean_mse": mean_mse,
        "mean_psnr": mean_psnr,
        "num_slices": len(mse_list),
    }

    with open(os.path.join(full_dir, "metrics_full.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Eval] Full-slice done. mean MSE={mean_mse:.6f}, mean PSNR={mean_psnr:.2f} dB")
    return metrics


def run_tiled_eval(netG: nn.Module,
                   test_dataset: SpineWebTestDataset,
                   out_dir: str,
                   tile_size: int = 64,
                   stride: int = 32) -> dict:
    """Patch-wise tiled evaluation to reconstruct full slices from patch predictions."""
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    tiled_dir = os.path.join(out_dir, "tiled_slice")
    grid_dir = os.path.join(out_dir, "patch_grids")
    os.makedirs(tiled_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    mse_list, psnr_list = [], []

    print("[Eval] Tiled (patch-wise) evaluation...")
    for idx, (artifact, clean, _) in enumerate(loader):
        artifact = artifact.to(device)  # [1,1,H,W]
        clean = clean.to(device)

        _, _, H, W = artifact.shape
        out_acc = torch.zeros_like(artifact)
        weight = torch.zeros_like(artifact)

        # Sliding window over the slice
        for r in range(0, H - tile_size + 1, stride):
            for c in range(0, W - tile_size + 1, stride):
                patch = artifact[:, :, r:r + tile_size, c:c + tile_size]
                with torch.no_grad():
                    pred_patch = netG(patch)
                out_acc[:, :, r:r + tile_size, c:c + tile_size] += pred_patch
                weight[:, :, r:r + tile_size, c:c + tile_size] += 1.0

        # Avoid division by zero
        mask = weight > 0
        pred = torch.zeros_like(out_acc)
        pred[mask] = out_acc[mask] / weight[mask]

        mse = F.mse_loss(pred, clean, reduction="mean").item()
        mse_list.append(mse)
        psnr = psnr_from_mse(mse)
        psnr_list.append(psnr)

        if idx < 10:
            # Save tiled triplet
            title = f"Slice {idx:03d} (tiled) | MSE={mse:.6f}, PSNR={psnr:.2f} dB"
            out_path = os.path.join(tiled_dir, f"slice_{idx:03d}.png")
            save_triplet_figure(artifact[0], pred[0], clean[0], out_path, title=title)

            # Also save a patch grid for this slice (artifact / pred / clean)
            art_den = (artifact + 1.0) / 2.0
            pred_den = (pred + 1.0) / 2.0
            clean_den = (clean + 1.0) / 2.0

            # Sample a few non-overlapping patches for visualization
            patches = []
            for r in range(0, H - tile_size + 1, tile_size):
                for c in range(0, W - tile_size + 1, tile_size):
                    patches.append(art_den[:, :, r:r + tile_size, c:c + tile_size])
                    patches.append(pred_den[:, :, r:r + tile_size, c:c + tile_size])
                    patches.append(clean_den[:, :, r:r + tile_size, c:c + tile_size])
            if patches:
                patches_tensor = torch.cat(patches, dim=0)  # [3*N,1,ts,ts]
                grid = vutils.make_grid(patches_tensor, nrow=3, padding=2, normalize=True)
                grid_path = os.path.join(grid_dir, f"slice_{idx:03d}_patchgrid.png")
                vutils.save_image(grid, grid_path)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(loader)} slices...")

    mean_mse = float(np.mean(mse_list)) if mse_list else None
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else None

    metrics = {
        "mean_mse": mean_mse,
        "mean_psnr": mean_psnr,
        "num_slices": len(mse_list),
        "tile_size": tile_size,
        "stride": stride,
    }

    with open(os.path.join(tiled_dir, "metrics_tiled.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Eval] Tiled done. mean MSE={mean_mse:.6f}, mean PSNR={mean_psnr:.2f} dB")
    return metrics


def main():
    # Hard-code defaults matching your current training setup; can be changed if needed.
    checkpoint_dir = "./training_checkpoints"
    # By default, use the latest fine-tuned checkpoint if present
    default_ckpt = None
    if os.path.isdir(checkpoint_dir):
        # Pick the most recent finetuned_spineweb checkpoint if available
        candidates = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.startswith("finetuned_spineweb_epoch_") and f.endswith(".pth")
        ]
        if candidates:
            default_ckpt = sorted(candidates)[-1]

    if default_ckpt is None:
        default_ckpt = os.path.join(checkpoint_dir, "bestcheckpoint.pth")

    checkpoint_path = default_ckpt

    artifact_dir = "/home/Drive-D/UWSpine_adn/test/synthesized_metal_transfer/"
    clean_dir = "/home/Drive-D/UWSpine_adn/test/no_metal/"

    if not (os.path.isdir(artifact_dir) and os.path.isdir(clean_dir)):
        raise RuntimeError(
            f"SpineWeb test folders not found. Expected:\n  {artifact_dir}\n  {clean_dir}"
        )

    timestamp = datetime.now().strftime("test_%Y%m%d_%H%M%S")
    out_root = "./finetune_results"
    out_dir = os.path.join(out_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print("═══════════════════════════════════════════════════════════════")
    print("Test fine-tuned NGswin Generator on SpineWeb test set")
    print("Checkpoint:", checkpoint_path)
    print("Artifact dir:", artifact_dir)
    print("Clean dir   :", clean_dir)
    print("Results dir :", out_dir)
    print("═══════════════════════════════════════════════════════════════")

    # Load model and dataset
    netG = load_generator(checkpoint_path, ngpu=1)
    test_dataset = SpineWebTestDataset(artifact_dir=artifact_dir, clean_dir=clean_dir)

    # Run full-slice evaluation
    metrics_full = run_full_slice_eval(netG, test_dataset, out_dir)

    # Run tiled (patch-wise) evaluation + patch grids
    metrics_tiled = run_tiled_eval(netG, test_dataset, out_dir, tile_size=64, stride=32)

    # Save summary
    summary = {
        "checkpoint": checkpoint_path,
        "artifact_dir": artifact_dir,
        "clean_dir": clean_dir,
        "metrics_full": metrics_full,
        "metrics_tiled": metrics_tiled,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[Done] Testing complete. Summary written to:", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
