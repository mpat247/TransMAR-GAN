#!/usr/bin/env python3
"""
Model Variant Training Launcher for TransMAR-GAN
Supports training 7 different model configurations for ablation studies
"""
import subprocess
import sys

VARIANTS = {
    "baseline": {
        "description": "NGswin Generator + DCGAN Discriminator + MSE Loss Only",
        "script": "training/train_baseline_variants.py",
        "args": "--loss_mode mse"
    },
    "v1": {
        "description": "Baseline + Adversarial Loss",
        "script": "training/train_baseline_variants.py",
        "args": "--loss_mode adversarial"
    },
    "v2": {
        "description": "NGswin + Multi-Scale PatchGAN + Adversarial",
        "script": "training/train_combined.py",
        "args": "--losses adversarial"
    },
    "v3": {
        "description": "V2 + Feature Matching Loss",
        "script": "training/train_combined.py",
        "args": "--losses adversarial feature_matching"
    },
    "v4": {
        "description": "V3 + Metal-Aware Reconstruction (Eq 3)",
        "script": "training/train_combined.py",
        "args": "--losses adversarial feature_matching mar"
    },
    "v5": {
        "description": "V4 + Metal-Aware Edge Loss (Eq 4)",
        "script": "training/train_combined.py",
        "args": "--losses adversarial feature_matching mar edge"
    },
    "full": {
        "description": "Complete Model: All Losses (Adversarial + FM + MAR + Edge + Physics + Metal Consistency)",
        "script": "training/train_combined.py",
        "args": "--losses adversarial feature_matching mar edge physics metal_consistency"
    }
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_model_variants.py <variant>")
        print("\nAvailable variants:")
        for k, v in VARIANTS.items():
            print(f"  {k}: {v['description']}")
        sys.exit(1)
    
    variant = sys.argv[1]
    if variant not in VARIANTS:
        print(f"Error: Unknown variant \"{variant}\"")
        sys.exit(1)
    
    config = VARIANTS[variant]
    print(f"Training {variant}: {config['description']}")
    cmd = f"python {config['script']} {config['args']}"
    subprocess.run(cmd, shell=True)
