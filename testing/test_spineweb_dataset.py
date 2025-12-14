"""
Test script to verify SpineWeb dataset implementation
Run this before starting fine-tuning to ensure data is loaded correctly
"""

import torch
import numpy as np
from data.datasets import SpineWebTrainDataset, SpineWebTestDataset
from torch.utils.data import DataLoader

print("═══════════════════════════════════════════════════════════════")
print("TESTING SPINEWEB DATASET IMPLEMENTATION")
print("═══════════════════════════════════════════════════════════════")

# Test parameters
artifact_dir = "/home/Drive-D/UWSpine_adn/train/metal_artifact/"
clean_dir = "/home/Drive-D/UWSpine_adn/train/no_metal/"
PATCH_SIZE = 64
BATCH_SIZE = 4

print("\n1. Creating SpineWeb Training Dataset...")
try:
    train_dataset = SpineWebTrainDataset(
        artifact_dir=artifact_dir,
        clean_dir=clean_dir,
        patchSize=PATCH_SIZE,
        paired=True,
        hu_range=(-1000, 2000)
    )
    print(f"✅ SUCCESS: Dataset created with {len(train_dataset)} samples")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

print("\n2. Testing single sample loading...")
try:
    sample = train_dataset[0]
    print(f"✅ SUCCESS: Sample loaded")
    print(f"   - Input shape: {sample[0].shape}")
    print(f"   - Target shape: {sample[1].shape}")
    print(f"   - Original shape: {sample[2].shape}")
    print(f"   - Input range: [{sample[0].min():.3f}, {sample[0].max():.3f}]")
    print(f"   - Target range: [{sample[1].min():.3f}, {sample[1].max():.3f}]")
    
    # Verify shapes and ranges
    assert sample[0].shape == torch.Size([1, 64, 64]), f"Wrong input shape: {sample[0].shape}"
    assert sample[1].shape == torch.Size([1, 64, 64]), f"Wrong target shape: {sample[1].shape}"
    assert -1.1 <= sample[0].min() <= -0.9, f"Input min out of range: {sample[0].min()}"
    assert 0.9 <= sample[0].max() <= 1.1, f"Input max out of range: {sample[0].max()}"
    print("✅ All validations passed!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

print("\n3. Creating DataLoader...")
try:
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    print(f"✅ SUCCESS: DataLoader created")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

print("\n4. Testing batch loading...")
try:
    batch = next(iter(dataloader))
    print(f"✅ SUCCESS: Batch loaded")
    print(f"   - Batch input shape: {batch[0].shape}")
    print(f"   - Batch target shape: {batch[1].shape}")
    print(f"   - Batch input range: [{batch[0].min():.3f}, {batch[0].max():.3f}]")
    
    # Verify batch shapes
    assert batch[0].shape == torch.Size([BATCH_SIZE, 1, 64, 64]), f"Wrong batch shape: {batch[0].shape}"
    print("✅ Batch validation passed!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

print("\n5. Testing multiple batches...")
try:
    for i, data in enumerate(dataloader):
        if i >= 2:  # Test 3 batches
            break
        print(f"   - Batch {i+1}: Input {data[0].shape}, Target {data[1].shape}")
    print("✅ SUCCESS: Multiple batches loaded correctly")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

print("\n═══════════════════════════════════════════════════════════════")
print("✅ ALL TESTS PASSED!")
print("═══════════════════════════════════════════════════════════════")
print("\nSpineWeb dataset is ready for fine-tuning!")
print("\nTo start fine-tuning:")
print("1. Open model.py")
print("2. Set USE_SPINEWEB = True (around line 35)")
print("3. Run: python model.py")
print("\nExpected behavior:")
print("- Learning rate: 0.00001 (lower for fine-tuning)")
print("- Epochs: 20 (fewer for fine-tuning)")
print("- Checkpoint: bestcheckpoint.pth will be loaded")
print("- Saves: finetuned_spineweb_epoch_N.pth")
