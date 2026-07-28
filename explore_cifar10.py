#!/usr/bin/env python3
"""
explore_cifar10.py — Run this on your server:
    conda activate qnn
    unset LD_LIBRARY_PATH
    cd ~/pytorch_resnet_cifar10
    python explore_cifar10.py
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Step 1: Raw pixels ──────────────────────────────────────
print("=" * 50)
print("STEP 1: RAW PIXELS (before normalization)")
print("=" * 50)

raw_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)
raw_loader = DataLoader(raw_dataset, batch_size=512, shuffle=False)
raw_images, _ = next(iter(raw_loader))

print(f"\nBatch shape : {raw_images.shape}  [batch, channels, H, W]")
print(f"Min  : {raw_images.min():.4f}")
print(f"Max  : {raw_images.max():.4f}")
print(f"Mean : {raw_images.mean():.4f}")
print(f"Std  : {raw_images.std():.4f}")

print("\nPer-channel stats:")
for i, ch in enumerate(['Red', 'Green', 'Blue']):
    c = raw_images[:, i]
    print(f"  {ch:5s}: min={c.min():.4f}  max={c.max():.4f}  "
          f"mean={c.mean():.4f}  std={c.std():.4f}")

print("\nFirst image, Red channel, first 5x5 pixels:")
for row in raw_images[0, 0, :5, :5]:
    print("  " + "  ".join(f"{v:.3f}" for v in row.tolist()))

# ── Step 2: Normalized pixels ────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: AFTER NORMALIZATION")
print("=" * 50)

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

norm_dataset = datasets.CIFAR10(
    root='./data', train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
)
norm_loader = DataLoader(norm_dataset, batch_size=512, shuffle=False)
norm_images, _ = next(iter(norm_loader))

print(f"\nBatch shape : {norm_images.shape}")
print(f"Min  : {norm_images.min():.4f}")
print(f"Max  : {norm_images.max():.4f}")
print(f"Mean : {norm_images.mean():.4f}")
print(f"Std  : {norm_images.std():.4f}")

print("\nPer-channel stats:")
for i, ch in enumerate(['Red', 'Green', 'Blue']):
    c = norm_images[:, i]
    print(f"  {ch:5s}: min={c.min():.4f}  max={c.max():.4f}  "
          f"mean={c.mean():.4f}  std={c.std():.4f}")

print("\nFirst image, Red channel, first 5x5 pixels:")
for row in norm_images[0, 0, :5, :5]:
    print("  " + "  ".join(f"{v:+.3f}" for v in row.tolist()))