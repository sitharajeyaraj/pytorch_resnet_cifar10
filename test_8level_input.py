#!/usr/bin/env python3
"""
test_8level_input.py — Test the Input8Level quantizer.

Run on your server:
    conda activate qnn
    unset LD_LIBRARY_PATH
    cd ~/pytorch_resnet_cifar10
    python test_8level_input.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ── The quantizer class ──────────────────────────────────────
class Input8Level(nn.Module):
    def __init__(self):
        super().__init__()
        levels = torch.linspace(-2.75, 2.75, 8)
        self.register_buffer('levels', levels)

    def forward(self, x):
        dists = (x.unsqueeze(-1) - self.levels).abs()  # [B, 3, 32, 32, 8]
        idx   = dists.argmin(dim=-1)                   # [B, 3, 32, 32]
        return self.levels[idx]                        # [B, 3, 32, 32]


# ── Load one batch of normalized CIFAR-10 ───────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

dataset = datasets.CIFAR10(
    root='./data', train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
)
loader = DataLoader(dataset, batch_size=512, shuffle=False)
images, _ = next(iter(loader))

# ── Run through quantizer ────────────────────────────────────
quantizer = Input8Level()
snapped   = quantizer(images)

# ── Print results ────────────────────────────────────────────
print("=" * 50)
print("8-LEVEL QUANTIZER TEST")
print("=" * 50)

print(f"\nThe 8 levels:")
for i, lv in enumerate(quantizer.levels.tolist()):
    print(f"  Level {i+1}: {lv:+.4f}")

print(f"\n--- Before quantization ---")
print(f"  Min  : {images.min():.4f}")
print(f"  Max  : {images.max():.4f}")
print(f"  Mean : {images.mean():.4f}")
print(f"  Unique values : {images.numel()} (every pixel is different)")

print(f"\n--- After quantization ---")
print(f"  Min  : {snapped.min():.4f}")
print(f"  Max  : {snapped.max():.4f}")
print(f"  Mean : {snapped.mean():.4f}")
print(f"  Unique values : {snapped.unique().numel()} (should be exactly 8)")
print(f"  Values used   : {[f'{v:+.4f}' for v in snapped.unique().tolist()]}")

print(f"\n--- How many pixels snapped to each level? ---")
for i, lv in enumerate(quantizer.levels.tolist()):
    count = (snapped == lv).sum().item()
    pct   = 100.0 * count / snapped.numel()
    bar   = "█" * int(pct / 2)
    print(f"  Level {i+1} ({lv:+.4f}): {count:7d} pixels  ({pct:5.1f}%)  {bar}")

print(f"\n--- First image, Red channel: before vs after ---")
print(f"  {'Before':>8}   {'After':>8}   {'Difference':>10}")
print(f"  {'------':>8}   {'-----':>8}   {'----------':>10}")
for r in range(5):
    for c in range(5):
        before = images[0, 0, r, c].item()
        after  = snapped[0, 0, r, c].item()
        diff   = abs(before - after)
        print(f"  {before:>8.4f}   {after:>8.4f}   {diff:>10.4f}")

print(f"\n--- Quantization error stats ---")
error = (images - snapped).abs()
print(f"  Mean error : {error.mean():.4f}")
print(f"  Max error  : {error.max():.4f}")
print(f"  (max error should be ≤ half step = {(2.75*2/7/2):.4f})")

print("\n" + "=" * 50)
print("If unique values = 8, the quantizer is working correctly.")
print("=" * 50)