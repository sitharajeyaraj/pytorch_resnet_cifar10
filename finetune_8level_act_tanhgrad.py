#!/usr/bin/env python3
"""
finetune_8level_act_tanhgrad.py
================================
Fine-tunes ResNet20 with:
  - 8-level input quantization
  - 8-level activation quantization (stacked tanh gradient, NOT STE)
  - Float weights
  - Fixed LR (no scheduler)
  - Beta annealing: beta grows from 1.0 to 20.0 over training
    beta=1  → smooth staircase, gradients flow freely (early training)
    beta=20 → sharp staircase, nearly identical to hard snap (late training)

Comparison target:
    8level-act (STE)        : 86.01%  (clip=1.0)
    8level-act-tanhgrad     : ??? %   (this script)

Run:
    conda activate qnn
    unset LD_LIBRARY_PATH
    cd ~/pytorch_resnet_cifar10
    python finetune_8level_act_tanhgrad.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import resnet
from resnet import Activation8Level

# ============================================================
# Config
# CLIP=1.0 was the best result with STE — keep same levels
# so the comparison is fair (only the backward pass changes)
CLIP = 1.0

# Beta annealing schedule
# beta starts at BETA_START (smooth) and grows to BETA_END (sharp)
# after BETA_END_EPOCH it stays fixed at BETA_END
BETA_START    = 1.0
BETA_END      = 20.0
BETA_END_EPOCH = 80     # reach full sharpness by epoch 80 out of 100
# ============================================================

# ── Paths ────────────────────────────────────────────────────
LOAD_PATH = './8level_act_clip1_best.pth'       # best checkpoint from STE run
SAVE_PATH = './8level_act_tanhgrad_best.pth'
PLOT_PATH = './8level_act_tanhgrad_plot.png'

EPOCHS     = 100
LR         = 1e-3
BATCH_SIZE = 128
WORKERS    = 2

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device      : {device}")
print(f"Clip range  : [-{CLIP}, +{CLIP}]")
print(f"LR          : {LR} (fixed, no scheduler)")
print(f"Beta        : {BETA_START} → {BETA_END} over {BETA_END_EPOCH} epochs")

# ── Data ─────────────────────────────────────────────────────
train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True, download=False,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True
)
test_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])),
    batch_size=256, shuffle=False, num_workers=WORKERS, pin_memory=True
)

# ── Model ─────────────────────────────────────────────────────
model = resnet.resnet20().to(device)
ckpt       = torch.load(LOAD_PATH, map_location=device)
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)

# Set activation levels from CLIP — updates all 19 quantizers
# Same levels as the STE run so comparison is fair
new_levels = torch.linspace(-CLIP, CLIP, 8).to(device)
model.act1.levels.copy_(new_levels)
for layer in [model.layer1, model.layer2, model.layer3]:
    for block in layer:
        block.act1.levels.copy_(new_levels)
        block.act2.levels.copy_(new_levels)

print(f"\nLoaded checkpoint  : {LOAD_PATH}")
print(f"Missing keys       : {missing}")
print(f"Unexpected keys    : {unexpected}")
print(f"\nQuantization setup:")
print(f"  Input      : 8 levels {model.input_quantizer.levels.tolist()}")
print(f"  Activations: 8 levels {model.act1.levels.tolist()}")
print(f"  Backward   : stacked tanh gradient (NOT STE)")
print(f"  Weights    : float (not yet quantized)")


def set_beta(model, beta):
    """Update beta on every Activation8Level module in the model."""
    for m in model.modules():
        if isinstance(m, Activation8Level):
            m.beta = beta


# ── Training ──────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

train_accs = []
val_accs   = []
betas      = []
best_acc   = 0.0

print(f"\nStarting fine-tuning for {EPOCHS} epochs ...\n")

for epoch in range(1, EPOCHS + 1):

    # ── Beta annealing ────────────────────────────────────────
    # Linear ramp from BETA_START to BETA_END over BETA_END_EPOCH epochs
    # After that, stays fixed at BETA_END
    progress = min((epoch - 1) / (BETA_END_EPOCH - 1), 1.0)
    beta     = BETA_START + progress * (BETA_END - BETA_START)
    set_beta(model, beta)
    betas.append(beta)
    # ─────────────────────────────────────────────────────────

    model.train()
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    train_acc = 100.0 * correct / total

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total   += labels.size(0)
    val_acc = 100.0 * correct / total

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'acc': best_acc,
                    'clip': CLIP,
                    'beta': beta}, SAVE_PATH)

    print(f"Epoch {epoch:3d}/{EPOCHS}  "
          f"Beta: {beta:5.1f}  "
          f"Train: {train_acc:.2f}%  "
          f"Val: {val_acc:.2f}%  "
          f"Best: {best_acc:.2f}%")

# ── Plot ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
epochs_range = range(1, EPOCHS + 1)

# Top: accuracy curves
ax1.plot(epochs_range, train_accs, label='Train accuracy', color='steelblue', linewidth=1.5)
ax1.plot(epochs_range, val_accs,   label='Val accuracy',   color='tomato',    linewidth=1.5)
ax1.axhline(y=86.01, color='orange', linestyle='--', linewidth=1, label='STE baseline (86.01%)')
ax1.axhline(y=89.02, color='green',  linestyle='--', linewidth=1, label='Input only (89.02%)')
ax1.axhline(y=91.47, color='purple', linestyle='--', linewidth=1, label='Float baseline (91.47%)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title(f'Stacked tanh gradient  |  clip=[-{CLIP},+{CLIP}]  |  Best val: {best_acc:.2f}%')
ax1.legend()
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Bottom: beta annealing curve
ax2.plot(epochs_range, betas, color='darkorchid', linewidth=1.5, label='beta')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Beta')
ax2.set_title('Beta annealing schedule')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(PLOT_PATH, dpi=120, bbox_inches='tight')
plt.close(fig)

print(f"\nPlot saved : {PLOT_PATH}")
print(f"\n{'='*50}")
print(f"RESULT SUMMARY")
print(f"{'='*50}")
print(f"  Float baseline              : 91.47%")
print(f"  + 8-level input             : 89.02%")
print(f"  + activations STE (clip=1)  : 86.01%")
print(f"  + activations TanhGrad      : {best_acc:.2f}%")
print(f"  Checkpoint saved            : {SAVE_PATH}")
print(f"  Plot saved                  : {PLOT_PATH}")
print(f"{'='*50}")