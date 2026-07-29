#!/usr/bin/env python3
"""
finetune_8level_weight_tanhgrad.py
====================================
Fine-tunes ResNet20 with:
  - 8-level input quantization
  - 8-level activation quantization (stacked tanh gradient)
  - 8-level weight quantization    (stacked tanh gradient)
  - LR starts at 1e-3, halved every 25 epochs (step decay)
  - Fixed beta for both activations and weights (no annealing)

Loading from:
  8level_act_tanhgrad_best.pth  (85.65%)

Run:
    conda activate qnn
    unset LD_LIBRARY_PATH
    cd ~/pytorch_resnet_cifar10
    python finetune_8level_weight_tanhgrad.py
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
from resnet import Activation8Level, QConv2d

# ============================================================
# CONFIG — change only here
# ============================================================
ACT_CLIP   = 1.0    # activation levels: linspace(-1, +1, 8)
W_CLIP     = 1.0    # weight levels:     linspace(-1, +1, 8)
ACT_BETA   = 5.0    # sharpness for activation backward
W_BETA     = 5.0    # sharpness for weight backward
EPOCHS     = 100
LR         = 1e-3   # starting LR — halved every 25 epochs
LR_STEP    = 25     # halve LR every this many epochs
LR_GAMMA   = 0.5    # multiply LR by this at each step
BATCH_SIZE = 128
WORKERS    = 2

LOAD_PATH  = './8level_act_tanhgrad_best.pth'
SAVE_PATH  = './8level_weight_tanhgrad_best.pth'
PLOT_PATH  = './8level_weight_tanhgrad_lr1e3_stepdecay_plot.png'
# ============================================================

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device      : {device}")
print(f"Act clip    : [-{ACT_CLIP}, +{ACT_CLIP}]")
print(f"Weight clip : [-{W_CLIP}, +{W_CLIP}]")
print(f"LR schedule : {LR} halved every {LR_STEP} epochs")
print(f"  Epoch  1-25 : LR = {LR}")
print(f"  Epoch 26-50 : LR = {LR*0.5}")
print(f"  Epoch 51-75 : LR = {LR*0.25}")
print(f"  Epoch 76-100: LR = {LR*0.125}")
print(f"ACT_BETA    : {ACT_BETA} (fixed)")
print(f"W_BETA      : {W_BETA} (fixed)")

# ── Data ──────────────────────────────────────────────────────
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

# ── Model ──────────────────────────────────────────────────────
model = resnet.resnet20(w_clip=W_CLIP).to(device)

ckpt       = torch.load(LOAD_PATH, map_location=device)
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)

# Set activation levels on all quantizers
new_act_levels = torch.linspace(-ACT_CLIP, ACT_CLIP, 8).to(device)
model.act1.levels.copy_(new_act_levels)
for layer in [model.layer1, model.layer2, model.layer3]:
    for block in layer:
        block.act1.levels.copy_(new_act_levels)
        block.act2.levels.copy_(new_act_levels)

# Set betas
model.set_act_beta(ACT_BETA)
model.set_weight_beta(W_BETA)

# Switch weight quantization ON
model.set_weight_quantization(True)

print(f"\nLoaded checkpoint  : {LOAD_PATH}")
print(f"Missing keys       : {missing}")
print(f"Unexpected keys    : {unexpected}")
print(f"\nQuantization setup:")
print(f"  Input      : 8 levels {model.input_quantizer.levels.tolist()}")
print(f"  Activations: 8 levels {model.act1.levels.tolist()},  beta={ACT_BETA}")
print(f"  Weights    : 8 levels {model.conv1.levels.tolist()},  beta={W_BETA}")
print(f"  Weight quantization : ON")

# ── Training ───────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

train_accs   = []
val_accs     = []
train_losses = []
val_losses   = []
best_acc     = 0.0

print(f"\nStarting fine-tuning for {EPOCHS} epochs ...\n")

for epoch in range(1, EPOCHS + 1):

    current_lr = optimizer.param_groups[0]['lr']

    # ── Train ──
    model.train()
    correct = total = 0
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    train_acc  = 100.0 * correct / total
    train_loss = total_loss / total

    # ── Validate ──
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += outputs.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
    val_acc  = 100.0 * correct / total
    val_loss = total_loss / total

    scheduler.step()

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'acc': best_acc,
                    'act_clip': ACT_CLIP,
                    'w_clip': W_CLIP,
                    'act_beta': ACT_BETA,
                    'w_beta': W_BETA}, SAVE_PATH)

    print(f"Epoch {epoch:3d}/{EPOCHS} | LR {current_lr:.2e} | "
          f"Train {train_acc:.2f}% loss {train_loss:.4f} | "
          f"Val {val_acc:.2f}% loss {val_loss:.4f} | "
          f"Best {best_acc:.2f}%")

# ── Plot ───────────────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)
lr_drops = [25, 50, 75]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(epochs_range, train_accs, label='Train accuracy', color='steelblue', linewidth=1.5)
ax1.plot(epochs_range, val_accs,   label='Val accuracy',   color='tomato',    linewidth=1.5)
for step in lr_drops:
    ax1.axvline(x=step, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax1.axhline(y=85.65, color='orange', linestyle='--', linewidth=1, label='Act TanhGrad, float weights (85.65%)')
ax1.axhline(y=86.01, color='brown',  linestyle='--', linewidth=1, label='Act STE, float weights (86.01%)')
ax1.axhline(y=89.02, color='green',  linestyle='--', linewidth=1, label='Input only (89.02%)')
ax1.axhline(y=91.47, color='purple', linestyle='--', linewidth=1, label='Float baseline (91.47%)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title(f'Accuracy | W_BETA={W_BETA} | Best: {best_acc:.2f}% (red = LR halved)')
ax1.legend(fontsize=7)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(epochs_range, train_losses, label='Train loss', color='steelblue', linewidth=1.5)
ax2.plot(epochs_range, val_losses,   label='Val loss',   color='tomato',    linewidth=1.5)
for step in lr_drops:
    ax2.axvline(x=step, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss (red lines = LR halved)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(PLOT_PATH, dpi=120, bbox_inches='tight')
plt.close(fig)

print(f"\nPlot saved : {PLOT_PATH}")
print(f"\n{'='*55}")
print(f"RESULT SUMMARY")
print(f"{'='*55}")
print(f"  Float baseline                          : 91.47%")
print(f"  + 8-level input                         : 89.02%")
print(f"  + activations STE,      weights float   : 86.01%")
print(f"  + activations TanhGrad, weights float   : 85.65%")
print(f"  + activations TanhGrad, weights TanhGrad: {best_acc:.2f}%")
print(f"  Checkpoint saved : {SAVE_PATH}")
print(f"  Plot saved       : {PLOT_PATH}")
print(f"{'='*55}")