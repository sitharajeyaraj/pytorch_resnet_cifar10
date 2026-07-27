#!/usr/bin/env python3
"""
sweep_activation_levels.py
===========================
Runs activation quantization fine-tuning for multiple clip ranges.
For each case produces:
  - Individual train/val accuracy plot
  - Combined plot comparing all cases
  - Saved checkpoint

Run:
    conda activate qnn
    unset LD_LIBRARY_PATH
    cd ~/pytorch_resnet_cifar10
    python sweep_activation_levels.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')   # no display needed — saves to file
import matplotlib.pyplot as plt
import os

# ── Cases to sweep ────────────────────────────────────────────
# Each case is (clip_value, label)
# levels = linspace(-clip, +clip, 8)
CASES = [
    (1.0, "[-1, +1]"),
    (2.0, "[-2, +2]"),
    (3.0, "[-3, +3]"),
    (4.0, "[-4, +4]"),
    (6.0, "[-6, +6]"),
]

# ── Config ────────────────────────────────────────────────────
LOAD_PATH  = './8level_input_best.pth'
EPOCHS     = 100
LR         = 1e-3
BATCH_SIZE = 128
WORKERS    = 2
SAVE_DIR   = './activation_sweep'
os.makedirs(SAVE_DIR, exist_ok=True)

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {device}")
print(f"Cases  : {[c[1] for c in CASES]}")
print(f"Epochs : {EPOCHS} per case")
print(f"Output : {SAVE_DIR}/\n")

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

# ── Helper: build model with given clip value ─────────────────
def build_model(clip):
    """Dynamically patch the activation levels before building."""
    import resnet
    import importlib
    # Patch the clip value in the module
    original_linspace = torch.linspace

    def patched_linspace(start, end, steps, **kwargs):
        # Only intercept the activation level linspace calls
        if start == -2.0 and end == 2.0 and steps == 8:
            return original_linspace(-clip, clip, steps, **kwargs)
        return original_linspace(start, end, steps, **kwargs)

    torch.linspace = patched_linspace
    importlib.reload(resnet)
    torch.linspace = original_linspace

    model = resnet.resnet20().to(device)

    # Load pretrained input-quantized weights
    ckpt       = torch.load(LOAD_PATH, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model, resnet


# ── Helper: train one epoch ───────────────────────────────────
def train_epoch(model, optimizer, criterion):
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
    return 100.0 * correct / total


# ── Helper: evaluate ─────────────────────────────────────────
@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ── Helper: plot individual case ─────────────────────────────
def plot_individual(train_accs, val_accs, label, clip, best_acc):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs  = range(1, len(train_accs) + 1)
    ax.plot(epochs, train_accs, label='Train accuracy', color='steelblue', linewidth=1.5)
    ax.plot(epochs, val_accs,   label='Val accuracy',   color='tomato',    linewidth=1.5)
    ax.axhline(y=89.02, color='green',  linestyle='--', linewidth=1, label='Input only (89.02%)')
    ax.axhline(y=91.47, color='purple', linestyle='--', linewidth=1, label='Float baseline (91.47%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Activation levels {label}  |  Best val: {best_acc:.2f}%')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fname = f'{SAVE_DIR}/case_clip{clip:.0f}.png'
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved individual plot: {fname}")


# ── Main sweep ────────────────────────────────────────────────
all_results = {}   # clip → (train_accs, val_accs, best_acc)

for clip, label in CASES:
    print("=" * 55)
    print(f"CASE: levels = linspace(-{clip}, +{clip}, 8)")
    levels_list = torch.linspace(-clip, clip, 8).tolist()
    print(f"  Levels : {[f'{v:+.3f}' for v in levels_list]}")
    print(f"  Step   : {levels_list[1]-levels_list[0]:.4f}")
    print("=" * 55)

    model, resnet_mod = build_model(clip)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)

    train_accs = []
    val_accs   = []
    best_acc   = 0.0
    save_path  = f'{SAVE_DIR}/best_clip{clip:.0f}.pth'

    for epoch in range(1, EPOCHS + 1):
        tr  = train_epoch(model, optimizer, criterion)
        val = evaluate(model)
        train_accs.append(tr)
        val_accs.append(val)
        scheduler.step()

        if val > best_acc:
            best_acc = val
            torch.save({'state_dict': model.state_dict(),
                        'clip': clip, 'acc': best_acc}, save_path)

        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"Train: {tr:.2f}%  Val: {val:.2f}%  Best: {best_acc:.2f}%")

    all_results[clip] = (train_accs, val_accs, best_acc, label)
    plot_individual(train_accs, val_accs, label, clip, best_acc)
    print(f"  Best accuracy for {label}: {best_acc:.2f}%\n")

# ── Combined plot ─────────────────────────────────────────────
print("Generating combined plot ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['royalblue', 'tomato', 'green', 'orange', 'purple']
epochs = range(1, EPOCHS + 1)

# Left: validation accuracy only (clean comparison)
ax = axes[0]
for (clip, (train_accs, val_accs, best_acc, label)), color in zip(all_results.items(), colors):
    ax.plot(epochs, val_accs, label=f'{label}  best={best_acc:.2f}%',
            color=color, linewidth=1.5)
ax.axhline(y=89.02, color='black', linestyle='--', linewidth=1, label='Input only (89.02%)')
ax.axhline(y=91.47, color='gray',  linestyle='--', linewidth=1, label='Float baseline (91.47%)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Validation accuracy — all cases')
ax.legend(fontsize=8)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Right: train vs val for each case
ax = axes[1]
for (clip, (train_accs, val_accs, best_acc, label)), color in zip(all_results.items(), colors):
    ax.plot(epochs, train_accs, color=color, linewidth=1,   linestyle='--', alpha=0.6)
    ax.plot(epochs, val_accs,   color=color, linewidth=1.5, label=f'{label}')
ax.axhline(y=89.02, color='black', linestyle=':', linewidth=1)
ax.axhline(y=91.47, color='gray',  linestyle=':', linewidth=1)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Train (dashed) vs Val (solid) — all cases')
ax.legend(fontsize=8)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

fig.suptitle('Activation level sweep — 8-level quantization', fontsize=13)
fig.tight_layout()
combined_path = f'{SAVE_DIR}/combined_all_cases.png'
fig.savefig(combined_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved combined plot: {combined_path}")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"  Float baseline          : 91.47%")
print(f"  + 8-level input only    : 89.02%")
for clip, (_, _, best_acc, label) in all_results.items():
    print(f"  + activations {label:10s} : {best_acc:.2f}%")
print("=" * 55)