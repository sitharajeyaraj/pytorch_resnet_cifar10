#!/usr/bin/env python3
"""
finetune_8level_input.py
=========================
Fine-tunes ResNet20 with 8-level input quantization.
Starts from the pretrained 91.47% master checkpoint.
Weights stay float — only the input is quantized.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import resnet

# ── Config ───────────────────────────────────────────────────
CHECKPOINT   = './pretrained_models/resnet20-12fca82f.th'
SAVE_PATH    = './8level_input_best.pth'
EPOCHS       = 100
LR           = 1e-3
LR_DECAY     = 0.1
LR_MILESTONES = [40, 70]   # epochs where LR is multiplied by LR_DECAY
BATCH_SIZE   = 128
WORKERS      = 2

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {device}")

# ── Data ─────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

train_set = datasets.CIFAR10(root='./data', train=True,  download=True,  transform=train_transform)
test_set  = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=WORKERS, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=WORKERS, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────
model = resnet.resnet20().to(device)

# Load pretrained weights — strip 'module.' prefix if saved with DataParallel
ckpt = torch.load(CHECKPOINT, map_location=device)
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
print(f"Loaded checkpoint : {CHECKPOINT}")
print(f"Input quantizer   : 8 levels, symmetric [-2.75, +2.75]")
print(f"Weights           : float (fine-tuned from 91.47% baseline)")

# ── Verify input quantizer is active ─────────────────────────
print(f"\n8 quantization levels:")
for i, lv in enumerate(model.input_quantizer.levels.tolist()):
    print(f"  Level {i+1}: {lv:+.4f}")

# ── Training ──────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_DECAY)

best_acc = 0.0

print(f"\nStarting fine-tuning for {EPOCHS} epochs ...\n")

for epoch in range(1, EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────
    model.train()
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    train_acc = 100.0 * correct / total

    # ── Evaluate ──────────────────────────────────────────────
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total   += labels.size(0)
    test_acc = 100.0 * correct / total

    # ── Save best ─────────────────────────────────────────────
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'acc': best_acc}, SAVE_PATH)

    scheduler.step()

    print(f"Epoch {epoch:3d}/{EPOCHS}  "
          f"Train: {train_acc:.2f}%  "
          f"Test: {test_acc:.2f}%  "
          f"Best: {best_acc:.2f}%  "
          f"LR: {scheduler.get_last_lr()[0]:.5f}")

print(f"\nDone. Best accuracy: {best_acc:.2f}%")
print(f"Checkpoint saved : {SAVE_PATH}")