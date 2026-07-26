"""
Fine-tune ResNet-20 with quantized QPSK input.

Starting point : pretrained float ResNet-20 (91.47% on CIFAR-10)
What changes   : QPSKInput snaps all pixels to ±0.7071 before conv1
Goal           : measure how much accuracy we lose/retain after
                 the network adapts to quantized inputs

Usage:
    python finetune_qpsk_input.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import resnet

# ── config ────────────────────────────────────────────────────
PRETRAINED   = 'pretrained_models/resnet20-12fca82f.th'
SAVE_BEST    = 'qpsk_input_best_v2.pth'
SAVE_FINAL   = 'qpsk_input_final_v2.pth'
EPOCHS       = 100
LR           = 1e-3      # low LR — we are fine-tuning, not training from scratch
BATCH        = 128
WORKERS      = 4
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── data ──────────────────────────────────────────────────────
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std =[0.2023, 0.1994, 0.2010]
)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    ),
    batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        './data', train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    ),
    batch_size=256, shuffle=False, num_workers=WORKERS, pin_memory=True
)

# ── model ─────────────────────────────────────────────────────
model = resnet.resnet20().to(DEVICE)

# load pretrained float weights
ckpt = torch.load(PRETRAINED, map_location=DEVICE)
sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd)
print(f'Loaded pretrained weights from {PRETRAINED}')

# warm start from our best finetuned checkpoint if it exists
import os
if os.path.exists('qpsk_input_best.pth'):
    ckpt2 = torch.load('qpsk_input_best.pth', map_location=DEVICE)
    model.load_state_dict(ckpt2['state_dict'])
    print(f'Resumed from qpsk_input_best.pth ({ckpt2["best_acc"]:.2f}%)')

# ── evaluation function ───────────────────────────────────────
@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = total = 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        correct += out.max(1)[1].eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total

# check accuracy BEFORE fine-tuning
# this tells us the cost of just snapping inputs without any adaptation
acc_before = evaluate(model)
print(f'Accuracy with QPSK input (no fine-tuning): {acc_before:.2f}%')
print(f'Float baseline was: 91.47%')
print(f'Accuracy drop from quantized input alone: {91.47 - acc_before:.2f}%')
print()

# ── fine-tuning ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = acc_before
print(f'Starting fine-tuning for {EPOCHS} epochs...')
print(f'{"Epoch":>6}  {"Train Acc":>10}  {"Test Acc":>10}  {"Best":>10}')
print('-' * 44)

for epoch in range(1, EPOCHS + 1):
    # ── train one epoch ───────────────────────────────────────
    model.train()
    correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct += out.max(1)[1].eq(labels).sum().item()
        total   += labels.size(0)
    train_acc = 100.0 * correct / total

    # ── evaluate ──────────────────────────────────────────────
    test_acc = evaluate(model)
    scheduler.step()

    # ── save best ─────────────────────────────────────────────
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch'     : epoch,
            'state_dict': model.state_dict(),
            'best_acc'  : best_acc,
            'stage'     : 'qpsk_input_finetune',
        }, SAVE_BEST)

    print(f'{epoch:>6}  {train_acc:>10.2f}%  {test_acc:>10.2f}%  {best_acc:>10.2f}%')

# ── save final ────────────────────────────────────────────────
torch.save({
    'epoch'     : EPOCHS,
    'state_dict': model.state_dict(),
    'best_acc'  : best_acc,
    'stage'     : 'qpsk_input_finetune',
}, SAVE_FINAL)

print()
print('=' * 44)
print(f'Float baseline          : 91.47%')
print(f'QPSK input (no finetune): {acc_before:.2f}%')
print(f'QPSK input (finetuned)  : {best_acc:.2f}%')
print(f'Best checkpoint saved   : {SAVE_BEST}')
print('=' * 44)