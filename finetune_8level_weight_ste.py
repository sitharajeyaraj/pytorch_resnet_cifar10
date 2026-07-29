"""
finetune_8level_weight_ste.py
=============================
Stage 6: Weight quantization with STE on top of the 8-level activation baseline.

Starting checkpoint : 8level_act_clip1_best.pth  (86.01% — activations already quantized)
Branch              : 8level-weight-ste

Weight quantization is switched ON from epoch 1.
LR starts at 1e-3, halved every 25 epochs (step decay).
  Epoch  1-25 : LR = 1e-3
  Epoch 26-50 : LR = 5e-4
  Epoch 51-75 : LR = 2.5e-4
  Epoch 76-100: LR = 1.25e-4
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from resnet import resnet20

# ============================================================
# TOP-LEVEL VARIABLES — change only here
# ============================================================
CHECKPOINT  = '8level_act_clip1_best.pth'   # starting point (86.01%)
SAVE_PATH   = '8level_weight_ste_best.pth'
W_CLIP      = 1.0    # outermost weight level (linspace -1 to +1)
EPOCHS      = 100
LR          = 1e-3   # starting LR — halved every 25 epochs
LR_STEP     = 25     # halve LR every this many epochs
LR_GAMMA    = 0.5    # multiply LR by this at each step
BATCH_SIZE  = 128
NUM_WORKERS = 4
# ============================================================


def get_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=100,
                                               shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total   += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def save_plot(train_accs, val_accs, train_losses, val_losses):
    epochs = list(range(1, len(train_accs) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_accs, label='Train accuracy')
    ax1.plot(epochs, val_accs,   label='Val accuracy')
    # mark LR drop points
    for step in [25, 50, 75]:
        ax1.axvline(x=step, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy (red lines = LR halved)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_losses, label='Train loss')
    ax2.plot(epochs, val_losses,   label='Val loss')
    for step in [25, 50, 75]:
        ax2.axvline(x=step, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss (red lines = LR halved)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('8level_weight_ste_lr1e3_stepdecay_plot.png')
    plt.close()
    print('Plot saved to 8level_weight_ste_lr1e3_stepdecay_plot.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_loader, test_loader = get_loaders()

    # Build model and load activation-quantized checkpoint
    model = resnet20().to(device)
    assert os.path.exists(CHECKPOINT), f'Checkpoint not found: {CHECKPOINT}'
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'Checkpoint epoch {ckpt["epoch"]}, acc {ckpt["acc"]:.2f}%')

    criterion = nn.CrossEntropyLoss()

    # Evaluate starting accuracy before any training
    _, start_acc = evaluate(model, test_loader, criterion, device)
    print(f'Starting val accuracy: {start_acc:.2f}%')

    # Switch weight quantization ON from epoch 1
    model.set_weight_quantization(True)
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=0.9, weight_decay=1e-4)

    # Halve LR every 25 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    best_acc = 0.0

    print(f'\n--- Weight quant ON from epoch 1 ({EPOCHS} epochs) ---')
    print(f'LR schedule: {LR} → {LR*0.5} → {LR*0.25} → {LR*0.125} (halved every {LR_STEP} epochs)\n')

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_accs.append(tr_acc)
        val_accs.append(vl_acc)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f'Epoch {epoch:3d}/{EPOCHS} | LR {current_lr:.2e} | '
              f'train {tr_acc:.2f}% loss {tr_loss:.4f} | '
              f'val {vl_acc:.2f}% loss {vl_loss:.4f} | '
              f'best {best_acc:.2f}%')

    print(f'\nBest val accuracy: {best_acc:.2f}%')
    print(f'Model saved to: {SAVE_PATH}')
    save_plot(train_accs, val_accs, train_losses, val_losses)


if __name__ == '__main__':
    main()