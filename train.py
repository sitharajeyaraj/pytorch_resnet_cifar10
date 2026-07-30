# ==============================================================
# train.py  —  Unified training script for 8-level QNN
# ResNet-20 on CIFAR-10
#
# HOW TO USE:
#   1. Edit the CONFIG section below for your experiment.
#   2. Run:  python train.py
#   3. To pick a specific GPU:
#        GPU_ID=1 python train.py
#
# OUTPUTS (auto-saved to experiments/<run_name>/):
#   config.txt          — exact settings used for this run
#   log.txt             — epoch-by-epoch train/val accuracy and loss
#   best_model.pth      — best checkpoint
#   training_curves.png — accuracy + loss plot
# ==============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import resnet

# ==============================================================
# CONFIG — change only this block for each experiment
# ==============================================================

EXPERIMENT       = '8level-weight-ste'

# --- Checkpoint paths ------------------------------------------
CHECKPOINT       = '8level_act_clip1_best.pth'   # where to START from

# --- What to quantize ------------------------------------------
QUANTIZE_INPUT   = True
QUANTIZE_ACT     = True
QUANTIZE_WEIGHTS = True

# --- Which backward estimator ----------------------------------
ACT_GRAD    = 'ste'       # 'ste' or 'tanhgrad'
WEIGHT_GRAD = 'ste'       # 'ste' or 'tanhgrad'

# --- Quantization levels ---------------------------------------
ACT_CLIP    = 1.0
W_CLIP      = 1.0
ACT_BETA    = 5.0         # ignored if ACT_GRAD='ste'
W_BETA      = 5.0         # ignored if WEIGHT_GRAD='ste'

# --- Training schedule -----------------------------------------
EPOCHS           = 1
LR               = 1e-3

USE_SCHEDULER    = False
STEP_SIZE        = 25     # halve LR every this many epochs
SCHEDULER_GAMMA  = 0.5

# --- Other hyperparameters ------------------------------------
BATCH_SIZE   = 128
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4

# --- GPU selection --------------------------------------------
GPU_ID = int(os.environ.get('GPU_ID', 0))

# ==============================================================
# END CONFIG
# ==============================================================


# ------ Build run name and output folder ----------------------
sched_tag = f'step{STEP_SIZE}' if USE_SCHEDULER else 'nosched'
lr_tag    = f'lr{LR:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
act_tag   = f'act-{ACT_GRAD}'
wgt_tag   = f'wgt-{WEIGHT_GRAD}' if QUANTIZE_WEIGHTS else 'wgt-float'

RUN_NAME   = f'{EXPERIMENT}_{act_tag}_{wgt_tag}_{lr_tag}_{sched_tag}'
OUTPUT_DIR = os.path.join('experiments', RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE   = os.path.join(OUTPUT_DIR, 'log.txt')
SAVE_PATH  = os.path.join(OUTPUT_DIR, 'best_model.pth')
PLOT_FILE  = os.path.join(OUTPUT_DIR, 'training_curves.png')
CFG_FILE   = os.path.join(OUTPUT_DIR, 'config.txt')


# ------ Helper: write to both terminal and log file -----------
log_fh = open(LOG_FILE, 'w')

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()


# ------ Save config -------------------------------------------
def save_config():
    lines = [
        f'RUN_NAME         = {RUN_NAME}',
        f'EXPERIMENT       = {EXPERIMENT}',
        f'CHECKPOINT       = {CHECKPOINT}',
        f'',
        f'QUANTIZE_INPUT   = {QUANTIZE_INPUT}',
        f'QUANTIZE_ACT     = {QUANTIZE_ACT}',
        f'QUANTIZE_WEIGHTS = {QUANTIZE_WEIGHTS}',
        f'',
        f'ACT_GRAD         = {ACT_GRAD}',
        f'WEIGHT_GRAD      = {WEIGHT_GRAD}',
        f'ACT_CLIP         = {ACT_CLIP}',
        f'W_CLIP           = {W_CLIP}',
        f'ACT_BETA         = {ACT_BETA}',
        f'W_BETA           = {W_BETA}',
        f'',
        f'EPOCHS           = {EPOCHS}',
        f'LR               = {LR}',
        f'USE_SCHEDULER    = {USE_SCHEDULER}',
        f'STEP_SIZE        = {STEP_SIZE}',
        f'SCHEDULER_GAMMA  = {SCHEDULER_GAMMA}',
        f'',
        f'BATCH_SIZE       = {BATCH_SIZE}',
        f'MOMENTUM         = {MOMENTUM}',
        f'WEIGHT_DECAY     = {WEIGHT_DECAY}',
        f'GPU_ID           = {GPU_ID}',
    ]
    with open(CFG_FILE, 'w') as f:
        f.write('\n'.join(lines))


# ------ Device setup ------------------------------------------
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

# ------ Data loaders ------------------------------------------
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True,  download=True, transform=train_transform)
test_dataset  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=256,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ------ Build model -------------------------------------------
model = resnet.resnet20(
    quantize_input   = QUANTIZE_INPUT,
    quantize_act     = QUANTIZE_ACT,
    quantize_weights = QUANTIZE_WEIGHTS,
    act_grad         = ACT_GRAD,
    weight_grad      = WEIGHT_GRAD,
    act_clip         = ACT_CLIP,
    w_clip           = W_CLIP,
    act_beta         = ACT_BETA,
    w_beta           = W_BETA,
).to(device)

# ------ Load checkpoint ---------------------------------------
if CHECKPOINT and os.path.isfile(CHECKPOINT):
    ckpt       = torch.load(CHECKPOINT, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
else:
    missing = unexpected = []

# ------ Optimizer and scheduler --------------------------------
optimizer = optim.SGD(model.parameters(),
                      lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
if USE_SCHEDULER:
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=STEP_SIZE,
                                          gamma=SCHEDULER_GAMMA)
else:
    scheduler = None

criterion = nn.CrossEntropyLoss()
cudnn.benchmark = True

# ------ Print and log header ----------------------------------
save_config()

header_lines = [
    '=' * 60,
    f'RUN    : {RUN_NAME}',
    f'OUTPUT : {OUTPUT_DIR}',
    f'Device : {device}',
    f'Start  : {CHECKPOINT}',
    f'Quant  : input={QUANTIZE_INPUT}  act={QUANTIZE_ACT}  weights={QUANTIZE_WEIGHTS}',
    f'Grad   : act={ACT_GRAD}  weight={WEIGHT_GRAD}',
    f'Clip   : act={ACT_CLIP}  weight={W_CLIP}',
    f'LR={LR}  Epochs={EPOCHS}  Scheduler={USE_SCHEDULER}',
    f'Missing keys (new layers): {len(missing)}',
    f'=' * 60,
    f'{"Epoch":>6}  {"TrainAcc":>9}  {"TrainLoss":>10}  {"ValAcc":>8}  {"ValLoss":>8}  {"LR":>9}',
    f'{"-"*60}',
]
for line in header_lines:
    log(line)


# ------ Training and evaluation functions ---------------------
def train_one_epoch():
    model.train()
    correct = total = 0
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted  = outputs.max(1)
        total        += targets.size(0)
        correct      += predicted.eq(targets).sum().item()
    return 100. * correct / total, running_loss / total


def evaluate():
    model.eval()
    correct = total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs      = model(inputs)
            loss         = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted  = outputs.max(1)
            total        += targets.size(0)
            correct      += predicted.eq(targets).sum().item()
    return 100. * correct / total, running_loss / total


# ------ Reference lines for plot ------------------------------
REFERENCE_LINES = {
    'Float baseline'            : 91.47,
    '+ 8-level input'           : 89.02,
    '+ Acts STE, float weights' : 86.01,
}

# ------ Training loop -----------------------------------------
train_accs, train_losses = [], []
val_accs,   val_losses   = [], []
best_val = 0.0

for epoch in range(1, EPOCHS + 1):
    train_acc, train_loss = train_one_epoch()
    val_acc,   val_loss   = evaluate()

    train_accs.append(train_acc)
    train_losses.append(train_loss)
    val_accs.append(val_acc)
    val_losses.append(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    if scheduler:
        scheduler.step()

    best_tag = ''
    if val_acc > best_val:
        best_val = val_acc
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_val,
                    'run_name': RUN_NAME},
                   SAVE_PATH)
        best_tag = '  ← best'

    log(f'{epoch:>6}  {train_acc:>8.2f}%  {train_loss:>10.4f}  '
        f'{val_acc:>7.2f}%  {val_loss:>8.4f}  {current_lr:>9.6f}{best_tag}')


# ------ Final summary -----------------------------------------
summary = [
    '-' * 60,
    f'Best val accuracy : {best_val:.2f}%',
    f'Checkpoint saved  : {SAVE_PATH}',
    f'Log saved         : {LOG_FILE}',
    f'Config saved      : {CFG_FILE}',
    f'Plot saved        : {PLOT_FILE}',
]
for line in summary:
    log(line)

log_fh.close()

# ------ Save plot ---------------------------------------------
epochs_axis = list(range(1, EPOCHS + 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'{RUN_NAME}\nBest val: {best_val:.2f}%', fontsize=11)

# Accuracy
ax1.plot(epochs_axis, train_accs, label='Train acc', color='steelblue')
ax1.plot(epochs_axis, val_accs,   label='Val acc',   color='darkorange')
colors = ['gray', 'green', 'red', 'purple', 'brown']
for i, (label, acc) in enumerate(REFERENCE_LINES.items()):
    ax1.axhline(y=acc, linestyle='--', color=colors[i % len(colors)],
                linewidth=1.0, label=label)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(epochs_axis, train_losses, label='Train loss', color='steelblue')
ax2.plot(epochs_axis, val_losses,   label='Val loss',   color='darkorange')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
