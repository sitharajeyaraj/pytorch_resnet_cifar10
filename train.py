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
# NOISE MODES (controlled by NOISE_MODE):
#   'none'      — no noise at all
#   'uniform'   — same noise variance for all layers
#   'layerwise' — different noise variance per layer group
#                 (stem, layer1, layer2, layer3)
#   For uniform and layerwise, NOISE_PHASE_EPOCHS controls
#   how many epochs noise is active. After that, clean.
#
# OUTPUTS (auto-saved to experiments/<run_name>/):
#   config.txt          — exact settings used
#   log.txt             — epoch-by-epoch results
#   best_model.pth      — best val accuracy checkpoint
#   last_model.pth      — checkpoint after final epoch
#   training_curves.png — accuracy + loss plot
# ==============================================================

import os
import math
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

EXPERIMENT       = '8level-weight-ste-noise-layerwise'

# --- Checkpoint -------------------------------------------
CHECKPOINT       = 'experiments/8level-weight-ste_act-ste_wgt-ste_lr1e-3_nosched/best_model.pth'

# --- What to quantize -------------------------------------
QUANTIZE_INPUT   = True
QUANTIZE_ACT     = True
QUANTIZE_WEIGHTS = True

# --- Which backward estimator -----------------------------
ACT_GRAD    = 'ste'
WEIGHT_GRAD = 'ste'

# --- Quantization levels ----------------------------------
ACT_CLIP    = 1.0
W_CLIP      = 1.0
ACT_BETA    = 5.0
W_BETA      = 5.0

# --- Noise settings ---------------------------------------
# NOISE_MODE:
#   'none'      — no noise
#   'uniform'   — same variance for all layers (use NOISE_VARIANCE)
#   'layerwise' — different variance per layer group
NOISE_MODE         = 'layerwise'
NOISE_PHASE_EPOCHS = 130          # noise ON for first N epochs, then clean

# Uniform noise variance (used when NOISE_MODE='uniform')
NOISE_VARIANCE     = 0.025

# Layerwise noise variances (used when NOISE_MODE='layerwise')
# More noise in early layers, less in later layers
NOISE_VAR_STEM     = 0.05         # stem act   (1 layer)   std=0.224
NOISE_VAR_LAYER1   = 0.025        # layer1 acts (6 layers)  std=0.158
NOISE_VAR_LAYER2   = 0.01         # layer2 acts (6 layers)  std=0.100
NOISE_VAR_LAYER3   = 0.001        # layer3 acts (6 layers)  std=0.032

# --- Training schedule ------------------------------------
EPOCHS           = 150
LR               = 1e-3

USE_SCHEDULER    = False
STEP_SIZE        = 25
SCHEDULER_GAMMA  = 0.5

# --- Other hyperparameters --------------------------------
BATCH_SIZE   = 128
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4

# --- GPU selection ----------------------------------------
GPU_ID = int(os.environ.get('GPU_ID', 0))

# ==============================================================
# END CONFIG
# ==============================================================

# Compute std values
NOISE_STD        = math.sqrt(NOISE_VARIANCE)
NOISE_STD_STEM   = math.sqrt(NOISE_VAR_STEM)
NOISE_STD_LAYER1 = math.sqrt(NOISE_VAR_LAYER1)
NOISE_STD_LAYER2 = math.sqrt(NOISE_VAR_LAYER2)
NOISE_STD_LAYER3 = math.sqrt(NOISE_VAR_LAYER3)

# Layerwise noise dict — passed to model.set_layerwise_noise()
LAYERWISE_NOISE_STD = {
    'stem'  : NOISE_STD_STEM,
    'layer1': NOISE_STD_LAYER1,
    'layer2': NOISE_STD_LAYER2,
    'layer3': NOISE_STD_LAYER3,
}

# ------ Build run name ------------------------------------
sched_tag  = f'step{STEP_SIZE}' if USE_SCHEDULER else 'nosched'
lr_tag     = f'lr{LR:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
act_tag    = f'act-{ACT_GRAD}'
wgt_tag    = f'wgt-{WEIGHT_GRAD}' if QUANTIZE_WEIGHTS else 'wgt-float'

if NOISE_MODE == 'layerwise':
    noise_tag = (f'noise-lw-s{NOISE_VAR_STEM}-l1{NOISE_VAR_LAYER1}'
                 f'-l2{NOISE_VAR_LAYER2}-l3{NOISE_VAR_LAYER3}-{NOISE_PHASE_EPOCHS}ep')
elif NOISE_MODE == 'uniform':
    noise_tag = f'noise-uniform-{NOISE_VARIANCE}-{NOISE_PHASE_EPOCHS}ep'
else:
    noise_tag = 'nonoise'

RUN_NAME   = f'{EXPERIMENT}_{act_tag}_{wgt_tag}_{lr_tag}_{sched_tag}_{noise_tag}'
OUTPUT_DIR = os.path.join('experiments', RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE   = os.path.join(OUTPUT_DIR, 'log.txt')
SAVE_PATH  = os.path.join(OUTPUT_DIR, 'best_model.pth')
LAST_PATH  = os.path.join(OUTPUT_DIR, 'last_model.pth')
PLOT_FILE  = os.path.join(OUTPUT_DIR, 'training_curves.png')
CFG_FILE   = os.path.join(OUTPUT_DIR, 'config.txt')

# ------ Helper: write to terminal and log -----------------
log_fh = open(LOG_FILE, 'w')

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()

# ------ Save config ---------------------------------------
def save_config():
    lines = [
        f'RUN_NAME           = {RUN_NAME}',
        f'EXPERIMENT         = {EXPERIMENT}',
        f'CHECKPOINT         = {CHECKPOINT}',
        f'',
        f'QUANTIZE_INPUT     = {QUANTIZE_INPUT}',
        f'QUANTIZE_ACT       = {QUANTIZE_ACT}',
        f'QUANTIZE_WEIGHTS   = {QUANTIZE_WEIGHTS}',
        f'',
        f'ACT_GRAD           = {ACT_GRAD}',
        f'WEIGHT_GRAD        = {WEIGHT_GRAD}',
        f'ACT_CLIP           = {ACT_CLIP}',
        f'W_CLIP             = {W_CLIP}',
        f'ACT_BETA           = {ACT_BETA}',
        f'W_BETA             = {W_BETA}',
        f'',
        f'NOISE_MODE         = {NOISE_MODE}',
        f'NOISE_PHASE_EPOCHS = {NOISE_PHASE_EPOCHS}',
        f'',
        f'NOISE_VAR_STEM     = {NOISE_VAR_STEM}   std={NOISE_STD_STEM:.4f}',
        f'NOISE_VAR_LAYER1   = {NOISE_VAR_LAYER1}  std={NOISE_STD_LAYER1:.4f}',
        f'NOISE_VAR_LAYER2   = {NOISE_VAR_LAYER2}   std={NOISE_STD_LAYER2:.4f}',
        f'NOISE_VAR_LAYER3   = {NOISE_VAR_LAYER3}  std={NOISE_STD_LAYER3:.4f}',
        f'',
        f'EPOCHS             = {EPOCHS}',
        f'LR                 = {LR}',
        f'USE_SCHEDULER      = {USE_SCHEDULER}',
        f'STEP_SIZE          = {STEP_SIZE}',
        f'SCHEDULER_GAMMA    = {SCHEDULER_GAMMA}',
        f'',
        f'BATCH_SIZE         = {BATCH_SIZE}',
        f'MOMENTUM           = {MOMENTUM}',
        f'WEIGHT_DECAY       = {WEIGHT_DECAY}',
        f'GPU_ID             = {GPU_ID}',
    ]
    with open(CFG_FILE, 'w') as f:
        f.write('\n'.join(lines))

# ------ Device setup --------------------------------------
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

# ------ Data loaders --------------------------------------
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

# ------ Build model ---------------------------------------
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
    noise_std        = 0.0,   # always start with 0 — set per layer below
).to(device)

# ------ Load checkpoint -----------------------------------
if CHECKPOINT and os.path.isfile(CHECKPOINT):
    ckpt       = torch.load(CHECKPOINT, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
else:
    missing = unexpected = []

# ------ Optimizer and scheduler ---------------------------
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

# ------ Print and log header ------------------------------
save_config()

header_lines = [
    '=' * 70,
    f'RUN      : {RUN_NAME}',
    f'OUTPUT   : {OUTPUT_DIR}',
    f'Device   : {device}',
    f'Start    : {CHECKPOINT}',
    f'Quant    : input={QUANTIZE_INPUT}  act={QUANTIZE_ACT}  weights={QUANTIZE_WEIGHTS}',
    f'Grad     : act={ACT_GRAD}  weight={WEIGHT_GRAD}',
    f'Noise    : mode={NOISE_MODE}  phase={NOISE_PHASE_EPOCHS} epochs',
    f'  Stem   : var={NOISE_VAR_STEM}   std={NOISE_STD_STEM:.4f}',
    f'  Layer1 : var={NOISE_VAR_LAYER1}  std={NOISE_STD_LAYER1:.4f}',
    f'  Layer2 : var={NOISE_VAR_LAYER2}   std={NOISE_STD_LAYER2:.4f}',
    f'  Layer3 : var={NOISE_VAR_LAYER3}  std={NOISE_STD_LAYER3:.4f}',
    f'LR={LR}  Epochs={EPOCHS}  Scheduler={USE_SCHEDULER}',
    f'Missing keys: {len(missing)}',
    '=' * 70,
    f'{"Epoch":>6}  {"Phase":>8}  {"TrainAcc":>9}  {"TrainLoss":>10}  '
    f'{"ValAcc":>8}  {"ValLoss":>8}  {"LR":>9}',
    '-' * 70,
]
for line in header_lines:
    log(line)

# ------ Training and evaluation ---------------------------
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


# ------ Reference lines for plot --------------------------
REFERENCE_LINES = {
    'Float baseline'             : 91.47,
    '+ 8-level input'            : 89.02,
    '+ Acts STE, float weights'  : 86.01,
    '+ Acts+Wgt STE (82.67%)'   : 82.67,
    '+ Noise var=0.025 (83.84%)' : 83.84,
}

# ------ Training loop -------------------------------------
train_accs, train_losses = [], []
val_accs,   val_losses   = [], []
best_val  = 0.0

for epoch in range(1, EPOCHS + 1):

    # --- Noise phase control ------------------------------
    if NOISE_MODE != 'none' and epoch <= NOISE_PHASE_EPOCHS:
        model.set_noise(True)
        if NOISE_MODE == 'layerwise':
            model.set_layerwise_noise(LAYERWISE_NOISE_STD)
        else:  # uniform
            model.set_noise_std(NOISE_STD)
        phase = 'NOISY'
    else:
        model.set_noise(False)
        phase = 'CLEAN'

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

    # Save last epoch checkpoint
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'last_val_acc': val_acc,
                'run_name': RUN_NAME},
               LAST_PATH)

    log(f'{epoch:>6}  {phase:>8}  {train_acc:>8.2f}%  {train_loss:>10.4f}  '
        f'{val_acc:>7.2f}%  {val_loss:>8.4f}  {current_lr:>9.6f}{best_tag}')

# ------ Final summary -------------------------------------
summary = [
    '-' * 70,
    f'Best val accuracy : {best_val:.2f}%',
    f'Best checkpoint   : {SAVE_PATH}',
    f'Last checkpoint   : {LAST_PATH}',
    f'Log saved         : {LOG_FILE}',
    f'Config saved      : {CFG_FILE}',
    f'Plot saved        : {PLOT_FILE}',
]
for line in summary:
    log(line)
log_fh.close()

# ------ Save plot -----------------------------------------
epochs_axis = list(range(1, EPOCHS + 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'{RUN_NAME}\nBest val: {best_val:.2f}%', fontsize=9)

# Shade noisy phase
if NOISE_MODE != 'none':
    for ax in [ax1, ax2]:
        ax.axvspan(1, NOISE_PHASE_EPOCHS, alpha=0.08, color='red',
                   label=f'Noisy phase (ep 1-{NOISE_PHASE_EPOCHS})')

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
ax1.legend(fontsize=6)
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(epochs_axis, train_losses, label='Train loss', color='steelblue')
ax2.plot(epochs_axis, val_losses,   label='Val loss',   color='darkorange')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)