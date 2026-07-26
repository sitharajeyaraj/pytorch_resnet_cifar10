"""
Clean evaluation script for ResNet on CIFAR-10.
Usage: python evaluate.py --arch resnet20 --resume pretrained_models/resnet20-12fca82f.th
"""
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='resnet20')
parser.add_argument('--resume', required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Build model
model = resnet.__dict__[args.arch]().to(device)

# Load checkpoint - strip DataParallel 'module.' prefix
ckpt = torch.load(args.resume, map_location=device)
sd = ckpt['state_dict']
sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
print(f'Loaded checkpoint. Saved accuracy: {ckpt["best_prec1"]:.2f}%')

# CIFAR-10 test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
testset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=256,
                                      shuffle=False, num_workers=4)

correct = total = 0
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        correct += out.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)

print(f'Test accuracy: {100*correct/total:.2f}%  ({correct}/{total})')
