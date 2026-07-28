import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


# ============================================================
# 8-LEVEL INPUT QUANTIZER
# Snaps each normalized pixel to the nearest of 8 uniformly
# spaced levels between -2.75 and +2.75.
# No STE needed — inputs are data, not learnable parameters.
# ============================================================
class Input8Level(nn.Module):
    def __init__(self):
        super(Input8Level, self).__init__()
        levels = torch.linspace(-2.75, 2.75, 8)
        self.register_buffer('levels', levels)

    def forward(self, x):
        dists = (x.unsqueeze(-1) - self.levels).abs()  # [B, 3, 32, 32, 8]
        idx   = dists.argmin(dim=-1)                   # [B, 3, 32, 32]
        return self.levels[idx]                        # [B, 3, 32, 32]


# ============================================================
# 8-LEVEL ACTIVATION QUANTIZER WITH STE
# Replaces ReLU after every BN.
# Forward : hard snap to nearest of 8 levels
# Backward: straight-through gradient inside clip window
#           clip = outermost level (set automatically from levels)
# STE is needed because activations are in the gradient path
# ============================================================
class Act8LevelSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, levels):
        ctx.save_for_backward(x)
        # clip = outermost level value (always positive since levels are symmetric)
        ctx.clip = levels.abs().max().item()
        dists = (x.unsqueeze(-1) - levels).abs()
        idx   = dists.argmin(dim=-1)
        return levels[idx]

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        # Pass gradient only where input is within the clip range
        # Values outside clip always snap to outermost level — no useful gradient
        mask = (x.abs() <= ctx.clip).float()
        return grad_out * mask, None   # None: levels has no gradient


class Activation8Level(nn.Module):
    def __init__(self):
        super(Activation8Level, self).__init__()
        # Default levels — overridden at runtime via levels.copy_() in training script
        levels = torch.linspace(-1.0, 1.0, 8)
        self.register_buffer('levels', levels)

    def forward(self, x):
        return Act8LevelSTE.apply(x, self.levels)


# ============================================================
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.act1  = Activation8Level()   # replaces F.relu after bn1
        self.act2  = Activation8Level()   # replaces F.relu after bn2 + skip

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # conv1 → BN → quantize  (replaces: conv1 → BN → ReLU)
        out = self.act1(self.bn1(self.conv1(x)))
        # conv2 → BN → skip add → quantize  (replaces: conv2 → BN → skip add → ReLU)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.input_quantizer = Input8Level()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.act1  = Activation8Level()   # replaces F.relu after stem BN

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_quantizer(x)
        out = self.act1(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()