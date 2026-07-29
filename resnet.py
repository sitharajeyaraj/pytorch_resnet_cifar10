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
        levels = torch.linspace(-1.0, 1.0, 8)
        self.register_buffer('levels', levels)

    def forward(self, x):
        return Act8LevelSTE.apply(x, self.levels)


# ============================================================
# 8-LEVEL WEIGHT QUANTIZER WITH STE
# Used inside QConv2d.
# Forward : hard argmin snap to nearest of 8 levels
# Backward: straight-through gradient, zeroed outside clip
#           clip = outermost level (auto-matched from levels)
# Same logic as Act8LevelSTE — only the tensor being snapped
# is weights instead of activations.
# ============================================================
class Weight8LevelSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w, levels):
        ctx.save_for_backward(w)
        ctx.clip = levels.abs().max().item()   # = W_CLIP = 1.0
        dists = (w.unsqueeze(-1) - levels).abs()
        idx   = dists.argmin(dim=-1)
        return levels[idx]

    @staticmethod
    def backward(ctx, grad_out):
        w, = ctx.saved_tensors
        # Zero gradient for weights that have drifted outside the level range.
        # Those weights already snap to the outermost level regardless of their
        # exact value — pushing them further gives zero improvement.
        mask = (w.abs() <= ctx.clip).float()
        return grad_out * mask, None   # None: levels has no gradient


# ============================================================
# QUANTIZED CONV2D
# Drop-in replacement for nn.Conv2d inside residual blocks.
# Stores a float latent weight (what the optimizer updates).
# In the forward pass, snaps the latent weight to 8 levels
# before the convolution — so the actual MAC uses discrete values.
# quantize_weights=False by default; switched on via
# model.set_weight_quantization(True) in the training script.
# ============================================================
class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False, w_clip=1.0):
        super(QConv2d, self).__init__()
        self.stride  = stride
        self.padding = padding

        # Latent weight — float, updated by optimizer every step
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        self.bias = None

        # 8 uniformly spaced levels from -w_clip to +w_clip
        levels = torch.linspace(-w_clip, w_clip, 8)
        self.register_buffer('levels', levels)

        # Off by default — training script turns this on after warm-up
        self.quantize_weights = False

    def forward(self, x):
        if self.quantize_weights:
            # Snap latent weight to nearest level (discrete copy, not in-place)
            w = Weight8LevelSTE.apply(self.weight, self.levels)
        else:
            # Phase 1: use float latent weight directly
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


# ============================================================
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    # Note: QConv2d initialises its own weight in __init__ via kaiming_normal_
    # so it does not need to be handled here.


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
        # QConv2d replaces nn.Conv2d — weights will be quantized during Phase 2
        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.act1  = Activation8Level()   # replaces F.relu after bn1
        self.act2  = Activation8Level()   # replaces F.relu after bn2 + skip

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                # Shortcut stays nn.Conv2d — it is part of the float skip highway
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.input_quantizer = Input8Level()

        # Stem conv stays nn.Conv2d — it sees the quantized input directly
        # and is not part of the weight quantization experiment
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.act1  = Activation8Level()

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

    # ----------------------------------------------------------
    # Helper: turn weight quantization on or off for all QConv2d
    # Call model.set_weight_quantization(True) at the start of Phase 2
    # ----------------------------------------------------------
    def set_weight_quantization(self, on: bool):
        for m in self.modules():
            if isinstance(m, QConv2d):
                m.quantize_weights = on

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