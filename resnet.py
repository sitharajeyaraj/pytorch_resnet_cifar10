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
# 8-LEVEL ACTIVATION QUANTIZER WITH STACKED TANH GRADIENT
# Replaces ReLU after every BN.
# Forward : hard snap to nearest of 8 levels  <- UNCHANGED
# Backward: gradient of smooth stacked tanh   <- CHANGED from identity STE
#           Large gradient near level boundaries,
#           small gradient inside flat regions.
#           beta set once before training, no annealing needed
# ============================================================
class Act8LevelTanhGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels, beta):
        transitions = (levels[:-1] + levels[1:]) / 2.0
        ctx.save_for_backward(x)
        ctx.transitions = transitions
        ctx.beta = beta
        dists = (x.unsqueeze(-1) - levels).abs()
        idx   = dists.argmin(dim=-1)
        return levels[idx]

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        beta = ctx.beta
        transitions = ctx.transitions
        grad_estimate = torch.zeros_like(x)
        for t in transitions:
            tanh_val = torch.tanh(beta * (x - t))
            grad_estimate = grad_estimate + beta * (1.0 - tanh_val ** 2)
        # Normalise by number of transitions so peak gradient magnitude ≈ 1
        grad_estimate = grad_estimate / len(transitions)
        return grad_out * grad_estimate, None, None


class Activation8Level(nn.Module):
    def __init__(self):
        super(Activation8Level, self).__init__()
        levels = torch.linspace(-1.0, 1.0, 8)
        self.register_buffer('levels', levels)
        # beta set by training script via set_act_beta()
        # fixed — no annealing needed since forward is always hard snap
        self.beta = 5.0

    def forward(self, x):
        return Act8LevelTanhGrad.apply(x, self.levels, self.beta)


# ============================================================
# 8-LEVEL WEIGHT QUANTIZER WITH STACKED TANH GRADIENT
# Applied inside QConv2d during the forward pass.
# Forward : hard snap to nearest of 8 levels  <- same idea as activations
# Backward: gradient of smooth stacked tanh   <- same idea as activations
#           but with a separate beta (W_BETA) tunable independently
#           from activation beta (ACT_BETA)
# ============================================================
class Weight8LevelTanhGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, levels, beta):
        transitions = (levels[:-1] + levels[1:]) / 2.0
        ctx.save_for_backward(w)
        ctx.transitions = transitions
        ctx.beta = beta
        # Forward: hard argmin snap — weights used in conv are discrete
        dists = (w.unsqueeze(-1) - levels).abs()
        idx   = dists.argmin(dim=-1)
        return levels[idx]

    @staticmethod
    def backward(ctx, grad_out):
        w, = ctx.saved_tensors
        beta = ctx.beta
        transitions = ctx.transitions
        # Same stacked tanh gradient as activations
        # but applied to weight values w instead of activation values x
        grad_estimate = torch.zeros_like(w)
        for t in transitions:
            tanh_val = torch.tanh(beta * (w - t))
            grad_estimate = grad_estimate + beta * (1.0 - tanh_val ** 2)
        # Normalise by number of transitions so peak gradient magnitude ≈ 1
        grad_estimate = grad_estimate / len(transitions)
        # Three return values: one per forward argument (w, levels, beta)
        return grad_out * grad_estimate, None, None


# ============================================================
# QUANTIZED CONV LAYER
# Drop-in replacement for nn.Conv2d.
# Float weights are always kept as the learnable parameter.
# During forward, weights are snapped to 8 levels on the fly.
# quantize_weights flag: False = float conv, True = quantized conv
# Switched on via model.set_weight_quantization(True) in training script.
# ============================================================
class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False, w_clip=1.0):
        super(QConv2d, self).__init__()
        self.stride  = stride
        self.padding = padding
        # Float weight — always kept, optimizer updates this
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        self.bias = None
        # 8 levels for weights: linspace(-w_clip, +w_clip, 8)
        levels = torch.linspace(-w_clip, w_clip, 8)
        self.register_buffer('levels', levels)
        # beta for weight backward — set by training script via set_weight_beta()
        self.beta = 5.0
        # quantize_weights off by default — switched on by training script
        self.quantize_weights = False

    def forward(self, x):
        if self.quantize_weights:
            # Snap float weights to nearest of 8 levels before conv
            # The snapped copy is temporary — float weight is unchanged
            w = Weight8LevelTanhGrad.apply(self.weight, self.levels, self.beta)
        else:
            # Float conv — used during initial loading and warmup
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


# ============================================================
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, QConv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', w_clip=1.0):
        super(BasicBlock, self).__init__()
        # QConv2d instead of nn.Conv2d — intercepts forward to snap weights
        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, w_clip=w_clip)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, w_clip=w_clip)
        self.bn2   = nn.BatchNorm2d(planes)
        self.act1  = Activation8Level()
        self.act2  = Activation8Level()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    QConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, w_clip=w_clip),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, w_clip=1.0):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.input_quantizer = Input8Level()
        # Stem conv — also replaced with QConv2d
        self.conv1 = QConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, w_clip=w_clip)
        self.bn1   = nn.BatchNorm2d(16)
        self.act1  = Activation8Level()
        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1, w_clip=w_clip)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2, w_clip=w_clip)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2, w_clip=w_clip)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, w_clip=1.0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, w_clip=w_clip))
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

    def set_weight_quantization(self, on: bool):
        """Switch weight quantization on or off for all QConv2d layers."""
        for m in self.modules():
            if isinstance(m, QConv2d):
                m.quantize_weights = on

    def set_weight_beta(self, beta: float):
        """Set beta for all QConv2d layers (weight backward sharpness)."""
        for m in self.modules():
            if isinstance(m, QConv2d):
                m.beta = beta

    def set_act_beta(self, beta: float):
        """Set beta for all Activation8Level layers (activation backward sharpness)."""
        for m in self.modules():
            if isinstance(m, Activation8Level):
                m.beta = beta


def resnet20(w_clip=1.0):
    return ResNet(BasicBlock, [3, 3, 3], w_clip=w_clip)

def resnet32(w_clip=1.0):
    return ResNet(BasicBlock, [5, 5, 5], w_clip=w_clip)

def resnet44(w_clip=1.0):
    return ResNet(BasicBlock, [7, 7, 7], w_clip=w_clip)

def resnet56(w_clip=1.0):
    return ResNet(BasicBlock, [9, 9, 9], w_clip=w_clip)

def resnet110(w_clip=1.0):
    return ResNet(BasicBlock, [18, 18, 18], w_clip=w_clip)

def resnet1202(w_clip=1.0):
    return ResNet(BasicBlock, [200, 200, 200], w_clip=w_clip)


def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1,
        net.parameters()))))

if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()