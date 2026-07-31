import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


# ============================================================
# 8-LEVEL INPUT QUANTIZER
# ============================================================
class Input8Level(nn.Module):
    def __init__(self):
        super(Input8Level, self).__init__()
        levels = torch.linspace(-2.75, 2.75, 8)
        self.register_buffer('levels', levels)

    def forward(self, x):
        dists = (x.unsqueeze(-1) - self.levels).abs()
        idx   = dists.argmin(dim=-1)
        return self.levels[idx]


# ============================================================
# ACTIVATION QUANTIZERS
# ============================================================

class Act8LevelSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels):
        dists = (x.unsqueeze(-1) - levels).abs()
        idx   = dists.argmin(dim=-1)
        lo, hi = levels[0], levels[-1]
        mask = ((x >= lo) & (x <= hi)).to(x.dtype)
        ctx.save_for_backward(mask)
        return levels[idx]

    @staticmethod
    def backward(ctx, grad_out):
        mask, = ctx.saved_tensors
        return grad_out * mask, None


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
        grad_estimate = grad_estimate / len(transitions)
        return grad_out * grad_estimate, None, None


class Activation8Level(nn.Module):
    """
    Drop-in replacement for ReLU.
    grad_type  : 'ste' or 'tanhgrad'
    act_clip   : outermost level (levels = linspace(-act_clip, +act_clip, 8))
    beta       : sharpness for tanhgrad backward (ignored for ste)
    noise_std  : std of Gaussian noise added AFTER snap during training.
                 0.0 = no noise (default).
                 Can be set per layer via model.set_layerwise_noise().
                 Noise is NEVER added during validation (model.eval()).
    add_noise  : master switch — set to False to disable noise completely.
    layer_group: which group this activation belongs to
                 'stem', 'layer1', 'layer2', 'layer3'
                 Used by set_layerwise_noise() to assign correct std.
    """
    def __init__(self, grad_type='ste', act_clip=1.0, beta=5.0,
                 noise_std=0.0, layer_group='stem'):
        super(Activation8Level, self).__init__()
        levels = torch.linspace(-act_clip, act_clip, 8)
        self.register_buffer('levels', levels)
        self.grad_type   = grad_type
        self.beta        = beta
        self.noise_std   = noise_std
        self.add_noise   = False
        self.layer_group = layer_group   # 'stem', 'layer1', 'layer2', 'layer3'

    def forward(self, x):
        if self.grad_type == 'tanhgrad':
            out = Act8LevelTanhGrad.apply(x, self.levels, self.beta)
        else:
            out = Act8LevelSTE.apply(x, self.levels)

        if self.add_noise and self.training and self.noise_std > 0.0:
            noise = torch.randn_like(out) * self.noise_std
            out   = out + noise

        return out


# ============================================================
# WEIGHT QUANTIZERS
# ============================================================

class Weight8LevelSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, levels):
        dists = (w.unsqueeze(-1) - levels).abs()
        idx   = dists.argmin(dim=-1)
        lo, hi = levels[0], levels[-1]
        mask = ((w >= lo) & (w <= hi)).to(w.dtype)
        ctx.save_for_backward(mask)
        return levels[idx]

    @staticmethod
    def backward(ctx, grad_out):
        mask, = ctx.saved_tensors
        return grad_out * mask, None


class Weight8LevelTanhGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, levels, beta):
        transitions = (levels[:-1] + levels[1:]) / 2.0
        ctx.save_for_backward(w)
        ctx.transitions = transitions
        ctx.beta = beta
        dists = (w.unsqueeze(-1) - levels).abs()
        idx   = dists.argmin(dim=-1)
        return levels[idx]

    @staticmethod
    def backward(ctx, grad_out):
        w, = ctx.saved_tensors
        beta = ctx.beta
        transitions = ctx.transitions
        grad_estimate = torch.zeros_like(w)
        for t in transitions:
            tanh_val = torch.tanh(beta * (w - t))
            grad_estimate = grad_estimate + beta * (1.0 - tanh_val ** 2)
        grad_estimate = grad_estimate / len(transitions)
        return grad_out * grad_estimate, None, None


# ============================================================
# QUANTIZED CONV LAYER
# ============================================================
class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False,
                 grad_type='ste', w_clip=1.0, w_beta=5.0):
        super(QConv2d, self).__init__()
        self.stride    = stride
        self.padding   = padding
        self.grad_type = grad_type
        self.w_beta    = w_beta
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        self.bias = None
        levels = torch.linspace(-w_clip, w_clip, 8)
        self.register_buffer('levels', levels)
        self.quantize_weights = False

    def forward(self, x):
        if self.quantize_weights:
            if self.grad_type == 'tanhgrad':
                w = Weight8LevelTanhGrad.apply(self.weight, self.levels, self.w_beta)
            else:
                w = Weight8LevelSTE.apply(self.weight, self.levels)
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


# ============================================================
def _weights_init(m):
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

    def __init__(self, in_planes, planes, stride=1, option='A',
                 act_grad='ste', act_clip=1.0, act_beta=5.0, noise_std=0.0,
                 weight_grad='ste', w_clip=1.0, w_beta=5.0,
                 layer_group='layer1'):
        super(BasicBlock, self).__init__()
        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride,
                             padding=1, bias=False,
                             grad_type=weight_grad, w_clip=w_clip, w_beta=w_beta)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1,
                             padding=1, bias=False,
                             grad_type=weight_grad, w_clip=w_clip, w_beta=w_beta)
        self.bn2   = nn.BatchNorm2d(planes)
        self.act1  = Activation8Level(grad_type=act_grad,
                                      act_clip=act_clip, beta=act_beta,
                                      noise_std=noise_std,
                                      layer_group=layer_group)
        self.act2  = Activation8Level(grad_type=act_grad,
                                      act_clip=act_clip, beta=act_beta,
                                      noise_std=noise_std,
                                      layer_group=layer_group)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    QConv2d(in_planes, self.expansion * planes, kernel_size=1,
                            stride=stride, bias=False,
                            grad_type=weight_grad, w_clip=w_clip, w_beta=w_beta),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,
                 quantize_input=True, quantize_act=True, quantize_weights=True,
                 act_grad='ste',    act_clip=1.0, act_beta=5.0, noise_std=0.0,
                 weight_grad='ste', w_clip=1.0,   w_beta=5.0):
        super(ResNet, self).__init__()
        self.in_planes        = 16
        self.quantize_input   = quantize_input
        self.quantize_act     = quantize_act
        self._quantize_weights = quantize_weights

        self.input_quantizer = Input8Level()

        self.conv1 = QConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                             grad_type=weight_grad, w_clip=w_clip, w_beta=w_beta)
        self.bn1   = nn.BatchNorm2d(16)

        if quantize_act:
            self.act1 = Activation8Level(grad_type=act_grad,
                                         act_clip=act_clip, beta=act_beta,
                                         noise_std=noise_std,
                                         layer_group='stem')
        else:
            self.act1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1,
                                       quantize_act=quantize_act,
                                       act_grad=act_grad, act_clip=act_clip,
                                       act_beta=act_beta, noise_std=noise_std,
                                       weight_grad=weight_grad, w_clip=w_clip,
                                       w_beta=w_beta, layer_group='layer1')
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2,
                                       quantize_act=quantize_act,
                                       act_grad=act_grad, act_clip=act_clip,
                                       act_beta=act_beta, noise_std=noise_std,
                                       weight_grad=weight_grad, w_clip=w_clip,
                                       w_beta=w_beta, layer_group='layer2')
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2,
                                       quantize_act=quantize_act,
                                       act_grad=act_grad, act_clip=act_clip,
                                       act_beta=act_beta, noise_std=noise_std,
                                       weight_grad=weight_grad, w_clip=w_clip,
                                       w_beta=w_beta, layer_group='layer3')
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

        if quantize_weights:
            self.set_weight_quantization(True)

    def _make_layer(self, block, planes, num_blocks, stride,
                    quantize_act=True,
                    act_grad='ste', act_clip=1.0, act_beta=5.0, noise_std=0.0,
                    weight_grad='ste', w_clip=1.0, w_beta=5.0,
                    layer_group='layer1'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s,
                                act_grad=act_grad, act_clip=act_clip,
                                act_beta=act_beta, noise_std=noise_std,
                                weight_grad=weight_grad, w_clip=w_clip,
                                w_beta=w_beta, layer_group=layer_group))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.quantize_input:
            x = self.input_quantizer(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def set_weight_quantization(self, on: bool):
        for m in self.modules():
            if isinstance(m, QConv2d):
                m.quantize_weights = on

    def set_noise(self, on: bool):
        """Switch noise on or off for ALL Activation8Level layers."""
        for m in self.modules():
            if isinstance(m, Activation8Level):
                m.add_noise = on

    def set_noise_std(self, std: float):
        """Set same noise std for ALL layers. Used for uniform noise."""
        for m in self.modules():
            if isinstance(m, Activation8Level):
                m.noise_std = std

    def set_layerwise_noise(self, noise_dict: dict):
        """
        Set different noise std for each layer group.
        noise_dict example:
            {
                'stem'  : 0.224,   # sqrt(0.05)
                'layer1': 0.158,   # sqrt(0.025)
                'layer2': 0.100,   # sqrt(0.01)
                'layer3': 0.032,   # sqrt(0.001)
            }
        Activations not in the dict are left unchanged.
        """
        for m in self.modules():
            if isinstance(m, Activation8Level):
                if m.layer_group in noise_dict:
                    m.noise_std = noise_dict[m.layer_group]

    def set_act_beta(self, beta: float):
        for m in self.modules():
            if isinstance(m, Activation8Level):
                m.beta = beta

    def set_weight_beta(self, beta: float):
        for m in self.modules():
            if isinstance(m, QConv2d):
                m.w_beta = beta


# ============================================================
# PUBLIC CONSTRUCTORS
# ============================================================
def resnet20(quantize_input=True, quantize_act=True, quantize_weights=True,
             act_grad='ste',    act_clip=1.0, act_beta=5.0, noise_std=0.0,
             weight_grad='ste', w_clip=1.0,   w_beta=5.0):
    return ResNet(BasicBlock, [3, 3, 3],
                  quantize_input=quantize_input,
                  quantize_act=quantize_act,
                  quantize_weights=quantize_weights,
                  act_grad=act_grad,    act_clip=act_clip,
                  act_beta=act_beta,   noise_std=noise_std,
                  weight_grad=weight_grad, w_clip=w_clip, w_beta=w_beta)

def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)

def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)

def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)

def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)

def resnet1202(**kwargs):
    return ResNet(BasicBlock, [200, 200, 200], **kwargs)


def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    # Print layer groups for verification
    print("\nLayer group assignments:")
    for name, m in net.named_modules():
        if isinstance(m, Activation8Level):
            print(f"  {name:40s} → group={m.layer_group}  noise_std={m.noise_std}")

if __name__ == "__main__":
    print("Testing resnet20 — layerwise noise check:")
    net = resnet20(quantize_input=True, quantize_act=True, quantize_weights=True,
                   act_grad='ste', weight_grad='ste', noise_std=0.0)
    # Set layerwise noise
    net.set_layerwise_noise({
        'stem'  : 0.224,
        'layer1': 0.158,
        'layer2': 0.100,
        'layer3': 0.032,
    })
    test(net)
    print("\nOK")