import torch
import math

QPSK_SCALE = 1.0 / math.sqrt(2.0)

class QPSKInput(torch.nn.Module):
    def forward(self, x):
        snapped = torch.sign(x)
        snapped[snapped == 0] = 1.0
        return snapped * QPSK_SCALE

quantizer = QPSKInput()

# ── Test 1: basic snapping ──────────────────────────────────────
x = torch.tensor([0.23, -0.88, 0.04, -0.71, 0.0])
out = quantizer(x)
print("=== Test 1: basic snapping ===")
print("Input: ", x)
print("Output:", out)
print()

# ── Test 2: RGB channels stay separate ─────────────────────────
print("=== Test 2: RGB channels ===")
x_rgb = torch.tensor([
    [[+0.43, -0.91], [-0.07, +0.55]],   # R channel
    [[-0.12, +0.30], [+0.88, -0.44]],   # G channel
    [[-0.88, +0.02], [+0.11, -0.66]],   # B channel
])
print("Input shape: ", x_rgb.shape)
out_rgb = quantizer(x_rgb)
print("Output shape:", out_rgb.shape)
print("R channel output:\n", out_rgb[0])
print("G channel output:\n", out_rgb[1])
print("B channel output:\n", out_rgb[2])
print()

# ── Test 3: full CIFAR-10 sized batch ──────────────────────────
print("=== Test 3: full batch ===")
fake_batch = torch.randn(64, 3, 32, 32)
out_batch = quantizer(fake_batch)
unique_vals = out_batch.unique()
print("Input shape: ", fake_batch.shape)
print("Output shape:", out_batch.shape)
print("Unique values in output:", unique_vals)
print("Only 2 unique values?", len(unique_vals) == 2)