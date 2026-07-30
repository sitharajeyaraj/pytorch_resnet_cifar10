"""
plot_forward_curves.py
======================
Plots the 8-level hard quantizer (forward pass staircase)
vs the smooth stacked tanh approximation on the same axis.

Run:
    python plot_forward_curves.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
BETA   = 20.0
LEVELS = np.linspace(-1.0, 1.0, 8)
SAVE   = 'forward_curves.png'
# ============================================================

transitions = (LEVELS[:-1] + LEVELS[1:]) / 2.0
x = np.linspace(-1.6, 1.6, 1000)

# ---- Hard quantizer (argmin snap) ----
def hard_quantize(x):
    idx = np.argmin(np.abs(x[:, None] - LEVELS[None, :]), axis=1)
    return LEVELS[idx]

# ---- Smooth stacked tanh ----
def smooth_tanh(x, beta):
    out = np.zeros_like(x)
    for t in transitions:
        out += np.tanh(beta * (x - t))
    out = out / len(transitions)
    return out

hard = hard_quantize(x)
soft = smooth_tanh(x, BETA)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, hard, label='Hard quantizer (forward)',
        color='steelblue', linewidth=2.5, drawstyle='steps-mid')
ax.plot(x, soft, label=f'Stacked tanh (β={BETA})',
        color='tomato', linewidth=2)
for t in transitions:
    ax.axvline(x=t, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
for l in LEVELS:
    ax.axhline(y=l, color='gray', linewidth=0.3, linestyle=':', alpha=0.3)
ax.scatter(LEVELS, LEVELS, color='steelblue', s=50, zorder=5,
           label='Quantization levels')

ax.set_xlabel('Input value x')
ax.set_ylabel('Output')
ax.set_title('Hard quantizer vs stacked tanh (forward pass)')
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.3, 1.3)

plt.tight_layout()
plt.savefig(SAVE, dpi=150)
plt.close()
print(f'Plot saved to {SAVE}')