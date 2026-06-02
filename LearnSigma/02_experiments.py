"""
03_experiments.py
=================
Research experiments for sigma (spike width) prediction from y(t).

All parameters fixed except sigma:
    mu    = 128 mm  (center of rod)
    amp   = 120 m/s (fixed amplitude above water)
    sigma = 5-30 mm (the only unknown — what we predict)

Experiment A — dataset size: MAE vs number of training samples
Experiment B — noise: how does SNR affect sigma prediction?
Experiment C — reconstructions: visualise predicted vs true spikes

Run:
    python 03_experiments.py

Requires:
    data/dataset.npz
    models/param_mlp.pt  (from 02_train.py)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os, copy

os.makedirs("figures", exist_ok=True)
os.makedirs("models",  exist_ok=True)

SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

SIGNAL_LEN = 512
N_POINTS   = 256
MU_FIXED   = 128
AMP_FIXED  = 120.0
SIGMA_MIN  = 5.0
SIGMA_MAX  = 30.0
C_WATER    = 1480.0

def denorm_sigma(v): return v * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN

def make_profile(sigma):
    x = np.arange(N_POINTS, dtype=np.float32)
    return C_WATER + AMP_FIXED * np.exp(-0.5 * ((x - MU_FIXED) / sigma)**2)

class ParamMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SIGNAL_LEN, 256), nn.Tanh(),
            nn.Linear(256,        256), nn.Tanh(),
            nn.Linear(256,        128), nn.Tanh(),
            nn.Linear(128,          1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)

def add_noise(signals, snr_db):
    power = np.mean(signals**2, axis=1, keepdims=True)
    noise_power = power / (10**(snr_db/10))
    return (signals + np.sqrt(noise_power) * np.random.randn(*signals.shape)).astype(np.float32)

def train_model(X_tr, Y_tr, n_epochs=200):
    ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
    n_v = max(1, int(len(ds)*0.15))
    tr, vl = random_split(ds, [len(ds)-n_v, n_v],
                          generator=torch.Generator().manual_seed(SEED))
    tr_l = DataLoader(tr, batch_size=64, shuffle=True)
    vl_l = DataLoader(vl, batch_size=64)
    m    = ParamMLP()
    opt  = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=12, factor=0.5)
    fn   = nn.MSELoss()
    best_val, best_w = float("inf"), None
    for _ in range(n_epochs):
        m.train()
        for xb, yb in tr_l:
            loss = fn(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        vl_loss = np.mean([fn(m(xb), yb).item() for xb, yb in vl_l])
        sch.step(vl_loss)
        if vl_loss < best_val:
            best_val = vl_loss
            best_w   = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_w); m.eval()
    return m

def evaluate(model, X_te, Y_te):
    with torch.no_grad():
        pred = model(torch.tensor(X_te)).numpy()
    return np.mean(np.abs(denorm_sigma(Y_te[:,0]) - denorm_sigma(pred[:,0])))

# ── load ──────────────────────────────────────────────────────────
print("Loading dataset...")
data      = np.load("data/dataset.npz")
y_signals = data["y_signals"].astype(np.float32)
params    = data["params"].astype(np.float32)
sigmas    = data["sigmas"]

N      = len(y_signals)
n_test = int(N * 0.10)
X_test = y_signals[-n_test:]; Y_test = params[-n_test:]
X_pool = y_signals[:-n_test]; Y_pool = params[:-n_test]
sigma_test = sigmas[-n_test:]
print(f"Pool: {len(X_pool)}  Test: {n_test}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT A: dataset size
# ══════════════════════════════════════════════════════════════════
print("\nExperiment A: dataset size...")
sizes  = [100, 200, 500, 1000, 2000, len(X_pool)]
maes_A = []

for n in sizes:
    print(f"  n={n}...", end=" ", flush=True)
    m   = train_model(X_pool[:n], Y_pool[:n])
    mae = evaluate(m, X_test, Y_test)
    maes_A.append(mae)
    print(f"MAE σ={mae:.2f}mm")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sizes, maes_A, "o-", color="#185FA5", linewidth=2, markersize=8)
ax.set_xlabel("Training samples")
ax.set_ylabel("MAE on σ (mm)")
ax.set_title("Experiment A: more training data → better width estimation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/03_experiment_A.png", dpi=150)
plt.show()
print("Saved: figures/03_experiment_A.png")

print("\nTraining clean model...")
model_clean = train_model(X_pool, Y_pool)

# ══════════════════════════════════════════════════════════════════
# FIGURE C: reconstructed spikes
# ══════════════════════════════════════════════════════════════════
print("\nGenerating reconstructions...")
model_clean.eval()
n_show  = 6
indices = np.random.choice(n_test, n_show, replace=False)
x_mm    = np.arange(N_POINTS)

with torch.no_grad():
    pred_params = model_clean(torch.tensor(X_test[indices])).numpy()

fig, axes = plt.subplots(n_show, 2, figsize=(12, 3*n_show))
fig.suptitle(f"Reconstructed spikes  (μ={MU_FIXED}mm, A={AMP_FIXED:.0f}m/s fixed)",
             fontsize=13)

for row, idx in enumerate(indices):
    s_true = sigma_test[idx]
    s_pred = denorm_sigma(pred_params[row, 0])

    c_true = make_profile(s_true)
    c_pred = make_profile(s_pred)

    axes[row,0].plot(x_mm, c_true, color="#185FA5", linewidth=2,
                     label=f"True  σ={s_true:.1f}mm")
    axes[row,0].plot(x_mm, c_pred, color="#993C1D", linewidth=2,
                     linestyle="--", label=f"Pred  σ={s_pred:.1f}mm")
    axes[row,0].axhline(C_WATER, color="gray", linestyle=":", linewidth=1)
    axes[row,0].set_ylim(1450, 1660)
    axes[row,0].set_ylabel("Speed (m/s)"); axes[row,0].set_xlabel("Position (mm)")
    axes[row,0].set_title(f"Sample {row+1}  Δσ={s_pred-s_true:+.1f}mm")
    axes[row,0].legend(fontsize=8); axes[row,0].grid(alpha=0.3)

    axes[row,1].plot(X_test[idx], color="#1D9E75", linewidth=1)
    axes[row,1].set_ylabel("Pressure (norm.)"); axes[row,1].set_xlabel("Time step")
    axes[row,1].set_title(f"Sample {row+1}: input signal y(t)")
    axes[row,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/03_reconstructions.png", dpi=150)
plt.show()
print("Saved: figures/03_reconstructions.png")