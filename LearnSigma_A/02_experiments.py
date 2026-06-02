"""
02_experiments.py  (MoreComplicatedVersion)
===========================================
Research experiments for JOINT estimation of spike amplitude A and width sigma.

Both A and sigma vary; mu is fixed at 128 mm.

  INPUT : y(t)             — 512 time-series values
  OUTPUT: (A_norm, sigma_norm) — two unknowns predicted simultaneously

Experiment A — dataset size: MAE vs number of training samples (for A and sigma)
Experiment B — reconstructions: visualise predicted vs true spikes

Run:
    python 02_experiments.py

Requires:
    data/dataset.npz
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
AMP_MIN    = 40.0
AMP_MAX    = 200.0
SIGMA_MIN  = 5.0
SIGMA_MAX  = 30.0
C_WATER    = 1480.0

def denorm_amp(a):   return a * (AMP_MAX   - AMP_MIN)   + AMP_MIN
def denorm_sigma(s): return s * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN

def make_profile(amp, sigma):
    x = np.arange(N_POINTS, dtype=np.float32)
    return C_WATER + amp * np.exp(-0.5 * ((x - MU_FIXED) / sigma)**2)

# Two-output MLP: predicts [amp_norm, sigma_norm] jointly
class ParamMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SIGNAL_LEN, 512), nn.Tanh(),
            nn.Linear(512,        256), nn.Tanh(),
            nn.Linear(256,        128), nn.Tanh(),
            nn.Linear(128,          2), nn.Sigmoid(),  # 2 outputs
        )
    def forward(self, x): return self.net(x)

def train_model(X_tr, Y_tr, n_epochs=250):
    ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
    n_v = max(1, int(len(ds)*0.15))
    tr, vl = random_split(ds, [len(ds)-n_v, n_v],
                          generator=torch.Generator().manual_seed(SEED))
    tr_l = DataLoader(tr, batch_size=64, shuffle=True)
    vl_l = DataLoader(vl, batch_size=64)
    m    = ParamMLP()
    opt  = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
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
    mae_amp   = np.mean(np.abs(denorm_amp(Y_te[:,0])   - denorm_amp(pred[:,0])))
    mae_sigma = np.mean(np.abs(denorm_sigma(Y_te[:,1]) - denorm_sigma(pred[:,1])))
    return mae_amp, mae_sigma

# ── load ──────────────────────────────────────────────────────────
print("Loading dataset...")
data      = np.load("data/dataset.npz")
y_signals = data["y_signals"].astype(np.float32)
params    = data["params"].astype(np.float32)   # (N, 2): [amp_norm, sigma_norm]
sigmas    = data["sigmas"]
amps      = data["amps"]

N      = len(y_signals)
n_test = int(N * 0.10)
X_test = y_signals[-n_test:]; Y_test = params[-n_test:]
X_pool = y_signals[:-n_test]; Y_pool = params[:-n_test]
amp_test   = amps[-n_test:]
sigma_test = sigmas[-n_test:]
print(f"Pool: {len(X_pool)}  Test: {n_test}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT A: dataset size
# ══════════════════════════════════════════════════════════════════
print("\nExperiment A: dataset size...")
sizes       = [200, 500, 1000, 2000, len(X_pool)]
maes_A_amp  = []
maes_A_sig  = []

for n in sizes:
    print(f"  n={n}...", end=" ", flush=True)
    m = train_model(X_pool[:n], Y_pool[:n])
    mae_amp, mae_sig = evaluate(m, X_test, Y_test)
    maes_A_amp.append(mae_amp)
    maes_A_sig.append(mae_sig)
    print(f"MAE A={mae_amp:.2f}m/s  σ={mae_sig:.2f}mm")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Experiment A: dataset size vs MAE  (joint A & σ prediction)", fontsize=13)

axes[0].plot(sizes, maes_A_amp, "o-", color="#185FA5", linewidth=2, markersize=8)
axes[0].set_xlabel("Training samples"); axes[0].set_ylabel("MAE on A (m/s)")
axes[0].set_title("Amplitude A"); axes[0].grid(alpha=0.3)

axes[1].plot(sizes, maes_A_sig, "o-", color="#993C1D", linewidth=2, markersize=8)
axes[1].set_xlabel("Training samples"); axes[1].set_ylabel("MAE on σ (mm)")
axes[1].set_title("Width σ"); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/02_experiment_A.png", dpi=150)
plt.show()
print("Saved: figures/02_experiment_A.png")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT B: reconstructed spikes
# ══════════════════════════════════════════════════════════════════
print("\nTraining model on full pool...")
model_clean = train_model(X_pool, Y_pool)

print("\nGenerating reconstructions...")
model_clean.eval()
n_show  = 6
indices = np.random.choice(n_test, n_show, replace=False)
x_mm    = np.arange(N_POINTS)

with torch.no_grad():
    pred_params = model_clean(torch.tensor(X_test[indices])).numpy()

fig, axes = plt.subplots(n_show, 2, figsize=(12, 3*n_show))
fig.suptitle(f"Reconstructed spikes  (μ={MU_FIXED}mm fixed, A and σ both predicted)",
             fontsize=13)

for row, idx in enumerate(indices):
    a_true = amp_test[idx];   s_true = sigma_test[idx]
    a_pred = denorm_amp(pred_params[row, 0])
    s_pred = denorm_sigma(pred_params[row, 1])

    c_true = make_profile(a_true, s_true)
    c_pred = make_profile(a_pred, s_pred)

    axes[row,0].plot(x_mm, c_true, color="#185FA5", linewidth=2,
                     label=f"True   A={a_true:.0f}m/s  σ={s_true:.1f}mm")
    axes[row,0].plot(x_mm, c_pred, color="#993C1D", linewidth=2,
                     linestyle="--",
                     label=f"Pred   A={a_pred:.0f}m/s  σ={s_pred:.1f}mm")
    axes[row,0].axhline(C_WATER, color="gray", linestyle=":", linewidth=1)
    axes[row,0].set_ylim(1400, 1750)
    axes[row,0].set_ylabel("Speed (m/s)"); axes[row,0].set_xlabel("Position (mm)")
    axes[row,0].set_title(f"Sample {row+1}  ΔA={a_pred-a_true:+.1f}m/s  Δσ={s_pred-s_true:+.1f}mm")
    axes[row,0].legend(fontsize=8); axes[row,0].grid(alpha=0.3)

    axes[row,1].plot(X_test[idx], color="#1D9E75", linewidth=1)
    axes[row,1].set_ylabel("Pressure (norm.)"); axes[row,1].set_xlabel("Time step")
    axes[row,1].set_title(f"Sample {row+1}: input signal y(t)")
    axes[row,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/02_reconstructions.png", dpi=150)
plt.show()
print("Saved: figures/02_reconstructions.png")

# ── save final model ──────────────────────────────────────────────
torch.save(model_clean.state_dict(), "models/param_mlp_2d.pt")
print("Saved: models/param_mlp_2d.pt")
