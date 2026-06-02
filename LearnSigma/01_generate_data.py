"""
01_generate_data.py
===================
Generate dataset for Gaussian spike parameter estimation.

Each sample:
  - Random Gaussian spike: mu (position), sigma (width), amp (height)
  - Simulate wave through it with finite differences
  - Record signal y(t) at far-end sensor

The TARGET is now just 2 numbers: (mu, sigma)
not the full 256-point c(x) profile.

This makes the inverse problem tractable:
  INPUT : y(t)  — 512 time-series values
  OUTPUT: (mu, sigma) — where is the spike, how wide is it

Run:
    python 01_generate_data.py

Output:
    data/dataset.npz   keys: y_signals, params
                              y_signals : (N, 512)
                              params    : (N, 2)  — [mu_norm, sigma_norm]
    figures/01_examples.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("data",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

SEED         = 42
N_SAMPLES    = 3000
N_POINTS     = 256
SIGNAL_LEN   = 512
DX           = 1e-3
T_TOTAL      = 4e-4
SOURCE_FREQ  = 300e3
C_WATER      = 1480.0
C_MAX_TISSUE = 1630.0

# all parameters fixed except sigma
MU_FIXED  = N_POINTS // 2     # always at position 128mm
AMP_FIXED = 120.0              # m/s above water — fixed amplitude
SIGMA_MIN = 5.0                # mm
SIGMA_MAX = 30.0               # mm

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

def make_profile(mu, sigma, amp):
    x = np.arange(N_POINTS, dtype=np.float64)
    c = C_WATER + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return c.astype(np.float32)

def simulate_wave(c):
    dt = 0.9 * DX / c.max()
    nt = int(T_TOTAL / dt)
    r2 = (c * dt / DX) ** 2
    u_prev = np.zeros(N_POINTS)
    u_curr = np.zeros(N_POINTS)
    u_next = np.zeros(N_POINTS)
    signal = np.zeros(nt)
    for it in range(nt):
        u_curr[0] += np.sin(2 * np.pi * SOURCE_FREQ * it * dt)
        u_next[1:-1] = (2*u_curr[1:-1] - u_prev[1:-1]
                        + r2[1:-1]*(u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))
        u_next[-1] = u_curr[-2]
        signal[it] = u_next[-1]
        u_prev[:] = u_curr; u_curr[:] = u_next
    x_old = np.linspace(0, 1, nt)
    x_new = np.linspace(0, 1, SIGNAL_LEN)
    sig   = np.interp(x_new, x_old, signal).astype(np.float32)
    # no per-sample normalisation here — preserves timing/phase info
    # that encodes spike position. Global normalisation applied after.
    return sig

# normalise sigma to [0,1] for training
def norm_sigma(s):   return (s - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN)
def denorm_sigma(s): return s * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN

print(f"Generating {N_SAMPLES} samples...")
y_signals = np.zeros((N_SAMPLES, SIGNAL_LEN), dtype=np.float32)
params    = np.zeros((N_SAMPLES, 1),          dtype=np.float32)  # [sigma_norm]
sigmas    = np.zeros(N_SAMPLES)
amps      = np.zeros(N_SAMPLES)

print(f"Generating {N_SAMPLES} samples (mu fixed at {MU_FIXED}mm)...")
for i in range(N_SAMPLES):
    if i % 500 == 0:
        print(f"  {i}/{N_SAMPLES}")
    sigma = rng.uniform(SIGMA_MIN, SIGMA_MAX)
    amp   = rng.uniform(AMP_FIXED,   AMP_FIXED)
    c     = make_profile(MU_FIXED, sigma, amp)
    sig   = simulate_wave(c)
    y_signals[i]  = sig
    params[i, 0]  = norm_sigma(sigma)
    sigmas[i]     = sigma
    amps[i]       = amp

print(f"Done. Applying global normalisation...")
global_mean = y_signals.mean()
global_std  = y_signals.std()
y_signals   = ((y_signals - global_mean) / global_std).astype(np.float32)
print(f"  signal mean: {y_signals.mean():.4f}  std: {y_signals.std():.4f}")

np.savez("data/dataset.npz",
         y_signals=y_signals, params=params,
         sigmas=sigmas, amps=amps,
         global_mean=np.array([global_mean]),
         global_std=np.array([global_std]))
print("Saved: data/dataset.npz")
print(f"  y_signals: {y_signals.shape}")
print(f"  params:    {params.shape}  (sigma_norm only)")

# visualise 4 examples
x_mm = np.arange(N_POINTS)
fig, axes = plt.subplots(4, 2, figsize=(11, 10))
fig.suptitle(f"Dataset: Gaussian spike (μ fixed={MU_FIXED}mm) → sensor signal",
             fontsize=13)
for row in range(4):
    sigma = sigmas[row];   amp = amps[row]
    c_plot = make_profile(MU_FIXED, sigma, amp)
    axes[row, 0].plot(x_mm, c_plot, color="#185FA5", linewidth=2)
    axes[row, 0].axhline(C_WATER, color="gray", linestyle="--", linewidth=1)
    axes[row, 0].set_ylim(1450, 1660)
    axes[row, 0].set_ylabel("Speed (m/s)"); axes[row, 0].set_xlabel("Position (mm)")
    axes[row, 0].set_title(f"Sample {row+1}: μ={MU_FIXED}mm  σ={sigma:.1f}mm  A={amp:.0f}m/s")
    axes[row, 0].grid(alpha=0.3)
    axes[row, 1].plot(y_signals[row], color="#993C1D", linewidth=1)
    axes[row, 1].set_ylabel("Pressure (norm.)"); axes[row, 1].set_xlabel("Time step")
    axes[row, 1].set_title(f"Sample {row+1}: y(t)")
    axes[row, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/01_examples.png", dpi=150)
plt.show()
print("Saved: figures/01_examples.png")