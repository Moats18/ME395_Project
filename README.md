# learnSigma
### "Learning to See Inside: Reconstructing Tissue from Sound Waves"

A research project in medical imaging, wave physics, and machine learning.

---

## What this project does

Ultrasound machines send sound pulses into the body and try to figure
out what is inside from the echoes. This is called an **inverse problem**.

This project solves it by learning to predict the shape of a Gaussian
tissue spike from the sensor signal it produces:

- **Generate data** — simulate sound travelling through fake tissue using real wave physics
- **Train a neural network** — teach an MLP to map y(t) → spike parameters
- **Experiment** — test how accuracy changes with dataset size

There are two versions, each building on the previous:

| Version | What varies | What the network predicts |
|---------|-------------|--------------------------|
| Root (`*.py`) | σ only (A fixed) | 1 parameter: spike width |
| `LearnSigma_A/` | A **and** σ | 2 parameters: amplitude + width |

---

## Setup (do this once)

Install Python (https://www.python.org) then run:

```bash
pip install numpy matplotlib torch
```

---

## Run the simple version (root folder)

### Step 1 — Generate the dataset
```bash
python 01_generate_data.py
```
Creates `data/dataset.npz` with 3000 samples (sigma varies, A and mu fixed).

### Step 2 — Run experiments
```bash
python 02_experiments.py
```
Trains a 1-output MLP and runs the dataset-size experiment.

---

## Run the more complicated version

```bash
cd LearnSigma_A
python 01_generate_data.py   # generates 5000 samples, both A and sigma vary
python 02_experiments.py     # trains 2-output MLP, same experiments
```

---

## Project structure

```
learnSigma/
├── 01_generate_data.py          # simple version: sigma varies only
├── 02_experiments.py            # simple version: experiments
├── data/
│   └── dataset.npz
├── models/
└── LearnSigma_A/
    ├── 01_generate_data.py      # both A and sigma vary
    ├── 02_experiments.py        # joint estimation experiments
    ├── data/
    └── models/
        └── param_mlp_2d.pt
```

---

## The physics (in plain English)

The rod of tissue is divided into 256 small segments.
Each segment has a speed of sound c(x):
- Water / fat:    ~1480 m/s
- Soft tissue:    ~1540–1650 m/s
- Dense tissue:   ~1650–1750 m/s

A sound pulse enters from the left end. It travels through the rod
and gets recorded by a sensor on the right end as a pressure signal y(t).

The wave equation that governs this is:

    d²u/dt² = c(x)² · d²u/dx²

We solve it numerically using **finite differences** — replacing the
derivatives with differences between neighbouring grid points.

The tissue spike is a Gaussian bump in speed of sound:

    c(x) = c_water + A · exp(−½ · ((x − μ) / σ)²)

where A is the amplitude (how much faster), σ is the width, and μ is the position.

---

## The machine learning (in plain English)

**Simple version (1 unknown):**
Sigma varies (5–30 mm); A and mu are fixed. The MLP reads the 512-point signal
and predicts one number: σ.

**Complicated version (2 unknowns):**
Both A (40–200 m/s) and sigma (5–30 mm) vary; mu stays fixed.
The MLP predicts two numbers simultaneously: (A, σ).

**Why is it harder with two unknowns?**
A affects the *amplitude* of the signal; σ affects the *timing* of the disturbance.
The network must learn to disentangle these two effects from the same waveform.

### MLP architecture (complicated version)

```
Input  →  512  (signal y(t))
Layer 1:  512 → 512, Tanh
Layer 2:  512 → 256, Tanh
Layer 3:  256 → 128, Tanh
Output:   128 →   2, Sigmoid   ← (A_norm, σ_norm)

~427,000 trainable parameters
Optimizer: Adam (lr=1e-3), ReduceLROnPlateau, 250 epochs
```

---

## Things to explore / extend

- Does the network struggle more with A or σ as dataset size shrinks?
- Does adding noise to y(t) hurt A and σ prediction equally?
- Can a 1D CNN outperform the MLP by exploiting the temporal structure of y(t)?

Each of these is a genuine research question with a real answer.
