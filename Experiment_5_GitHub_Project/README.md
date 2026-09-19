# Experiment 5 — DPCM and Delta Modulation

Digital Communication Laboratory implementation for:

- First-order DPCM prediction
- Prediction error and uniform quantization
- DPCM reconstruction and MSE
- Delta modulation for small, moderate, and large step sizes
- Granular noise and slope-overload demonstration
- MSE versus step size
- Mandatory validation: every delta-modulator output change is exactly `+Δ` or `−Δ`

## Run

```bash
pip install -r requirements.txt
python experiment5_dpcm_delta.py
```

The script creates `results/figures/` and `results/metrics.txt`.

## Core equations

First-order predictor:

`x_hat[n] = x[n-1]`

Prediction error:

`e[n] = x[n] - x_hat[n]`

Quantized error:

`e_q[n] = Q_Δ(e[n])`

DPCM reconstruction:

`x_r[n] = x_hat[n] + e_q[n]`

Delta modulation:

`x_r[n] = x_r[n-1] + Δ` if `x[n] >= x_r[n-1]`

`x_r[n] = x_r[n-1] - Δ` otherwise.

## Validation

For delta modulation, check:

`x_r[n] - x_r[n-1] ∈ {+Δ, -Δ}`

This is the required diagnostic for the lab.
