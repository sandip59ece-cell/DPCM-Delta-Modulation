"""
Experiment 5 — DPCM and Delta Modulation
Digital Communication Laboratory

Run:
    python experiment5_dpcm_delta.py

Outputs:
    results/figures/*.png
    results/metrics.txt
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FS = 1000
DPCM_STEP = 0.08
DM_STEPS = [0.03, 0.10, 0.25]

def quantize(v, step):
    return step * np.round(v / step)

def dpcm_first_order(x, step):
    """First-order predictor: x_hat[n] = x[n-1]."""
    x_hat = np.zeros_like(x)
    x_hat[1:] = x[:-1]
    error = x - x_hat
    q_error = quantize(error, step)
    reconstruction = x_hat + q_error
    mse = np.mean((x - reconstruction) ** 2)
    return x_hat, error, q_error, reconstruction, mse

def delta_modulate(x, delta):
    """
    One-bit delta modulation.
    If input >= previous reconstruction, transmit 1 and step up by +delta.
    Otherwise transmit 0 and step down by -delta.
    """
    bits = np.zeros(len(x), dtype=int)
    reconstruction = np.zeros_like(x)

    for n in range(1, len(x)):
        if x[n] >= reconstruction[n - 1]:
            bits[n] = 1
            reconstruction[n] = reconstruction[n - 1] + delta
        else:
            bits[n] = 0
            reconstruction[n] = reconstruction[n - 1] - delta

    mse = np.mean((x - reconstruction) ** 2)
    return bits, reconstruction, mse

def main():
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)

    fs = FS
    t = np.arange(0, 1, 1 / fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.25 * np.sin(2 * np.pi * 10 * t)

    # ----- DPCM -----
    x_hat, error, q_error, x_dpcm, dpcm_mse = dpcm_first_order(x, DPCM_STEP)

    plt.figure(figsize=(9, 4.8))
    idx = np.arange(300)
    plt.plot(t[idx], x[idx], label="Original")
    plt.plot(t[idx], x_dpcm[idx], label="DPCM reconstructed")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("DPCM: original vs reconstructed")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "dpcm_reconstruction.png", dpi=180)
    plt.close()

    # ----- Delta modulation -----
    metrics = [f"DPCM MSE = {dpcm_mse:.8f}"]

    for delta in DM_STEPS:
        bits, x_dm, mse = delta_modulate(x, delta)
        metrics.append(f"DM delta={delta:.2f}, MSE={mse:.8f}")

        if abs(delta - 0.10) < 1e-9:
            plt.figure(figsize=(9, 4.8))
            idx = np.arange(250)
            plt.plot(t[idx], x[idx], label="Input")
            plt.step(t[idx], x_dm[idx], where="post", label="DM reconstruction")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Delta modulation staircase (delta=0.10)")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "delta_staircase.png", dpi=180)
            plt.close()

    # MSE vs step
    mse_values = []
    for delta in DM_STEPS:
        _, _, mse = delta_modulate(x, delta)
        mse_values.append(mse)

    plt.figure(figsize=(8, 4.6))
    plt.plot(DM_STEPS, mse_values, marker="o")
    plt.xlabel("Step size delta")
    plt.ylabel("MSE")
    plt.title("Delta modulation: MSE versus step size")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "mse_vs_step_size.png", dpi=180)
    plt.close()

    (Path("results") / "metrics.txt").write_text("\n".join(metrics), encoding="utf-8")
    print("\n".join(metrics))
    print("Figures saved in results/figures/")

if __name__ == "__main__":
    main()
