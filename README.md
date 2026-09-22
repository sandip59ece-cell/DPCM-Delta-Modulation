# Experiment 5 — DPCM and Delta Modulation
# Software-Based Digital Communication Laboratory
# Google Colab / Python

import numpy as np
import matplotlib.pyplot as plt


def uniform_quantize_fixed(x, bits, xmin, xmax):
    """Uniform mid-rise quantizer over a fixed interval."""
    levels = 2 ** bits
    step = (xmax - xmin) / levels
    if step <= 0:
        return np.full_like(x, xmin), np.zeros_like(x, dtype=int)
    idx = np.clip(np.floor((x - xmin) / step).astype(int), 0, levels - 1)
    xq = xmin + (idx + 0.5) * step
    return xq, idx


def pcm_quantize(x, bits=4):
    xmin, xmax = -1.0, 1.0
    xq, idx = uniform_quantize_fixed(x, bits, xmin, xmax)
    return xq, idx


def dpcm_encode_decode(x, bits=4, error_range=0.5):
    """First-order DPCM with a closed-loop previous reconstructed sample."""
    n = len(x)
    pred = np.zeros(n)
    err = np.zeros(n)
    err_q = np.zeros(n)
    recon = np.zeros(n)
    indices = np.zeros(n, dtype=int)

    # Initial sample is transmitted separately in this simple simulation.
    recon[0] = x[0]

    for k in range(1, n):
        pred[k] = recon[k - 1]
        err[k] = x[k] - pred[k]
        err_q[k:k+1], indices[k:k+1] = uniform_quantize_fixed(
            np.array([err[k]]), bits, -error_range, error_range
        )
        recon[k] = pred[k] + err_q[k]

    return pred, err, err_q, recon, indices


def delta_modulate(x, delta):
    """1-bit delta modulation using a feedback staircase."""
    staircase = np.zeros(len(x))
    bits = np.zeros(len(x), dtype=int)
    for k in range(1, len(x)):
        if x[k] >= staircase[k - 1]:
            bits[k] = 1
            staircase[k] = staircase[k - 1] + delta
        else:
            bits[k] = 0
            staircase[k] = staircase[k - 1] - delta
    return bits, staircase


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def run_experiment(out_dir="."):
    fs = 1000
    duration = 1.0
    t = np.arange(0, duration, 1 / fs)
    slow = np.sin(2 * np.pi * 5 * t)
    fast = np.sin(2 * np.pi * 40 * t)

    # DPCM vs PCM on the slow input.
    pcm, _ = pcm_quantize(slow, bits=4)
    pred, err, err_q, dpcm_recon, _ = dpcm_encode_decode(slow, bits=4)
    pcm_mse = mse(slow, pcm)
    dpcm_mse = mse(slow, dpcm_recon)

    print(f"PCM MSE (4-bit):  {pcm_mse:.6f}")
    print(f"DPCM MSE (4-bit): {dpcm_mse:.6f}")

    # Delta modulation sweep.
    deltas = np.linspace(0.01, 0.20, 20)
    mse_slow = []
    mse_fast = []
    for d in deltas:
        _, s1 = delta_modulate(slow, d)
        _, s2 = delta_modulate(fast, d)
        mse_slow.append(mse(slow, s1))
        mse_fast.append(mse(fast, s2))

    # Representative step sizes.
    representative = [0.02, 0.08, 0.20]
    labels = ["Small Δ = 0.02", "Moderate Δ = 0.08", "Large Δ = 0.20"]

    # Plot 1: original vs prediction.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, slow, label="Original input", linewidth=2)
    ax.plot(t, pred, label="1st-order predicted", linewidth=1.6)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("DPCM: Original and Predicted Samples")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/exp5_original_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: prediction error.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, err, linewidth=1.5, label="Prediction error e[n]")
    ax.plot(t, err_q, linewidth=1.2, label="Quantized error")
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error amplitude")
    ax.set_title("DPCM Prediction Error and Quantized Error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/exp5_prediction_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: delta staircase, slow input.
    _, stair = delta_modulate(slow, 0.05)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, slow, label="Slow input (5 Hz)", linewidth=2)
    ax.step(t, stair, where="post", label="DM staircase, Δ = 0.05", linewidth=1.5)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Delta Modulation Staircase")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/exp5_delta_staircase.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: small/moderate/large step sizes on fast input.
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    for ax, d, lab in zip(axes, representative, labels):
        _, s = delta_modulate(fast, d)
        ax.plot(t, fast, label="40 Hz input", linewidth=1.8)
        ax.step(t, s, where="post", label=lab, linewidth=1.2)
        ax.set_xlim(0, 0.25)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Delta Modulation: Effect of Step Size on a Rapid Input", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{out_dir}/exp5_step_size_cases.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: slow vs fast input at moderate delta.
    d = 0.05
    _, ss = delta_modulate(slow, d)
    _, sf = delta_modulate(fast, d)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
    axes[0].plot(t, slow, label="5 Hz input", linewidth=2)
    axes[0].step(t, ss, where="post", label="DM staircase", linewidth=1.2)
    axes[0].set_title("Slowly Varying Input")
    axes[1].plot(t, fast, label="40 Hz input", linewidth=2)
    axes[1].step(t, sf, where="post", label="DM staircase", linewidth=1.2)
    axes[1].set_title("Rapidly Varying Input — Slope Overload")
    for ax in axes:
        ax.set_xlim(0, 0.25)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/exp5_slow_fast_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 6: MSE vs step size.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(deltas, mse_slow, "o-", label="5 Hz input")
    ax.plot(deltas, mse_fast, "s-", label="40 Hz input")
    ax.axvline(0.05, linestyle="--", linewidth=1)
    ax.set_xlabel("Step size Δ")
    ax.set_ylabel("MSE")
    ax.set_title("MSE versus Delta-Modulation Step Size")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/exp5_mse_vs_delta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Mandatory validation data.
    bits, staircase = delta_modulate(slow, 0.05)
    diffs = np.diff(staircase)[1:]
    validation = np.all(np.isclose(np.abs(diffs), 0.05))
    print("DM mandatory step validation:", validation)
    print("Unique non-zero step magnitudes:", np.unique(np.round(np.abs(diffs), 6)))

    # Slope comparison for analytical explanation.
    A = 1.0
    Ts = 1 / fs
    for f in [5, 40]:
        max_input_slope = 2 * np.pi * f * A
        dm_slope = 0.05 / Ts
        print(f"f={f} Hz: max input slope={max_input_slope:.3f}, DM slope={dm_slope:.3f}, ratio={max_input_slope/dm_slope:.2f}")

    # Text summary for slide content.
    with open(f"{out_dir}/exp5_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"PCM_MSE_4bit={pcm_mse:.8f}\n")
        f.write(f"DPCM_MSE_4bit={dpcm_mse:.8f}\n")
        f.write(f"DM_STEP_VALIDATION={validation}\n")
        f.write(f"SLOPE_LIMIT_D0.05={0.05/Ts:.6f}\n")
        for freq in [5, 40]:
            max_slope = 2*np.pi*freq*A
            f.write(f"MAX_INPUT_SLOPE_{freq}Hz={max_slope:.6f}\n")


if __name__ == "__main__":
    run_experiment(".")
