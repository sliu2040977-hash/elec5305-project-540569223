from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "full_experiment"
)

INPUT_CSV = RESULTS_DIR / "oracle_phase_benefit.csv"

FIGURE_DIR = RESULTS_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

noise_order = ["Cafe", "Traffic", "White"]
snr_order = [0, 5, 10]


# ============================================================
# Figure 1
# Oracle-phase PESQ benefit vs input SNR
# ============================================================

plt.figure(figsize=(8, 5))

for noise_type in noise_order:

    subset = df[
        df["Noise_Type"] == noise_type
    ]

    grouped = (
        subset.groupby("Target_SNR_dB")
        ["SS_Delta_PESQ"]
        .mean()
        .reindex(snr_order)
    )

    plt.plot(
        snr_order,
        grouped.values,
        marker="o",
        linewidth=2,
        label=noise_type,
    )

plt.xlabel("Input SNR (dB)")
plt.ylabel("Oracle-Phase PESQ Benefit (ΔPESQ)")
plt.title(
    "Oracle-Phase PESQ Benefit vs Input SNR\n"
    "Spectral Subtraction"
)
plt.xticks(snr_order)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    FIGURE_DIR
    / "oracle_phase_pesq_benefit_vs_snr_full.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 2
# Phase disagreement vs PESQ oracle benefit
# ============================================================

plt.figure(figsize=(8, 5))

for noise_type in noise_order:

    subset = df[
        df["Noise_Type"] == noise_type
    ]

    plt.scatter(
        subset["Phase_Disagreement_rad"],
        subset["SS_Delta_PESQ"],
        alpha=0.55,
        label=noise_type,
    )

# Linear trend across all observations
x = df["Phase_Disagreement_rad"].to_numpy()
y = df["SS_Delta_PESQ"].to_numpy()

valid = np.isfinite(x) & np.isfinite(y)

if np.sum(valid) >= 2:

    slope, intercept = np.polyfit(
        x[valid],
        y[valid],
        1,
    )

    x_line = np.linspace(
        np.min(x[valid]),
        np.max(x[valid]),
        200,
    )

    y_line = slope * x_line + intercept

    plt.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=2,
        label="Overall linear trend",
    )

    correlation = np.corrcoef(
        x[valid],
        y[valid],
    )[0, 1]

else:
    correlation = np.nan

plt.xlabel("Phase Disagreement (rad)")
plt.ylabel("Oracle-Phase PESQ Benefit (ΔPESQ)")
plt.title(
    "Phase Disagreement vs Oracle-Phase PESQ Benefit\n"
    f"Spectral Subtraction, Pearson r = {correlation:.3f}"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    FIGURE_DIR
    / "phase_disagreement_vs_pesq_benefit_full.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 3
# Phase disagreement vs STOI oracle benefit
# ============================================================

plt.figure(figsize=(8, 5))

for noise_type in noise_order:

    subset = df[
        df["Noise_Type"] == noise_type
    ]

    plt.scatter(
        subset["Phase_Disagreement_rad"],
        subset["SS_Delta_STOI"],
        alpha=0.55,
        label=noise_type,
    )

x = df["Phase_Disagreement_rad"].to_numpy()
y = df["SS_Delta_STOI"].to_numpy()

valid = np.isfinite(x) & np.isfinite(y)

if np.sum(valid) >= 2:

    slope, intercept = np.polyfit(
        x[valid],
        y[valid],
        1,
    )

    x_line = np.linspace(
        np.min(x[valid]),
        np.max(x[valid]),
        200,
    )

    y_line = slope * x_line + intercept

    plt.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=2,
        label="Overall linear trend",
    )

    stoi_correlation = np.corrcoef(
        x[valid],
        y[valid],
    )[0, 1]

else:
    stoi_correlation = np.nan

plt.xlabel("Phase Disagreement (rad)")
plt.ylabel("Oracle-Phase STOI Benefit (ΔSTOI)")
plt.title(
    "Phase Disagreement vs Oracle-Phase STOI Benefit\n"
    f"Spectral Subtraction, Pearson r = {stoi_correlation:.3f}"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    FIGURE_DIR
    / "phase_disagreement_vs_stoi_benefit_full.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Print numerical correlations
# ============================================================

print()
print("=" * 65)
print("FULL-EXPERIMENT PHASE ANALYSIS")
print("=" * 65)

print(
    f"Phase disagreement vs SS ΔPESQ: "
    f"r = {correlation:.4f}"
)

print(
    f"Phase disagreement vs SS ΔSTOI: "
    f"r = {stoi_correlation:.4f}"
)

print()
print("Figures created:")

print(
    FIGURE_DIR
    / "oracle_phase_pesq_benefit_vs_snr_full.png"
)

print(
    FIGURE_DIR
    / "phase_disagreement_vs_pesq_benefit_full.png"
)

print(
    FIGURE_DIR
    / "phase_disagreement_vs_stoi_benefit_full.png"
)