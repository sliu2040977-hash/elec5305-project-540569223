from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FULL_METRICS = ROOT / "results" / "full_experiment" / "full_metrics.csv"

MATCHED_ALL = (
    ROOT
    / "results"
    / "matched_comparison"
    / "matched_all_methods.csv"
)

MPSENET_METRICS = (
    ROOT
    / "results"
    / "mpsenet"
    / "mpsenet_metrics.csv"
)

OUTPUT_DIR = ROOT / "results" / "final_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    }
)


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Created: {path}")


# ============================================================
# Load data
# ============================================================

full = pd.read_csv(FULL_METRICS)
matched = pd.read_csv(MATCHED_ALL)
mpsenet = pd.read_csv(MPSENET_METRICS)

print("=" * 70)
print("FINAL RESULT FIGURE GENERATION")
print("=" * 70)

print(f"Full classical rows: {len(full)}")
print(f"Matched comparison rows: {len(matched)}")
print(f"MP-SENet rows: {len(mpsenet)}")


# ============================================================
# 1. Oracle phase benefit vs SNR
#    Separate curves for Cafe / Traffic / White
# ============================================================

key_cols = [
    "Clean_File",
    "Speaker",
    "Noise_Type",
    "Target_SNR_dB",
]

ss_noisy = (
    full[full["Condition"] == "SS + noisy phase"]
    [key_cols + ["STOI", "PESQ", "SI_SDR_dB", "Phase_Disagreement_rad"]]
    .rename(
        columns={
            "STOI": "SS_Noisy_STOI",
            "PESQ": "SS_Noisy_PESQ",
            "SI_SDR_dB": "SS_Noisy_SI_SDR",
        }
    )
)

ss_oracle = (
    full[full["Condition"] == "SS + clean oracle phase"]
    [key_cols + ["STOI", "PESQ", "SI_SDR_dB"]]
    .rename(
        columns={
            "STOI": "SS_Oracle_STOI",
            "PESQ": "SS_Oracle_PESQ",
            "SI_SDR_dB": "SS_Oracle_SI_SDR",
        }
    )
)

oracle = ss_noisy.merge(ss_oracle, on=key_cols, how="inner")

oracle["Delta_STOI"] = (
    oracle["SS_Oracle_STOI"] - oracle["SS_Noisy_STOI"]
)

oracle["Delta_PESQ"] = (
    oracle["SS_Oracle_PESQ"] - oracle["SS_Noisy_PESQ"]
)

oracle["Delta_SI_SDR"] = (
    oracle["SS_Oracle_SI_SDR"] - oracle["SS_Noisy_SI_SDR"]
)


noise_order = ["Cafe", "Traffic", "White"]
snr_order = [0, 5, 10]


# PESQ oracle benefit
fig, ax = plt.subplots(figsize=(7.2, 4.8))

for noise in noise_order:
    subset = oracle[oracle["Noise_Type"] == noise]

    summary = (
        subset.groupby("Target_SNR_dB")["Delta_PESQ"]
        .agg(["mean", "std"])
        .reindex(snr_order)
    )

    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        label=noise,
    )

ax.axhline(0, linewidth=1)
ax.set_xlabel("Input SNR (dB)")
ax.set_ylabel("Oracle-phase PESQ benefit (ΔPESQ)")
ax.set_title("Oracle-Phase PESQ Benefit vs Input SNR")
ax.set_xticks(snr_order)
ax.legend(title="Noise type")
ax.grid(alpha=0.25)

save_figure(
    fig,
    "01_oracle_phase_pesq_benefit_vs_snr.png",
)


# STOI oracle benefit
fig, ax = plt.subplots(figsize=(7.2, 4.8))

for noise in noise_order:
    subset = oracle[oracle["Noise_Type"] == noise]

    summary = (
        subset.groupby("Target_SNR_dB")["Delta_STOI"]
        .agg(["mean", "std"])
        .reindex(snr_order)
    )

    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        label=noise,
    )

ax.axhline(0, linewidth=1)
ax.set_xlabel("Input SNR (dB)")
ax.set_ylabel("Oracle-phase STOI benefit (ΔSTOI)")
ax.set_title("Oracle-Phase STOI Benefit vs Input SNR")
ax.set_xticks(snr_order)
ax.legend(title="Noise type")
ax.grid(alpha=0.25)

save_figure(
    fig,
    "02_oracle_phase_stoi_benefit_vs_snr.png",
)


# SI-SDR oracle benefit
fig, ax = plt.subplots(figsize=(7.2, 4.8))

for noise in noise_order:
    subset = oracle[oracle["Noise_Type"] == noise]

    summary = (
        subset.groupby("Target_SNR_dB")["Delta_SI_SDR"]
        .agg(["mean", "std"])
        .reindex(snr_order)
    )

    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        label=noise,
    )

ax.axhline(0, linewidth=1)
ax.set_xlabel("Input SNR (dB)")
ax.set_ylabel("Oracle-phase SI-SDR benefit (dB)")
ax.set_title("Oracle-Phase SI-SDR Benefit vs Input SNR")
ax.set_xticks(snr_order)
ax.legend(title="Noise type")
ax.grid(alpha=0.25)

save_figure(
    fig,
    "03_oracle_phase_sisdr_benefit_vs_snr.png",
)


# ============================================================
# 2. Phase disagreement vs oracle benefit
# ============================================================

x = oracle["Phase_Disagreement_rad"].to_numpy()
y_pesq = oracle["Delta_PESQ"].to_numpy()
y_stoi = oracle["Delta_STOI"].to_numpy()
y_sisdr = oracle["Delta_SI_SDR"].to_numpy()


def correlation(x_values, y_values):
    mask = np.isfinite(x_values) & np.isfinite(y_values)

    if mask.sum() < 2:
        return np.nan

    return np.corrcoef(
        x_values[mask],
        y_values[mask],
    )[0, 1]


r_pesq = correlation(x, y_pesq)
r_stoi = correlation(x, y_stoi)
r_sisdr = correlation(x, y_sisdr)


# PESQ scatter
fig, ax = plt.subplots(figsize=(7.2, 4.8))

for noise in noise_order:
    subset = oracle[oracle["Noise_Type"] == noise]

    ax.scatter(
        subset["Phase_Disagreement_rad"],
        subset["Delta_PESQ"],
        alpha=0.65,
        s=28,
        label=noise,
    )

ax.axhline(0, linewidth=1)
ax.set_xlabel("Phase disagreement (rad)")
ax.set_ylabel("Oracle-phase PESQ benefit (ΔPESQ)")
ax.set_title(
    f"Phase Disagreement vs Oracle PESQ Benefit\n"
    f"Pearson r = {r_pesq:.3f}"
)
ax.legend(title="Noise type")
ax.grid(alpha=0.25)

save_figure(
    fig,
    "04_phase_disagreement_vs_pesq_benefit.png",
)


# STOI scatter
fig, ax = plt.subplots(figsize=(7.2, 4.8))

for noise in noise_order:
    subset = oracle[oracle["Noise_Type"] == noise]

    ax.scatter(
        subset["Phase_Disagreement_rad"],
        subset["Delta_STOI"],
        alpha=0.65,
        s=28,
        label=noise,
    )

ax.axhline(0, linewidth=1)
ax.set_xlabel("Phase disagreement (rad)")
ax.set_ylabel("Oracle-phase STOI benefit (ΔSTOI)")
ax.set_title(
    f"Phase Disagreement vs Oracle STOI Benefit\n"
    f"Pearson r = {r_stoi:.3f}"
)
ax.legend(title="Noise type")
ax.grid(alpha=0.25)

save_figure(
    fig,
    "05_phase_disagreement_vs_stoi_benefit.png",
)


# SI-SDR scatter
fig, ax = plt.subplots(figsize=(7.2, 4.8))

for noise in noise_order:
    subset = oracle[oracle["Noise_Type"] == noise]

    ax.scatter(
        subset["Phase_Disagreement_rad"],
        subset["Delta_SI_SDR"],
        alpha=0.65,
        s=28,
        label=noise,
    )

ax.axhline(0, linewidth=1)
ax.set_xlabel("Phase disagreement (rad)")
ax.set_ylabel("Oracle-phase SI-SDR benefit (dB)")
ax.set_title(
    f"Phase Disagreement vs Oracle SI-SDR Benefit\n"
    f"Pearson r = {r_sisdr:.3f}"
)
ax.legend(title="Noise type")
ax.grid(alpha=0.25)

save_figure(
    fig,
    "06_phase_disagreement_vs_sisdr_benefit.png",
)


# ============================================================
# 3. Matched 167-sample comparison
# ============================================================

method_order = [
    "Noisy",
    "SS + noisy phase",
    "SS + clean oracle phase",
    "Wiener + noisy phase",
    "Wiener + clean oracle phase",
    "Perfect magnitude + noisy phase",
    "MP-SENet",
]

matched_summary = (
    matched.groupby("Condition")
    .agg(
        STOI_Mean=("STOI", "mean"),
        STOI_Std=("STOI", "std"),
        PESQ_Mean=("PESQ", "mean"),
        PESQ_Std=("PESQ", "std"),
        SI_SDR_Mean=("SI_SDR_dB", "mean"),
        SI_SDR_Std=("SI_SDR_dB", "std"),
        N=("STOI", "count"),
    )
    .reindex(method_order)
)


short_labels = [
    "Noisy",
    "SS\nNoisy phase",
    "SS\nOracle phase",
    "Wiener\nNoisy phase",
    "Wiener\nOracle phase",
    "Perfect mag.\nNoisy phase",
    "MP-SENet",
]


def method_bar_plot(
    mean_column,
    std_column,
    ylabel,
    title,
    filename,
):
    values = matched_summary[mean_column].to_numpy()
    errors = matched_summary[std_column].to_numpy()

    x_positions = np.arange(len(method_order))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    ax.bar(
        x_positions,
        values,
        yerr=errors,
        capsize=4,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, filename)


method_bar_plot(
    "STOI_Mean",
    "STOI_Std",
    "STOI",
    "Matched Comparison of Speech Enhancement Methods: STOI",
    "07_matched_method_comparison_stoi.png",
)

method_bar_plot(
    "PESQ_Mean",
    "PESQ_Std",
    "PESQ",
    "Matched Comparison of Speech Enhancement Methods: PESQ",
    "08_matched_method_comparison_pesq.png",
)

method_bar_plot(
    "SI_SDR_Mean",
    "SI_SDR_Std",
    "SI-SDR (dB)",
    "Matched Comparison of Speech Enhancement Methods: SI-SDR",
    "09_matched_method_comparison_sisdr.png",
)


# ============================================================
# 4. MP-SENet performance vs SNR
# ============================================================

mp_by_snr = (
    mpsenet.groupby("Target_SNR_dB")
    .agg(
        STOI=("STOI", "mean"),
        PESQ=("PESQ", "mean"),
        SI_SDR=("SI_SDR_dB", "mean"),
        N=("STOI", "count"),
    )
    .reindex(snr_order)
)


fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.plot(
    mp_by_snr.index,
    mp_by_snr["PESQ"],
    marker="o",
)

ax.set_xlabel("Input SNR (dB)")
ax.set_ylabel("PESQ")
ax.set_title("MP-SENet Performance vs Input SNR")
ax.set_xticks(snr_order)
ax.grid(alpha=0.25)

save_figure(
    fig,
    "10_mpsenet_pesq_vs_snr.png",
)


# ============================================================
# 5. MP-SENet PESQ by noise and SNR
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8))

for noise in noise_order:
    subset = mpsenet[mpsenet["Noise_Type"] == noise]

    summary = (
        subset.groupby("Target_SNR_dB")["PESQ"]
        .agg(["mean", "std"])
        .reindex(snr_order)
    )

    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        label=noise,
    )

ax.set_xlabel("Input SNR (dB)")
ax.set_ylabel("PESQ")
ax.set_title("MP-SENet PESQ by Noise Type and Input SNR")
ax.set_xticks(snr_order)
ax.legend(title="Noise type")
ax.grid(alpha=0.25)

save_figure(
    fig,
    "11_mpsenet_pesq_by_noise_snr.png",
)


# ============================================================
# 6. Export final matched table
# ============================================================

final_table = matched_summary.reset_index()

final_table.to_csv(
    OUTPUT_DIR / "final_matched_method_summary.csv",
    index=False,
)


# ============================================================
# Final console summary
# ============================================================

print()
print("=" * 70)
print("FINAL FIGURES COMPLETE")
print("=" * 70)

print(f"Matched MP-SENet samples: {len(mpsenet)}")
print(f"Classical mixtures used for oracle analysis: {len(oracle)}")

print()
print("Phase disagreement correlations:")
print(f"  ΔPESQ:   r = {r_pesq:.4f}")
print(f"  ΔSTOI:   r = {r_stoi:.4f}")
print(f"  ΔSI-SDR: r = {r_sisdr:.4f}")

print()
print("Matched method summary:")
print(
    matched_summary[
        [
            "STOI_Mean",
            "PESQ_Mean",
            "SI_SDR_Mean",
            "N",
        ]
    ].round(4)
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print("=" * 70)