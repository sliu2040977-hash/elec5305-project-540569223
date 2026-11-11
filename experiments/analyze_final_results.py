"""Recompute the final descriptive analysis from the archived score rows."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "final_analysis"
KEY = ["Clean_File", "Noise_Type", "Target_SNR_dB"]
METRICS = ["STOI", "PESQ", "SI_SDR_dB"]
LABELS = ["STOI", "PESQ", "SI-SDR (dB)"]
METHODS = ["SS", "Wiener"]
ORACLE_MAG = "Perfect magnitude + noisy phase"


def paired_scores(rows):
    """Require a single score per mixture and condition."""
    if not np.isfinite(rows[METRICS].to_numpy()).all():
        raise ValueError("Non-finite score in input.")
    wide = rows.pivot(index=KEY, columns="Condition", values=METRICS)
    if wide.isna().any().any():
        raise ValueError("Missing method score for at least one mixture.")
    return wide


def phase_effects(full, wide):
    phase = full.loc[full.Condition.eq("Noisy")].set_index(KEY)["Phase_Disagreement_rad"]
    phase = phase.reindex(wide.index)
    effects = []
    for method in METHODS:
        delta = pd.DataFrame({
            metric: wide[metric, f"{method} + clean oracle phase"]
            - wide[metric, f"{method} + noisy phase"] for metric in METRICS
        })
        delta["Phase_rad"] = phase
        delta["Method"] = method
        effects.append(delta.reset_index())
    return pd.concat(effects, ignore_index=True)


def describe(effects):
    records = []
    for method, rows in effects.groupby("Method", sort=False):
        cells = rows.groupby(["Noise_Type", "Target_SNR_dB"])
        columns = ["Phase_rad", *METRICS]
        centered = rows[columns] - cells[columns].transform("mean")
        for metric in METRICS:
            values = rows[metric]
            records.append({
                "Method": method, "Metric": metric, "N": len(rows),
                "Mean": values.mean(), "Median": values.median(),
                "Positive": int((values > 0).sum()),
                "Negative": int((values < 0).sum()),
                "Tied": int((values == 0).sum()),
                "Pooled_r": rows["Phase_rad"].corr(values),
                "Cell_centered_r": centered["Phase_rad"].corr(centered[metric]),
            })
    return pd.DataFrame(records)


def comparison(wide, matched):
    matched_full = wide.loc[matched.index]
    for metric in METRICS:
        for condition in wide[metric].columns:
            if not np.allclose(matched_full[metric, condition], matched[metric, condition],
                               rtol=0, atol=1e-10):
                raise ValueError(f"Archived matched scores differ for {condition}, {metric}.")
    remainder = wide.drop(index=matched.index)
    records = []
    for subset, rows in [("All classical", wide), ("Matched", matched),
                         ("Excluded from matched", remainder)]:
        for metric in METRICS:
            for condition, scores in rows[metric].items():
                records.append({
                    "Subset": subset, "Condition": condition, "Metric": metric,
                    "N": len(scores), "Mean": scores.mean(),
                })
    return pd.DataFrame(records)


def plot_phase_cells(cells):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.25), layout="constrained")
    for ax, metric, label in zip(axes, METRICS, LABELS):
        for noise, group in cells[cells.Method.eq("SS")].groupby("Noise_Type"):
            ax.plot(group.Target_SNR_dB, group[metric], marker="o", label=noise)
        ax.set(xlabel="Input SNR (dB)", ylabel=f"Change in {label}", xticks=[0, 5, 10])
        ax.grid(alpha=.24)
    axes[0].legend(frameon=False)
    fig.savefig(OUTPUT / "phase_effect_by_cell.png", dpi=200)
    plt.close(fig)


def plot_correlations(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.05), layout="constrained", sharey=True)
    x = np.arange(len(METRICS))
    for ax, method in zip(axes, METHODS):
        rows = summary.set_index(["Method", "Metric"]).loc[method].loc[METRICS]
        ax.bar(x - .18, rows.Pooled_r, .36, color="#255a83", label="Pooled")
        ax.bar(x + .18, rows.Cell_centered_r, .36, color="#bd662c", label="Cell-centered")
        ax.axhline(0, color="#434b52", linewidth=.8)
        ax.set(title=method, ylim=(-1, 1), xticks=x, xticklabels=LABELS)
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Pearson r")
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(OUTPUT / "phase_correlations.png", dpi=200)
    plt.close(fig)


def main():
    full = pd.read_csv(RESULTS / "full_experiment/full_metrics.csv")
    matched_rows = pd.read_csv(RESULTS / "matched_comparison/matched_all_methods.csv")
    wide = paired_scores(full)
    matched = paired_scores(matched_rows)
    effects = phase_effects(full, wide)
    summary = describe(effects)
    cells = (effects.groupby(["Method", "Noise_Type", "Target_SNR_dB"])
             [["Phase_rad", *METRICS]].mean().reset_index())
    cohorts = comparison(wide, matched)
    OUTPUT.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT / "paired_summary.csv", index=False)
    cells.to_csv(OUTPUT / "noise_snr_means.csv", index=False)
    cohorts.to_csv(OUTPUT / "cohort_means.csv", index=False)
    plot_phase_cells(cells)
    plot_correlations(summary)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Verified {len(wide)} full and {len(matched)} matched mixture IDs.")
    print(f"Outputs: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
