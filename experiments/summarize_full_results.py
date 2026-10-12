from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "full_experiment"
    / "full_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "full_experiment"
)

SUMMARY_CSV = OUTPUT_DIR / "summary_by_noise_snr.csv"
OVERALL_CSV = OUTPUT_DIR / "summary_overall.csv"
ORACLE_CSV = OUTPUT_DIR / "oracle_phase_benefit.csv"


def main():

    df = pd.read_csv(INPUT_CSV)

    print("=" * 70)
    print("FULL EXPERIMENT CHECK")
    print("=" * 70)

    print(f"Total rows: {len(df)}")
    print(f"Clean files: {df['Clean_File'].nunique()}")
    print(f"Speakers: {df['Speaker'].nunique()}")
    print(f"Noise types: {df['Noise_Type'].nunique()}")
    print(f"SNR levels: {df['Target_SNR_dB'].nunique()}")
    print(f"Conditions: {df['Condition'].nunique()}")

    print()
    print("Conditions:")
    for condition in df["Condition"].unique():
        print(f"  - {condition}")

    # ========================================================
    # Summary by noise type, SNR and condition
    # ========================================================

    summary = (
        df.groupby(
            [
                "Noise_Type",
                "Target_SNR_dB",
                "Condition",
            ]
        )
        .agg(
            STOI_mean=("STOI", "mean"),
            STOI_std=("STOI", "std"),
            PESQ_mean=("PESQ", "mean"),
            PESQ_std=("PESQ", "std"),
            SI_SDR_mean=("SI_SDR_dB", "mean"),
            SI_SDR_std=("SI_SDR_dB", "std"),
            Phase_Disagreement_mean=(
                "Phase_Disagreement_rad",
                "mean",
            ),
            N=("Clean_File", "count"),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # ========================================================
    # Overall summary across all noises and SNRs
    # ========================================================

    overall = (
        df.groupby("Condition")
        .agg(
            STOI_mean=("STOI", "mean"),
            STOI_std=("STOI", "std"),
            PESQ_mean=("PESQ", "mean"),
            PESQ_std=("PESQ", "std"),
            SI_SDR_mean=("SI_SDR_dB", "mean"),
            SI_SDR_std=("SI_SDR_dB", "std"),
            N=("Clean_File", "count"),
        )
        .reset_index()
    )

    overall.to_csv(
        OVERALL_CSV,
        index=False,
    )

    # ========================================================
    # Oracle-phase benefit
    # ========================================================

    index_columns = [
        "Clean_File",
        "Speaker",
        "Noise_Type",
        "Target_SNR_dB",
    ]

    metric_columns = [
        "STOI",
        "PESQ",
        "SI_SDR_dB",
    ]

    ss_noisy = (
        df[df["Condition"] == "SS + noisy phase"]
        .set_index(index_columns)
    )

    ss_oracle = (
        df[df["Condition"] == "SS + clean oracle phase"]
        .set_index(index_columns)
    )

    wiener_noisy = (
        df[df["Condition"] == "Wiener + noisy phase"]
        .set_index(index_columns)
    )

    wiener_oracle = (
        df[df["Condition"] == "Wiener + clean oracle phase"]
        .set_index(index_columns)
    )

    oracle_rows = []

    common_index = (
        ss_noisy.index
        .intersection(ss_oracle.index)
        .intersection(wiener_noisy.index)
        .intersection(wiener_oracle.index)
    )

    for idx in common_index:

        clean_file, speaker, noise_type, snr_db = idx

        original_row = df[
            (df["Clean_File"] == clean_file)
            & (df["Noise_Type"] == noise_type)
            & (df["Target_SNR_dB"] == snr_db)
        ].iloc[0]

        row = {
            "Clean_File": clean_file,
            "Speaker": speaker,
            "Noise_Type": noise_type,
            "Target_SNR_dB": snr_db,
            "Phase_Disagreement_rad":
                original_row["Phase_Disagreement_rad"],
        }

        for metric in metric_columns:

            row[f"SS_Delta_{metric}"] = (
                ss_oracle.loc[idx, metric]
                - ss_noisy.loc[idx, metric]
            )

            row[f"Wiener_Delta_{metric}"] = (
                wiener_oracle.loc[idx, metric]
                - wiener_noisy.loc[idx, metric]
            )

        oracle_rows.append(row)

    oracle_df = pd.DataFrame(oracle_rows)

    oracle_df.to_csv(
        ORACLE_CSV,
        index=False,
    )

    # ========================================================
    # Print important overall results
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL MEAN RESULTS")
    print("=" * 70)

    display_columns = [
        "Condition",
        "STOI_mean",
        "PESQ_mean",
        "SI_SDR_mean",
    ]

    print(
        overall[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("MEAN ORACLE-PHASE BENEFIT BY SNR")
    print("=" * 70)

    oracle_by_snr = (
        oracle_df.groupby("Target_SNR_dB")
        .agg(
            Phase_Disagreement=(
                "Phase_Disagreement_rad",
                "mean",
            ),
            SS_Delta_STOI=(
                "SS_Delta_STOI",
                "mean",
            ),
            SS_Delta_PESQ=(
                "SS_Delta_PESQ",
                "mean",
            ),
            SS_Delta_SI_SDR=(
                "SS_Delta_SI_SDR_dB",
                "mean",
            ),
            Wiener_Delta_STOI=(
                "Wiener_Delta_STOI",
                "mean",
            ),
            Wiener_Delta_PESQ=(
                "Wiener_Delta_PESQ",
                "mean",
            ),
            Wiener_Delta_SI_SDR=(
                "Wiener_Delta_SI_SDR_dB",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("Target_SNR_dB")
    )

    print(
        oracle_by_snr
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(SUMMARY_CSV)
    print(OVERALL_CSV)
    print(ORACLE_CSV)


if __name__ == "__main__":
    main()