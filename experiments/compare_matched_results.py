from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MPSENET_CSV = (
    PROJECT_ROOT
    / "results"
    / "mpsenet"
    / "mpsenet_metrics.csv"
)

FULL_CSV = (
    PROJECT_ROOT
    / "results"
    / "full_experiment"
    / "full_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "matched_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHED_ROWS_CSV = OUTPUT_DIR / "matched_all_methods.csv"
SUMMARY_CSV = OUTPUT_DIR / "matched_summary.csv"
SUMMARY_BY_SNR_CSV = OUTPUT_DIR / "matched_summary_by_snr.csv"
SUMMARY_BY_NOISE_CSV = OUTPUT_DIR / "matched_summary_by_noise.csv"
SUMMARY_BY_NOISE_SNR_CSV = OUTPUT_DIR / "matched_summary_by_noise_snr.csv"


KEY_COLUMNS = [
    "Clean_File",
    "Speaker",
    "Noise_Type",
    "Target_SNR_dB",
]


CONDITION_ORDER = [
    "Noisy",
    "SS + noisy phase",
    "SS + clean oracle phase",
    "Wiener + noisy phase",
    "Wiener + clean oracle phase",
    "Perfect magnitude + noisy phase",
    "MP-SENet",
]


def build_summary(df, group_columns):
    summary = (
        df.groupby(group_columns, as_index=False)
        .agg(
            STOI_Mean=("STOI", "mean"),
            STOI_Std=("STOI", "std"),
            PESQ_Mean=("PESQ", "mean"),
            PESQ_Std=("PESQ", "std"),
            SI_SDR_Mean=("SI_SDR_dB", "mean"),
            SI_SDR_Std=("SI_SDR_dB", "std"),
            N=("Condition", "size"),
        )
    )

    return summary


def main():
    mp = pd.read_csv(MPSENET_CSV)
    full = pd.read_csv(FULL_CSV)

    print("=" * 78)
    print("MATCHED CLASSICAL / ORACLE / MP-SENET COMPARISON")
    print("=" * 78)

    print(f"MP-SENet rows: {len(mp)}")
    print(f"Full classical rows: {len(full)}")

    # ------------------------------------------------------------
    # 1. Verify MP-SENet mixture keys are unique
    # ------------------------------------------------------------

    duplicate_mp = mp.duplicated(KEY_COLUMNS, keep=False)

    if duplicate_mp.any():
        print()
        print("WARNING: Duplicate MP-SENet mixture keys found:")
        print(
            mp.loc[
                duplicate_mp,
                KEY_COLUMNS + ["Enhanced_File"]
            ].to_string(index=False)
        )
        raise RuntimeError(
            "MP-SENet mixture keys are not unique."
        )

    mp_keys = mp[KEY_COLUMNS].drop_duplicates()

    print(f"Unique MP-SENet mixtures: {len(mp_keys)}")

    # ------------------------------------------------------------
    # 2. Select exactly the same mixtures from full experiment
    # ------------------------------------------------------------

    matched_classical = full.merge(
        mp_keys,
        on=KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )

    expected_classical_rows = len(mp_keys) * 6

    print(
        f"Matched classical rows: "
        f"{len(matched_classical)} "
        f"(expected {expected_classical_rows})"
    )

    if len(matched_classical) != expected_classical_rows:
        counts = (
            matched_classical.groupby(KEY_COLUMNS)
            .size()
            .reset_index(name="Condition_Count")
        )

        bad = counts[counts["Condition_Count"] != 6]

        if not bad.empty:
            print()
            print("Mixtures without exactly six classical conditions:")
            print(bad.to_string(index=False))

        raise RuntimeError(
            "Matched classical dataset is incomplete."
        )

    # ------------------------------------------------------------
    # 3. Prepare MP-SENet rows in the same format
    # ------------------------------------------------------------

    mp_common = mp[
        KEY_COLUMNS
        + [
            "Condition",
            "STOI",
            "PESQ",
            "SI_SDR_dB",
        ]
    ].copy()

    mp_common["Measured_Input_SNR_dB"] = pd.NA
    mp_common["Phase_Disagreement_rad"] = pd.NA

    classical_common = matched_classical[
        KEY_COLUMNS
        + [
            "Measured_Input_SNR_dB",
            "Condition",
            "STOI",
            "PESQ",
            "SI_SDR_dB",
            "Phase_Disagreement_rad",
        ]
    ].copy()

    mp_common = mp_common[
        classical_common.columns
    ]

    # ------------------------------------------------------------
    # 4. Combine all seven conditions
    # ------------------------------------------------------------

    combined = pd.concat(
        [
            classical_common,
            mp_common,
        ],
        ignore_index=True,
    )

    combined["Condition"] = pd.Categorical(
        combined["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(
        KEY_COLUMNS + ["Condition"]
    ).reset_index(drop=True)

    combined.to_csv(
        MATCHED_ROWS_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # 5. Verify condition counts
    # ------------------------------------------------------------

    condition_counts = (
        combined.groupby(
            "Condition",
            observed=False
        )
        .size()
    )

    print()
    print("Rows per condition:")
    print(condition_counts.to_string())

    expected_n = len(mp_keys)

    bad_conditions = condition_counts[
        condition_counts != expected_n
    ]

    if not bad_conditions.empty:
        print()
        print("WARNING: Not every condition has the same N.")
        print(bad_conditions)

    # ------------------------------------------------------------
    # 6. Overall matched summary
    # ------------------------------------------------------------

    overall = build_summary(
        combined,
        ["Condition"],
    )

    overall["Condition"] = pd.Categorical(
        overall["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    overall = overall.sort_values(
        "Condition"
    )

    overall.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # 7. Summary by SNR
    # ------------------------------------------------------------

    by_snr = build_summary(
        combined,
        [
            "Target_SNR_dB",
            "Condition",
        ],
    )

    by_snr["Condition"] = pd.Categorical(
        by_snr["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    by_snr = by_snr.sort_values(
        [
            "Target_SNR_dB",
            "Condition",
        ]
    )

    by_snr.to_csv(
        SUMMARY_BY_SNR_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # 8. Summary by noise
    # ------------------------------------------------------------

    by_noise = build_summary(
        combined,
        [
            "Noise_Type",
            "Condition",
        ],
    )

    by_noise["Condition"] = pd.Categorical(
        by_noise["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    by_noise = by_noise.sort_values(
        [
            "Noise_Type",
            "Condition",
        ]
    )

    by_noise.to_csv(
        SUMMARY_BY_NOISE_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # 9. Summary by noise + SNR
    # ------------------------------------------------------------

    by_noise_snr = build_summary(
        combined,
        [
            "Noise_Type",
            "Target_SNR_dB",
            "Condition",
        ],
    )

    by_noise_snr["Condition"] = pd.Categorical(
        by_noise_snr["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    by_noise_snr = by_noise_snr.sort_values(
        [
            "Noise_Type",
            "Target_SNR_dB",
            "Condition",
        ]
    )

    by_noise_snr.to_csv(
        SUMMARY_BY_NOISE_SNR_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # 10. Print final matched table
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL MATCHED OVERALL RESULTS")
    print("=" * 78)

    display_columns = [
        "Condition",
        "STOI_Mean",
        "PESQ_Mean",
        "SI_SDR_Mean",
        "N",
    ]

    print(
        overall[
            display_columns
        ].round(4).to_string(index=False)
    )

    # ------------------------------------------------------------
    # 11. Direct MP-SENet comparisons
    # ------------------------------------------------------------

    mp_row = overall[
        overall["Condition"] == "MP-SENet"
    ].iloc[0]

    print()
    print("=" * 78)
    print("MP-SENET DIFFERENCE RELATIVE TO SELECTED BASELINES")
    print("=" * 78)

    compare_conditions = [
        "Noisy",
        "SS + noisy phase",
        "SS + clean oracle phase",
        "Wiener + noisy phase",
        "Wiener + clean oracle phase",
        "Perfect magnitude + noisy phase",
    ]

    comparison_rows = []

    for condition in compare_conditions:
        baseline = overall[
            overall["Condition"] == condition
        ].iloc[0]

        comparison_rows.append(
            {
                "Compared_With": condition,
                "Delta_STOI": (
                    mp_row["STOI_Mean"]
                    - baseline["STOI_Mean"]
                ),
                "Delta_PESQ": (
                    mp_row["PESQ_Mean"]
                    - baseline["PESQ_Mean"]
                ),
                "Delta_SI_SDR_dB": (
                    mp_row["SI_SDR_Mean"]
                    - baseline["SI_SDR_Mean"]
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison_df.round(4).to_string(
            index=False
        )
    )

    comparison_df.to_csv(
        OUTPUT_DIR
        / "mpsenet_vs_baselines.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("MATCHED COMPARISON COMPLETE")
    print("=" * 78)

    print(f"Matched mixtures: {expected_n}")
    print(
        f"Total comparison rows: {len(combined)} "
        f"(expected {expected_n * 7})"
    )

    print()
    print("Files created:")
    print(MATCHED_ROWS_CSV)
    print(SUMMARY_CSV)
    print(SUMMARY_BY_SNR_CSV)
    print(SUMMARY_BY_NOISE_CSV)
    print(SUMMARY_BY_NOISE_SNR_CSV)
    print(
        OUTPUT_DIR
        / "mpsenet_vs_baselines.csv"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()