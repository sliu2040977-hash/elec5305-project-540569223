import os

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

METRICS_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "preliminary",
    "metrics.csv"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "preliminary",
    "figures"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "preliminary"
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load experimental metrics
    # -----------------------------------------------------

    df = pd.read_csv(METRICS_PATH)

    snr_values = sorted(
        df["Target_SNR_dB"].unique()
    )

    results = []

    # -----------------------------------------------------
    # Calculate oracle-phase benefit
    # -----------------------------------------------------

    for snr in snr_values:

        snr_df = df[
            df["Target_SNR_dB"] == snr
        ]

        ss_noisy = snr_df[
            snr_df["Condition"]
            == "SS + noisy phase"
        ].iloc[0]

        ss_oracle = snr_df[
            snr_df["Condition"]
            == "SS + clean oracle phase"
        ].iloc[0]

        wiener_noisy = snr_df[
            snr_df["Condition"]
            == "Wiener + noisy phase"
        ].iloc[0]

        wiener_oracle = snr_df[
            snr_df["Condition"]
            == "Wiener + clean oracle phase"
        ].iloc[0]

        results.append({
            "SNR_dB": snr,

            "SS_Delta_STOI":
                ss_oracle["STOI"]
                - ss_noisy["STOI"],

            "SS_Delta_PESQ":
                ss_oracle["PESQ"]
                - ss_noisy["PESQ"],

            "SS_Delta_SI_SDR_dB":
                ss_oracle["SI_SDR_dB"]
                - ss_noisy["SI_SDR_dB"],

            "Wiener_Delta_STOI":
                wiener_oracle["STOI"]
                - wiener_noisy["STOI"],

            "Wiener_Delta_PESQ":
                wiener_oracle["PESQ"]
                - wiener_noisy["PESQ"],

            "Wiener_Delta_SI_SDR_dB":
                wiener_oracle["SI_SDR_dB"]
                - wiener_noisy["SI_SDR_dB"],

            "Phase_Disagreement_rad":
                snr_df[
                    "Phase_Disagreement_rad"
                ].iloc[0]
        })

    benefit_df = pd.DataFrame(results)

    # -----------------------------------------------------
    # Save numerical oracle-phase benefit table
    # -----------------------------------------------------

    benefit_path = os.path.join(
        RESULTS_DIR,
        "oracle_phase_benefit.csv"
    )

    benefit_df.to_csv(
        benefit_path,
        index=False
    )

    # -----------------------------------------------------
    # Figure 1:
    # Oracle-phase PESQ benefit vs input SNR
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        benefit_df["SNR_dB"],
        benefit_df["SS_Delta_PESQ"],
        marker="o",
        linewidth=2,
        label="Spectral Subtraction"
    )

    plt.plot(
        benefit_df["SNR_dB"],
        benefit_df["Wiener_Delta_PESQ"],
        marker="s",
        linewidth=2,
        label="Wiener"
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Input SNR (dB)"
    )

    plt.ylabel(
        "Oracle-Phase PESQ Benefit (ΔPESQ)"
    )

    plt.title(
        "Oracle-Phase Benefit vs Input SNR\nPCAFETER Noise"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pesq_figure_path = os.path.join(
        FIGURE_DIR,
        "oracle_phase_pesq_benefit_vs_snr.png"
    )

    plt.savefig(
        pesq_figure_path,
        dpi=200
    )

    plt.close()

    # -----------------------------------------------------
    # Figure 2:
    # Phase disagreement vs oracle-phase PESQ benefit
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        benefit_df["Phase_Disagreement_rad"],
        benefit_df["SS_Delta_PESQ"],
        marker="o",
        linewidth=2,
        label="Spectral Subtraction"
    )

    plt.plot(
        benefit_df["Phase_Disagreement_rad"],
        benefit_df["Wiener_Delta_PESQ"],
        marker="s",
        linewidth=2,
        label="Wiener"
    )

    for _, row in benefit_df.iterrows():

        plt.annotate(
            f'{int(row["SNR_dB"])} dB',
            (
                row["Phase_Disagreement_rad"],
                row["SS_Delta_PESQ"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel(
        "Phase Disagreement Dφ (rad)"
    )

    plt.ylabel(
        "Oracle-Phase PESQ Benefit (ΔPESQ)"
    )

    plt.title(
        "Phase Disagreement vs Oracle-Phase Benefit\nPCAFETER Noise"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    phase_figure_path = os.path.join(
        FIGURE_DIR,
        "phase_disagreement_vs_oracle_pesq_benefit.png"
    )

    plt.savefig(
        phase_figure_path,
        dpi=200
    )

    plt.close()

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print(
        "\nOracle-phase benefit results:"
    )

    print(
        benefit_df.round(4).to_string(
            index=False
        )
    )

    print(
        "\nOracle-phase analysis completed successfully."
    )

    print(
        f"Table saved to: {benefit_path}"
    )

    print(
        f"Figure saved to: {pesq_figure_path}"
    )

    print(
        f"Figure saved to: {phase_figure_path}"
    )


if __name__ == "__main__":
    main()