import os
import sys
import csv
import numpy as np
import soundfile as sf

# Allow imports from the src directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from audio_utils import load_audio
from mixing import mix_at_snr
from stft_utils import compute_stft, get_magnitude_phase
from spectral_subtraction import spectral_subtraction_magnitude
from wiener_filter import wiener_magnitude
from phase_reconstruction import (
    reconstruct_with_noisy_phase,
    reconstruct_with_clean_phase,
    calculate_phase_disagreement,
)
from metrics import evaluate_signal


SAMPLE_RATE = 16000
SNR_LEVELS = [10, 5, 0]

CLEAN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "p232_001.wav"
)

NOISE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "ch01.wav"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "preliminary"
)

AUDIO_DIR = os.path.join(
    RESULTS_DIR,
    "audio"
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "metrics.csv"
)


def save_audio(filename, signal):
    path = os.path.join(AUDIO_DIR, filename)

    sf.write(
        path,
        signal,
        SAMPLE_RATE
    )


def evaluate_condition(
    clean,
    processed,
    condition,
    snr_db,
    phase_disagreement
):
    metrics = evaluate_signal(
        clean,
        processed,
        SAMPLE_RATE
    )

    return {
        "SNR_dB": snr_db,
        "Condition": condition,
        "STOI": metrics["STOI"],
        "PESQ": metrics["PESQ"],
        "SI_SDR_dB": metrics["SI-SDR"],
        "Phase_Disagreement_rad": phase_disagreement,
    }


def run_experiment():
    os.makedirs(
        AUDIO_DIR,
        exist_ok=True
    )

    clean = load_audio(
        CLEAN_PATH,
        sample_rate=SAMPLE_RATE
    )

    noise = load_audio(
        NOISE_PATH,
        sample_rate=SAMPLE_RATE
    )

    all_results = []

    for snr_db in SNR_LEVELS:

        noisy, scaled_noise = mix_at_snr(
            clean,
            noise,
            snr_db
        )

        clean_stft = compute_stft(clean)
        noisy_stft = compute_stft(noisy)
        noise_stft = compute_stft(scaled_noise)

        _, noisy_phase = get_magnitude_phase(
            noisy_stft
        )

        _, clean_phase = get_magnitude_phase(
            clean_stft
        )

        phase_disagreement = (
            calculate_phase_disagreement(
                clean_stft,
                noisy_stft
            )
        )

        # --------------------------------------------------
        # Spectral subtraction
        # --------------------------------------------------

        ss_magnitude = (
            spectral_subtraction_magnitude(
                noisy_stft,
                noise_stft
            )
        )

        ss_noisy_phase = (
            reconstruct_with_noisy_phase(
                ss_magnitude,
                noisy_stft,
                length=len(clean)
            )
        )

        ss_oracle_phase = (
            reconstruct_with_clean_phase(
                ss_magnitude,
                clean_stft,
                length=len(clean)
            )
        )

        # --------------------------------------------------
        # Wiener filtering
        # --------------------------------------------------

        wiener_magnitude_estimate = (
            wiener_magnitude(
                noisy_stft,
                noise_stft
            )
        )

        wiener_noisy_phase = (
            reconstruct_with_noisy_phase(
                wiener_magnitude_estimate,
                noisy_stft,
                length=len(clean)
            )
        )

        wiener_oracle_phase = (
            reconstruct_with_clean_phase(
                wiener_magnitude_estimate,
                clean_stft,
                length=len(clean)
            )
        )

        # --------------------------------------------------
        # Save audio
        # --------------------------------------------------

        prefix = f"{snr_db}dB"

        save_audio(
            f"{prefix}_noisy.wav",
            noisy
        )

        save_audio(
            f"{prefix}_ss_noisy_phase.wav",
            ss_noisy_phase
        )

        save_audio(
            f"{prefix}_ss_oracle_phase.wav",
            ss_oracle_phase
        )

        save_audio(
            f"{prefix}_wiener_noisy_phase.wav",
            wiener_noisy_phase
        )

        save_audio(
            f"{prefix}_wiener_oracle_phase.wav",
            wiener_oracle_phase
        )

        # --------------------------------------------------
        # Objective evaluation
        # --------------------------------------------------

        conditions = [
            (
                "Noisy",
                noisy
            ),
            (
                "SS + noisy phase",
                ss_noisy_phase
            ),
            (
                "SS + clean oracle phase",
                ss_oracle_phase
            ),
            (
                "Wiener + noisy phase",
                wiener_noisy_phase
            ),
            (
                "Wiener + clean oracle phase",
                wiener_oracle_phase
            ),
        ]

        for condition_name, signal in conditions:

            result = evaluate_condition(
                clean,
                signal,
                condition_name,
                snr_db,
                phase_disagreement
            )

            all_results.append(result)

    # ------------------------------------------------------
    # Save metrics table
    # ------------------------------------------------------

    fieldnames = [
        "SNR_dB",
        "Condition",
        "STOI",
        "PESQ",
        "SI_SDR_dB",
        "Phase_Disagreement_rad",
    ]

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print(
        "Preliminary experiment completed."
    )

    print(
        f"Results saved to: {CSV_PATH}"
    )


if __name__ == "__main__":
    run_experiment()
