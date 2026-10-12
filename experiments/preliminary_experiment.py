import os
import sys
import csv

import numpy as np
import soundfile as sf

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------
# Project modules
# ---------------------------------------------------------

from audio_utils import (
    load_audio,
    match_length
)

from mixing import (
    mix_at_snr,
    measure_snr
)

from stft_utils import (
    compute_stft,
    compute_istft
)

from spectral_subtraction import (
    spectral_subtraction_magnitude
)

from wiener_filter import (
    wiener_magnitude
)

from phase_reconstruction import (
    reconstruct_with_noisy_phase,
    reconstruct_with_clean_phase,
    calculate_phase_disagreement
)

from metrics import (
    evaluate_signal
)

# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

SAMPLE_RATE = 16000

SNR_LEVELS = [
    10,
    5,
    0
]

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

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def save_audio(filename, signal):
    """
    Save one experimental audio output.
    """

    path = os.path.join(
        AUDIO_DIR,
        filename
    )

    sf.write(
        path,
        signal,
        SAMPLE_RATE
    )


def evaluate_condition(
    clean,
    processed,
    condition,
    target_snr,
    measured_snr,
    phase_disagreement
):
    """
    Evaluate one experimental condition.
    """

    metrics = evaluate_signal(
        clean,
        processed,
        SAMPLE_RATE
    )

    return {
        "Target_SNR_dB": target_snr,
        "Measured_Input_SNR_dB": measured_snr,
        "Condition": condition,
        "STOI": metrics["STOI"],
        "PESQ": metrics["PESQ"],
        "SI_SDR_dB": metrics["SI-SDR"],
        "Phase_Disagreement_rad": phase_disagreement
    }


def reconstruct_perfect_magnitude_with_noisy_phase(
    clean_stft,
    noisy_stft,
    length
):
    """
    Reconstruct speech using the perfect clean magnitude spectrum
    together with the phase of the noisy mixture.

    This is a diagnostic oracle-magnitude condition used to isolate
    the limitation caused by retaining noisy phase even when the
    magnitude estimate is perfect.
    """

    clean_magnitude = np.abs(clean_stft)
    noisy_phase = np.angle(noisy_stft)

    reconstructed_stft = (
        clean_magnitude
        * np.exp(1j * noisy_phase)
    )

    reconstructed_signal = compute_istft(
        reconstructed_stft,
        length=length
    )

    return reconstructed_signal


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def run_experiment():

    os.makedirs(
        AUDIO_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load clean VoiceBank speech
    # -----------------------------------------------------

    clean, clean_sr = load_audio(
        CLEAN_PATH,
        target_sr=SAMPLE_RATE
    )

    # -----------------------------------------------------
    # Load DEMAND cafeteria noise
    # -----------------------------------------------------

    noise, noise_sr = load_audio(
        NOISE_PATH,
        target_sr=SAMPLE_RATE
    )

    if clean_sr != SAMPLE_RATE:
        raise ValueError(
            "Clean speech sampling rate is not 16 kHz."
        )

    if noise_sr != SAMPLE_RATE:
        raise ValueError(
            "Noise sampling rate is not 16 kHz."
        )

    # Match DEMAND noise length to VoiceBank utterance

    noise = match_length(
        noise,
        len(clean)
    )

    all_results = []

    # Clean STFT is identical for all SNR conditions

    clean_stft = compute_stft(
        clean
    )

    # -----------------------------------------------------
    # Run 10, 5 and 0 dB experiments
    # -----------------------------------------------------

    for snr_db in SNR_LEVELS:

        print(
            f"Processing {snr_db} dB SNR..."
        )

        # Controlled clean + noise mixture

        noisy, scaled_noise = mix_at_snr(
            clean,
            noise,
            snr_db
        )

        measured_snr = measure_snr(
            clean,
            scaled_noise
        )

        # -------------------------------------------------
        # STFT analysis
        # -------------------------------------------------

        noisy_stft = compute_stft(
            noisy
        )

        noise_stft = compute_stft(
            scaled_noise
        )

        # Measure clean/noisy phase disagreement

        phase_disagreement = (
            calculate_phase_disagreement(
                clean_stft,
                noisy_stft
            )
        )

        # =================================================
        # Spectral subtraction
        # =================================================

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

        # =================================================
        # Wiener filtering
        # =================================================

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

        # =================================================
        # Perfect magnitude + noisy phase
        # =================================================

        perfect_magnitude_noisy_phase = (
            reconstruct_perfect_magnitude_with_noisy_phase(
                clean_stft,
                noisy_stft,
                length=len(clean)
            )
        )

        # =================================================
        # Save audio examples
        # =================================================

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

        save_audio(
            f"{prefix}_perfect_magnitude_noisy_phase.wav",
            perfect_magnitude_noisy_phase
        )

        # =================================================
        # Objective evaluation
        # =================================================

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
            (
                "Perfect magnitude + noisy phase",
                perfect_magnitude_noisy_phase
            )
        ]

        for condition_name, signal in conditions:

            result = evaluate_condition(
                clean=clean,
                processed=signal,
                condition=condition_name,
                target_snr=snr_db,
                measured_snr=measured_snr,
                phase_disagreement=phase_disagreement
            )

            all_results.append(
                result
            )

    # -----------------------------------------------------
    # Save metrics CSV
    # -----------------------------------------------------

    fieldnames = [
        "Target_SNR_dB",
        "Measured_Input_SNR_dB",
        "Condition",
        "STOI",
        "PESQ",
        "SI_SDR_dB",
        "Phase_Disagreement_rad"
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

        writer.writerows(
            all_results
        )

    print()

    print(
        "Preliminary experiment completed successfully."
    )

    print(
        f"Metrics saved to: {CSV_PATH}"
    )

    print(
        f"Audio outputs saved to: {AUDIO_DIR}"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_experiment()