import os
import sys
import csv
from pathlib import Path

import numpy as np

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio_utils import load_audio, match_length
from mixing import mix_at_snr, measure_snr
from stft_utils import compute_stft, compute_istft
from spectral_subtraction import spectral_subtraction_magnitude
from wiener_filter import wiener_magnitude
from phase_reconstruction import (
    reconstruct_with_noisy_phase,
    reconstruct_with_clean_phase,
    calculate_phase_disagreement,
)
from metrics import evaluate_signal


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
NOISE_DIR = PROJECT_ROOT / "data" / "noise"

CAFE_PATH = NOISE_DIR / "cafe.wav"
TRAFFIC_PATH = NOISE_DIR / "traff.wav"

RESULTS_DIR = PROJECT_ROOT / "results" / "full_experiment"
CSV_PATH = RESULTS_DIR / "full_metrics.csv"

SNR_LEVELS = [10, 5, 0]

# Fixed seed makes generated white noise reproducible.
WHITE_NOISE_SEED = 5305


# ============================================================
# Utility functions
# ============================================================

def reconstruct_perfect_magnitude_with_noisy_phase(
    clean_stft,
    noisy_stft,
    length,
):
    """
    Diagnostic oracle baseline.

    Uses the true clean-speech magnitude together with the
    phase of the noisy mixture.

    This is not a deployable enhancement method. It is used
    to investigate how much noisy phase limits reconstruction
    even when magnitude information is perfect.
    """
    clean_magnitude = np.abs(clean_stft)
    noisy_phase = np.angle(noisy_stft)

    reconstructed_stft = (
        clean_magnitude * np.exp(1j * noisy_phase)
    )

    return compute_istft(
        reconstructed_stft,
        length=length,
    )


def create_white_noise(length, rng):
    """
    Generate reproducible zero-mean Gaussian white noise.
    Actual power is adjusted later by mix_at_snr().
    """
    noise = rng.standard_normal(length)
    noise = noise - np.mean(noise)

    return noise.astype(np.float32)


def prepare_record(
    clean_file,
    noise_name,
    snr_db,
    measured_snr,
    condition,
    metrics,
    phase_disagreement,
):
    """
    Convert one evaluation result into a CSV row.
    """
    return {
        "Clean_File": clean_file,
        "Speaker": clean_file.split("_")[0],
        "Noise_Type": noise_name,
        "Target_SNR_dB": snr_db,
        "Measured_Input_SNR_dB": measured_snr,
        "Condition": condition,
        "STOI": metrics["STOI"],
        "PESQ": metrics["PESQ"],
        "SI_SDR_dB": metrics["SI-SDR"],
        "Phase_Disagreement_rad": phase_disagreement,
    }


# ============================================================
# Process one clean/noise/SNR combination
# ============================================================

def process_mixture(
    clean,
    clean_stft,
    clean_filename,
    noise,
    noise_name,
    snr_db,
):
    """
    Run all enhancement conditions for one controlled mixture.
    """

    # --------------------------------------------------------
    # Controlled mixture
    # --------------------------------------------------------

    noise = match_length(
        noise,
        len(clean),
    )

    noisy, scaled_noise = mix_at_snr(
        clean,
        noise,
        snr_db,
    )

    measured_snr = measure_snr(
        clean,
        scaled_noise,
    )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    noisy_stft = compute_stft(noisy)
    noise_stft = compute_stft(scaled_noise)

    # --------------------------------------------------------
    # Phase disagreement
    # --------------------------------------------------------

    phase_disagreement = calculate_phase_disagreement(
        clean_stft,
        noisy_stft,
    )

    # --------------------------------------------------------
    # Spectral subtraction
    # --------------------------------------------------------

    ss_magnitude = spectral_subtraction_magnitude(
        noisy_stft,
        noise_stft,
    )

    ss_noisy_phase = reconstruct_with_noisy_phase(
        ss_magnitude,
        noisy_stft,
        length=len(clean),
    )

    ss_oracle_phase = reconstruct_with_clean_phase(
        ss_magnitude,
        clean_stft,
        length=len(clean),
    )

    # --------------------------------------------------------
    # Wiener filtering
    # --------------------------------------------------------

    wiener_mag = wiener_magnitude(
        noisy_stft,
        noise_stft,
    )

    wiener_noisy_phase = reconstruct_with_noisy_phase(
        wiener_mag,
        noisy_stft,
        length=len(clean),
    )

    wiener_oracle_phase = reconstruct_with_clean_phase(
        wiener_mag,
        clean_stft,
        length=len(clean),
    )

    # --------------------------------------------------------
    # Perfect magnitude + noisy phase
    # --------------------------------------------------------

    perfect_magnitude_noisy_phase = (
        reconstruct_perfect_magnitude_with_noisy_phase(
            clean_stft,
            noisy_stft,
            length=len(clean),
        )
    )

    # --------------------------------------------------------
    # Objective evaluation
    # --------------------------------------------------------

    conditions = [
        (
            "Noisy",
            noisy,
        ),
        (
            "SS + noisy phase",
            ss_noisy_phase,
        ),
        (
            "SS + clean oracle phase",
            ss_oracle_phase,
        ),
        (
            "Wiener + noisy phase",
            wiener_noisy_phase,
        ),
        (
            "Wiener + clean oracle phase",
            wiener_oracle_phase,
        ),
        (
            "Perfect magnitude + noisy phase",
            perfect_magnitude_noisy_phase,
        ),
    ]

    rows = []

    for condition_name, signal in conditions:

        metrics = evaluate_signal(
            clean,
            signal,
            SAMPLE_RATE,
        )

        row = prepare_record(
            clean_file=clean_filename,
            noise_name=noise_name,
            snr_db=snr_db,
            measured_snr=measured_snr,
            condition=condition_name,
            metrics=metrics,
            phase_disagreement=phase_disagreement,
        )

        rows.append(row)

    return rows


# ============================================================
# Main experiment
# ============================================================

def run_experiment():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Locate clean speech files
    # --------------------------------------------------------

    clean_files = sorted(
        CLEAN_DIR.glob("*.wav")
    )

    if len(clean_files) == 0:
        raise FileNotFoundError(
            f"No WAV files found in {CLEAN_DIR}"
        )

    print(
        f"Found {len(clean_files)} clean speech files."
    )

    if len(clean_files) != 40:
        print(
            f"WARNING: Expected 40 clean files, "
            f"but found {len(clean_files)}."
        )

    # --------------------------------------------------------
    # Load environmental noises
    # --------------------------------------------------------

    if not CAFE_PATH.exists():
        raise FileNotFoundError(
            f"Missing cafeteria noise: {CAFE_PATH}"
        )

    if not TRAFFIC_PATH.exists():
        raise FileNotFoundError(
            f"Missing traffic noise: {TRAFFIC_PATH}"
        )

    cafe_noise, cafe_sr = load_audio(
        CAFE_PATH,
        target_sr=SAMPLE_RATE,
    )

    traffic_noise, traffic_sr = load_audio(
        TRAFFIC_PATH,
        target_sr=SAMPLE_RATE,
    )

    if cafe_sr != SAMPLE_RATE:
        raise ValueError(
            f"Cafe noise sample rate is {cafe_sr}, "
            f"expected {SAMPLE_RATE}."
        )

    if traffic_sr != SAMPLE_RATE:
        raise ValueError(
            f"Traffic noise sample rate is {traffic_sr}, "
            f"expected {SAMPLE_RATE}."
        )

    # --------------------------------------------------------
    # Reproducible white-noise generator
    # --------------------------------------------------------

    rng = np.random.default_rng(
        WHITE_NOISE_SEED
    )

    # --------------------------------------------------------
    # CSV setup
    # --------------------------------------------------------

    fieldnames = [
        "Clean_File",
        "Speaker",
        "Noise_Type",
        "Target_SNR_dB",
        "Measured_Input_SNR_dB",
        "Condition",
        "STOI",
        "PESQ",
        "SI_SDR_dB",
        "Phase_Disagreement_rad",
    ]

    total_mixtures = (
        len(clean_files)
        * 3
        * len(SNR_LEVELS)
    )

    completed_mixtures = 0
    total_rows = 0

    # Write results progressively.
    # This means completed results are preserved even if
    # Codespaces is interrupted during a long experiment.
    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        # ----------------------------------------------------
        # Main clean-speech loop
        # ----------------------------------------------------

        for clean_index, clean_path in enumerate(
            clean_files,
            start=1,
        ):

            print()
            print(
                f"Clean file "
                f"{clean_index}/{len(clean_files)}: "
                f"{clean_path.name}"
            )

            clean, clean_sr = load_audio(
                clean_path,
                target_sr=SAMPLE_RATE,
            )

            if clean_sr != SAMPLE_RATE:
                raise ValueError(
                    f"{clean_path.name} sample rate is "
                    f"{clean_sr}, expected {SAMPLE_RATE}."
                )

            if len(clean) == 0:
                print(
                    f"Skipping empty file: "
                    f"{clean_path.name}"
                )
                continue

            clean_stft = compute_stft(clean)

            # Generate a separate reproducible white-noise
            # sequence for this utterance.
            white_noise = create_white_noise(
                len(clean),
                rng,
            )

            noise_sources = {
                "Cafe": cafe_noise,
                "Traffic": traffic_noise,
                "White": white_noise,
            }

            # ------------------------------------------------
            # Noise loop
            # ------------------------------------------------

            for noise_name, noise in noise_sources.items():

                # --------------------------------------------
                # SNR loop
                # --------------------------------------------

                for snr_db in SNR_LEVELS:

                    completed_mixtures += 1

                    print(
                        f"  [{completed_mixtures}/"
                        f"{total_mixtures}] "
                        f"{noise_name}, "
                        f"{snr_db} dB"
                    )

                    rows = process_mixture(
                        clean=clean,
                        clean_stft=clean_stft,
                        clean_filename=clean_path.name,
                        noise=noise,
                        noise_name=noise_name,
                        snr_db=snr_db,
                    )

                    writer.writerows(rows)

                    total_rows += len(rows)

                    # Immediately write results to disk.
                    csv_file.flush()

    # --------------------------------------------------------
    # Completion report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Full experiment completed successfully.")
    print(f"Clean utterances: {len(clean_files)}")
    print(f"Noise types: 3")
    print(f"SNR levels: {len(SNR_LEVELS)}")
    print(f"Total mixtures: {completed_mixtures}")
    print(f"Metric rows: {total_rows}")
    print(f"Results saved to: {CSV_PATH}")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_experiment()