import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------
# Project modules
# ---------------------------------------------------------

from audio_utils import load_audio, match_length
from mixing import mix_at_snr
from stft_utils import (
    compute_stft,
    reconstruct_from_magnitude_phase
)
from spectral_subtraction import spectral_subtraction_magnitude
from wiener_filter import wiener_magnitude


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SAMPLE_RATE = 16000
SNR_DB = 0
HOP_LENGTH = 256

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

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "preliminary",
    "figures"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def waveform_to_db_spectrogram(signal):

    stft_matrix = compute_stft(signal)

    magnitude = np.abs(stft_matrix)

    return librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )


def add_spectrogram(ax, signal, title):

    db_spec = waveform_to_db_spectrogram(signal)

    image = librosa.display.specshow(
        db_spec,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    return image


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load audio
    # -----------------------------------------------------

    clean, _ = load_audio(
        CLEAN_PATH,
        target_sr=SAMPLE_RATE
    )

    noise, _ = load_audio(
        NOISE_PATH,
        target_sr=SAMPLE_RATE
    )

    noise = match_length(
        noise,
        len(clean)
    )

    # -----------------------------------------------------
    # Controlled 0 dB mixture
    # -----------------------------------------------------

    noisy, scaled_noise = mix_at_snr(
        clean,
        noise,
        SNR_DB
    )

    # -----------------------------------------------------
    # STFT
    # -----------------------------------------------------

    clean_stft = compute_stft(clean)
    noisy_stft = compute_stft(noisy)
    noise_stft = compute_stft(scaled_noise)

    clean_phase = np.angle(clean_stft)
    noisy_phase = np.angle(noisy_stft)

    # -----------------------------------------------------
    # Magnitude enhancement
    # -----------------------------------------------------

    ss_magnitude = spectral_subtraction_magnitude(
        noisy_stft,
        noise_stft
    )

    wiener_enhanced_magnitude = wiener_magnitude(
        noisy_stft,
        noise_stft
    )

    # -----------------------------------------------------
    # Spectral subtraction reconstruction
    # -----------------------------------------------------

    ss_noisy_phase = reconstruct_from_magnitude_phase(
        ss_magnitude,
        noisy_phase,
        length=len(clean)
    )

    ss_oracle_phase = reconstruct_from_magnitude_phase(
        ss_magnitude,
        clean_phase,
        length=len(clean)
    )

    # -----------------------------------------------------
    # Wiener reconstruction
    # -----------------------------------------------------

    wiener_noisy_phase = reconstruct_from_magnitude_phase(
        wiener_enhanced_magnitude,
        noisy_phase,
        length=len(clean)
    )

    wiener_oracle_phase = reconstruct_from_magnitude_phase(
        wiener_enhanced_magnitude,
        clean_phase,
        length=len(clean)
    )

    # -----------------------------------------------------
    # Comparison figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        constrained_layout=True
    )

    image = add_spectrogram(
        axes[0, 0],
        clean,
        "Clean Speech"
    )

    add_spectrogram(
        axes[0, 1],
        noisy,
        "Noisy Speech - PCAFETER, 0 dB"
    )

    add_spectrogram(
        axes[1, 0],
        ss_noisy_phase,
        "Spectral Subtraction + Noisy Phase"
    )

    add_spectrogram(
        axes[1, 1],
        ss_oracle_phase,
        "Spectral Subtraction + Clean Oracle Phase"
    )

    add_spectrogram(
        axes[2, 0],
        wiener_noisy_phase,
        "Wiener + Noisy Phase"
    )

    add_spectrogram(
        axes[2, 1],
        wiener_oracle_phase,
        "Wiener + Clean Oracle Phase"
    )

    fig.colorbar(
        image,
        ax=axes,
        format="%+2.0f dB",
        label="Magnitude (dB)",
        shrink=0.8
    )

    fig.suptitle(
        "Noisy-Phase vs Oracle Clean-Phase Reconstruction at 0 dB SNR",
        fontsize=16
    )

    output_path = os.path.join(
        FIGURE_DIR,
        "reconstruction_comparison_0dB.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Reconstruction comparison generated successfully."
    )

    print(
        f"Figure saved to: {output_path}"
    )


if __name__ == "__main__":
    main()