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
from stft_utils import compute_stft
from spectral_subtraction import spectral_subtraction_magnitude


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SAMPLE_RATE = 16000
SNR_DB = 0

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
# Plot helpers
# ---------------------------------------------------------

def magnitude_to_db(magnitude):
    return librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )


def save_spectrogram(
    magnitude,
    title,
    filename
):
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        magnitude_to_db(magnitude),
        sr=SAMPLE_RATE,
        hop_length=256,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURE_DIR, filename),
        dpi=200
    )

    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

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

    noisy, scaled_noise = mix_at_snr(
        clean,
        noise,
        SNR_DB
    )

    clean_stft = compute_stft(clean)
    noisy_stft = compute_stft(noisy)
    noise_stft = compute_stft(scaled_noise)

    clean_magnitude = np.abs(clean_stft)
    noisy_magnitude = np.abs(noisy_stft)

    ss_magnitude = spectral_subtraction_magnitude(
        noisy_stft,
        noise_stft
    )

    clean_phase = np.angle(clean_stft)
    noisy_phase = np.angle(noisy_stft)

    phase_difference = np.angle(
        np.exp(
            1j * (
                noisy_phase - clean_phase
            )
        )
    )

    # -----------------------------------------------------
    # 1. Clean spectrogram
    # -----------------------------------------------------

    save_spectrogram(
        clean_magnitude,
        "Clean Speech Spectrogram",
        "clean_spectrogram.png"
    )

    # -----------------------------------------------------
    # 2. Noisy spectrogram
    # -----------------------------------------------------

    save_spectrogram(
        noisy_magnitude,
        "Noisy Speech Spectrogram - PCAFETER, 0 dB",
        "noisy_0dB_spectrogram.png"
    )

    # -----------------------------------------------------
    # 3. Enhanced magnitude
    # -----------------------------------------------------

    save_spectrogram(
        ss_magnitude,
        "Spectral Subtraction Enhanced Magnitude - 0 dB",
        "ss_enhanced_magnitude_0dB.png"
    )

    # -----------------------------------------------------
    # 4. Phase difference
    # -----------------------------------------------------

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        phase_difference,
        sr=SAMPLE_RATE,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi
    )

    plt.colorbar(
        label="Phase difference (rad)"
    )

    plt.title(
        "Noisy-Clean Phase Difference - PCAFETER, 0 dB"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "phase_difference_0dB.png"
        ),
        dpi=200
    )

    plt.close()

    print(
        "Preliminary figures generated successfully."
    )

    print(
        f"Figures saved to: {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()