from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_utils import (
    load_audio,
    match_length,
    save_audio,
)
from src.mixing import mix_at_snr
from src.stft_utils import (
    compute_stft,
    get_magnitude_phase,
)
from src.spectral_subtraction import spectral_subtraction_magnitude
from src.wiener_filter import wiener_magnitude
from src.phase_reconstruction import (
    reconstruct_with_noisy_phase,
    reconstruct_with_clean_phase,
)
from src.metrics import evaluate_signal


# ============================================================
# Experiment configuration
# ============================================================

CLEAN_FILE = PROJECT_ROOT / "data" / "clean" / "p232_001.wav"
NOISE_FILE = PROJECT_ROOT / "data" / "noise" / "cafe.wav"

MPSENET_FILE = (
    PROJECT_ROOT
    / "results"
    / "mpsenet"
    / "output_enhanced"
    / "p232_001__cafe__snr0db.wav"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "representative_example"
AUDIO_DIR = OUTPUT_DIR / "audio"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SNR_DB = 0
SR = 16000


# ============================================================
# Helper functions
# ============================================================

def align_signal(signal, target_length):
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) >= target_length:
        return signal[:target_length]

    return np.pad(
        signal,
        (0, target_length - len(signal)),
        mode="constant",
    )


def db_spectrogram(stft_matrix):
    magnitude = np.abs(stft_matrix)

    return librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
    )


def save_method_audio(name, signal):
    output_path = AUDIO_DIR / name
    save_audio(output_path, signal, SR)
    print(f"Saved audio: {output_path}")


def evaluate_method(clean, signal, method_name):
    metrics = evaluate_signal(clean, signal, SR)

    return {
        "Method": method_name,
        "STOI": metrics["STOI"],
        "PESQ": metrics["PESQ"],
        "SI_SDR_dB": metrics["SI-SDR"],
    }


# ============================================================
# Load clean speech and noise
# ============================================================

print("=" * 78)
print("GENERATING REPRESENTATIVE EXAMPLE")
print("=" * 78)

print(f"Clean speech: {CLEAN_FILE}")
print(f"Noise: {NOISE_FILE}")
print(f"Target SNR: {TARGET_SNR_DB} dB")

clean, sr_clean = load_audio(CLEAN_FILE, target_sr=SR)
noise, sr_noise = load_audio(NOISE_FILE, target_sr=SR)

if sr_clean != SR or sr_noise != SR:
    raise RuntimeError("Unexpected sampling rate.")

noise = match_length(noise, len(clean))

noisy, scaled_noise = mix_at_snr(
    clean,
    noise,
    TARGET_SNR_DB,
)


# ============================================================
# STFT
# ============================================================

clean_stft = compute_stft(clean)
noisy_stft = compute_stft(noisy)
noise_stft = compute_stft(scaled_noise)

clean_mag, clean_phase = get_magnitude_phase(clean_stft)
noisy_mag, noisy_phase = get_magnitude_phase(noisy_stft)


# ============================================================
# Spectral subtraction
# ============================================================

ss_mag = spectral_subtraction_magnitude(
    noisy_stft,
    noise_stft,
)

ss_noisy_phase = reconstruct_with_noisy_phase(
    ss_mag,
    noisy_stft,
    length=len(clean),
)

ss_oracle_phase = reconstruct_with_clean_phase(
    ss_mag,
    clean_stft,
    length=len(clean),
)


# ============================================================
# Wiener filtering
# ============================================================

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


# ============================================================
# MP-SENet output
# ============================================================

if not MPSENET_FILE.exists():
    raise FileNotFoundError(
        f"MP-SENet output not found:\n{MPSENET_FILE}"
    )

mpsenet, sr_mp = load_audio(
    MPSENET_FILE,
    target_sr=SR,
)

mpsenet = align_signal(
    mpsenet,
    len(clean),
)


# ============================================================
# Save representative audio
# ============================================================

save_method_audio("clean.wav", clean)
save_method_audio("noisy.wav", noisy)

save_method_audio(
    "ss_noisy_phase.wav",
    ss_noisy_phase,
)

save_method_audio(
    "ss_oracle_phase.wav",
    ss_oracle_phase,
)

save_method_audio(
    "wiener_noisy_phase.wav",
    wiener_noisy_phase,
)

save_method_audio(
    "wiener_oracle_phase.wav",
    wiener_oracle_phase,
)

save_method_audio(
    "mpsenet.wav",
    mpsenet,
)


# ============================================================
# Objective metrics
# ============================================================

metric_rows = []

metric_rows.append(
    evaluate_method(
        clean,
        noisy,
        "Noisy",
    )
)

metric_rows.append(
    evaluate_method(
        clean,
        ss_noisy_phase,
        "SS + noisy phase",
    )
)

metric_rows.append(
    evaluate_method(
        clean,
        ss_oracle_phase,
        "SS + clean oracle phase",
    )
)

metric_rows.append(
    evaluate_method(
        clean,
        wiener_noisy_phase,
        "Wiener + noisy phase",
    )
)

metric_rows.append(
    evaluate_method(
        clean,
        wiener_oracle_phase,
        "Wiener + clean oracle phase",
    )
)

metric_rows.append(
    evaluate_method(
        clean,
        mpsenet,
        "MP-SENet",
    )
)

metrics_df = pd.DataFrame(metric_rows)

metrics_path = OUTPUT_DIR / "metrics.csv"
metrics_df.to_csv(metrics_path, index=False)


# ============================================================
# Phase difference
# ============================================================

phase_difference = np.angle(
    np.exp(
        1j * (
            clean_phase
            - noisy_phase
        )
    )
)

absolute_phase_difference = np.abs(
    phase_difference
)


# ============================================================
# Figure data
# ============================================================

time_axis = np.arange(len(clean)) / SR

clean_db = db_spectrogram(clean_stft)
noisy_db = db_spectrogram(noisy_stft)

ss_db = librosa.amplitude_to_db(
    ss_mag,
    ref=np.max,
)

ss_noisy_stft = compute_stft(
    ss_noisy_phase
)

ss_oracle_stft = compute_stft(
    ss_oracle_phase
)

ss_noisy_db = db_spectrogram(
    ss_noisy_stft
)

ss_oracle_db = db_spectrogram(
    ss_oracle_stft
)


# ============================================================
# Create representative analysis figure
# ============================================================

fig, axes = plt.subplots(
    4,
    2,
    figsize=(14, 15),
)

# ------------------------------------------------------------
# 1. Clean waveform
# ------------------------------------------------------------

axes[0, 0].plot(
    time_axis,
    clean,
    linewidth=0.7,
)

axes[0, 0].set_title(
    "(a) Clean Speech Waveform"
)

axes[0, 0].set_xlabel(
    "Time (s)"
)

axes[0, 0].set_ylabel(
    "Amplitude"
)

axes[0, 0].grid(
    alpha=0.2
)


# ------------------------------------------------------------
# 2. Noisy waveform
# ------------------------------------------------------------

axes[0, 1].plot(
    time_axis,
    noisy,
    linewidth=0.7,
)

axes[0, 1].set_title(
    "(b) Noisy Speech Waveform — Cafe, 0 dB"
)

axes[0, 1].set_xlabel(
    "Time (s)"
)

axes[0, 1].set_ylabel(
    "Amplitude"
)

axes[0, 1].grid(
    alpha=0.2
)


# ------------------------------------------------------------
# 3. Clean spectrogram
# ------------------------------------------------------------

img = librosa.display.specshow(
    clean_db,
    sr=SR,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[1, 0],
)

axes[1, 0].set_title(
    "(c) Clean Speech Spectrogram"
)

fig.colorbar(
    img,
    ax=axes[1, 0],
    format="%+2.0f dB",
)


# ------------------------------------------------------------
# 4. Noisy spectrogram
# ------------------------------------------------------------

img = librosa.display.specshow(
    noisy_db,
    sr=SR,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[1, 1],
)

axes[1, 1].set_title(
    "(d) Noisy Speech Spectrogram"
)

fig.colorbar(
    img,
    ax=axes[1, 1],
    format="%+2.0f dB",
)


# ------------------------------------------------------------
# 5. Enhanced magnitude
# ------------------------------------------------------------

img = librosa.display.specshow(
    ss_db,
    sr=SR,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[2, 0],
)

axes[2, 0].set_title(
    "(e) Spectral Subtraction Enhanced Magnitude"
)

fig.colorbar(
    img,
    ax=axes[2, 0],
    format="%+2.0f dB",
)


# ------------------------------------------------------------
# 6. Phase difference
# ------------------------------------------------------------

img = librosa.display.specshow(
    absolute_phase_difference,
    sr=SR,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[2, 1],
)

axes[2, 1].set_title(
    "(f) |Clean Phase − Noisy Phase|"
)

fig.colorbar(
    img,
    ax=axes[2, 1],
    format="%.1f rad",
)


# ------------------------------------------------------------
# 7. Normal reconstruction
# ------------------------------------------------------------

img = librosa.display.specshow(
    ss_noisy_db,
    sr=SR,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[3, 0],
)

axes[3, 0].set_title(
    "(g) SS Reconstruction with Noisy Phase"
)

fig.colorbar(
    img,
    ax=axes[3, 0],
    format="%+2.0f dB",
)


# ------------------------------------------------------------
# 8. Oracle reconstruction
# ------------------------------------------------------------

img = librosa.display.specshow(
    ss_oracle_db,
    sr=SR,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[3, 1],
)

axes[3, 1].set_title(
    "(h) SS Reconstruction with Clean Oracle Phase"
)

fig.colorbar(
    img,
    ax=axes[3, 1],
    format="%+2.0f dB",
)


fig.suptitle(
    "Representative Example: p232_001 + Cafe Noise at 0 dB SNR",
    fontsize=16,
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.98]
)

figure_path = (
    OUTPUT_DIR
    / "representative_analysis.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 78)
print("REPRESENTATIVE EXAMPLE COMPLETE")
print("=" * 78)

print()
print("Condition:")
print("  Clean file: p232_001.wav")
print("  Noise: Cafe")
print("  Input SNR: 0 dB")

print()
print("Objective metrics:")
print(
    metrics_df.round(4).to_string(
        index=False
    )
)

print()
print("Files created:")
print(f"  {figure_path}")
print(f"  {metrics_path}")
print(f"  {AUDIO_DIR}")

print()
print("Audio files:")
for audio_file in sorted(
    AUDIO_DIR.glob("*.wav")
):
    print(
        f"  {audio_file.name}"
    )

print("=" * 78)