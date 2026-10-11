import numpy as np
import librosa


# STFT parameters required by the project
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512


def compute_stft(signal):
    """
    Compute the short-time Fourier transform (STFT).

    Parameters
    ----------
    signal : np.ndarray
        Time-domain audio signal.

    Returns
    -------
    stft_matrix : np.ndarray
        Complex-valued STFT matrix.
    """
    signal = np.asarray(signal, dtype=np.float32)

    stft_matrix = librosa.stft(
        signal,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    return stft_matrix


def compute_istft(stft_matrix, length=None):
    """
    Reconstruct a time-domain signal using inverse STFT.

    Parameters
    ----------
    stft_matrix : np.ndarray
        Complex-valued STFT matrix.
    length : int, optional
        Desired output signal length.

    Returns
    -------
    signal : np.ndarray
        Reconstructed time-domain signal.
    """
    signal = librosa.istft(
        stft_matrix,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        length=length
    )

    return signal.astype(np.float32)


def get_magnitude_phase(stft_matrix):
    """
    Separate a complex STFT into magnitude and phase.

    Returns
    -------
    magnitude : np.ndarray
        Magnitude spectrum.

    phase : np.ndarray
        Phase angle in radians.
    """
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    return magnitude, phase


def reconstruct_from_magnitude_phase(magnitude, phase, length=None):
    """
    Reconstruct a waveform from a magnitude spectrum
    and a specified phase.

    This function is used for both:
    1. ordinary noisy-phase reconstruction
    2. oracle clean-phase reconstruction
    """
    complex_stft = magnitude * np.exp(1j * phase)

    return compute_istft(complex_stft, length=length)


def phase_difference(phase_a, phase_b):
    """
    Calculate wrapped phase difference in the range [-pi, pi].
    """
    return np.angle(np.exp(1j * (phase_a - phase_b)))
