import numpy as np


def estimate_noise_power(noise_stft):
    """
    Estimate the average noise power spectrum.

    Parameters
    ----------
    noise_stft : np.ndarray
        Complex STFT of a noise-only signal.

    Returns
    -------
    noise_power : np.ndarray
        Mean noise power for each frequency bin.
    """
    noise_power = np.mean(
        np.abs(noise_stft) ** 2,
        axis=1,
        keepdims=True
    )

    return noise_power


def wiener_magnitude(
    noisy_stft,
    noise_stft,
    gain_floor=0.02
):
    """
    Estimate an enhanced magnitude spectrum using Wiener filtering.

    The function returns magnitude only. Phase is intentionally
    excluded so that the same enhanced magnitude can later be
    reconstructed using either noisy phase or oracle clean phase.

    Parameters
    ----------
    noisy_stft : np.ndarray
        Complex STFT of noisy speech.

    noise_stft : np.ndarray
        Complex STFT of a noise-only signal.

    gain_floor : float
        Minimum Wiener gain used to avoid excessive attenuation.

    Returns
    -------
    enhanced_magnitude : np.ndarray
        Wiener-filtered magnitude spectrum.
    """

    noisy_power = np.abs(noisy_stft) ** 2

    noise_power = estimate_noise_power(noise_stft)

    estimated_speech_power = np.maximum(
        noisy_power - noise_power,
        0.0
    )

    wiener_gain = estimated_speech_power / (
        estimated_speech_power + noise_power + 1e-12
    )

    wiener_gain = np.maximum(
        wiener_gain,
        gain_floor
    )

    noisy_magnitude = np.abs(noisy_stft)

    enhanced_magnitude = wiener_gain * noisy_magnitude

    return enhanced_magnitude
