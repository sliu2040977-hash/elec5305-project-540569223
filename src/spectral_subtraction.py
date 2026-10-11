import numpy as np


def estimate_noise_magnitude(noise_stft):
    """
    Estimate the average noise magnitude spectrum.

    Parameters
    ----------
    noise_stft : np.ndarray
        Complex STFT of a noise-only signal.

    Returns
    -------
    noise_magnitude : np.ndarray
        Mean noise magnitude for each frequency bin.
    """
    noise_magnitude = np.mean(
        np.abs(noise_stft),
        axis=1,
        keepdims=True
    )

    return noise_magnitude


def spectral_subtraction_magnitude(
    noisy_stft,
    noise_stft,
    alpha=1.0,
    floor_factor=0.02
):
    """
    Perform magnitude-domain spectral subtraction.

    The function estimates an enhanced magnitude spectrum only.
    Phase is intentionally not applied here so that the same
    magnitude estimate can later be reconstructed using either
    noisy phase or oracle clean phase.

    Parameters
    ----------
    noisy_stft : np.ndarray
        Complex STFT of noisy speech.

    noise_stft : np.ndarray
        Complex STFT of a noise-only signal.

    alpha : float
        Spectral subtraction factor.

    floor_factor : float
        Spectral floor used to reduce excessive subtraction
        and musical-noise artefacts.

    Returns
    -------
    enhanced_magnitude : np.ndarray
        Estimated clean-speech magnitude spectrum.
    """

    noisy_magnitude = np.abs(noisy_stft)

    noise_magnitude = estimate_noise_magnitude(noise_stft)

    subtracted = noisy_magnitude - alpha * noise_magnitude

    spectral_floor = floor_factor * noisy_magnitude

    enhanced_magnitude = np.maximum(
        subtracted,
        spectral_floor
    )

    return enhanced_magnitude
