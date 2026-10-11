import numpy as np

def calculate_power(signal):

    """

    Calculate the average power of a signal.

    """

    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) == 0:

        raise ValueError("Signal is empty.")

    return np.mean(signal ** 2)

def mix_at_snr(clean, noise, snr_db):

    """

    Mix clean speech with noise at a specified SNR.

    Parameters

    ----------

    clean : np.ndarray

        Clean speech waveform.

    noise : np.ndarray

        Noise waveform with the same length as clean speech.

    snr_db : float

        Desired signal-to-noise ratio in dB.

    Returns

    -------

    noisy : np.ndarray

        Speech and scaled noise mixture.

    scaled_noise : np.ndarray

        Noise after scaling to the requested SNR.

    """

    clean = np.asarray(clean, dtype=np.float64)

    noise = np.asarray(noise, dtype=np.float64)

    if len(clean) != len(noise):

        raise ValueError(

            "Clean speech and noise must have the same length."

        )

    clean_power = calculate_power(clean)

    noise_power = calculate_power(noise)

    if clean_power < 1e-12:

        raise ValueError("Clean speech has zero or negligible power.")

    if noise_power < 1e-12:

        raise ValueError("Noise has zero or negligible power.")

    target_noise_power = clean_power / (10 ** (snr_db / 10.0))

    scale = np.sqrt(target_noise_power / noise_power)

    scaled_noise = noise * scale

    noisy = clean + scaled_noise

    return noisy.astype(np.float32), scaled_noise.astype(np.float32)

def measure_snr(clean, noise):

    """

    Measure the SNR between clean speech and noise.

    Returns

    -------

    snr_db : float

        Measured SNR in dB.

    """

    clean_power = calculate_power(clean)

    noise_power = calculate_power(noise)

    if noise_power < 1e-12:

        return np.inf

    return 10 * np.log10(clean_power / noise_power)
