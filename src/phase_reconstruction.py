import numpy as np

from stft_utils import (
    compute_stft,
    get_magnitude_phase,
    reconstruct_from_magnitude_phase,
    phase_difference,
)


def reconstruct_with_noisy_phase(
    enhanced_magnitude,
    noisy_stft,
    length=None
):
    """
    Reconstruct enhanced speech using the phase
    of the noisy speech mixture.
    """

    _, noisy_phase = get_magnitude_phase(noisy_stft)

    return reconstruct_from_magnitude_phase(
        enhanced_magnitude,
        noisy_phase,
        length=length
    )


def reconstruct_with_clean_phase(
    enhanced_magnitude,
    clean_stft,
    length=None
):
    """
    Oracle experiment:
    reconstruct the SAME enhanced magnitude using
    the phase of the clean reference speech.

    Clean phase is used only for diagnostic analysis
    and is not available in a deployable system.
    """

    _, clean_phase = get_magnitude_phase(clean_stft)

    return reconstruct_from_magnitude_phase(
        enhanced_magnitude,
        clean_phase,
        length=length
    )


def calculate_phase_disagreement(
    clean_stft,
    noisy_stft,
    magnitude_weighted=True
):
    """
    Measure disagreement between clean and noisy phase.

    Parameters
    ----------
    clean_stft : np.ndarray
        STFT of clean reference speech.

    noisy_stft : np.ndarray
        STFT of noisy speech.

    magnitude_weighted : bool
        If True, weight phase disagreement by
        clean-speech magnitude.

    Returns
    -------
    disagreement : float
        Mean absolute wrapped phase difference in radians.
    """

    clean_magnitude, clean_phase = get_magnitude_phase(clean_stft)
    _, noisy_phase = get_magnitude_phase(noisy_stft)

    difference = np.abs(
        phase_difference(clean_phase, noisy_phase)
    )

    if magnitude_weighted:
        weights = clean_magnitude + 1e-12

        disagreement = np.sum(
            difference * weights
        ) / np.sum(weights)

    else:
        disagreement = np.mean(difference)

    return float(disagreement)


def prepare_phase_reconstructions(
    clean,
    noisy,
    enhanced_magnitude
):
    """
    Generate the two reconstruction conditions required
    for the oracle-phase experiment.

    Returns
    -------
    noisy_phase_output :
        Enhanced magnitude + noisy phase.

    clean_phase_output :
        Same enhanced magnitude + oracle clean phase.

    phase_disagreement :
        Magnitude-weighted clean/noisy phase disagreement.
    """

    clean_stft = compute_stft(clean)
    noisy_stft = compute_stft(noisy)

    noisy_phase_output = reconstruct_with_noisy_phase(
        enhanced_magnitude,
        noisy_stft,
        length=len(clean)
    )

    clean_phase_output = reconstruct_with_clean_phase(
        enhanced_magnitude,
        clean_stft,
        length=len(clean)
    )

    disagreement = calculate_phase_disagreement(
        clean_stft,
        noisy_stft
    )

    return (
        noisy_phase_output,
        clean_phase_output,
        disagreement
    )
