import numpy as np
from pystoi import stoi
from pesq import pesq


def align_signals(reference, estimate):
    """
    Trim two signals to the same length.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    length = min(len(reference), len(estimate))

    return reference[:length], estimate[:length]


def calculate_stoi(clean, processed, sample_rate=16000):
    """
    Calculate Short-Time Objective Intelligibility (STOI).

    Higher values indicate better speech intelligibility.
    """
    clean, processed = align_signals(clean, processed)

    score = stoi(
        clean,
        processed,
        sample_rate,
        extended=False
    )

    return float(score)


def calculate_pesq(clean, processed, sample_rate=16000):
    """
    Calculate wideband PESQ.

    PESQ evaluates perceptual speech quality.
    The project uses 16 kHz audio, so wideband mode is used.
    """
    clean, processed = align_signals(clean, processed)

    if sample_rate != 16000:
        raise ValueError(
            "This project uses 16 kHz audio for wideband PESQ."
        )

    score = pesq(
        sample_rate,
        clean,
        processed,
        "wb"
    )

    return float(score)


def calculate_si_sdr(clean, processed):
    """
    Calculate scale-invariant signal-to-distortion ratio (SI-SDR).

    Returns
    -------
    score : float
        SI-SDR in dB.
    """
    clean, processed = align_signals(clean, processed)

    clean = clean - np.mean(clean)
    processed = processed - np.mean(processed)

    clean_energy = np.sum(clean ** 2) + 1e-12

    scale = np.sum(processed * clean) / clean_energy

    target = scale * clean
    residual = processed - target

    target_energy = np.sum(target ** 2)
    residual_energy = np.sum(residual ** 2) + 1e-12

    score = 10.0 * np.log10(
        (target_energy + 1e-12) / residual_energy
    )

    return float(score)


def evaluate_signal(clean, processed, sample_rate=16000):
    """
    Evaluate one processed speech signal using all
    principal objective metrics.

    Returns
    -------
    results : dict
        STOI, PESQ and SI-SDR scores.
    """

    results = {
        "STOI": calculate_stoi(
            clean,
            processed,
            sample_rate
        ),

        "PESQ": calculate_pesq(
            clean,
            processed,
            sample_rate
        ),

        "SI-SDR": calculate_si_sdr(
            clean,
            processed
        )
    }

    return results


def calculate_oracle_phase_benefit(
    clean,
    noisy_phase_output,
    clean_phase_output,
    sample_rate=16000
):
    """
    Measure the improvement obtained by replacing noisy phase
    with oracle clean phase while keeping magnitude unchanged.

    Positive values indicate a benefit from oracle clean phase.
    """

    noisy_metrics = evaluate_signal(
        clean,
        noisy_phase_output,
        sample_rate
    )

    oracle_metrics = evaluate_signal(
        clean,
        clean_phase_output,
        sample_rate
    )

    benefit = {
        "Delta_STOI": (
            oracle_metrics["STOI"]
            - noisy_metrics["STOI"]
        ),

        "Delta_PESQ": (
            oracle_metrics["PESQ"]
            - noisy_metrics["PESQ"]
        ),

        "Delta_SI-SDR": (
            oracle_metrics["SI-SDR"]
            - noisy_metrics["SI-SDR"]
        )
    }

    return noisy_metrics, oracle_metrics, benefit
