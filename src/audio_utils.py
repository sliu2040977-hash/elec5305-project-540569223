import numpy as np

import librosa

import soundfile as sf

def load_audio(file_path, target_sr=16000):

    """

    Load an audio file as mono and resample it to the target sampling rate.

    Parameters

    ----------

    file_path : str

        Path to the input audio file.

    target_sr : int

        Target sampling rate in Hz.

    Returns

    -------

    audio : np.ndarray

        Mono audio waveform.

    sr : int

        Sampling rate.

    """

    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

    return audio.astype(np.float32), sr

def normalize_audio(audio, peak=0.99):

    """

    Peak-normalize an audio signal while avoiding clipping.

    """

    audio = np.asarray(audio, dtype=np.float32)

    max_value = np.max(np.abs(audio))

    if max_value < 1e-12:

        return audio.copy()

    return audio * (peak / max_value)

def match_length(signal, target_length):

    """

    Trim or repeat a signal so that it has the required number of samples.

    """

    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) == 0:

        raise ValueError("Input signal is empty.")

    if len(signal) >= target_length:

        return signal[:target_length]

    repetitions = int(np.ceil(target_length / len(signal)))

    repeated = np.tile(signal, repetitions)

    return repeated[:target_length]

def save_audio(file_path, audio, sr=16000):

    """

    Save an audio waveform to a WAV file.

    """

    sf.write(file_path, audio, sr)
