# elec5305-project-540569223

## 1.Project Title
Comparative Evaluation of STFT-Based Spectral Subtraction and Wiener Filtering for Speech Enhancement

## 2. Student Information

| Field | Details |
|---|---|
| Full Name | Suxin Liu |
| Student ID | 540569223 |
| GitHub Username | sliu2040977-hash |
| Repository | [elec5305-project-540569223](https://github.com/sliu2040977-hash/elec5305-project-540569223) |
| Project Site | [GitHub Pages](https://sliu2040977-hash.github.io/elec5305-project-540569223/) |

## 3. Project Overview

Environmental noise can reduce the quality and intelligibility of recorded speech. Enhancement is therefore important for communication, voice interfaces, hearing assistance, and speech recognition. Recent Deep Noise Suppression challenges confirm that producing intelligible, natural speech under changing noise remains difficult [1].

This project will implement and compare two classical single-channel enhancement methods: **spectral subtraction** and **Wiener filtering**. Both operate in the short-time Fourier transform (STFT) domain and require substantially less computation than modern trained systems. Clean speech will be mixed with stationary and non-stationary noise at controlled signal-to-noise ratios (SNRs). Each algorithm will suppress the estimated noise spectrum before reconstructing the enhanced waveform through the inverse STFT (ISTFT).

The research question is:

> How do spectral subtraction and Wiener filtering differ in noise reduction, speech intelligibility, and speech distortion under stationary and non-stationary noise?

Objective metrics, spectrograms, and listening examples will reveal how algorithm choice, noise type, and input SNR affect the balance between noise attenuation and speech preservation.

## 4. Background and Motivation

Spectral subtraction estimates the noise magnitude spectrum during noise-only periods and subtracts it from each noisy speech frame. It is efficient and interpretable, but estimation errors can leave isolated peaks perceived as musical noise. Wiener filtering instead calculates a frequency-dependent gain from estimated speech and noise power. Noise-dominated bins are attenuated while speech-dominated bins are preserved. It often produces smoother output, although strong attenuation can distort speech.

Recent systems demonstrate major advances: FullSubNet combines full-band and sub-band information [2], MetricGAN+ targets perceptual quality [3], DeepFilterNet uses low-complexity deep filtering [4], and diffusion models demonstrate cross-dataset generalisation [5]. Reproducing them, however, requires large datasets, training, and substantial computing resources.

A controlled comparison of two transparent classical methods is feasible and directly demonstrates ELEC5305 concepts: framing, windowing, STFT, spectral estimation, filtering, phase reuse, ISTFT, and overlap-add. Recent research will guide the evaluation while the classical methods provide reproducible baselines.

## 5. Proposed Methodology

### Tools and Data

The project will use **Python**, NumPy, SciPy, librosa, Matplotlib, soundfile, and pystoi. A Jupyter Notebook and documented functions will provide a reproducible workflow.

Five clean VoiceBank utterances will be combined with café and traffic noise from DEMAND and white noise generated in Python. VoiceBank-DEMAND remains widely used in recent enhancement research [3]. Signals will be converted to mono at 16 kHz and normalised.

Each utterance will be mixed with the three noise types at **0 dB, 5 dB, and 10 dB input SNR**. A short noise-only segment will provide a consistent noise estimate. Automated loops will generate all mixtures and enhanced outputs.

### Signal Processing

Signals will be divided into overlapping frames using a 512-sample Hann window, 256-sample hop, and 512-point FFT. At 16 kHz, this gives a 32 ms frame and 50% overlap.

Spectral subtraction will subtract the estimated noise magnitude from the noisy magnitude spectrum. Half-wave rectification and a small spectral floor will prevent negative values and limit artefacts. Wiener filtering will calculate a frequency-dependent gain from the estimated noise power and apply it to the complex noisy STFT. Both algorithms will reuse the noisy phase and reconstruct audio through ISTFT and overlap-add. Identical inputs, noise estimates, and STFT settings will ensure a fair comparison.

### Evaluation

Performance will be evaluated against the clean references using:

- **Output SNR and SNR improvement** for noise reduction;
- **Short-Time Objective Intelligibility (STOI)** for intelligibility preservation;
- spectrograms for residual noise and speech distortion;
- listening examples of clean, noisy, and enhanced speech.

Tables and grouped bar charts will compare algorithms across noise types and input SNRs. Representative four-panel spectrograms will show clean, noisy, spectral-subtraction, and Wiener-filter outputs. The discussion will identify whether improved numerical SNR also corresponds to improved intelligibility and perceived quality [1], [3].

## 6. Expected Outcomes

The project will deliver a reproducible prototype implementing both algorithms. Both methods are expected to improve some noisy conditions, with more stable performance for stationary noise than rapidly changing environmental noise. Spectral subtraction may provide stronger attenuation but introduce musical-noise artefacts, while Wiener filtering may sound smoother but cause some speech distortion. The experiment will quantify these trade-offs instead of assuming one method is always superior.

Deliverables will include code, audio examples, spectrograms, metric tables, charts, the proposal PDF, and a GitHub Pages demonstration. Sensitivity to noise estimation will also be reported.

## 8. References

1. C. K. A. Reddy et al., “ICASSP 2021 Deep Noise Suppression Challenge,” *ICASSP*, pp. 6623–6627, 2021. https://doi.org/10.1109/ICASSP39728.2021.9415105

2. X. Hao et al., “FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement,” *ICASSP*, pp. 6633–6637, 2021. https://doi.org/10.1109/ICASSP39728.2021.9414177

3. S.-W. Fu et al., “MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement,” *Interspeech*, pp. 201–205, 2021. https://doi.org/10.21437/Interspeech.2021-599

4. H. Schröter et al., “DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio Based on Deep Filtering,” *ICASSP*, pp. 7407–7411, 2022. https://doi.org/10.1109/ICASSP43922.2022.9747055

5. J. Richter et al., “Speech Enhancement and Dereverberation With Diffusion-Based Generative Models,” *IEEE/ACM TASLP*, vol. 31, pp. 2351–2364, 2023. https://doi.org/10.1109/TASLP.2023.3285241
