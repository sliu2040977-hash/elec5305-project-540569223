# elec5305-project-540569223

## 1. Project Title

**Investigating Phase Limitations in STFT-Based Speech Enhancement Using Oracle-Phase Analysis**

## 2. Student Information

- **Full Name:** Suxin Liu
- **Student ID (SID):** 540569223
- **GitHub Username:** sliu2040977-hash
- **GitHub Project Repository:** https://github.com/sliu2040977-hash/elec5305-project-540569223

## 3. Project Overview

Single-channel speech enhancement aims to recover intelligible and high-quality speech from recordings corrupted by environmental noise. Many classical STFT-based enhancement methods estimate an improved magnitude spectrum but reconstruct the signal using the phase of the noisy mixture.

This project investigates when this noisy-phase assumption becomes a significant limitation.

Spectral subtraction and Wiener filtering are retained as classical enhancement baselines. Rather than treating their direct performance comparison as the main research question, the project uses them to investigate the separate effects of magnitude estimation and phase information.

For each classical method, two reconstructions will be evaluated:

1. the estimated enhanced magnitude combined with the original noisy phase;
2. the same estimated enhanced magnitude combined with the oracle clean-speech phase.

Because the magnitude estimate remains unchanged, the difference between these two conditions isolates the potential improvement associated with phase information.

The main research question is:

> **For classical STFT-based speech enhancement, when does reuse of the noisy phase become a significant limitation, how much improvement is achievable if the clean phase is known, and how does this compare with a modern phase-aware pretrained enhancer?**

A pretrained **MP-SENet** model will be used as a contemporary phase-aware reference. The project will not train a neural network.

## 4. Background and Motivation

A noisy speech signal can be represented in the STFT domain as

$begin:math:display$
X\(k\,m\)\=\|X\(k\,m\)\|e\^\{j\\phi\_X\(k\,m\)\}\.
$end:math:display$

Classical spectral enhancement methods generally estimate a clean-speech magnitude

$begin:math:display$
\|\\hat\{S\}\(k\,m\)\|
$end:math:display$

and reconstruct the enhanced spectrum using the noisy mixture phase:

$begin:math:display$
\\hat\{S\}\(k\,m\)\=\|\\hat\{S\}\(k\,m\)\|e\^\{j\\phi\_X\(k\,m\)\}\.
$end:math:display$

Therefore, even if the magnitude estimate improves substantially, reconstruction may remain limited by inaccurate phase information.

The importance of phase in speech enhancement is already established in previous research, particularly under difficult low-SNR conditions. Therefore, this project does not attempt to demonstrate that phase matters in general. Instead, it measures the size of the phase-related limitation for the particular classical methods, SNRs, and noise structures studied here.

Spectral subtraction and Wiener filtering provide transparent classical baselines for this investigation. Both use STFT-domain processing and normally retain the noisy phase, making them suitable for controlled oracle-phase experiments.

The project also connects these classical methods with modern phase-aware enhancement. MP-SENet explicitly estimates magnitude and phase spectra and will be used as a pretrained reference system. DCCRN will be included as relevant background literature on complex-domain and phase-aware speech enhancement.

## 5. Research Questions

### Primary Research Question

> When does reuse of the noisy phase become a significant limitation in STFT-based speech enhancement, and how much of the remaining enhancement error is associated with magnitude estimation versus phase error?

### Secondary Research Question

> How does the performance of classical noisy-phase and oracle-phase enhancement compare with a modern pretrained phase-aware speech-enhancement model?

The investigation will also examine:

- whether the oracle-phase benefit changes with input SNR;
- whether phase limitations differ across white, café, and traffic noise;
- whether measured phase disagreement predicts the improvement obtained from oracle clean phase.

## 6. Dataset and Experimental Conditions

The experiments will use clean speech from **VoiceBank** and environmental noise from **DEMAND**.

For controlled experiments, selected clean VoiceBank utterances will be mixed with:

- white noise;
- café noise;
- traffic noise.

The primary input SNR conditions will be:

- 0 dB;
- 5 dB;
- 10 dB.

A -5 dB condition may be added if 0 dB does not provide a sufficiently difficult regime.

The full experiment will use approximately **30–50 clean utterances from several speakers**. Using 40 utterances, three noise types, and three SNR conditions would produce:

$begin:math:display$
40 \\times 3 \\times 3 \= 360
$end:math:display$

controlled noisy mixtures.

All signals will be converted to mono at 16 kHz and normalised consistently.

## 7. STFT Configuration

The same signal-processing configuration will be used for all classical experiments:

- sampling rate: 16 kHz;
- frame length: 512 samples;
- Hann analysis window;
- hop length: 256 samples;
- overlap: 50%;
- FFT size: 512.

The same noisy signals, STFT parameters, and noise estimates will be used for spectral subtraction and Wiener filtering to maintain a controlled comparison.

## 8. Classical Enhancement Baselines

### 8.1 Spectral Subtraction

Spectral subtraction estimates the noise magnitude spectrum and subtracts it from the noisy magnitude spectrum. A spectral floor will be used to prevent negative magnitudes and reduce numerical problems.

The method produces an estimated magnitude:

$begin:math:display$
\|\\hat\{S\}\_\{SS\}\(k\,m\)\|\.
$end:math:display$

### 8.2 Wiener Filtering

Wiener filtering estimates a frequency-dependent gain from speech and noise power estimates and applies the gain to the noisy spectrum.

The method produces an estimated magnitude:

$begin:math:display$
\|\\hat\{S\}\_\{W\}\(k\,m\)\|\.
$end:math:display$

Spectral subtraction and Wiener filtering are used as **classical baselines**, rather than as the final research comparison.

## 9. Oracle-Phase Experiments

The clean reference signal has STFT

$begin:math:display$
S\(k\,m\)\=\|S\(k\,m\)\|e\^\{j\\phi\_S\(k\,m\)\}\.
$end:math:display$

The noisy signal has STFT

$begin:math:display$
X\(k\,m\)\=\|X\(k\,m\)\|e\^\{j\\phi\_X\(k\,m\)\}\.
$end:math:display$

### 9.1 Perfect-Magnitude / Noisy-Phase Experiment

The first diagnostic oracle experiment reconstructs:

$begin:math:display$
S\_\{\\text\{oracle\-mag\}\}\(k\,m\)
\=
\|S\(k\,m\)\|e\^\{j\\phi\_X\(k\,m\)\}\.
$end:math:display$

This condition uses the **perfect clean-speech magnitude** while retaining the **noisy mixture phase**.

Comparing this reconstruction with the true clean signal

$begin:math:display$
S\(k\,m\)\=\|S\(k\,m\)\|e\^\{j\\phi\_S\(k\,m\)\}
$end:math:display$

isolates the limitation caused by using noisy phase even when magnitude estimation is perfect.

### 9.2 Spectral Subtraction Phase Experiment

Normal spectral subtraction:

$begin:math:display$
\\hat\{S\}\_\{SS\,\\text\{noisy\-phase\}\}
\=
\|\\hat\{S\}\_\{SS\}\|e\^\{j\\phi\_X\}\.
$end:math:display$

Oracle-phase spectral subtraction:

$begin:math:display$
\\hat\{S\}\_\{SS\,\\text\{clean\-phase\}\}
\=
\|\\hat\{S\}\_\{SS\}\|e\^\{j\\phi\_S\}\.
$end:math:display$

The magnitude estimate is identical in both cases. Only the phase changes.

### 9.3 Wiener Phase Experiment

Normal Wiener reconstruction:

$begin:math:display$
\\hat\{S\}\_\{W\,\\text\{noisy\-phase\}\}
\=
\|\\hat\{S\}\_\{W\}\|e\^\{j\\phi\_X\}\.
$end:math:display$

Oracle-phase Wiener reconstruction:

$begin:math:display$
\\hat\{S\}\_\{W\,\\text\{clean\-phase\}\}
\=
\|\\hat\{S\}\_\{W\}\|e\^\{j\\phi\_S\}\.
$end:math:display$

These clean-phase versions are diagnostic experimental controls, not practical deployable enhancement systems.

## 10. Separating Magnitude and Phase Limitations

The oracle experiments allow the enhancement error to be investigated in terms of two components:

**magnitude-estimation error + phase-related error**

For an objective metric $begin:math:text$Q$end:math:text$, the oracle-phase benefit will be defined as

$begin:math:display$
\\Delta Q\_\{\\phi\}
\=
Q\_\{\\text\{clean\-phase\}\}
\-
Q\_\{\\text\{noisy\-phase\}\}\.
$end:math:display$

For example:

$begin:math:display$
\\Delta PESQ\_\{\\phi\}
\=
PESQ\_\{\\text\{clean\-phase\}\}
\-
PESQ\_\{\\text\{noisy\-phase\}\}\.
$end:math:display$

A small oracle-phase benefit indicates that phase is not the dominant limitation under that condition.

A large oracle-phase benefit indicates that substantial improvement remains possible if more accurate phase information is available.

## 11. Phase-Disagreement Measurement

The phase difference between clean and noisy speech will be calculated as

$begin:math:display$
\\Delta\\phi\(k\,m\)
\=
\\operatorname\{wrap\}
\\left\(
\\phi\_S\(k\,m\)\-\\phi\_X\(k\,m\)
\\right\)\.
$end:math:display$

An energy-weighted phase-disagreement measure will be calculated as

$begin:math:display$
D\_\{\\phi\}
\=
\\frac\{
\\sum\_\{k\,m\}\|S\(k\,m\)\|\^2
\\left\[1\-\\cos\\Delta\\phi\(k\,m\)\\right\]
\}\{
\\sum\_\{k\,m\}\|S\(k\,m\)\|\^2\+\\epsilon
\}\.
$end:math:display$

The analysis will test whether larger phase disagreement corresponds to a larger oracle-phase improvement.

A key analysis will therefore compare:

$begin:math:display$
D\_\{\\phi\}
\\quad \\text\{versus\} \\quad
\\Delta PESQ\_\{\\phi\}
$end:math:display$

and equivalent changes in other objective metrics.

## 12. Objective Evaluation

The principal evaluation metrics will be:

- **STOI** — speech intelligibility;
- **PESQ** — perceptual speech quality;
- **SI-SDR or output SNR** — signal reconstruction performance;
- **phase disagreement $begin:math:text$D\_\{\\phi\}$end:math:text$** — difference between clean and noisy phase.

For each method and experimental condition, the evaluation will distinguish between noisy-phase and oracle clean-phase reconstruction.

A typical result table will contain:

| Condition | Phase Used | STOI | PESQ | SI-SDR |
|---|---|---:|---:|---:|
| Noisy input | noisy | TBD | TBD | TBD |
| Spectral subtraction | noisy | TBD | TBD | TBD |
| Spectral subtraction | clean oracle | TBD | TBD | TBD |
| Wiener filtering | noisy | TBD | TBD | TBD |
| Wiener filtering | clean oracle | TBD | TBD | TBD |
| Perfect magnitude | noisy | TBD | TBD | TBD |
| MP-SENet | estimated phase | TBD | TBD | TBD |
| Clean reference | clean | TBD | TBD | TBD |

The `TBD` values will be replaced by measured experimental results.

## 13. Analysis Across SNR

The oracle-phase benefit will be evaluated at 0, 5, and 10 dB input SNR.

For each SNR:

$begin:math:display$
\\Delta Q\_\{\\phi\}
\=
Q\_\{\\text\{clean\-phase\}\}
\-
Q\_\{\\text\{noisy\-phase\}\}\.
$end:math:display$

The project will plot oracle-phase benefit against input SNR to determine whether the importance of accurate phase increases as the acoustic mixture becomes more difficult.

Separate curves will be produced for the selected noise types and enhancement methods.

## 14. Analysis Across Noise Types

Three different noise structures will be studied:

- white noise;
- café noise;
- traffic noise.

These noises differ in temporal variation, frequency structure, and stationarity.

The experiments will determine whether the phase-related limitation is similar across interference types or whether some noise structures produce substantially greater phase disagreement and oracle-phase benefit.

No outcome will be assumed in advance.

## 15. Modern Phase-Aware Reference: MP-SENet

A pretrained **MP-SENet (Magnitude-and-Phase Speech Enhancement Network)** will be used as a contemporary reference.

MP-SENet estimates magnitude and phase information in parallel. The pretrained VoiceBank+DEMAND checkpoint will be used for inference on the same noisy examples evaluated by the classical methods.

The model will **not be trained or tuned** as part of this project.

The purpose of MP-SENet is to determine how a practical modern phase-aware system compares with:

- classical noisy-phase enhancement;
- classical oracle-phase enhancement;
- the theoretical improvement indicated by the oracle experiments.

DCCRN will be discussed as an additional background example of complex-domain phase-aware speech enhancement, but implementation of both neural models is not required.

## 16. Preliminary Experiment

The first implementation milestone will use **one clean VoiceBank utterance** and **café noise**.

Three mixtures will be generated:

- 10 dB;
- 5 dB;
- 0 dB.

For each mixture, the following will be produced:

1. clean spectrogram;
2. noisy spectrogram;
3. spectral-subtraction output;
4. Wiener-filter output;
5. spectral subtraction with oracle clean phase;
6. Wiener filtering with oracle clean phase.

The following conditions will then be evaluated:

| Condition | STOI | PESQ | SI-SDR |
|---|---:|---:|---:|
| Noisy | TBD | TBD | TBD |
| SS + noisy phase | TBD | TBD | TBD |
| SS + clean oracle phase | TBD | TBD | TBD |
| Wiener + noisy phase | TBD | TBD | TBD |
| Wiener + clean oracle phase | TBD | TBD | TBD |

This preliminary experiment will verify that the central experimental pipeline works before scaling to the full dataset.

## 17. Visual and Acoustic Analysis

Representative signals will be analysed using:

- clean and noisy waveforms;
- clean and noisy spectrograms;
- enhanced magnitude spectra;
- phase-difference visualisation;
- normal noisy-phase reconstructions;
- oracle clean-phase reconstructions.

Representative audio examples will include:

- noisy speech;
- spectral subtraction;
- oracle-phase spectral subtraction;
- Wiener filtering;
- oracle-phase Wiener filtering;
- MP-SENet.

The acoustic discussion will consider audible effects including residual background noise, musical noise, speech attenuation, roughness, and phasiness.

## 18. Planned Key Figures

One principal figure will show **oracle-phase benefit versus input SNR**.

For example:

$begin:math:display$
\\Delta PESQ\_\{\\phi\}
\=
PESQ\_\{\\text\{clean\-phase\}\}
\-
PESQ\_\{\\text\{noisy\-phase\}\}\.
$end:math:display$

Separate curves will be shown for white, café, and traffic noise.

A second analysis will plot:

$begin:math:display$
D\_\{\\phi\}
\\quad \\text\{versus\} \\quad
\\Delta PESQ\_\{\\phi\}\.
$end:math:display$

This will test whether measured phase disagreement predicts the potential benefit of improved phase information.

## 19. Implementation Scope

The project implementation will focus on:

- audio loading and preprocessing;
- controlled SNR mixing;
- STFT and ISTFT;
- spectral subtraction;
- Wiener filtering;
- magnitude/phase decomposition;
- oracle clean-phase reconstruction;
- phase-disagreement measurement;
- STOI, PESQ and SI-SDR evaluation;
- pretrained MP-SENet inference;
- quantitative and acoustic interpretation.

The project will **not train a neural network** and will not add unnecessary additional enhancement algorithms.

## 20. Repository Structure

The planned repository structure is:

```text
elec5305-project-540569223/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── audio_utils.py
│   ├── mixing.py
│   ├── stft_utils.py
│   ├── spectral_subtraction.py
│   ├── wiener_filter.py
│   ├── oracle_phase.py
│   ├── metrics.py
│   └── phase_metrics.py
│
├── experiments/
│   └── preliminary_experiment.py
│
├── results/
│   ├── figures/
│   └── tables/
│
└── audio_examples/
```

Dataset files and large pretrained model files will not be committed unnecessarily. Instructions for obtaining the required datasets and pretrained model will be documented.

## 21. Resources

The project will use the following resources:

- VoiceBank+DEMAND speech-enhancement dataset;
- DEMAND environmental-noise database;
- MP-SENet official implementation and pretrained checkpoint;
- DCCRN / DeepComplexCRN as phase-aware background literature;
- `pystoi` for STOI evaluation;
- Python PESQ implementation for perceptual quality evaluation.

## 22. Literature Review Direction

The final literature review will contain approximately 8–10 relevant sources covering:

- spectral subtraction;
- Wiener filtering;
- STFT-based speech enhancement;
- magnitude and phase in speech enhancement;
- oracle-phase experiments;
- complex ratio masks and complex-domain enhancement;
- DCCRN;
- MP-SENet;
- VoiceBank+DEMAND;
- STOI and PESQ evaluation.

The literature review will describe the progression from magnitude-only enhancement, through recognition of phase limitations, to modern complex-domain and phase-aware enhancement.

## 23. Expected Project Outcome

The project will quantify when noisy-phase reuse limits classical STFT-based speech enhancement rather than merely comparing two established classical algorithms.

The final analysis will determine:

1. the improvement available when noisy phase is replaced by oracle clean phase;
2. how this improvement changes with SNR;
3. how it changes across different noise structures;
4. whether phase disagreement predicts oracle-phase benefit;
5. how a pretrained phase-aware MP-SENet system compares with the classical baselines and their oracle-phase controls.

The final repository will contain the source code, reproducible experiment instructions, objective results, plots, and representative audio examples.

## 24. References

1. K. K. Paliwal, K. Wójcicki, and B. Shannon, “The importance of phase in speech enhancement,” *Speech Communication*, vol. 53, no. 4, pp. 465–494, 2011.

2. S. Boll, “Suppression of acoustic noise in speech using spectral subtraction,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 27, no. 2, pp. 113–120, 1979.

3. C. H. You, S. N. Koh, and S. Rahardja, “Beta-order MMSE spectral amplitude estimation for speech enhancement,” *IEEE Transactions on Speech and Audio Processing*, vol. 13, no. 4, pp. 475–486, 2005.

4. C. K. A. Reddy et al., “ICASSP 2021 Deep Noise Suppression Challenge,” *ICASSP*, pp. 6623–6627, 2021.

5. X. Hao et al., “FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement,” *ICASSP*, pp. 6633–6637, 2021.

6. S.-W. Fu et al., “MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement,” *Interspeech*, pp. 201–205, 2021.

7. H. Schröter et al., “DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio Based on Deep Filtering,” *ICASSP*, pp. 7407–7411, 2022.

8. J. Richter et al., “Speech Enhancement and Dereverberation With Diffusion-Based Generative Models,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 31, pp. 2351–2364, 2023.

Further references on MP-SENet, DCCRN, VoiceBank+DEMAND, STOI, PESQ, and complex-domain enhancement will be incorporated as implementation progresses.
