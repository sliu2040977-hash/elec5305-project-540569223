# Experimental Results

This directory stores outputs from the controlled speech-enhancement experiments.

The preliminary hub experiment evaluates one clean VoiceBank utterance mixed with café noise at 10, 5, and 0 dB SNR.

For each SNR condition, the experiment will generate:

1. clean reference;
2. noisy mixture;
3. spectral subtraction with noisy phase;
4. spectral subtraction with oracle clean phase;
5. Wiener filtering with noisy phase;
6. Wiener filtering with oracle clean phase.

Evaluation metrics:

- STOI
- PESQ
- SI-SDR

Spectrograms and phase-related analysis are also generated for representative conditions.
