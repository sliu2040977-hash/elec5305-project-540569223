from pathlib import Path
import csv

import numpy as np

from src.audio_utils import load_audio, match_length, save_audio
from src.mixing import mix_at_snr, measure_snr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
NOISE_DIR = PROJECT_ROOT / "data" / "noise"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "mpsenet"
INPUT_DIR = OUTPUT_ROOT / "input_noisy"
MANIFEST_PATH = OUTPUT_ROOT / "input_manifest.csv"

SAMPLE_RATE = 16000
SNR_LEVELS = [10, 5, 0]

WHITE_NOISE_SEED = 5305


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(CLEAN_DIR.glob("*.wav"))

    if len(clean_files) == 0:
        raise RuntimeError(
            f"No clean WAV files found in {CLEAN_DIR}"
        )

    print(f"Clean utterances found: {len(clean_files)}")

    # --------------------------------------------------------
    # Load real noise recordings
    # --------------------------------------------------------

    cafe_path = NOISE_DIR / "cafe.wav"
    traffic_path = NOISE_DIR / "traff.wav"

    if not cafe_path.exists():
        raise FileNotFoundError(
            f"Cafe noise not found: {cafe_path}"
        )

    if not traffic_path.exists():
        raise FileNotFoundError(
            f"Traffic noise not found: {traffic_path}"
        )

    cafe_noise, cafe_sr = load_audio(
        cafe_path,
        target_sr=SAMPLE_RATE,
    )

    traffic_noise, traffic_sr = load_audio(
        traffic_path,
        target_sr=SAMPLE_RATE,
    )

    if cafe_sr != SAMPLE_RATE:
        raise RuntimeError("Cafe noise sample-rate error.")

    if traffic_sr != SAMPLE_RATE:
        raise RuntimeError("Traffic noise sample-rate error.")

    noise_sources = {
        "Cafe": cafe_noise,
        "Traffic": traffic_noise,
    }

    # IMPORTANT:
    # Same deterministic seed used by full_experiment.py.
    rng = np.random.default_rng(WHITE_NOISE_SEED)

    manifest_rows = []

    total_expected = (
        len(clean_files)
        * 3
        * len(SNR_LEVELS)
    )

    counter = 0

    # --------------------------------------------------------
    # Reproduce the same experiment order:
    #
    # clean file
    #   -> Cafe
    #   -> Traffic
    #   -> White
    #       -> 10, 5, 0 dB
    # --------------------------------------------------------

    for clean_path in clean_files:

        clean, sr = load_audio(
            clean_path,
            target_sr=SAMPLE_RATE,
        )

        if sr != SAMPLE_RATE:
            raise RuntimeError(
                f"Unexpected sample rate for {clean_path.name}"
            )

        clean_length = len(clean)

        speaker = clean_path.stem.split("_")[0]

        # ====================================================
        # Cafe and Traffic
        # ====================================================

        for noise_type in ["Cafe", "Traffic"]:

            base_noise = noise_sources[noise_type]

            noise = match_length(
                base_noise,
                clean_length,
            )

            for snr_db in SNR_LEVELS:

                noisy, scaled_noise = mix_at_snr(
                    clean,
                    noise,
                    snr_db,
                )

                measured_snr = measure_snr(
                    clean,
                    scaled_noise,
                )

                filename = (
                    f"{clean_path.stem}"
                    f"__{noise_type.lower()}"
                    f"__snr{snr_db}db.wav"
                )

                output_path = INPUT_DIR / filename

                save_audio(
                    output_path,
                    noisy,
                    SAMPLE_RATE,
                )

                counter += 1

                manifest_rows.append(
                    {
                        "Input_File": filename,
                        "Clean_File": clean_path.name,
                        "Speaker": speaker,
                        "Noise_Type": noise_type,
                        "Target_SNR_dB": snr_db,
                        "Measured_Input_SNR_dB": measured_snr,
                    }
                )

                print(
                    f"[{counter}/{total_expected}] "
                    f"{clean_path.name} | "
                    f"{noise_type} | "
                    f"{snr_db} dB"
                )

        # ====================================================
        # White noise
        #
        # Generate once per clean utterance, exactly as the
        # controlled full experiment does.
        # ====================================================

        white_noise = rng.standard_normal(
            clean_length
        ).astype(np.float32)

        for snr_db in SNR_LEVELS:

            noisy, scaled_noise = mix_at_snr(
                clean,
                white_noise,
                snr_db,
            )

            measured_snr = measure_snr(
                clean,
                scaled_noise,
            )

            filename = (
                f"{clean_path.stem}"
                f"__white"
                f"__snr{snr_db}db.wav"
            )

            output_path = INPUT_DIR / filename

            save_audio(
                output_path,
                noisy,
                SAMPLE_RATE,
            )

            counter += 1

            manifest_rows.append(
                {
                    "Input_File": filename,
                    "Clean_File": clean_path.name,
                    "Speaker": speaker,
                    "Noise_Type": "White",
                    "Target_SNR_dB": snr_db,
                    "Measured_Input_SNR_dB": measured_snr,
                }
            )

            print(
                f"[{counter}/{total_expected}] "
                f"{clean_path.name} | "
                f"White | "
                f"{snr_db} dB"
            )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    fieldnames = [
        "Input_File",
        "Clean_File",
        "Speaker",
        "Noise_Type",
        "Target_SNR_dB",
        "Measured_Input_SNR_dB",
    ]

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    generated_files = sorted(
        INPUT_DIR.glob("*.wav")
    )

    print()
    print("=" * 65)
    print("MP-SENET INPUT GENERATION COMPLETE")
    print("=" * 65)

    print(
        f"Clean utterances: {len(clean_files)}"
    )

    print("Noise types: 3")
    print("SNR levels: 3")

    print(
        f"Expected mixtures: {total_expected}"
    )

    print(
        f"Generated WAV files: {len(generated_files)}"
    )

    print(
        f"Manifest rows: {len(manifest_rows)}"
    )

    print(
        f"Input directory: {INPUT_DIR}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    if len(generated_files) != total_expected:
        raise RuntimeError(
            "Generated WAV count does not match "
            "expected mixture count."
        )

    if len(manifest_rows) != total_expected:
        raise RuntimeError(
            "Manifest row count does not match "
            "expected mixture count."
        )

    print()
    print("Verification passed: 360 controlled mixtures ready.")
    print("=" * 65)


if __name__ == "__main__":
    main()