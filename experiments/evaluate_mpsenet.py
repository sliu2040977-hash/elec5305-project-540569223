from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_utils import load_audio
from src.metrics import evaluate_signal


SR = 16000

MANIFEST_PATH = PROJECT_ROOT / "results" / "mpsenet" / "input_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "mpsenet" / "output_enhanced"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

RESULT_DIR = PROJECT_ROOT / "results" / "mpsenet"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULT_DIR / "mpsenet_metrics.csv"
SUMMARY_CSV = RESULT_DIR / "mpsenet_summary.csv"


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Output directory not found: {OUTPUT_DIR}")

    manifest = pd.read_csv(MANIFEST_PATH)

    print("=" * 70)
    print("MP-SENET OUTPUT EVALUATION")
    print("=" * 70)

    output_files = sorted(OUTPUT_DIR.glob("*.wav"))

    print(f"Manifest entries: {len(manifest)}")
    print(f"Enhanced WAV files found: {len(output_files)}")
    print()

    rows = []

    for index, enhanced_path in enumerate(output_files, start=1):
        filename = enhanced_path.name

        match = manifest[manifest["Output_Filename"] == filename] \
            if "Output_Filename" in manifest.columns else pd.DataFrame()

        if match.empty:
            if "Filename" in manifest.columns:
                match = manifest[manifest["Filename"] == filename]

        if match.empty:
            # Fall back to filename parsing
            stem = enhanced_path.stem

            parts = stem.split("__")

            if len(parts) != 3:
                print(f"[SKIP] Could not parse: {filename}")
                continue

            clean_stem = parts[0]
            noise_type = parts[1].capitalize()

            snr_text = parts[2].lower()
            snr_text = snr_text.replace("snr", "").replace("db", "")
            target_snr = int(snr_text)

            clean_filename = clean_stem + ".wav"

        else:
            row = match.iloc[0]

            possible_clean_columns = [
                "Clean_File",
                "Clean_Filename",
                "clean_file",
                "clean_filename",
            ]

            clean_filename = None

            for column in possible_clean_columns:
                if column in manifest.columns:
                    clean_filename = str(row[column])
                    break

            if clean_filename is None:
                clean_filename = filename.split("__")[0] + ".wav"

            possible_noise_columns = [
                "Noise_Type",
                "noise_type",
                "Noise",
                "noise",
            ]

            noise_type = None

            for column in possible_noise_columns:
                if column in manifest.columns:
                    noise_type = str(row[column])
                    break

            if noise_type is None:
                noise_type = filename.split("__")[1].capitalize()

            possible_snr_columns = [
                "Target_SNR_dB",
                "SNR_dB",
                "snr_db",
                "SNR",
            ]

            target_snr = None

            for column in possible_snr_columns:
                if column in manifest.columns:
                    target_snr = int(float(row[column]))
                    break

            if target_snr is None:
                snr_text = filename.split("__")[2].lower()
                snr_text = snr_text.replace(".wav", "")
                snr_text = snr_text.replace("snr", "").replace("db", "")
                target_snr = int(snr_text)

        clean_path = CLEAN_DIR / clean_filename

        if not clean_path.exists():
            print(f"[SKIP] Clean reference not found: {clean_path}")
            continue

        clean, sr_clean = load_audio(clean_path, target_sr=SR)
        enhanced, sr_enhanced = load_audio(enhanced_path, target_sr=SR)

        min_length = min(len(clean), len(enhanced))

        if min_length == 0:
            print(f"[SKIP] Empty audio: {filename}")
            continue

        clean = clean[:min_length]
        enhanced = enhanced[:min_length]

        metrics = evaluate_signal(
            clean,
            enhanced,
            SR,
        )

        speaker = clean_filename.split("_")[0]

        rows.append(
            {
                "Enhanced_File": filename,
                "Clean_File": clean_filename,
                "Speaker": speaker,
                "Noise_Type": noise_type,
                "Target_SNR_dB": target_snr,
                "Condition": "MP-SENet",
                "STOI": metrics["STOI"],
                "PESQ": metrics["PESQ"],
                "SI_SDR_dB": metrics["SI-SDR"],
            }
        )

        print(
            f"[{index}/{len(output_files)}] "
            f"{filename} | "
            f"STOI={metrics['STOI']:.4f} | "
            f"PESQ={metrics['PESQ']:.4f} | "
            f"SI-SDR={metrics['SI-SDR']:.2f} dB"
        )

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No MP-SENet files were successfully evaluated.")

    results.to_csv(OUTPUT_CSV, index=False)

    summary = (
        results.groupby(
            ["Noise_Type", "Target_SNR_dB"],
            as_index=False,
        )
        .agg(
            STOI_Mean=("STOI", "mean"),
            STOI_Std=("STOI", "std"),
            PESQ_Mean=("PESQ", "mean"),
            PESQ_Std=("PESQ", "std"),
            SI_SDR_Mean=("SI_SDR_dB", "mean"),
            SI_SDR_Std=("SI_SDR_dB", "std"),
            N=("Enhanced_File", "count"),
        )
        .sort_values(["Noise_Type", "Target_SNR_dB"])
    )

    summary.to_csv(SUMMARY_CSV, index=False)

    print()
    print("=" * 70)
    print("MP-SENET EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Successfully evaluated: {len(results)}")
    print(f"Unique clean utterances: {results['Clean_File'].nunique()}")
    print(f"Speakers: {results['Speaker'].nunique()}")
    print(f"Noise types: {results['Noise_Type'].nunique()}")
    print(f"SNR levels: {results['Target_SNR_dB'].nunique()}")

    print()
    print("Overall MP-SENet performance:")
    print(
        results[
            ["STOI", "PESQ", "SI_SDR_dB"]
        ].mean().round(4)
    )

    print()
    print("Performance by noise type and SNR:")
    print(summary.round(4).to_string(index=False))

    print()
    print("Files created:")
    print(OUTPUT_CSV)
    print(SUMMARY_CSV)
    print("=" * 70)


if __name__ == "__main__":
    main()