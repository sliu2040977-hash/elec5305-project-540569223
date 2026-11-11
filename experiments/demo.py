"""Run the bundled VoiceBank/café sample without changing archived results."""

import argparse
import csv
from pathlib import Path

import preliminary_experiment as preliminary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    preliminary.RESULTS_DIR = args.output
    preliminary.AUDIO_DIR = args.output / "audio"
    preliminary.CSV_PATH = args.output / "metrics.csv"
    preliminary.run_experiment()
    with preliminary.CSV_PATH.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["Target_SNR_dB"] == "0"]
    print(f"\nCafe, {float(rows[0]['Measured_Input_SNR_dB']):.2f} dB input SNR")
    print(f"{'Condition':36} {'STOI':>7} {'PESQ':>7} {'SI-SDR':>7}")
    for row in rows:
        print(f"{row['Condition']:36} {float(row['STOI']):7.4f} "
              f"{float(row['PESQ']):7.4f} {float(row['SI_SDR_dB']):7.2f}")


if __name__ == "__main__":
    main()
