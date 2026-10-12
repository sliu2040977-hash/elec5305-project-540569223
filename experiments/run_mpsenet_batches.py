from pathlib import Path
import shutil
import subprocess
import sys
import time


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "mpsenet"
    / "input_noisy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "mpsenet"
    / "output_enhanced"
)

BATCH_DIR = (
    PROJECT_ROOT
    / "results"
    / "mpsenet"
    / "current_batch"
)

MPSENET_DIR = (
    PROJECT_ROOT
    / "external"
    / "MP-SENet"
)

INFERENCE_SCRIPT = MPSENET_DIR / "inference.py"

CHECKPOINT = (
    MPSENET_DIR
    / "best_ckpt"
    / "g_best_vb"
)

# Smaller batch for Codespaces stability
BATCH_SIZE = 5

# Pause after each successful batch
PAUSE_SECONDS = 15

# If a batch fails, stop rather than repeatedly retrying
# and consuming Codespaces resources.
MAX_RETRIES = 0


# ============================================================
# Utility functions
# ============================================================

def get_input_files():
    return sorted(INPUT_DIR.glob("*.wav"))


def get_completed_names():
    return {
        path.name
        for path in OUTPUT_DIR.glob("*.wav")
    }


def get_remaining_files():
    completed = get_completed_names()

    return [
        path
        for path in get_input_files()
        if path.name not in completed
    ]


def prepare_batch(files):

    if BATCH_DIR.exists():
        shutil.rmtree(BATCH_DIR)

    BATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source in files:

        destination = BATCH_DIR / source.name

        destination.symlink_to(
            source.resolve()
        )


def run_mpsenet():

    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--input_noisy_wavs_dir",
        str(BATCH_DIR),
        "--output_dir",
        str(OUTPUT_DIR),
        "--checkpoint_file",
        str(CHECKPOINT),
    ]

    print()
    print("Starting MP-SENet inference...")
    print()

    result = subprocess.run(
        command,
        cwd=MPSENET_DIR,
    )

    return result.returncode


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Inference script not found: {INFERENCE_SCRIPT}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    total_inputs = len(get_input_files())

    if total_inputs == 0:
        raise RuntimeError(
            "No MP-SENet input WAV files were found."
        )

    print("=" * 70)
    print("AUTOMATIC MP-SENET BATCH INFERENCE")
    print("=" * 70)

    print(f"Total input files: {total_inputs}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Already complete:  {len(get_completed_names())}")
    print(f"Remaining:         {len(get_remaining_files())}")

    batch_number = 0

    # --------------------------------------------------------
    # Batch loop
    # --------------------------------------------------------

    while True:

        completed_before = get_completed_names()
        remaining = get_remaining_files()

        completed_count = len(completed_before)
        remaining_count = len(remaining)

        print()
        print("=" * 70)
        print(
            f"Progress: {completed_count}/{total_inputs} complete"
        )
        print(
            f"Remaining: {remaining_count}"
        )
        print("=" * 70)

        if remaining_count == 0:
            break

        batch_number += 1

        current_batch = remaining[:BATCH_SIZE]

        print()
        print(
            f"Preparing batch {batch_number}: "
            f"{len(current_batch)} files"
        )

        prepare_batch(current_batch)

        for file_path in current_batch:
            print(f"  {file_path.name}")

        # ----------------------------------------------------
        # Run MP-SENet in a separate process
        # ----------------------------------------------------

        start_time = time.time()

        return_code = run_mpsenet()

        elapsed = time.time() - start_time

        print()
        print(
            f"Batch runtime: {elapsed / 60:.2f} minutes"
        )

        completed_after = get_completed_names()

        new_files = (
            completed_after
            - completed_before
        )

        print(
            f"New output files: {len(new_files)}"
        )

        print(
            f"Total completed: "
            f"{len(completed_after)}/{total_inputs}"
        )

        # ----------------------------------------------------
        # Handle subprocess failure
        # ----------------------------------------------------

        if return_code != 0:

            print()
            print("=" * 70)
            print("MP-SENet batch failed.")
            print("=" * 70)

            print(f"Return code: {return_code}")

            if return_code == -9:
                print(
                    "The subprocess received SIGKILL. "
                    "This can occur when the environment "
                    "terminates the process."
                )

            print(
                "All successfully created output files "
                "have been preserved."
            )

            print(
                "Run this script again later to resume."
            )

            sys.exit(return_code)

        # ----------------------------------------------------
        # Prevent infinite loop
        # ----------------------------------------------------

        if len(new_files) == 0:

            print()
            print("=" * 70)
            print(
                "ERROR: inference exited successfully "
                "but created no new output files."
            )
            print("=" * 70)

            sys.exit(1)

        # ----------------------------------------------------
        # Verify this batch
        # ----------------------------------------------------

        missing_outputs = [
            path.name
            for path in current_batch
            if path.name not in completed_after
        ]

        if missing_outputs:

            print()
            print(
                "WARNING: The following batch files "
                "have no output:"
            )

            for name in missing_outputs:
                print(f"  {name}")

            print()
            print(
                "Stopping here. Existing results "
                "have been preserved."
            )

            sys.exit(1)

        # ----------------------------------------------------
        # Pause before next model process
        # ----------------------------------------------------

        remaining_after = (
            total_inputs
            - len(completed_after)
        )

        if remaining_after > 0:

            print()
            print(
                f"Batch completed successfully."
            )

            print(
                f"Pausing {PAUSE_SECONDS} seconds "
                f"before starting the next batch..."
            )

            time.sleep(PAUSE_SECONDS)

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    final_completed = len(
        get_completed_names()
    )

    print()
    print("=" * 70)
    print("MP-SENET FULL INFERENCE COMPLETE")
    print("=" * 70)

    print(f"Input WAV files:  {total_inputs}")
    print(f"Output WAV files: {final_completed}")

    if final_completed == total_inputs:

        print()
        print(
            "Verification passed: "
            "all controlled mixtures were processed."
        )

    else:

        print()
        print(
            "WARNING: output count does not match "
            "input count."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
