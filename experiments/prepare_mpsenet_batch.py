from pathlib import Path
import shutil


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

BATCH_SIZE = 5


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove previous temporary batch
    if BATCH_DIR.exists():
        shutil.rmtree(BATCH_DIR)

    BATCH_DIR.mkdir(parents=True)

    input_files = sorted(INPUT_DIR.glob("*.wav"))

    completed_names = {
        path.name
        for path in OUTPUT_DIR.glob("*.wav")
    }

    remaining_files = [
        path
        for path in input_files
        if path.name not in completed_names
    ]

    print("=" * 60)
    print("MP-SENET BATCH PREPARATION")
    print("=" * 60)

    print(f"Total inputs:     {len(input_files)}")
    print(f"Already complete: {len(completed_names)}")
    print(f"Remaining:        {len(remaining_files)}")

    if not remaining_files:
        print()
        print("All MP-SENet files have already been processed.")
        return

    current_batch = remaining_files[:BATCH_SIZE]

    for source in current_batch:

        destination = BATCH_DIR / source.name

        # Symlink instead of copying audio data
        destination.symlink_to(source.resolve())

    print(f"Current batch:    {len(current_batch)}")
    print()
    print(f"Batch directory: {BATCH_DIR}")

    print()
    print("Files selected:")

    for path in current_batch:
        print(f"  {path.name}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()