import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

SPLITS = ["train", "val", "test"]

PATIENT_PATTERN = re.compile(r"^(person\d+)_")


def extract_patient_id(filename: str) -> str | None:
    match = PATIENT_PATTERN.match(filename)

    if match is None:
        return None

    return match.group(1)


def main() -> None:
    patients_by_split = {}

    for split in SPLITS:
        pneumonia_dir = DATASET_ROOT / split / "PNEUMONIA"

        if not pneumonia_dir.exists():
            print(f"[ERROR] Missing directory: {pneumonia_dir}")
            patients_by_split[split] = set()
            continue

        patient_counts = Counter()
        unparsed_files = []

        for file_path in pneumonia_dir.iterdir():
            if not file_path.is_file():
                continue

            patient_id = extract_patient_id(file_path.name)

            if patient_id is None:
                unparsed_files.append(file_path.name)
                continue

            patient_counts[patient_id] += 1

        patients_by_split[split] = set(patient_counts.keys())

        repeated_patients = {
            patient_id: count
            for patient_id, count in patient_counts.items()
            if count > 1
        }

        print(f"[{split.upper()}]")
        print(f"  Images: {sum(patient_counts.values())}")
        print(f"  Unique patients: {len(patient_counts)}")
        print(
            f"  Patients with multiple images: "
            f"{len(repeated_patients)}"
        )
        print(f"  Unparsed files: {len(unparsed_files)}")
        print()

        print("  Most images per patient:")

        for patient_id, count in patient_counts.most_common(10):
            print(f"    {patient_id}: {count}")

        print()

    print("[CROSS-SPLIT PATIENT OVERLAP]")

    train_patients = patients_by_split["train"]
    val_patients = patients_by_split["val"]
    test_patients = patients_by_split["test"]

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    print(f"  Train ∩ Val: {len(train_val_overlap)}")
    print(f"  Train ∩ Test: {len(train_test_overlap)}")
    print(f"  Val ∩ Test: {len(val_test_overlap)}")
    print()

    if train_val_overlap:
        print("  Examples Train ∩ Val:")

        for patient_id in sorted(train_val_overlap)[:10]:
            print(f"    {patient_id}")

        print()

    if train_test_overlap:
        print("  Examples Train ∩ Test:")

        for patient_id in sorted(train_test_overlap)[:10]:
            print(f"    {patient_id}")

        print()

    if val_test_overlap:
        print("  Examples Val ∩ Test:")

        for patient_id in sorted(val_test_overlap)[:10]:
            print(f"    {patient_id}")

        print()


if __name__ == "__main__":
    main()