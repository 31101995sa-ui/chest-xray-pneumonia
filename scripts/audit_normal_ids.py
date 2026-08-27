import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

NORMAL_PATTERN = re.compile(
    r"^((?:NORMAL2-)?IM-\d+)-"
)


def extract_normal_id(filename: str) -> str | None:
    match = NORMAL_PATTERN.match(filename)

    if match is None:
        return None

    return match.group(1)


def main() -> None:
    ids_by_split = {}

    for split in SPLITS:
        normal_dir = DATASET_ROOT / split / "NORMAL"

        if not normal_dir.exists():
            print(f"[ERROR] Missing directory: {normal_dir}")
            ids_by_split[split] = set()
            continue

        id_counts = Counter()
        unparsed_files = []

        for file_path in normal_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            normal_id = extract_normal_id(file_path.name)

            if normal_id is None:
                unparsed_files.append(file_path.name)
                continue

            id_counts[normal_id] += 1

        ids_by_split[split] = set(id_counts.keys())

        repeated_ids = {
            normal_id: count
            for normal_id, count in id_counts.items()
            if count > 1
        }

        print(f"[{split.upper()}]")
        print(f"  Images parsed: {sum(id_counts.values())}")
        print(f"  Unique IDs: {len(id_counts)}")
        print(f"  IDs with multiple images: {len(repeated_ids)}")
        print(f"  Unparsed files: {len(unparsed_files)}")
        print()

        print("  Most images per ID:")

        for normal_id, count in id_counts.most_common(10):
            print(f"    {normal_id}: {count}")

        print()

        if unparsed_files:
            print("  Unparsed examples:")

            for filename in unparsed_files[:10]:
                print(f"    {filename}")

            print()

    train_ids = ids_by_split["train"]
    val_ids = ids_by_split["val"]
    test_ids = ids_by_split["test"]

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("[CROSS-SPLIT NORMAL ID OVERLAP]")
    print(f"  Train ∩ Val: {len(train_val_overlap)}")
    print(f"  Train ∩ Test: {len(train_test_overlap)}")
    print(f"  Val ∩ Test: {len(val_test_overlap)}")
    print()

    if train_test_overlap:
        print("  Examples Train ∩ Test:")

        for normal_id in sorted(train_test_overlap)[:10]:
            print(f"    {normal_id}")

        print()


if __name__ == "__main__":
    main()