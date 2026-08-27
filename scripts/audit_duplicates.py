import hashlib
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def main() -> None:
    files_by_hash = defaultdict(list)
    total_files = 0

    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = DATASET_ROOT / split / class_name

            if not class_dir.exists():
                print(f"[ERROR] Missing directory: {class_dir}")
                continue

            for file_path in class_dir.iterdir():
                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                file_hash = calculate_sha256(file_path)

                files_by_hash[file_hash].append(
                    {
                        "split": split,
                        "class": class_name,
                        "path": file_path,
                    }
                )

                total_files += 1

    duplicate_groups = {
        file_hash: records
        for file_hash, records in files_by_hash.items()
        if len(records) > 1
    }

    cross_split_duplicates = {}

    for file_hash, records in duplicate_groups.items():
        splits = {record["split"] for record in records}

        if len(splits) > 1:
            cross_split_duplicates[file_hash] = records

    cross_class_duplicates = {}

    for file_hash, records in duplicate_groups.items():
        classes = {record["class"] for record in records}

        if len(classes) > 1:
            cross_class_duplicates[file_hash] = records

    duplicate_groups_by_location = Counter()

    for records in duplicate_groups.values():
        splits = {record["split"] for record in records}
        classes = {record["class"] for record in records}

        if len(splits) == 1 and len(classes) == 1:
            split = next(iter(splits))
            class_name = next(iter(classes))

            duplicate_groups_by_location[(split, class_name)] += 1

    print("[EXACT DUPLICATE AUDIT]")
    print(f"  Total image files hashed: {total_files}")
    print(f"  Unique SHA-256 hashes: {len(files_by_hash)}")
    print(f"  Duplicate hash groups: {len(duplicate_groups)}")
    print(
        f"  Cross-split duplicate groups: "
        f"{len(cross_split_duplicates)}"
    )
    print(
        f"  Cross-class duplicate groups: "
        f"{len(cross_class_duplicates)}"
    )
    print()

    print("[DUPLICATE GROUPS BY LOCATION]")

    for split in SPLITS:
        for class_name in CLASSES:
            count = duplicate_groups_by_location[(split, class_name)]

            print(f"  {split} | {class_name}: {count}")

    print()

    overlap_counts = {
        "train-val": 0,
        "train-test": 0,
        "val-test": 0,
    }

    for records in cross_split_duplicates.values():
        splits = {record["split"] for record in records}

        if "train" in splits and "val" in splits:
            overlap_counts["train-val"] += 1

        if "train" in splits and "test" in splits:
            overlap_counts["train-test"] += 1

        if "val" in splits and "test" in splits:
            overlap_counts["val-test"] += 1

    print("[CROSS-SPLIT EXACT DUPLICATES]")
    print(f"  Train ∩ Val: {overlap_counts['train-val']}")
    print(f"  Train ∩ Test: {overlap_counts['train-test']}")
    print(f"  Val ∩ Test: {overlap_counts['val-test']}")
    print()

    if cross_split_duplicates:
        print("[CROSS-SPLIT EXAMPLES]")

        for index, records in enumerate(
            cross_split_duplicates.values(),
            start=1,
        ):
            if index > 10:
                break

            print(f"  Duplicate group {index}:")

            for record in records:
                relative_path = record["path"].relative_to(DATASET_ROOT)

                print(
                    f"    {record['split']} | "
                    f"{record['class']} | "
                    f"{relative_path}"
                )

            print()

    if cross_class_duplicates:
        print("[CROSS-CLASS DUPLICATES]")

        for index, records in enumerate(
            cross_class_duplicates.values(),
            start=1,
        ):
            if index > 10:
                break

            print(f"  Conflict group {index}:")

            for record in records:
                relative_path = record["path"].relative_to(DATASET_ROOT)

                print(
                    f"    {record['split']} | "
                    f"{record['class']} | "
                    f"{relative_path}"
                )

            print()


if __name__ == "__main__":
    main()