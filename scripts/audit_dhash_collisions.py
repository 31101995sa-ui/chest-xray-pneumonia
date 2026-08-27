from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

HASH_SIZE = 8
MAX_EXAMPLES = 20


def calculate_dhash(file_path: Path) -> int:
    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")

        image = image.resize(
            (HASH_SIZE + 1, HASH_SIZE),
            Image.Resampling.LANCZOS,
        )

        pixels = list(image.get_flattened_data())

    hash_value = 0

    for row in range(HASH_SIZE):
        row_start = row * (HASH_SIZE + 1)

        for column in range(HASH_SIZE):
            left_pixel = pixels[row_start + column]
            right_pixel = pixels[row_start + column + 1]

            bit = int(left_pixel > right_pixel)

            hash_value = (hash_value << 1) | bit

    return hash_value


def main() -> None:
    records_by_hash = defaultdict(list)
    unreadable_files = []
    total_images = 0

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

                try:
                    image_hash = calculate_dhash(file_path)

                except Exception:
                    unreadable_files.append(file_path)
                    continue

                records_by_hash[image_hash].append(
                    {
                        "split": split,
                        "class": class_name,
                        "path": file_path,
                    }
                )

                total_images += 1

    cross_split_groups = {}

    for image_hash, records in records_by_hash.items():
        splits = {
            record["split"]
            for record in records
        }

        if len(splits) > 1:
            cross_split_groups[image_hash] = records

    pair_counts = {
        "train-val": 0,
        "train-test": 0,
        "val-test": 0,
    }

    cross_split_pairs = []

    for image_hash, records in cross_split_groups.items():
        for left_record, right_record in combinations(records, 2):
            left_split = left_record["split"]
            right_split = right_record["split"]

            if left_split == right_split:
                continue

            split_pair = {left_split, right_split}

            if split_pair == {"train", "val"}:
                pair_counts["train-val"] += 1

            elif split_pair == {"train", "test"}:
                pair_counts["train-test"] += 1

            elif split_pair == {"val", "test"}:
                pair_counts["val-test"] += 1

            cross_split_pairs.append(
                (
                    image_hash,
                    left_record,
                    right_record,
                )
            )

    print("[dHASH COLLISION AUDIT]")
    print(f"  Images hashed: {total_images}")
    print(f"  Unique dHashes: {len(records_by_hash)}")
    print(f"  Unreadable images: {len(unreadable_files)}")
    print(
        f"  Cross-split identical-dHash groups: "
        f"{len(cross_split_groups)}"
    )
    print()

    print("[IDENTICAL dHASH CROSS-SPLIT PAIRS]")
    print(f"  Train ∩ Val: {pair_counts['train-val']}")
    print(f"  Train ∩ Test: {pair_counts['train-test']}")
    print(f"  Val ∩ Test: {pair_counts['val-test']}")
    print()

    if cross_split_pairs:
        print("[EXAMPLES]")

        for index, (
            image_hash,
            left_record,
            right_record,
        ) in enumerate(
            cross_split_pairs[:MAX_EXAMPLES],
            start=1,
        ):
            left_path = left_record["path"].relative_to(DATASET_ROOT)
            right_path = right_record["path"].relative_to(DATASET_ROOT)

            print(f"  Pair {index}:")
            print(f"    dHash: {image_hash:016x}")

            print(
                f"    {left_record['split']} | "
                f"{left_record['class']} | "
                f"{left_path}"
            )

            print(
                f"    {right_record['split']} | "
                f"{right_record['class']} | "
                f"{right_path}"
            )

            print()


if __name__ == "__main__":
    main()