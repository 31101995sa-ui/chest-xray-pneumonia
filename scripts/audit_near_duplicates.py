from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

HASH_SIZE = 8
DISTANCE_THRESHOLDS = [0, 2, 4]
MAX_EXAMPLES = 10


def calculate_dhash(file_path: Path) -> int:
    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")

        image = image.resize(
            (HASH_SIZE + 1, HASH_SIZE),
            Image.Resampling.LANCZOS,
        )

        pixels = list(image.getdata())

    hash_value = 0

    for row in range(HASH_SIZE):
        row_start = row * (HASH_SIZE + 1)

        for column in range(HASH_SIZE):
            left_pixel = pixels[row_start + column]
            right_pixel = pixels[row_start + column + 1]

            bit = int(left_pixel > right_pixel)

            hash_value = (hash_value << 1) | bit

    return hash_value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


def collect_records() -> tuple[dict[str, list[dict]], list[Path]]:
    records_by_split = {
        split: []
        for split in SPLITS
    }

    unreadable_files = []

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

                records_by_split[split].append(
                    {
                        "split": split,
                        "class": class_name,
                        "path": file_path,
                        "hash": image_hash,
                    }
                )

    return records_by_split, unreadable_files


def compare_splits(
    left_split: str,
    right_split: str,
    records_by_split: dict[str, list[dict]],
) -> None:
    left_records = records_by_split[left_split]
    right_records = records_by_split[right_split]

    threshold_counts = {
        threshold: 0
        for threshold in DISTANCE_THRESHOLDS
    }

    candidate_examples = []

    minimum_distance = 65

    for left_record in left_records:
        for right_record in right_records:
            distance = hamming_distance(
                left_record["hash"],
                right_record["hash"],
            )

            if distance < minimum_distance:
                minimum_distance = distance

            for threshold in DISTANCE_THRESHOLDS:
                if distance <= threshold:
                    threshold_counts[threshold] += 1

            if (
                distance <= max(DISTANCE_THRESHOLDS)
                and len(candidate_examples) < MAX_EXAMPLES
            ):
                candidate_examples.append(
                    (
                        distance,
                        left_record,
                        right_record,
                    )
                )

    print(
        f"[{left_split.upper()} ∩ "
        f"{right_split.upper()}]"
    )

    print(f"  Minimum dHash distance: {minimum_distance}")

    for threshold in DISTANCE_THRESHOLDS:
        print(
            f"  Candidate pairs with distance <= "
            f"{threshold}: {threshold_counts[threshold]}"
        )

    print()

    if candidate_examples:
        print("  Candidate examples:")

        candidate_examples.sort(
            key=lambda item: item[0]
        )

        for distance, left_record, right_record in candidate_examples:
            left_path = left_record["path"].relative_to(DATASET_ROOT)
            right_path = right_record["path"].relative_to(DATASET_ROOT)

            print(f"    Distance: {distance}")

            print(
                f"      {left_record['class']} | "
                f"{left_path}"
            )

            print(
                f"      {right_record['class']} | "
                f"{right_path}"
            )

            print()


def main() -> None:
    records_by_split, unreadable_files = collect_records()

    total_images = sum(
        len(records)
        for records in records_by_split.values()
    )

    print("[PERCEPTUAL HASH AUDIT]")
    print(f"  Images hashed: {total_images}")
    print(f"  Hash method: 64-bit dHash")
    print(f"  Unreadable images: {len(unreadable_files)}")
    print()

    compare_splits(
        "train",
        "val",
        records_by_split,
    )

    compare_splits(
        "train",
        "test",
        records_by_split,
    )

    compare_splits(
        "val",
        "test",
        records_by_split,
    )


if __name__ == "__main__":
    main()