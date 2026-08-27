from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def count_images(directory: Path) -> int:
    count = 0

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1

    return count


def inspect_images(directory: Path) -> tuple[Counter, Counter, list[Path]]:
    size_counts = Counter()
    mode_counts = Counter()
    broken_files = []

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(file_path) as image:
                size_counts[image.size] += 1
                mode_counts[image.mode] += 1
                image.verify()

        except Exception:
            broken_files.append(file_path)

    return size_counts, mode_counts, broken_files


def main() -> None:
    print(f"Dataset root: {DATASET_ROOT}")
    print()

    total_images = 0
    all_size_counts = Counter()
    all_mode_counts = Counter()
    all_broken_files = []

    for split in SPLITS:
        split_dir = DATASET_ROOT / split

        if not split_dir.exists():
            print(f"[ERROR] Missing split directory: {split_dir}")
            continue

        print(f"[{split.upper()}]")

        split_counts = {}

        for class_name in CLASSES:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"  [ERROR] Missing class directory: {class_name}")
                continue

            image_count = count_images(class_dir)

            size_counts, mode_counts, broken_files = inspect_images(class_dir)

            split_counts[class_name] = image_count
            total_images += image_count

            all_size_counts.update(size_counts)
            all_mode_counts.update(mode_counts)
            all_broken_files.extend(broken_files)

            print(f"  {class_name}: {image_count}")

        split_total = sum(split_counts.values())

        if split_total > 0:
            print("  Distribution:")

            for class_name, image_count in split_counts.items():
                percentage = image_count / split_total * 100
                print(f"    {class_name}: {percentage:.1f}%")

        print()

    print(f"Total images: {total_images}")
    print()

    print("[IMAGE MODES]")

    for mode, count in all_mode_counts.most_common():
        print(f"  {mode}: {count}")

    print()

    print(f"Unique image sizes: {len(all_size_counts)}")
    print("Most common image sizes:")

    for size, count in all_size_counts.most_common(10):
        print(f"  {size}: {count}")

    if all_size_counts:
        widths = [size[0] for size in all_size_counts]
        heights = [size[1] for size in all_size_counts]

        print()
        print("[IMAGE DIMENSIONS]")
        print(f"  Min width: {min(widths)}")
        print(f"  Max width: {max(widths)}")
        print(f"  Min height: {min(heights)}")
        print(f"  Max height: {max(heights)}")

        landscape = 0
        portrait = 0
        square = 0

        for (width, height), count in all_size_counts.items():
            if width > height:
                landscape += count
            elif height > width:
                portrait += count
            else:
                square += count

        print()
        print("[ORIENTATION]")
        print(f"  Landscape: {landscape}")
        print(f"  Portrait: {portrait}")
        print(f"  Square: {square}")

    print()
    print(f"Broken images: {len(all_broken_files)}")

    for file_path in all_broken_files:
        print(f"  {file_path}")


if __name__ == "__main__":
    main()