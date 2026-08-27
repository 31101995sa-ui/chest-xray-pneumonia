import csv
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_PATH = OUTPUT_DIR / "split_manifest.csv"

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

VAL_FRACTION = 0.15
RANDOM_SEED = 42

CLASS_LABELS = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}

PNEUMONIA_PATTERN = re.compile(r"^(person\d+)_")
NORMAL_PATTERN = re.compile(r"^((?:NORMAL2-)?IM-\d+)-")


def extract_group_id(
    filename: str,
    class_name: str,
) -> str | None:
    if class_name == "PNEUMONIA":
        match = PNEUMONIA_PATTERN.match(filename)

    elif class_name == "NORMAL":
        match = NORMAL_PATTERN.match(filename)

    else:
        return None

    if match is None:
        return None

    return match.group(1)


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def collect_class_groups(
    class_name: str,
) -> dict[str, list[Path]]:
    class_dir = DATASET_ROOT / "train" / class_name

    groups = defaultdict(list)

    for file_path in class_dir.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        group_id = extract_group_id(
            file_path.name,
            class_name,
        )

        if group_id is None:
            raise ValueError(
                f"Could not extract group ID: {file_path}"
            )

        groups[group_id].append(file_path)

    return dict(groups)


def select_validation_groups(
    groups: dict[str, list[Path]],
    seed: int,
) -> set[str]:
    group_items = list(groups.items())

    rng = random.Random(seed)
    rng.shuffle(group_items)

    total_images = sum(
        len(paths)
        for _, paths in group_items
    )

    target_val_images = round(
        total_images * VAL_FRACTION
    )

    selected_groups = set()
    selected_images = 0

    for group_id, paths in group_items:
        if selected_images >= target_val_images:
            break

        selected_groups.add(group_id)
        selected_images += len(paths)

    return selected_groups


def check_exact_hash_overlap(
    rows: list[dict[str, str]],
) -> int:
    hashes_by_split = {
        "train": set(),
        "val": set(),
    }

    for row in rows:
        target_split = row["target_split"]

        if target_split not in hashes_by_split:
            continue

        file_path = DATASET_ROOT / row["relative_path"]

        file_hash = calculate_sha256(file_path)

        hashes_by_split[target_split].add(file_hash)

    overlap = (
        hashes_by_split["train"]
        & hashes_by_split["val"]
    )

    return len(overlap)


def main() -> None:
    rows = []

    validation_groups_by_class = {}

    for class_index, class_name in enumerate(
        CLASS_LABELS
    ):
        groups = collect_class_groups(class_name)

        validation_groups = select_validation_groups(
            groups,
            seed=RANDOM_SEED + class_index,
        )

        validation_groups_by_class[class_name] = (
            validation_groups
        )

        for group_id, paths in groups.items():
            if group_id in validation_groups:
                target_split = "val"
            else:
                target_split = "train"

            for file_path in paths:
                relative_path = file_path.relative_to(
                    DATASET_ROOT
                )

                rows.append(
                    {
                        "relative_path": str(relative_path),
                        "class": class_name,
                        "label": str(
                            CLASS_LABELS[class_name]
                        ),
                        "group_id": group_id,
                        "source_split": "train",
                        "target_split": target_split,
                    }
                )

    for class_name in CLASS_LABELS:
        test_dir = DATASET_ROOT / "test" / class_name

        for file_path in test_dir.iterdir():
            if not file_path.is_file():
                continue

            if (
                file_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            group_id = extract_group_id(
                file_path.name,
                class_name,
            )

            if group_id is None:
                group_id = "UNKNOWN"

            relative_path = file_path.relative_to(
                DATASET_ROOT
            )

            rows.append(
                {
                    "relative_path": str(relative_path),
                    "class": class_name,
                    "label": str(
                        CLASS_LABELS[class_name]
                    ),
                    "group_id": group_id,
                    "source_split": "test",
                    "target_split": "test",
                }
            )

    for class_name in CLASS_LABELS:
        original_val_dir = (
            DATASET_ROOT
            / "val"
            / class_name
        )

        for file_path in original_val_dir.iterdir():
            if not file_path.is_file():
                continue

            if (
                file_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            group_id = extract_group_id(
                file_path.name,
                class_name,
            )

            if group_id is None:
                group_id = "UNKNOWN"

            relative_path = file_path.relative_to(
                DATASET_ROOT
            )

            rows.append(
                {
                    "relative_path": str(relative_path),
                    "class": class_name,
                    "label": str(
                        CLASS_LABELS[class_name]
                    ),
                    "group_id": group_id,
                    "source_split": "val",
                    "target_split": "legacy_val",
                }
            )

    train_groups = {
        class_name: {
            row["group_id"]
            for row in rows
            if row["class"] == class_name
            and row["target_split"] == "train"
        }
        for class_name in CLASS_LABELS
    }

    val_groups = {
        class_name: {
            row["group_id"]
            for row in rows
            if row["class"] == class_name
            and row["target_split"] == "val"
        }
        for class_name in CLASS_LABELS
    }

    for class_name in CLASS_LABELS:
        overlap = (
            train_groups[class_name]
            & val_groups[class_name]
        )

        if overlap:
            raise RuntimeError(
                f"Group leakage detected for "
                f"{class_name}: {len(overlap)} groups"
            )

    exact_hash_overlap = check_exact_hash_overlap(
        rows
    )

    if exact_hash_overlap:
        raise RuntimeError(
            f"Exact image leakage detected between "
            f"new train and val: "
            f"{exact_hash_overlap} hashes"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "relative_path",
                "class",
                "label",
                "group_id",
                "source_split",
                "target_split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("[SPLIT MANIFEST CREATED]")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Validation fraction target: {VAL_FRACTION}")
    print()

    for target_split in [
        "train",
        "val",
        "test",
        "legacy_val",
    ]:
        split_rows = [
            row
            for row in rows
            if row["target_split"] == target_split
        ]

        normal_count = sum(
            row["class"] == "NORMAL"
            for row in split_rows
        )

        pneumonia_count = sum(
            row["class"] == "PNEUMONIA"
            for row in split_rows
        )

        print(f"[{target_split.upper()}]")
        print(f"  NORMAL: {normal_count}")
        print(f"  PNEUMONIA: {pneumonia_count}")
        print(f"  Total: {len(split_rows)}")
        print()

    print("[LEAKAGE CHECK]")
    print("  Group overlap train ↔ val: 0")
    print(
        f"  Exact SHA-256 overlap train ↔ val: "
        f"{exact_hash_overlap}"
    )


if __name__ == "__main__":
    main()