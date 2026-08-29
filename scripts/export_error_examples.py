import csv
import shutil
from pathlib import Path

from src.data import DATASET_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERRORS_CSV = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "resnet18_test_errors.csv"
)

FALSE_POSITIVE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "false_positives"
)

FALSE_NEGATIVE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "false_negatives"
)

TOP_FALSE_POSITIVES = 10


def read_errors() -> list[dict]:
    with ERRORS_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def prepare_directory(
    directory: Path,
) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in directory.iterdir():
        if item.is_file():
            item.unlink()


def copy_error_image(
    row: dict,
    destination_dir: Path,
    rank: int,
) -> Path:
    source_path = (
        DATASET_ROOT
        / Path(row["path"])
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Image not found: {source_path}"
        )

    confidence = float(
        row["prediction_confidence"]
    )

    probability = float(
        row["p_pneumonia"]
    )

    destination_name = (
        f"{rank:02d}"
        f"_conf-{confidence:.4f}"
        f"_pneu-{probability:.4f}"
        f"_{source_path.name}"
    )

    destination_path = (
        destination_dir
        / destination_name
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    return destination_path


def main() -> None:
    rows = read_errors()

    false_positives = [
        row
        for row in rows
        if row["result_type"]
        == "FALSE_POSITIVE"
    ]

    false_negatives = [
        row
        for row in rows
        if row["result_type"]
        == "FALSE_NEGATIVE"
    ]

    false_positives.sort(
        key=lambda row: float(
            row["prediction_confidence"]
        ),
        reverse=True,
    )

    false_negatives.sort(
        key=lambda row: float(
            row["prediction_confidence"]
        ),
        reverse=True,
    )

    selected_false_positives = (
        false_positives[
            :TOP_FALSE_POSITIVES
        ]
    )

    prepare_directory(
        FALSE_POSITIVE_DIR
    )

    prepare_directory(
        FALSE_NEGATIVE_DIR
    )

    print(
        "=== Exporting false positives ==="
    )

    for rank, row in enumerate(
        selected_false_positives,
        start=1,
    ):
        output_path = copy_error_image(
            row=row,
            destination_dir=(
                FALSE_POSITIVE_DIR
            ),
            rank=rank,
        )

        print(output_path.name)

    print()
    print(
        "=== Exporting false negatives ==="
    )

    for rank, row in enumerate(
        false_negatives,
        start=1,
    ):
        output_path = copy_error_image(
            row=row,
            destination_dir=(
                FALSE_NEGATIVE_DIR
            ),
            rank=rank,
        )

        print(output_path.name)

    print()
    print(
        "False positives exported:",
        len(selected_false_positives),
    )

    print(
        "False negatives exported:",
        len(false_negatives),
    )


if __name__ == "__main__":
    main()