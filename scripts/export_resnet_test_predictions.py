import csv
from pathlib import Path

import torch

from src.data import create_data_loader
from src.evaluate import collect_predictions
from src.models import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_baseline_repro_best.pth"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "resnet18_test_predictions.csv"
)

ERRORS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "resnet18_test_errors.csv"
)

LABEL_NAMES = {
    0: "NORMAL",
    1: "PNEUMONIA",
}


def get_result_type(
    true_label: int,
    predicted_label: int,
) -> str:
    if true_label == predicted_label:
        return "CORRECT"

    if true_label == 0 and predicted_label == 1:
        return "FALSE_POSITIVE"

    return "FALSE_NEGATIVE"


def write_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "path",
        "true_label_id",
        "true_label",
        "predicted_label_id",
        "predicted_label",
        "p_pneumonia",
        "prediction_confidence",
        "result_type",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_top_errors(
    rows: list[dict],
    error_type: str,
    limit: int = 10,
) -> None:
    selected = [
        row
        for row in rows
        if row["result_type"] == error_type
    ]

    selected.sort(
        key=lambda row: row["prediction_confidence"],
        reverse=True,
    )

    print()
    print(
        f"=== Top {error_type} "
        f"by model confidence ==="
    )

    for row in selected[:limit]:
        print(
            f"{row['prediction_confidence']:.6f} | "
            f"P(pneumonia)="
            f"{row['p_pneumonia']:.6f} | "
            f"{row['path']}"
        )


def main() -> None:
    device = torch.device("cpu")

    test_loader = create_data_loader(
        target_split="test",
        batch_size=32,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = create_resnet18(
        pretrained=False,
        freeze_backbone=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    y_true, y_pred, y_prob = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    records = test_loader.dataset.records

    if len(records) != len(y_true):
        raise RuntimeError(
            "Dataset records and predictions "
            "have different lengths."
        )

    rows = []

    for (
        record,
        true_label,
        predicted_label,
        p_pneumonia,
    ) in zip(
        records,
        y_true,
        y_pred,
        y_prob,
    ):
        manifest_label = int(
            record["label"]
        )

        if manifest_label != true_label:
            raise RuntimeError(
                "Prediction order does not match "
                "dataset record order."
            )

        p_pneumonia = float(
            p_pneumonia
        )

        if predicted_label == 1:
            prediction_confidence = (
                p_pneumonia
            )
        else:
            prediction_confidence = (
                1.0 - p_pneumonia
            )

        result_type = get_result_type(
            true_label=true_label,
            predicted_label=predicted_label,
        )

        rows.append(
            {
                "path": record["relative_path"],
                "true_label_id": true_label,
                "true_label": LABEL_NAMES[
                    true_label
                ],
                "predicted_label_id": (
                    predicted_label
                ),
                "predicted_label": LABEL_NAMES[
                    predicted_label
                ],
                "p_pneumonia": (
                    p_pneumonia
                ),
                "prediction_confidence": (
                    prediction_confidence
                ),
                "result_type": result_type,
            }
        )

    error_rows = [
        row
        for row in rows
        if row["result_type"] != "CORRECT"
    ]

    error_rows.sort(
        key=lambda row: row[
            "prediction_confidence"
        ],
        reverse=True,
    )

    write_csv(
        PREDICTIONS_PATH,
        rows,
    )

    write_csv(
        ERRORS_PATH,
        error_rows,
    )

    false_positives = sum(
        row["result_type"]
        == "FALSE_POSITIVE"
        for row in rows
    )

    false_negatives = sum(
        row["result_type"]
        == "FALSE_NEGATIVE"
        for row in rows
    )

    correct = sum(
        row["result_type"] == "CORRECT"
        for row in rows
    )

    print(
        "=== ResNet18 test "
        "prediction export ==="
    )
    print(
        f"Total predictions: {len(rows)}"
    )
    print(
        f"Correct: {correct}"
    )
    print(
        f"False positives: "
        f"{false_positives}"
    )
    print(
        f"False negatives: "
        f"{false_negatives}"
    )
    print(
        f"Total errors: "
        f"{len(error_rows)}"
    )

    print()
    print(
        "Predictions CSV:",
        PREDICTIONS_PATH,
    )

    print(
        "Errors CSV:",
        ERRORS_PATH,
    )

    print_top_errors(
        rows,
        "FALSE_POSITIVE",
    )

    print_top_errors(
        rows,
        "FALSE_NEGATIVE",
    )


if __name__ == "__main__":
    main()