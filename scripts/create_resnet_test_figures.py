import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "resnet18_test_predictions.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

CONFUSION_MATRIX_PATH = (
    FIGURES_DIR
    / "resnet18_test_confusion_matrix.png"
)

ROC_CURVE_PATH = (
    FIGURES_DIR
    / "resnet18_test_roc_curve.png"
)


def load_predictions():
    y_true = []
    y_pred = []
    y_prob = []

    with PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            y_true.append(
                int(row["true_label_id"])
            )

            y_pred.append(
                int(row["predicted_label_id"])
            )

            y_prob.append(
                float(row["p_pneumonia"])
            )

    return y_true, y_pred, y_prob


def create_confusion_matrix(
    y_true,
    y_pred,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    if (
        tn != 130
        or fp != 104
        or fn != 5
        or tp != 385
    ):
        raise RuntimeError(
            "Confusion matrix does not match "
            "the sealed-test result."
        )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "NORMAL",
            "PNEUMONIA",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "ResNet18 — Sealed Test Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Confusion matrix:",
        matrix.tolist(),
    )

    print(
        "Saved:",
        CONFUSION_MATRIX_PATH,
    )


def create_roc_curve(
    y_true,
    y_prob,
) -> None:
    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            y_prob,
        )
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate,
    )

    if abs(
        roc_auc - 0.955775
    ) > 0.00001:
        raise RuntimeError(
            "ROC-AUC does not match "
            "the sealed-test result."
        )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=(
            f"ResNet18 "
            f"(AUC = {roc_auc:.4f})"
        ),
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ResNet18 — Sealed Test ROC Curve"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        ROC_CURVE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )

    print(
        "Saved:",
        ROC_CURVE_PATH,
    )


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_true, y_pred, y_prob = (
        load_predictions()
    )

    print(
        "=== ResNet18 sealed-test figures ==="
    )

    print(
        "Samples:",
        len(y_true),
    )

    print()

    create_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
    )

    print()

    create_roc_curve(
        y_true=y_true,
        y_prob=y_prob,
    )


if __name__ == "__main__":
    main()