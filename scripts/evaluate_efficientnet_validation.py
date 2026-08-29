import json
from pathlib import Path

import torch

from src.data import create_data_loader
from src.evaluate import collect_predictions
from src.metrics import calculate_binary_metrics
from src.models import create_efficientnet_b0


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_baseline_best.pth"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "efficientnet_b0_baseline_validation.json"
)


def main() -> None:
    device = torch.device("cpu")

    val_loader = create_data_loader(
        target_split="val",
        batch_size=32,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = create_efficientnet_b0(
        pretrained=False,
        freeze_backbone=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    y_true, y_pred, y_prob = collect_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    metrics = calculate_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    result = {
        "model": "efficientnet_b0_baseline",
        "split": "validation",
        "samples": len(y_true),
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_epoch": checkpoint["epoch"],
        "metrics": metrics,
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        "=== EfficientNet-B0 baseline "
        "validation evaluation ==="
    )

    print(
        f"Samples: {len(y_true)}"
    )

    print()

    for name, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{name}: {value:.6f}"
            )
        else:
            print(
                f"{name}: {value}"
            )

    print()

    print(
        "Metrics saved:",
        METRICS_PATH,
    )


if __name__ == "__main__":
    main()