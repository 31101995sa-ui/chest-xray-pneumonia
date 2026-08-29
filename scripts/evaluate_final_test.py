import json
from pathlib import Path

import torch

from src.data import create_data_loader
from src.evaluate import collect_predictions
from src.metrics import calculate_binary_metrics
from src.models import (
    create_efficientnet_b0,
    create_resnet18,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "final_test_comparison.json"
)


MODEL_CONFIGS = [
    {
        "name": "resnet18_baseline",
        "checkpoint": (
            PROJECT_ROOT
            / "models"
            / "resnet18_baseline_repro_best.pth"
        ),
        "factory": create_resnet18,
    },
    {
        "name": "efficientnet_b0_baseline",
        "checkpoint": (
            PROJECT_ROOT
            / "models"
            / "efficientnet_b0_baseline_best.pth"
        ),
        "factory": create_efficientnet_b0,
    },
]


def evaluate_model(
    model_name,
    checkpoint_path,
    model_factory,
    test_loader,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = model_factory(
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

    metrics = calculate_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    return {
        "model": model_name,
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": checkpoint["epoch"],
        "samples": len(y_true),
        "metrics": metrics,
    }


def main() -> None:
    device = torch.device("cpu")

    test_loader = create_data_loader(
        target_split="test",
        batch_size=32,
        num_workers=0,
    )

    results = []

    print("=== FINAL SEALED TEST EVALUATION ===")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()

    for config in MODEL_CONFIGS:
        result = evaluate_model(
            model_name=config["name"],
            checkpoint_path=config["checkpoint"],
            model_factory=config["factory"],
            test_loader=test_loader,
            device=device,
        )

        results.append(result)

        print(f"--- {result['model']} ---")

        for name, value in result["metrics"].items():
            if isinstance(value, float):
                print(f"{name}: {value:.6f}")
            else:
                print(f"{name}: {value}")

        print()

    output = {
        "split": "test",
        "test_policy": (
            "Models and training configurations were "
            "selected using validation data before "
            "opening the test split."
        ),
        "models": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "Final test comparison saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()