import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import AdamW

from src.data import (
    DATASET_ROOT as DEFAULT_DATASET_ROOT,
    ChestXRayDataset,
    calculate_class_weights,
    create_data_loader,
)
from src.models import create_resnet18
from src.train import fit_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "split_manifest.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_weighted_best.pth"
)

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "resnet18_weighted_001.json"
)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def calculate_sha256(
    file_path: Path,
) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest().upper()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(
    dataset_root: Path,
    git_commit: str,
) -> None:
    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    manifest_sha256 = calculate_sha256(
        MANIFEST_PATH
    )

    train_dataset = ChestXRayDataset(
        target_split="train",
        dataset_root=dataset_root,
    )

    class_weights = calculate_class_weights(
        train_dataset
    ).to(device)

    train_loader = create_data_loader(
        target_split="train",
        batch_size=BATCH_SIZE,
        num_workers=0,
        dataset_root=dataset_root,
    )

    val_loader = create_data_loader(
        target_split="val",
        batch_size=BATCH_SIZE,
        num_workers=0,
        dataset_root=dataset_root,
    )

    model = create_resnet18(
        pretrained=True,
        freeze_backbone=True,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "=== ResNet18 weighted experiment ==="
    )
    print("Git commit:", git_commit)
    print(
        "Manifest SHA256:",
        manifest_sha256,
    )
    print("Random seed:", RANDOM_SEED)
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Train samples:",
        len(train_loader.dataset),
    )
    print(
        "Validation samples:",
        len(val_loader.dataset),
    )
    print(
        "Class weights:",
        class_weights.detach().cpu(),
    )
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print(
        "Learning rate:",
        LEARNING_RATE,
    )
    print()

    start_time = time.perf_counter()

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        model_name="resnet18_weighted",
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    checkpoint_sha256 = calculate_sha256(
        CHECKPOINT_PATH
    )

    experiment = {
        "experiment_id": (
            "resnet18_weighted_001"
        ),
        "status": "completed",

        "provenance": {
            "git_commit": git_commit,
            "manifest_sha256": (
                manifest_sha256
            ),
            "random_seed": RANDOM_SEED,
        },

        "model": {
            "architecture": "resnet18",
            "pretrained": True,
            "pretrained_source": (
                "ImageNet weights via torchvision"
            ),
            "freeze_backbone": True,
            "num_classes": 2,
            "total_parameters": (
                total_parameters
            ),
            "trainable_parameters": (
                trainable_parameters
            ),
        },

        "dataset": {
            "train_samples": len(
                train_loader.dataset
            ),
            "validation_samples": len(
                val_loader.dataset
            ),
            "sealed_test_samples": 624,
        },

        "class_weighting": {
            "enabled": True,
            "method": (
                "inverse frequency"
            ),
            "normal_weight": (
                class_weights[0].item()
            ),
            "pneumonia_weight": (
                class_weights[1].item()
            ),
        },

        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": (
                LEARNING_RATE
            ),
            "optimizer": "AdamW",
            "loss": (
                "CrossEntropyLoss"
            ),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "training_time_seconds": (
                elapsed_time
            ),
        },

        "environment": {
            "torch": str(
                torch.__version__
            ),
            "torchvision": str(
                torchvision.__version__
            ),
            "platform": (
                "Kaggle"
                if Path("/kaggle").exists()
                else "local"
            ),
        },

        "history": history,

        "best_checkpoint": {
            "epoch": checkpoint["epoch"],
            "val_loss": (
                checkpoint["val_loss"]
            ),
            "val_accuracy": (
                checkpoint[
                    "val_accuracy"
                ]
            ),
            "filename": (
                CHECKPOINT_PATH.name
            ),
            "sha256": (
                checkpoint_sha256
            ),
            "size_bytes": (
                CHECKPOINT_PATH
                .stat()
                .st_size
            ),
        },

        "evaluation_status": {
            "precision": (
                "not_evaluated"
            ),
            "recall": (
                "not_evaluated"
            ),
            "f1": "not_evaluated",
            "roc_auc": (
                "not_evaluated"
            ),
            "confusion_matrix": (
                "not_evaluated"
            ),
            "sealed_test": (
                "not_evaluated"
            ),
        },
    }

    EXPERIMENT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EXPERIMENT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
            indent=2,
        )

    print()
    print(
        "=== Weighted experiment finished ==="
    )
    print(
        f"Training time: "
        f"{elapsed_time:.2f} seconds"
    )
    print(
        "Best epoch:",
        checkpoint["epoch"],
    )
    print(
        "Best val loss:",
        checkpoint["val_loss"],
    )
    print(
        "Best val accuracy:",
        checkpoint[
            "val_accuracy"
        ],
    )
    print(
        "Checkpoint SHA256:",
        checkpoint_sha256,
    )
    print(
        "Experiment record:",
        EXPERIMENT_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--git-commit",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        dataset_root=args.dataset_root,
        git_commit=args.git_commit,
    )