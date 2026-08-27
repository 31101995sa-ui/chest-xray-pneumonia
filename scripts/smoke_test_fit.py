from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.data import ChestXRayDataset
from src.models import create_resnet18
from src.train import fit_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_smoke.pth"
)


def select_balanced_indices(
    dataset: ChestXRayDataset,
    samples_per_class: int,
) -> list[int]:
    normal_indices = []
    pneumonia_indices = []

    for index, record in enumerate(dataset.records):
        label = record["label"]

        if (
            label == 0
            and len(normal_indices) < samples_per_class
        ):
            normal_indices.append(index)

        elif (
            label == 1
            and len(pneumonia_indices) < samples_per_class
        ):
            pneumonia_indices.append(index)

        if (
            len(normal_indices) == samples_per_class
            and len(pneumonia_indices) == samples_per_class
        ):
            break

    return normal_indices + pneumonia_indices


def main() -> None:
    device = torch.device("cpu")

    train_dataset = ChestXRayDataset("train")
    val_dataset = ChestXRayDataset("val")

    train_indices = select_balanced_indices(
        train_dataset,
        samples_per_class=32,
    )

    val_indices = select_balanced_indices(
        val_dataset,
        samples_per_class=32,
    )

    small_train_dataset = Subset(
        train_dataset,
        train_indices,
    )

    small_val_dataset = Subset(
        val_dataset,
        val_indices,
    )

    train_loader = DataLoader(
        small_train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        small_val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = create_resnet18(
        pretrained=True,
        freeze_backbone=True,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=0.001,
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=2,
        checkpoint_path=CHECKPOINT_PATH,
        model_name="resnet18",
    )

    print()
    print(f"Epochs completed: {len(history)}")
    print(
        f"Checkpoint exists: "
        f"{CHECKPOINT_PATH.exists()}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    print(
        f"Best checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best checkpoint val loss: "
        f"{checkpoint['val_loss']:.6f}"
    )


if __name__ == "__main__":
    main()