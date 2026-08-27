from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import create_data_loader
from src.models import create_resnet18
from src.train import fit_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_baseline_best.pth"
)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    train_loader = create_data_loader(
        target_split="train",
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    val_loader = create_data_loader(
        target_split="val",
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print()

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
        lr=LEARNING_RATE,
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        model_name="resnet18_baseline",
    )

    print()
    print(f"Training completed: {len(history)} epochs")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()