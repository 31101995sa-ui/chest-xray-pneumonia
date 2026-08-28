import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import (
    ChestXRayDataset,
    calculate_class_weights,
    create_data_loader,
)
from src.models import create_resnet18


def main() -> None:
    device = torch.device("cpu")

    train_dataset = ChestXRayDataset("train")

    class_weights = calculate_class_weights(
        train_dataset
    ).to(device)

    train_loader = create_data_loader(
        target_split="train",
        batch_size=4,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

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
        lr=0.001,
    )

    optimizer.zero_grad()

    logits = model(images)

    loss = criterion(
        logits,
        labels,
    )

    loss.backward()

    gradient_norm = (
        model.fc.weight.grad.norm().item()
    )

    optimizer.step()

    print("Class weights:", class_weights)
    print("Labels:", labels)
    print("Logits shape:", logits.shape)
    print(f"Weighted loss: {loss.item():.6f}")
    print(
        f"FC gradient norm: "
        f"{gradient_norm:.6f}"
    )


if __name__ == "__main__":
    main()