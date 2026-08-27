import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import create_data_loader
from src.models import create_resnet18


def main() -> None:
    loader = create_data_loader(
        "train",
        batch_size=4,
    )

    images, labels = next(iter(loader))

    model = create_resnet18(
        pretrained=True,
        freeze_backbone=True,
    )

    model.eval()
    model.fc.train()

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=0.001,
    )

    weights_before = model.fc.weight.detach().clone()

    logits = model(images)

    loss = criterion(
        logits,
        labels,
    )

    print(f"Input batch: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss before step: {loss.item():.6f}")

    optimizer.zero_grad()

    loss.backward()

    gradient_norm = model.fc.weight.grad.norm().item()

    print(f"FC gradient norm: {gradient_norm:.6f}")

    optimizer.step()

    weights_after = model.fc.weight.detach().clone()

    max_weight_change = (
        weights_after - weights_before
    ).abs().max().item()

    print(
        f"Max FC weight change: "
        f"{max_weight_change:.8f}"
    )

    with torch.no_grad():
        logits_after = model(images)

        loss_after = criterion(
            logits_after,
            labels,
        )

    print(
        f"Loss after step: "
        f"{loss_after.item():.6f}"
    )


if __name__ == "__main__":
    main()