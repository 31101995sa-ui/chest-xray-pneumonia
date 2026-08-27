import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.data import ChestXRayDataset
from src.models import create_resnet18
from src.train import train_one_epoch


def main() -> None:
    device = torch.device("cpu")

    full_dataset = ChestXRayDataset("train")

    small_dataset = Subset(
        full_dataset,
        range(64),
    )

    small_loader = DataLoader(
        small_dataset,
        batch_size=32,
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

    average_loss, accuracy = train_one_epoch(
        model=model,
        data_loader=small_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print(f"Samples used: {len(small_dataset)}")
    print(f"Batches used: {len(small_loader)}")
    print(f"Average loss: {average_loss:.6f}")
    print(f"Training accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()