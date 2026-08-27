import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import ChestXRayDataset
from src.models import create_resnet18
from src.train import validate_one_epoch


def main() -> None:
    device = torch.device("cpu")

    full_dataset = ChestXRayDataset("val")

    normal_indices = []
    pneumonia_indices = []

    for index, record in enumerate(full_dataset.records):
        label = record["label"]

        if label == 0 and len(normal_indices) < 32:
            normal_indices.append(index)

        elif label == 1 and len(pneumonia_indices) < 32:
            pneumonia_indices.append(index)

        if (
            len(normal_indices) == 32
            and len(pneumonia_indices) == 32
        ):
            break

    selected_indices = (
        normal_indices
        + pneumonia_indices
    )

    small_dataset = Subset(
        full_dataset,
        selected_indices,
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

    weights_before = (
        model.fc.weight
        .detach()
        .clone()
    )

    average_loss, accuracy = validate_one_epoch(
        model=model,
        data_loader=small_loader,
        criterion=criterion,
        device=device,
    )

    weights_after = (
        model.fc.weight
        .detach()
        .clone()
    )

    max_weight_change = (
        weights_after
        - weights_before
    ).abs().max().item()

    print(f"Samples used: {len(small_dataset)}")
    print(f"NORMAL samples: {len(normal_indices)}")
    print(
        f"PNEUMONIA samples: "
        f"{len(pneumonia_indices)}"
    )
    print(f"Batches used: {len(small_loader)}")
    print(f"Validation loss: {average_loss:.6f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(
        f"Max FC weight change: "
        f"{max_weight_change:.8f}"
    )


if __name__ == "__main__":
    main()