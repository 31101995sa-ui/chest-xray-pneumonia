from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        running_loss / total_samples
    )

    accuracy = (
        correct_predictions / total_samples
    )

    return average_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = logits.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    average_loss = (
        running_loss / total_samples
    )

    accuracy = (
        correct_predictions / total_samples
    )

    return average_loss, accuracy


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    model_name: str,
) -> list[dict]:
    history = []

    best_val_loss = float("inf")

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            print(
                f"  Saved best checkpoint: "
                f"{checkpoint_path}"
            )

    return history