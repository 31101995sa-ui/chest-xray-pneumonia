import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = logits.argmax(
                dim=1,
            )

            pneumonia_probabilities = (
                probabilities[:, 1]
            )

            y_true.extend(
                labels.cpu().tolist()
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

            y_prob.extend(
                pneumonia_probabilities
                .cpu()
                .tolist()
            )

    return y_true, y_pred, y_prob