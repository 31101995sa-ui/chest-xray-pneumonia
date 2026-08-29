from pathlib import Path

import torch
from PIL import Image

from src.data import preprocess_image
from src.models import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_baseline_repro_best.pth"
)

LABEL_NAMES = {
    0: "NORMAL",
    1: "PNEUMONIA",
}


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = create_resnet18(
        pretrained=False,
        freeze_backbone=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, device


def predict(
    image: Image.Image,
    model,
    device: torch.device,
) -> dict:
    image_tensor = preprocess_image(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    pneumonia_probability = (
        probabilities[0, 1]
        .item()
    )

    predicted_label_id = (
        probabilities
        .argmax(dim=1)
        .item()
    )

    predicted_label = LABEL_NAMES[
        predicted_label_id
    ]

    return {
        "prediction": predicted_label,
        "probability": (
            pneumonia_probability
        ),
        "model": "resnet18_baseline",
        "disclaimer": (
            "Educational/research use only. "
            "Not intended for clinical diagnosis."
        ),
    }


def predict_file(
    image_path: Path,
    model,
    device: torch.device,
) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    with Image.open(
        image_path
    ) as image:
        image = image.convert("RGB")

        return predict(
            image=image,
            model=model,
            device=device,
        )