import torch

from src.data import create_data_loader
from src.models import create_efficientnet_b0


def main() -> None:
    loader = create_data_loader(
        target_split="train",
        batch_size=4,
    )

    images, labels = next(iter(loader))

    model = create_efficientnet_b0(
        pretrained=True,
        freeze_backbone=True,
    )

    model.eval()

    print(f"Input batch: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Model training mode: {model.training}")

    with torch.no_grad():
        logits = model(images)

    print(f"Output logits: {logits.shape}")
    print(logits)


if __name__ == "__main__":
    main()