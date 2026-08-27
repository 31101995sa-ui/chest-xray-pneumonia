import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"

CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

SAMPLES_PER_CLASS = 3


def get_image_paths(directory: Path) -> list[Path]:
    image_paths = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(file_path)

    return image_paths


def main() -> None:
    train_dir = DATASET_ROOT / "train"

    figure, axes = plt.subplots(
        len(CLASSES),
        SAMPLES_PER_CLASS,
        figsize=(12, 7),
    )

    for row_index, class_name in enumerate(CLASSES):
        class_dir = train_dir / class_name
        image_paths = get_image_paths(class_dir)

        selected_paths = random.sample(
            image_paths,
            k=SAMPLES_PER_CLASS,
        )

        for column_index, image_path in enumerate(selected_paths):
            with Image.open(image_path) as image:
                axes[row_index, column_index].imshow(
                    image,
                    cmap="gray",
                )

            axes[row_index, column_index].set_title(class_name)
            axes[row_index, column_index].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()