import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "split_manifest.csv"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_base_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.convert("RGB")
            ),
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def preprocess_image(
    image_path: Path,
) -> torch.Tensor:
    transform = build_base_transform()

    with Image.open(image_path) as image:
        tensor = transform(image)

    return tensor


class ChestXRayDataset(Dataset):
    def __init__(
        self,
        target_split: str,
        manifest_path: Path = MANIFEST_PATH,
        dataset_root: Path = DATASET_ROOT,
    ) -> None:
        self.dataset_root = dataset_root
        self.transform = build_base_transform()
        self.records = []

        with manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                if row["target_split"] != target_split:
                    continue

                self.records.append(
                    {
                        "relative_path": row["relative_path"],
                        "class_name": row["class"],
                        "label": int(row["label"]),
                    }
                )

        if not self.records:
            raise ValueError(
                f"No records found for split: "
                f"{target_split}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        record = self.records[index]

        image_path = (
            self.dataset_root
            / Path(record["relative_path"])
        )

        with Image.open(image_path) as image:
            image_tensor = self.transform(image)

        label = record["label"]

        return image_tensor, label


def calculate_class_weights(
    dataset: ChestXRayDataset,
) -> torch.Tensor:
    class_counts = {
        0: 0,
        1: 0,
    }

    for record in dataset.records:
        label = record["label"]
        class_counts[label] += 1

    total_samples = len(dataset)
    num_classes = len(class_counts)

    weights = []

    for class_index in range(num_classes):
        class_count = class_counts[class_index]

        class_weight = (
            total_samples
            / (num_classes * class_count)
        )

        weights.append(class_weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def create_data_loader(
    target_split: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    manifest_path: Path = MANIFEST_PATH,
    dataset_root: Path = DATASET_ROOT,
) -> DataLoader:
    dataset = ChestXRayDataset(
        target_split=target_split,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )

    shuffle = target_split == "train"

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )