from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.predict import (
    load_model,
    predict,
    predict_file,
)


class DummyModel(nn.Module):
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = images.shape[0]

        logits = torch.tensor(
            [[0.0, 2.0]],
            dtype=torch.float32,
            device=images.device,
        )

        return logits.repeat(
            batch_size,
            1,
        )


def test_predict_returns_valid_result():
    device = torch.device("cpu")

    model = DummyModel()
    model.eval()

    image = Image.new(
        "L",
        (100, 100),
        color=128,
    )

    result = predict(
        image=image,
        model=model,
        device=device,
    )

    assert result["prediction"] == "PNEUMONIA"

    assert (
        0.0
        <= result["probability"]
        <= 1.0
    )

    assert result["probability"] > 0.5

    assert (
        result["model"]
        == "resnet18_baseline"
    )

    assert "disclaimer" in result


def test_predict_file_missing_image():
    device = torch.device("cpu")

    model = DummyModel()
    model.eval()

    missing_path = Path(
        "this_file_does_not_exist.jpeg"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        predict_file(
            image_path=missing_path,
            model=model,
            device=device,
        )


def test_load_model_missing_checkpoint():
    missing_checkpoint = Path(
        "missing_checkpoint.pth"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_model(
            checkpoint_path=(
                missing_checkpoint
            ),
            device=torch.device("cpu"),
        )