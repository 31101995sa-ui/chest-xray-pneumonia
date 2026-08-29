from io import BytesIO

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main


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


def fake_load_model():
    model = DummyModel()
    model.eval()

    device = torch.device("cpu")

    return model, device


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "load_model",
        fake_load_model,
    )

    with TestClient(
        api_main.app
    ) as test_client:
        yield test_client


def create_test_image_bytes() -> bytes:
    image = Image.new(
        "L",
        (100, 100),
        color=128,
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
    )

    return buffer.getvalue()


def test_health_returns_200(client):
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert (
        data["service"]
        == "chest-xray-pneumonia-api"
    )


def test_predict_valid_image_returns_200(
    client,
):
    image_bytes = (
        create_test_image_bytes()
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpeg",
                image_bytes,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["prediction"]
        == "PNEUMONIA"
    )

    assert (
        0.0
        <= data["probability"]
        <= 1.0
    )

    assert (
        data["model"]
        == "resnet18_baseline"
    )

    assert "disclaimer" in data


def test_predict_rejects_non_image(
    client,
):
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "Uploaded file must be "
            "an image."
        )
    }


def test_predict_rejects_corrupted_image(
    client,
):
    response = client.post(
        "/predict",
        files={
            "file": (
                "broken.jpeg",
                b"this is not really jpeg data",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "Invalid or unsupported "
            "image file."
        )
    }