from pathlib import Path
import sys

import torch
from PIL import Image

from src.data import MANIFEST_PATH
from src.predict import DEFAULT_CHECKPOINT_PATH, load_model, predict, predict_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
    / "NORMAL"
    / "NORMAL2-IM-0256-0001.jpeg"
)


def print_check(name: str, passed: bool, details: str = "") -> None:
    status = "OK" if passed else "FAILED"
    message = f"[{status}] {name}"

    if details:
        message += f": {details}"

    print(message)


def print_skipped(name: str, details: str = "") -> None:
    message = f"[SKIPPED] {name}"

    if details:
        message += f": {details}"

    print(message)


def main() -> int:
    print("Chest X-Ray Pneumonia AI Project")
    print("Installation / project verification")
    print("-" * 50)

    print_check(
        "Python",
        True,
        sys.version.split()[0],
    )

    print_check(
        "PyTorch",
        True,
        torch.__version__,
    )

    manifest_exists = MANIFEST_PATH.exists()
    print_check(
        "Split manifest",
        manifest_exists,
        str(MANIFEST_PATH),
    )

    if not manifest_exists:
        return 1

    checkpoint_exists = DEFAULT_CHECKPOINT_PATH.exists()
    print_check(
        "Model checkpoint",
        checkpoint_exists,
        str(DEFAULT_CHECKPOINT_PATH),
    )

    if not checkpoint_exists:
        print()
        print(
            "The source repository is valid, but real model inference "
            "cannot be verified without the trained checkpoint."
        )
        print("Expected checkpoint:")
        print(DEFAULT_CHECKPOINT_PATH)
        return 1

    try:
        model, device = load_model()
    except Exception as error:
        print_check(
            "Model loading",
            False,
            str(error),
        )
        return 1

    print_check(
        "Model loading",
        True,
        f"device={device}",
    )

    # Always perform a real forward pass without requiring the raw dataset.
    try:
        synthetic_image = Image.new(
            mode="L",
            size=(224, 224),
            color=128,
        )

        synthetic_result = predict(
            image=synthetic_image,
            model=model,
            device=device,
        )

        probability = synthetic_result["probability"]

        synthetic_valid = (
            synthetic_result["prediction"] in {"NORMAL", "PNEUMONIA"}
            and 0.0 <= probability <= 1.0
            and synthetic_result["model"] == "resnet18_baseline"
        )

    except Exception as error:
        print_check(
            "Synthetic inference",
            False,
            str(error),
        )
        return 1

    print_check(
        "Synthetic inference",
        synthetic_valid,
        (
            f"prediction={synthetic_result['prediction']}, "
            f"P(PNEUMONIA)={probability:.6f}"
        ),
    )

    if not synthetic_valid:
        return 1

    # Optional reproducibility check using a known dataset sample.
    if not TEST_IMAGE_PATH.exists():
        print_skipped(
            "Known reference inference",
            "raw dataset is not available",
        )

        print()
        print("PROJECT VERIFICATION PASSED")
        return 0

    try:
        result = predict_file(
            image_path=TEST_IMAGE_PATH,
            model=model,
            device=device,
        )
    except Exception as error:
        print_check(
            "Known reference inference",
            False,
            str(error),
        )
        return 1

    probability = result["probability"]

    expected_prediction = "PNEUMONIA"
    expected_probability = 0.9999955892562866

    prediction_matches = result["prediction"] == expected_prediction
    probability_matches = abs(probability - expected_probability) < 1e-5

    print_check(
        "Known reference inference",
        prediction_matches and probability_matches,
        (
            f"prediction={result['prediction']}, "
            f"P(PNEUMONIA)={probability:.6f}"
        ),
    )

    if not prediction_matches or not probability_matches:
        return 1

    print()
    print("PROJECT VERIFICATION PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())