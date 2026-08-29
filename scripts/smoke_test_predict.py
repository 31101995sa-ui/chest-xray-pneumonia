from src.data import DATASET_ROOT
from src.predict import (
    load_model,
    predict_file,
)


TEST_IMAGE_PATH = (
    DATASET_ROOT
    / "test"
    / "NORMAL"
    / "NORMAL2-IM-0256-0001.jpeg"
)


def main() -> None:
    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: "
            f"{TEST_IMAGE_PATH}"
        )

    model, device = load_model()

    result = predict_file(
        image_path=TEST_IMAGE_PATH,
        model=model,
        device=device,
    )

    required_keys = {
        "prediction",
        "probability",
        "model",
        "disclaimer",
    }

    if set(result.keys()) != required_keys:
        raise RuntimeError(
            "Unexpected prediction "
            f"result keys: {result.keys()}"
        )

    if result["prediction"] not in {
        "NORMAL",
        "PNEUMONIA",
    }:
        raise RuntimeError(
            "Unexpected prediction label: "
            f"{result['prediction']}"
        )

    probability = result[
        "probability"
    ]

    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            "Probability is outside "
            f"[0, 1]: {probability}"
        )

    print(
        "=== predict() smoke test ==="
    )
    print(
        "Image:",
        TEST_IMAGE_PATH.name,
    )
    print(
        "Device:",
        device,
    )
    print(
        "Prediction:",
        result["prediction"],
    )
    print(
        "P(PNEUMONIA):",
        f"{probability:.6f}",
    )
    print(
        "Model:",
        result["model"],
    )

    print()

    print(
        "Valid result structure: True"
    )
    print(
        "Probability in [0, 1]: True"
    )

    known_result_reproduced = (
        result["prediction"]
        == "PNEUMONIA"
        and probability > 0.99
    )

    print(
        "Known sealed-test result "
        "reproduced:",
        known_result_reproduced,
    )

    if not known_result_reproduced:
        raise RuntimeError(
            "Inference result does not "
            "reproduce the known "
            "sealed-test prediction."
        )


if __name__ == "__main__":
    main()