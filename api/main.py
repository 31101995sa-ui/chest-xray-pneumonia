from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError

from src.predict import (
    load_model,
    predict,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, device = load_model()

    app.state.model = model
    app.state.device = device

    print(
        f"Model loaded on device: {device}"
    )

    yield


app = FastAPI(
    title="Chest X-Ray Pneumonia AI API",
    version="0.1.0",
    description=(
        "Educational/research API for "
        "NORMAL vs PNEUMONIA classification."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "chest-xray-pneumonia-api",
    }


@app.post("/predict")
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    if (
        file.content_type is not None
        and not file.content_type.startswith(
            "image/"
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        file_bytes = await file.read()

        with Image.open(
            BytesIO(file_bytes)
        ) as image:
            image = image.convert("RGB")

            result = predict(
                image=image,
                model=request.app.state.model,
                device=request.app.state.device,
            )

    except (
        UnidentifiedImageError,
        OSError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid or unsupported "
                "image file."
            ),
        ) from error

    return result