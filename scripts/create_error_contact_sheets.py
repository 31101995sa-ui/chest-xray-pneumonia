from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FALSE_POSITIVE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "false_positives"
)

FALSE_NEGATIVE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "errors"
    / "false_negatives"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FALSE_POSITIVE_SHEET = (
    FIGURES_DIR
    / "resnet18_false_positives_contact_sheet.png"
)

FALSE_NEGATIVE_SHEET = (
    FIGURES_DIR
    / "resnet18_false_negatives_contact_sheet.png"
)

TILE_WIDTH = 300
IMAGE_HEIGHT = 300
TEXT_HEIGHT = 80
TILE_HEIGHT = IMAGE_HEIGHT + TEXT_HEIGHT

COLUMNS = 5


def load_font(
    size: int,
) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "arial.ttf",
            size=size,
        )
    except OSError:
        return ImageFont.load_default()


def split_filename(
    filename: str,
) -> tuple[str, str]:
    parts = filename.split(
        "_",
        maxsplit=3,
    )

    if len(parts) < 4:
        return filename, ""

    summary = (
        f"{parts[0]} | "
        f"{parts[1]} | "
        f"{parts[2]}"
    )

    original_name = parts[3]

    return summary, original_name


def create_contact_sheet(
    image_directory: Path,
    output_path: Path,
    title: str,
) -> None:
    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in "
            f"{image_directory}"
        )

    rows = (
        len(image_paths)
        + COLUMNS
        - 1
    ) // COLUMNS

    title_height = 70

    sheet_width = (
        COLUMNS
        * TILE_WIDTH
    )

    sheet_height = (
        title_height
        + rows * TILE_HEIGHT
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    title_font = load_font(24)
    text_font = load_font(15)

    draw.text(
        (20, 20),
        title,
        fill="black",
        font=title_font,
    )

    for index, image_path in enumerate(
        image_paths
    ):
        row = index // COLUMNS
        column = index % COLUMNS

        x = column * TILE_WIDTH
        y = (
            title_height
            + row * TILE_HEIGHT
        )

        with Image.open(
            image_path
        ) as image:
            image = image.convert("RGB")

            fitted = ImageOps.contain(
                image,
                (
                    TILE_WIDTH - 20,
                    IMAGE_HEIGHT - 20,
                ),
            )

        image_x = (
            x
            + (
                TILE_WIDTH
                - fitted.width
            ) // 2
        )

        image_y = (
            y
            + (
                IMAGE_HEIGHT
                - fitted.height
            ) // 2
        )

        sheet.paste(
            fitted,
            (
                image_x,
                image_y,
            ),
        )

        summary, original_name = (
            split_filename(
                image_path.name
            )
        )

        draw.text(
            (
                x + 8,
                y + IMAGE_HEIGHT + 5,
            ),
            summary,
            fill="black",
            font=text_font,
        )

        draw.text(
            (
                x + 8,
                y + IMAGE_HEIGHT + 30,
            ),
            original_name,
            fill="black",
            font=text_font,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        format="PNG",
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Images: {len(image_paths)}"
    )


def main() -> None:
    print(
        "=== False Positive "
        "contact sheet ==="
    )

    create_contact_sheet(
        image_directory=(
            FALSE_POSITIVE_DIR
        ),
        output_path=(
            FALSE_POSITIVE_SHEET
        ),
        title=(
            "ResNet18 — "
            "Top 10 False Positives"
        ),
    )

    print()

    print(
        "=== False Negative "
        "contact sheet ==="
    )

    create_contact_sheet(
        image_directory=(
            FALSE_NEGATIVE_DIR
        ),
        output_path=(
            FALSE_NEGATIVE_SHEET
        ),
        title=(
            "ResNet18 — "
            "All 5 False Negatives"
        ),
    )


if __name__ == "__main__":
    main()