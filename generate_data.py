import argparse
import csv
from pathlib import Path
import shutil

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 1000

BOTTLE_HEIGHT = 700
BODY_WIDTH = 300
NECK_WIDTH = 120

NORMAL_BODY_VARIATION = 3.0
NORMAL_NECK_VARIATION = 2.0
NORMAL_HEIGHT_VARIATION = 4

CENTER_X = IMAGE_WIDTH // 2
BOTTOM_Y = 850

DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_RANDOM_SEED = 42

DEFECT_TYPES = (
    "normal",
    "bulge",
    "shrink",
    "lean",
    "height",
    "neck",
)

METADATA_FIELDNAMES = (
    "filename",
    "defect_type",
    "body_width",
    "neck_width",
    "bottle_height",
    "magnitude",
    "center",
    "sigma",
    "lean_shift",
    "height_change",
)


# ============================================================
# Directory management
# ============================================================

def clean_output_directory(
    output_dir: Path,
) -> None:
    """
    Remove previously generated dataset artifacts.

    Only generator-managed paths are removed.
    The output directory itself is preserved.
    """

    output_dir = Path(
        output_dir
    )

    images_dir = (
        output_dir
        / "images"
    )

    masks_dir = (
        output_dir
        / "masks"
    )

    metadata_path = (
        output_dir
        / "metadata.csv"
    )

    for directory in (
        images_dir,
        masks_dir,
    ):
        if directory.exists():
            shutil.rmtree(
                directory
            )

    if metadata_path.exists():
        metadata_path.unlink()

def create_directories(
    images_dir: Path,
    masks_dir: Path,
) -> None:
    """
    Create the directory structure required for
    generated images and ground-truth masks.
    """

    (images_dir / "normal").mkdir(
        parents=True,
        exist_ok=True,
    )

    (masks_dir / "normal").mkdir(
        parents=True,
        exist_ok=True,
    )

    for defect_type in DEFECT_TYPES:

        if defect_type == "normal":
            continue

        (
            images_dir
            / "defective"
            / defect_type
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            masks_dir
            / "defective"
            / defect_type
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Base bottle geometry
# ============================================================

def get_bottle_width(
    y_normalized: float,
    body_width: float,
    neck_width: float,
) -> float:
    """
    Return the nominal bottle width at a normalized
    vertical position.

    0.0 represents the bottle top.
    1.0 represents the bottle bottom.
    """

    if y_normalized < 0.18:
        return neck_width

    if y_normalized < 0.32:

        t = (
            (y_normalized - 0.18)
            / (0.32 - 0.18)
        )

        return (
            neck_width
            + t
            * (body_width - neck_width)
        )

    return body_width


def get_bottle_bounds(
    bottle_height: int,
    defect_type: str,
    height_change: int,
) -> tuple[int, int]:
    """
    Calculate the vertical bottle boundaries.
    """

    final_height = bottle_height

    if defect_type == "height":
        final_height += height_change

    if final_height <= 0:
        raise ValueError(
            "Final bottle height must be positive."
        )

    bottom_y = BOTTOM_Y
    top_y = bottom_y - final_height

    return top_y, bottom_y


# ============================================================
# Defect models
# ============================================================

def apply_bulge(
    width: float,
    y_normalized: float,
    center: float,
    magnitude: float,
    sigma: float,
) -> float:
    """
    Apply a smooth outward deformation.
    """

    deformation = magnitude * np.exp(
        -0.5
        * (
            (y_normalized - center)
            / sigma
        ) ** 2
    )

    return width + deformation


def apply_shrink(
    width: float,
    y_normalized: float,
    center: float,
    magnitude: float,
    sigma: float,
) -> float:
    """
    Apply a smooth inward deformation.
    """

    deformation = magnitude * np.exp(
        -0.5
        * (
            (y_normalized - center)
            / sigma
        ) ** 2
    )

    return width - deformation


def apply_neck_defect(
    width: float,
    y_normalized: float,
    magnitude: float,
) -> float:
    """
    Apply a width deformation to the bottle neck.
    """

    if y_normalized < 0.18:
        return width + magnitude

    return width


def get_bottle_center(
    y_normalized: float,
    defect_type: str,
    lean_shift: float,
) -> float:
    """
    Return the horizontal center of the bottle
    at the requested vertical position.
    """

    if defect_type == "lean":

        shift = (
            lean_shift
            * (1.0 - y_normalized)
        )

        return CENTER_X + shift

    return float(CENTER_X)


# ============================================================
# Mask generation
# ============================================================

def create_bottle_mask(
    defect_type: str,
    body_width: float,
    neck_width: float,
    bottle_height: int,
    magnitude: float = 0.0,
    center: float = 0.0,
    sigma: float = 0.0,
    lean_shift: float = 0.0,
    height_change: int = 0,
) -> np.ndarray:
    """
    Generate a ground-truth bottle mask.
    """

    mask = np.zeros(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ),
        dtype=np.uint8,
    )

    top_y, bottom_y = (
        get_bottle_bounds(
            bottle_height=bottle_height,
            defect_type=defect_type,
            height_change=height_change,
        )
    )

    if top_y < 0:
        raise ValueError(
            "Bottle extends beyond the image boundary."
        )

    if bottom_y >= IMAGE_HEIGHT:
        raise ValueError(
            "Bottle extends beyond the image boundary."
        )

    current_height = (
        bottom_y - top_y
    )

    left_points = []
    right_points = []

    for y in range(
        top_y,
        bottom_y + 1,
    ):

        y_normalized = (
            (y - top_y)
            / current_height
        )

        width = get_bottle_width(
            y_normalized,
            body_width=body_width,
            neck_width=neck_width,
        )

        if defect_type == "bulge":

            width = apply_bulge(
                width=width,
                y_normalized=y_normalized,
                center=center,
                magnitude=magnitude,
                sigma=sigma,
            )

        elif defect_type == "shrink":

            width = apply_shrink(
                width=width,
                y_normalized=y_normalized,
                center=center,
                magnitude=magnitude,
                sigma=sigma,
            )

        elif defect_type == "neck":

            width = apply_neck_defect(
                width=width,
                y_normalized=y_normalized,
                magnitude=magnitude,
            )

        center_x = get_bottle_center(
            y_normalized=y_normalized,
            defect_type=defect_type,
            lean_shift=lean_shift,
        )

        left_x = int(
            center_x - width / 2
        )

        right_x = int(
            center_x + width / 2
        )

        left_points.append(
            [left_x, y]
        )

        right_points.append(
            [right_x, y]
        )

    bottle_points = np.asarray(
        left_points
        + right_points[::-1],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [bottle_points],
        255,
    )

    return mask


# ============================================================
# Synthetic image rendering
# ============================================================

def render_image(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Render a synthetic backlight-style camera image
    from a clean ground-truth mask.

    Noise and imaging effects are applied only to the
    rendered image. The ground-truth mask remains clean.
    """

    background_brightness = rng.uniform(
        215.0,
        245.0,
    )

    bottle_brightness = rng.uniform(
        25.0,
        65.0,
    )

    image = np.full(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ),
        background_brightness,
        dtype=np.float32,
    )

    image[
        mask == 255
    ] = bottle_brightness

    # Simulate non-uniform illumination.
    gradient_strength = rng.uniform(
        -15.0,
        15.0,
    )

    gradient = np.linspace(
        -gradient_strength,
        gradient_strength,
        IMAGE_WIDTH,
        dtype=np.float32,
    )

    image += gradient[
        np.newaxis,
        :
    ]

    # Simulate optical blur.
    blur_kernel = int(
        rng.choice(
            [3, 5, 7]
        )
    )

    image = cv2.GaussianBlur(
        image,
        (
            blur_kernel,
            blur_kernel,
        ),
        0,
    )

    # Simulate sensor noise.
    noise_sigma = rng.uniform(
        1.0,
        5.0,
    )

    noise = rng.normal(
        0.0,
        noise_sigma,
        image.shape,
    )

    image += noise

    image = np.clip(
        image,
        0,
        255,
    )

    return image.astype(
        np.uint8
    )


# ============================================================
# Random parameter generation
# ============================================================

def generate_normal_geometry(
    rng: np.random.Generator,
) -> dict:
    """
    Generate normal manufacturing variation around
    the nominal bottle geometry.
    """

    body_width = rng.uniform(
        BODY_WIDTH
        - NORMAL_BODY_VARIATION,
        BODY_WIDTH
        + NORMAL_BODY_VARIATION,
    )

    neck_width = rng.uniform(
        NECK_WIDTH
        - NORMAL_NECK_VARIATION,
        NECK_WIDTH
        + NORMAL_NECK_VARIATION,
    )

    bottle_height = int(
        rng.integers(
            BOTTLE_HEIGHT
            - NORMAL_HEIGHT_VARIATION,
            BOTTLE_HEIGHT
            + NORMAL_HEIGHT_VARIATION
            + 1,
        )
    )

    return {
        "body_width": body_width,
        "neck_width": neck_width,
        "bottle_height": bottle_height,
    }


def generate_random_parameters(
    defect_type: str,
    rng: np.random.Generator,
) -> dict:
    """
    Generate random parameters for the selected
    defect type.
    """

    if defect_type not in DEFECT_TYPES:
        raise ValueError(
            f"Unsupported defect type: {defect_type}"
        )

    parameters = {
        "magnitude": 0.0,
        "center": 0.0,
        "sigma": 0.0,
        "lean_shift": 0.0,
        "height_change": 0,
    }

    if defect_type in {
        "bulge",
        "shrink",
    }:

        parameters["magnitude"] = float(
            rng.uniform(
                5.0,
                40.0,
            )
        )

        parameters["center"] = float(
            rng.uniform(
                0.40,
                0.80,
            )
        )

        parameters["sigma"] = float(
            rng.uniform(
                0.03,
                0.10,
            )
        )

    elif defect_type == "lean":

        direction = float(
            rng.choice(
                [-1.0, 1.0]
            )
        )

        parameters["lean_shift"] = (
            direction
            * float(
                rng.uniform(
                    5.0,
                    40.0,
                )
            )
        )

    elif defect_type == "height":

        direction = int(
            rng.choice(
                [-1, 1]
            )
        )

        parameters["height_change"] = int(
            direction
            * rng.integers(
                10,
                61,
            )
        )

    elif defect_type == "neck":

        direction = float(
            rng.choice(
                [-1.0, 1.0]
            )
        )

        parameters["magnitude"] = (
            direction
            * float(
                rng.uniform(
                    5.0,
                    35.0,
                )
            )
        )

    return parameters


# ============================================================
# Path helpers
# ============================================================

def get_sample_paths(
    images_dir: Path,
    masks_dir: Path,
    defect_type: str,
    filename: str,
    mask_filename: str,
) -> tuple[Path, Path]:
    """
    Return output paths for an image and its mask.
    """

    if defect_type == "normal":

        image_path = (
            images_dir
            / "normal"
            / filename
        )

        mask_path = (
            masks_dir
            / "normal"
            / mask_filename
        )

    else:

        image_path = (
            images_dir
            / "defective"
            / defect_type
            / filename
        )

        mask_path = (
            masks_dir
            / "defective"
            / defect_type
            / mask_filename
        )

    return (
        image_path,
        mask_path,
    )


# ============================================================
# Sample generation
# ============================================================

def generate_sample(
    defect_type: str,
    sample_index: int,
    rng: np.random.Generator,
    images_dir: Path,
    masks_dir: Path,
) -> dict:
    """
    Generate one synthetic bottle sample and save
    both the rendered image and ground-truth mask.
    """

    geometry = generate_normal_geometry(
        rng
    )

    parameters = (
        generate_random_parameters(
            defect_type,
            rng,
        )
    )

    mask = create_bottle_mask(
        defect_type=defect_type,
        body_width=geometry[
            "body_width"
        ],
        neck_width=geometry[
            "neck_width"
        ],
        bottle_height=geometry[
            "bottle_height"
        ],
        magnitude=parameters[
            "magnitude"
        ],
        center=parameters[
            "center"
        ],
        sigma=parameters[
            "sigma"
        ],
        lean_shift=parameters[
            "lean_shift"
        ],
        height_change=parameters[
            "height_change"
        ],
    )

    image = render_image(
        mask,
        rng,
    )

    filename = (
        f"{defect_type}_"
        f"{sample_index:04d}.png"
    )

    mask_filename = (
        f"{defect_type}_"
        f"{sample_index:04d}_mask.png"
    )

    (
        image_path,
        mask_path,
    ) = get_sample_paths(
        images_dir=images_dir,
        masks_dir=masks_dir,
        defect_type=defect_type,
        filename=filename,
        mask_filename=mask_filename,
    )

    image_saved = cv2.imwrite(
        str(image_path),
        image,
    )

    if not image_saved:
        raise RuntimeError(
            f"Image could not be saved: {image_path}"
        )

    mask_saved = cv2.imwrite(
        str(mask_path),
        mask,
    )

    if not mask_saved:
        raise RuntimeError(
            f"Mask could not be saved: {mask_path}"
        )

    return {
        "filename": filename,
        "defect_type": defect_type,
        "body_width": geometry[
            "body_width"
        ],
        "neck_width": geometry[
            "neck_width"
        ],
        "bottle_height": geometry[
            "bottle_height"
        ],
        "magnitude": parameters[
            "magnitude"
        ],
        "center": parameters[
            "center"
        ],
        "sigma": parameters[
            "sigma"
        ],
        "lean_shift": parameters[
            "lean_shift"
        ],
        "height_change": parameters[
            "height_change"
        ],
    }


# ============================================================
# Metadata
# ============================================================

def save_metadata(
    metadata: list[dict],
    metadata_path: Path,
) -> None:
    """
    Save generated sample metadata as CSV.
    """

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=METADATA_FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(
            metadata
        )


# ============================================================
# Dataset generation pipeline
# ============================================================

def generate_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    """
    Generate the complete synthetic bottle dataset.

    Returns
    -------
    Path
        Path to the generated metadata.csv file.
    """

    output_dir = Path(
        output_dir
    )

    if samples_per_class <= 0:
        raise ValueError(
            "samples_per_class must be greater than zero."
        )

    images_dir = (
        output_dir
        / "images"
    )

    masks_dir = (
        output_dir
        / "masks"
    )

    metadata_path = (
        output_dir
        / "metadata.csv"
    )

    clean_output_directory(
        output_dir
    )
    
    create_directories(
        images_dir=images_dir,
        masks_dir=masks_dir,
    )

    rng = np.random.default_rng(
        seed
    )

    metadata = []

    for defect_type in DEFECT_TYPES:

        print(
            f"Generating "
            f"{defect_type} samples..."
        )

        for sample_index in range(
            1,
            samples_per_class + 1,
        ):

            sample_metadata = (
                generate_sample(
                    defect_type=defect_type,
                    sample_index=sample_index,
                    rng=rng,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                )
            )

            metadata.append(
                sample_metadata
            )

    save_metadata(
        metadata=metadata,
        metadata_path=metadata_path,
    )

    total_samples = len(
        metadata
    )

    print()
    print(
        "Dataset generation completed."
    )
    print(
        f"Total samples: {total_samples}"
    )
    print(
        f"Samples per class: "
        f"{samples_per_class}"
    )
    print(
        f"Random seed: {seed}"
    )
    print(
        f"Output directory: "
        f"{output_dir}"
    )
    print(
        f"Metadata: "
        f"{metadata_path}"
    )

    return metadata_path


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic dataset for "
            "glass bottle geometry inspection."
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=(
            "Number of samples generated for "
            "each bottle class. "
            f"Default: {DEFAULT_SAMPLES_PER_CLASS}"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed used for reproducible "
            "dataset generation. "
            f"Default: {DEFAULT_RANDOM_SEED}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Dataset output directory. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    CLI entry point.
    """

    args = parse_arguments()

    generate_dataset(
        output_dir=args.output_dir,
        samples_per_class=(
            args.samples_per_class
        ),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
