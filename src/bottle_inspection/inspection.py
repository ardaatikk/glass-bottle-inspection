import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEFAULT_OUTPUT_DIR = Path("outputs/inspection")

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Image loading
# ============================================================

def load_image(
    image_path: Path,
) -> np.ndarray:
    """
    Load an image from disk in grayscale.

    Parameters
    ----------
    image_path:
        Path to the input image.

    Returns
    -------
    np.ndarray
        Grayscale image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Image path is not a file: {image_path}"
        )

    if (
        image_path.suffix.lower()
        not in SUPPORTED_IMAGE_EXTENSIONS
    ):
        raise ValueError(
            "Unsupported image format: "
            f"{image_path.suffix}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Image could not be loaded: {image_path}"
        )

    return image


# ============================================================
# Image preparation
# ============================================================

def prepare_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert an input image to grayscale when necessary.

    This allows the core inspection pipeline to accept
    grayscale images as well as BGR camera/video frames.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if image.ndim == 2:
        return image.copy()

    if (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    if (
        image.ndim == 3
        and image.shape[2] == 4
    ):
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )

    raise ValueError(
        "Unsupported image shape: "
        f"{image.shape}"
    )


# ============================================================
# Segmentation
# ============================================================

def segment_bottle(
    image: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Segment the bottle from the background using
    Otsu thresholding.

    The current pipeline assumes that the bottle is
    darker than the background.
    """

    threshold_value, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    return binary, float(
        threshold_value
    )


def clean_binary(
    binary: np.ndarray,
) -> np.ndarray:
    """
    Reduce small gaps and discontinuities in the
    segmented bottle using morphological closing.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return cleaned


def extract_bottle_contour(
    binary: np.ndarray,
) -> np.ndarray:
    """
    Extract the largest external contour.

    The largest external contour is assumed to
    represent the bottle.
    """

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise RuntimeError(
            "No contour detected."
        )

    bottle_contour = max(
        contours,
        key=cv2.contourArea,
    )

    contour_area = cv2.contourArea(
        bottle_contour
    )

    if contour_area <= 0:
        raise RuntimeError(
            "Detected contour has no area."
        )

    return bottle_contour


def create_silhouette(
    image_shape: tuple[int, int],
    contour: np.ndarray,
) -> np.ndarray:
    """
    Create a completely filled binary bottle
    silhouette from the detected contour.
    """

    silhouette = np.zeros(
        image_shape,
        dtype=np.uint8,
    )

    cv2.drawContours(
        silhouette,
        [contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    return silhouette


# ============================================================
# Basic geometry
# ============================================================

def measure_bottle_bounds(
    silhouette: np.ndarray,
) -> dict:
    """
    Measure the bounding geometry of the bottle.
    """

    y_coordinates, x_coordinates = np.where(
        silhouette == 255
    )

    if len(x_coordinates) == 0:
        raise RuntimeError(
            "Bottle silhouette is empty."
        )

    left = int(
        x_coordinates.min()
    )

    right = int(
        x_coordinates.max()
    )

    top = int(
        y_coordinates.min()
    )

    bottom = int(
        y_coordinates.max()
    )

    width = (
        right
        - left
        + 1
    )

    height = (
        bottom
        - top
        + 1
    )

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "width": width,
        "height": height,
    }


# ============================================================
# Width profile
# ============================================================

def calculate_width_profile(
    silhouette: np.ndarray,
    top: int,
    bottom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure bottle width at every horizontal row.
    """

    y_values = []
    widths = []

    for y in range(
        top,
        bottom + 1,
    ):

        x_coordinates = np.where(
            silhouette[y] == 255
        )[0]

        if len(x_coordinates) == 0:
            continue

        left = int(
            x_coordinates.min()
        )

        right = int(
            x_coordinates.max()
        )

        width = (
            right
            - left
            + 1
        )

        y_values.append(
            y
        )

        widths.append(
            width
        )

    if not y_values:
        raise RuntimeError(
            "Width profile could not be calculated."
        )

    return (
        np.asarray(
            y_values,
            dtype=np.int32,
        ),
        np.asarray(
            widths,
            dtype=np.float64,
        ),
    )


# ============================================================
# Center profile
# ============================================================

def calculate_center_profile(
    silhouette: np.ndarray,
    top: int,
    bottom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the horizontal center of the bottle
    at every row.
    """

    y_values = []
    centers = []

    for y in range(
        top,
        bottom + 1,
    ):

        x_coordinates = np.where(
            silhouette[y] == 255
        )[0]

        if len(x_coordinates) == 0:
            continue

        left = float(
            x_coordinates.min()
        )

        right = float(
            x_coordinates.max()
        )

        center = (
            left + right
        ) / 2.0

        y_values.append(
            y
        )

        centers.append(
            center
        )

    if not y_values:
        raise RuntimeError(
            "Center profile could not be calculated."
        )

    return (
        np.asarray(
            y_values,
            dtype=np.int32,
        ),
        np.asarray(
            centers,
            dtype=np.float64,
        ),
    )


# ============================================================
# Normalized vertical coordinates
# ============================================================

def normalize_vertical_coordinates(
    y_values: np.ndarray,
    top: int,
    bottom: int,
) -> np.ndarray:
    """
    Convert image Y coordinates to normalized bottle
    coordinates between 0.0 and 1.0.

    0.0 = bottle top
    1.0 = bottle bottom
    """

    bottle_height = (
        bottom - top
    )

    if bottle_height <= 0:
        raise RuntimeError(
            "Invalid bottle height."
        )

    return (
        (y_values - top)
        / bottle_height
    )


# ============================================================
# Measurements
# ============================================================

def measure_body_width(
    y_values: np.ndarray,
    widths: np.ndarray,
    top: int,
    bottom: int,
) -> float:
    """
    Measure median width in a stable lower-body region.
    """

    normalized_y = (
        normalize_vertical_coordinates(
            y_values,
            top,
            bottom,
        )
    )

    body_region = (
        (normalized_y >= 0.65)
        & (normalized_y <= 0.90)
    )

    body_widths = widths[
        body_region
    ]

    if len(body_widths) == 0:
        raise RuntimeError(
            "Body region could not be measured."
        )

    return float(
        np.median(
            body_widths
        )
    )


def measure_neck_width(
    y_values: np.ndarray,
    widths: np.ndarray,
    top: int,
    bottom: int,
) -> float:
    """
    Measure median width in the bottle neck region.
    """

    normalized_y = (
        normalize_vertical_coordinates(
            y_values,
            top,
            bottom,
        )
    )

    neck_region = (
        (normalized_y >= 0.05)
        & (normalized_y <= 0.15)
    )

    neck_widths = widths[
        neck_region
    ]

    if len(neck_widths) == 0:
        raise RuntimeError(
            "Neck region could not be measured."
        )

    return float(
        np.median(
            neck_widths
        )
    )


def measure_lean(
    y_values: np.ndarray,
    centers: np.ndarray,
    top: int,
    bottom: int,
) -> float:
    """
    Measure horizontal displacement between the
    upper and lower bottle regions.

    Positive:
        upper region is shifted to the right.

    Negative:
        upper region is shifted to the left.
    """

    normalized_y = (
        normalize_vertical_coordinates(
            y_values,
            top,
            bottom,
        )
    )

    top_region = (
        (normalized_y >= 0.05)
        & (normalized_y <= 0.15)
    )

    bottom_region = (
        (normalized_y >= 0.85)
        & (normalized_y <= 0.95)
    )

    top_centers = centers[
        top_region
    ]

    bottom_centers = centers[
        bottom_region
    ]

    if (
        len(top_centers) == 0
        or len(bottom_centers) == 0
    ):
        raise RuntimeError(
            "Lean could not be measured."
        )

    top_center = float(
        np.median(
            top_centers
        )
    )

    bottom_center = float(
        np.median(
            bottom_centers
        )
    )

    return (
        top_center
        - bottom_center
    )


# ============================================================
# Bottle measurement pipeline
# ============================================================

def measure_bottle(
    silhouette: np.ndarray,
) -> dict:
    """
    Extract geometric measurements and profiles
    from a bottle silhouette.
    """

    bounds = measure_bottle_bounds(
        silhouette
    )

    (
        width_y_values,
        width_profile,
    ) = calculate_width_profile(
        silhouette,
        bounds["top"],
        bounds["bottom"],
    )

    (
        center_y_values,
        center_profile,
    ) = calculate_center_profile(
        silhouette,
        bounds["top"],
        bounds["bottom"],
    )

    body_width = measure_body_width(
        width_y_values,
        width_profile,
        bounds["top"],
        bounds["bottom"],
    )

    neck_width = measure_neck_width(
        width_y_values,
        width_profile,
        bounds["top"],
        bounds["bottom"],
    )

    lean = measure_lean(
        center_y_values,
        center_profile,
        bounds["top"],
        bounds["bottom"],
    )

    normalized_width_y = (
        normalize_vertical_coordinates(
            width_y_values,
            bounds["top"],
            bounds["bottom"],
        )
    )

    normalized_center_y = (
        normalize_vertical_coordinates(
            center_y_values,
            bounds["top"],
            bounds["bottom"],
        )
    )

    return {
        "bounds": bounds,

        "height": bounds["height"],
        "bounding_width": bounds["width"],

        "body_width": body_width,
        "neck_width": neck_width,
        "lean": lean,

        "width_y_values": width_y_values,
        "normalized_width_y": normalized_width_y,
        "width_profile": width_profile,

        "center_y_values": center_y_values,
        "normalized_center_y": normalized_center_y,
        "center_profile": center_profile,
    }


# ============================================================
# Core inspection pipeline
# ============================================================

def inspect_array(
    image: np.ndarray,
) -> dict:
    """
    Inspect an image already loaded in memory.

    This is the core inspection function.

    It can later receive images from:
        - image files
        - industrial cameras
        - video frames
        - other Python modules

    No metadata or ground-truth masks are used.
    """

    grayscale = prepare_grayscale(
        image
    )

    binary, threshold_value = (
        segment_bottle(
            grayscale
        )
    )

    cleaned_binary = clean_binary(
        binary
    )

    contour = extract_bottle_contour(
        cleaned_binary
    )

    silhouette = create_silhouette(
        grayscale.shape,
        contour,
    )

    measurements = measure_bottle(
        silhouette
    )

    return {
        "image": grayscale,
        "binary": binary,
        "cleaned_binary": cleaned_binary,
        "contour": contour,
        "silhouette": silhouette,

        "threshold": threshold_value,

        "contour_area": float(
            cv2.contourArea(
                contour
            )
        ),

        "measurements": measurements,
    }


def inspect_image(
    image_path: Path,
) -> dict:
    """
    Inspect a bottle image stored on disk.

    This function is a file-based wrapper around
    inspect_array().
    """

    image_path = Path(
        image_path
    )

    image = load_image(
        image_path
    )

    result = inspect_array(
        image
    )

    result["image_path"] = (
        image_path
    )

    return result


# ============================================================
# Debug output
# ============================================================

def save_debug_outputs(
    result: dict,
    output_dir: Path,
) -> None:
    """
    Save useful intermediate pipeline outputs.

    These images are intended for development,
    debugging and demonstration purposes.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_input.png"
        ),
        result["image"],
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_otsu_binary.png"
        ),
        result["binary"],
    )

    cv2.imwrite(
        str(
            output_dir
            / "03_cleaned_binary.png"
        ),
        result["cleaned_binary"],
    )

    cv2.imwrite(
        str(
            output_dir
            / "04_silhouette.png"
        ),
        result["silhouette"],
    )


# ============================================================
# Console output
# ============================================================

def print_results(
    result: dict,
    output_dir: Path | None = None,
) -> None:
    """
    Print a human-readable inspection summary.
    """

    measurements = result[
        "measurements"
    ]

    image_path = result.get(
        "image_path"
    )

    if image_path is not None:
        print(
            f"Image: {image_path}"
        )

    print(
        f"Otsu threshold: "
        f"{result['threshold']:.2f}"
    )

    print(
        f"Contour area: "
        f"{result['contour_area']:.2f}"
    )

    if output_dir is not None:
        print(
            f"Results saved to: "
            f"{output_dir}"
        )

    print()

    print(
        "Bottle measurements"
    )

    print(
        "-------------------"
    )

    print(
        f"Height: "
        f"{measurements['height']} px"
    )

    print(
        f"Bounding width: "
        f"{measurements['bounding_width']} px"
    )

    print(
        f"Body width: "
        f"{measurements['body_width']:.2f} px"
    )

    print(
        f"Neck width: "
        f"{measurements['neck_width']:.2f} px"
    )

    print(
        f"Lean: "
        f"{measurements['lean']:.2f} px"
    )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for single-image
    inspection.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Inspect bottle geometry using "
            "classical computer vision."
        ),
    )

    parser.add_argument(
        "image",
        type=Path,
        help=(
            "Path to a bottle image."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for debug outputs. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Run inspection without saving "
            "intermediate images."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    CLI entry point for inspecting one image.

    Batch processing will reuse inspect_image()
    instead of launching this script repeatedly.
    """

    args = parse_arguments()

    result = inspect_image(
        args.image
    )

    output_dir = None

    if not args.no_save:

        save_debug_outputs(
            result,
            args.output_dir,
        )

        output_dir = (
            args.output_dir
        )

    print_results(
        result,
        output_dir,
    )


if __name__ == "__main__":
    main()