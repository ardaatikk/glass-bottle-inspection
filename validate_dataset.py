import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEFAULT_DATA_DIR = Path("data")

GEOMETRY_TOLERANCE = 2.1
HEIGHT_TOLERANCE = 1
LEAN_TOP_REGION = (0.05, 0.15)
LEAN_BOTTOM_REGION = (0.85, 0.95)

DEFECT_TYPES = (
    "normal",
    "bulge",
    "shrink",
    "lean",
    "height",
    "neck",
)

REQUIRED_METADATA_FIELDS = (
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
# Metadata loading
# ============================================================

def load_metadata(
    metadata_path: Path,
) -> list[dict]:
    """
    Load dataset metadata from CSV.
    """

    metadata_path = Path(
        metadata_path
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist: "
            f"{metadata_path}"
        )

    if not metadata_path.is_file():
        raise ValueError(
            f"Metadata path is not a file: "
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Metadata file has no header."
            )

        missing_fields = (
            set(REQUIRED_METADATA_FIELDS)
            - set(reader.fieldnames)
        )

        if missing_fields:
            raise ValueError(
                "Metadata is missing required fields: "
                + ", ".join(
                    sorted(missing_fields)
                )
            )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "Metadata file contains no samples."
        )

    return rows


# ============================================================
# Metadata parsing
# ============================================================

def parse_metadata_row(
    row: dict,
) -> dict:
    """
    Convert CSV string values to their expected
    Python numeric types.
    """

    defect_type = row[
        "defect_type"
    ]

    if defect_type not in DEFECT_TYPES:
        raise ValueError(
            f"Unsupported defect type: "
            f"{defect_type}"
        )

    return {
        "filename": row[
            "filename"
        ],

        "defect_type": defect_type,

        "body_width": float(
            row["body_width"]
        ),

        "neck_width": float(
            row["neck_width"]
        ),

        "bottle_height": int(
            float(
                row["bottle_height"]
            )
        ),

        "magnitude": float(
            row["magnitude"]
        ),

        "center": float(
            row["center"]
        ),

        "sigma": float(
            row["sigma"]
        ),

        "lean_shift": float(
            row["lean_shift"]
        ),

        "height_change": int(
            float(
                row["height_change"]
            )
        ),
    }


# ============================================================
# Dataset path resolution
# ============================================================

def get_sample_paths(
    images_dir: Path,
    masks_dir: Path,
    metadata: dict,
) -> tuple[Path, Path]:
    """
    Resolve image and mask paths for a metadata row.
    """

    filename = metadata[
        "filename"
    ]

    defect_type = metadata[
        "defect_type"
    ]

    image_stem = Path(
        filename
    ).stem

    mask_filename = (
        f"{image_stem}_mask.png"
    )

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
# File validation
# ============================================================

def load_mask(
    mask_path: Path,
) -> np.ndarray:
    """
    Load a ground-truth mask as grayscale.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Mask could not be loaded: "
            f"{mask_path}"
        )

    return mask


def validate_binary_mask(
    mask: np.ndarray,
) -> bool:
    """
    Check whether a mask contains only binary
    pixel values 0 and 255.
    """

    unique_values = np.unique(
        mask
    )

    return bool(
        np.all(
            np.isin(
                unique_values,
                [0, 255],
            )
        )
    )


# ============================================================
# Mask geometry
# ============================================================

def get_mask_bounds(
    mask: np.ndarray,
) -> dict:
    """
    Extract bottle bounding geometry from a
    ground-truth mask.
    """

    y_coordinates, x_coordinates = np.where(
        mask == 255
    )

    if len(x_coordinates) == 0:
        raise RuntimeError(
            "Mask contains no bottle pixels."
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
        right - left + 1
    )

    height = (
        bottom - top + 1
    )

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "width": width,
        "height": height,
    }


def calculate_width_profile(
    mask: np.ndarray,
    top: int,
    bottom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate bottle width at every valid row
    of a ground-truth mask.
    """

    y_values = []
    widths = []

    for y in range(
        top,
        bottom + 1,
    ):

        x_coordinates = np.where(
            mask[y] == 255
        )[0]

        if len(x_coordinates) == 0:
            continue

        width = (
            int(x_coordinates.max())
            - int(x_coordinates.min())
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
            "Mask width profile could not "
            "be calculated."
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


def calculate_center_profile(
    mask: np.ndarray,
    top: int,
    bottom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate bottle center at every valid row
    of a ground-truth mask.
    """

    y_values = []
    centers = []

    for y in range(
        top,
        bottom + 1,
    ):

        x_coordinates = np.where(
            mask[y] == 255
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
            "Mask center profile could not "
            "be calculated."
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


def normalize_vertical_coordinates(
    y_values: np.ndarray,
    top: int,
    bottom: int,
) -> np.ndarray:
    """
    Normalize bottle vertical coordinates.

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
# Geometry measurements
# ============================================================

def measure_body_width(
    y_values: np.ndarray,
    widths: np.ndarray,
    top: int,
    bottom: int,
) -> float:
    """
    Measure median width in the stable body region.
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
    Measure median width in the neck region.
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
    """

    normalized_y = (
        normalize_vertical_coordinates(
            y_values,
            top,
            bottom,
        )
    )

    top_region = (
        (normalized_y >= LEAN_TOP_REGION[0])
        & (normalized_y <= LEAN_TOP_REGION[1])
    )

    bottom_region = (
        (normalized_y >= LEAN_BOTTOM_REGION[0])
        & (normalized_y <= LEAN_BOTTOM_REGION[1])
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

    return float(
        np.median(top_centers)
        - np.median(bottom_centers)
    )


def measure_mask_geometry(
    mask: np.ndarray,
) -> dict:
    """
    Measure geometry directly from a ground-truth
    bottle mask.
    """

    bounds = get_mask_bounds(
        mask
    )

    (
        width_y_values,
        width_profile,
    ) = calculate_width_profile(
        mask,
        bounds["top"],
        bounds["bottom"],
    )

    (
        center_y_values,
        center_profile,
    ) = calculate_center_profile(
        mask,
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

    return {
        "height": bounds[
            "height"
        ],
        "bounding_width": bounds[
            "width"
        ],
        "body_width": body_width,
        "neck_width": neck_width,
        "lean": lean,
        "width_y_values": width_y_values,
        "width_profile": width_profile,
    }


# ============================================================
# Expected geometry
# ============================================================

def get_expected_height(
    metadata: dict,
) -> int:
    """
    Return expected bottle height from metadata.
    """

    expected_height = metadata[
        "bottle_height"
    ]

    if (
        metadata["defect_type"]
        == "height"
    ):
        expected_height += metadata[
            "height_change"
        ]

    # Generated masks include both top and bottom rows.
    return expected_height + 1


def get_expected_geometry(
    metadata: dict,
) -> float | None:
    """
    Return the primary expected geometry value
    for the sample's defect type.

    This value is used as a compact validation
    summary and for defect-specific checks.
    """

    defect_type = metadata[
        "defect_type"
    ]

    if defect_type == "normal":
        return metadata[
            "body_width"
        ]

    if defect_type == "bulge":
        return (
            metadata["body_width"]
            + metadata["magnitude"]
        )

    if defect_type == "shrink":
        return (
            metadata["body_width"]
            - metadata["magnitude"]
        )

    if defect_type == "lean":

        top_center = (
            LEAN_TOP_REGION[0]
            + LEAN_TOP_REGION[1]
        ) / 2.0

        bottom_center = (
            LEAN_BOTTOM_REGION[0]
            + LEAN_BOTTOM_REGION[1]
        ) / 2.0

        expected_lean_factor = (
            bottom_center
            - top_center
        )

        return (
            metadata["lean_shift"]
            * expected_lean_factor
        )

    if defect_type == "height":
        return float(
            get_expected_height(
                metadata
            )
        )

    if defect_type == "neck":
        return (
            metadata["neck_width"]
            + metadata["magnitude"]
        )

    return None


# ============================================================
# Defect-specific geometry
# ============================================================

def measure_local_deformation(
    metadata: dict,
    measurements: dict,
) -> float:
    """
    Measure local width around the configured
    bulge/shrink center.
    """

    y_values = measurements[
        "width_y_values"
    ]

    widths = measurements[
        "width_profile"
    ]

    height = measurements[
        "height"
    ]

    if height <= 1:
        raise RuntimeError(
            "Invalid bottle height."
        )

    top = int(
        y_values.min()
    )

    normalized_y = (
        (y_values - top)
        / (height - 1)
    )

    center = metadata[
        "center"
    ]

    center_index = int(
        np.argmin(
            np.abs(
                normalized_y - center
            )
        )
    )

    return float(
        widths[
            center_index
        ]
    )


# ============================================================
# Sample validation
# ============================================================

def validate_sample(
    metadata: dict,
    image_path: Path,
    mask_path: Path,
) -> dict:
    """
    Validate one generated sample against its
    metadata and ground-truth mask.
    """

    errors = []

    if not image_path.exists():
        errors.append(
            "missing_image"
        )

    if not mask_path.exists():
        errors.append(
            "missing_mask"
        )

    if errors:
        return {
            "filename": metadata[
                "filename"
            ],
            "passed": False,
            "errors": ",".join(
                errors
            ),
            "expected_height": None,
            "measured_height": None,
            "expected_geometry": None,
            "measured_geometry": None,
        }

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        errors.append(
            "invalid_image"
        )

    try:
        mask = load_mask(
            mask_path
        )

    except RuntimeError:
        errors.append(
            "invalid_mask"
        )

        return {
            "filename": metadata[
                "filename"
            ],
            "passed": False,
            "errors": ",".join(
                errors
            ),
            "expected_height": None,
            "measured_height": None,
            "expected_geometry": None,
            "measured_geometry": None,
        }

    if not validate_binary_mask(
        mask
    ):
        errors.append(
            "non_binary_mask"
        )

    try:
        measurements = (
            measure_mask_geometry(
                mask
            )
        )

    except RuntimeError:
        errors.append(
            "geometry_measurement_failed"
        )

        return {
            "filename": metadata[
                "filename"
            ],
            "passed": False,
            "errors": ",".join(
                errors
            ),
            "expected_height": None,
            "measured_height": None,
            "expected_geometry": None,
            "measured_geometry": None,
        }

    expected_height = (
        get_expected_height(
            metadata
        )
    )

    measured_height = measurements[
        "height"
    ]

    if (
        abs(
            measured_height
            - expected_height
        )
        > HEIGHT_TOLERANCE
    ):
        errors.append(
            "height_mismatch"
        )

    defect_type = metadata[
        "defect_type"
    ]

    expected_geometry = (
        get_expected_geometry(
            metadata
        )
    )

    measured_geometry = None

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    if defect_type == "normal":

        measured_geometry = measurements[
            "body_width"
        ]

        if (
            abs(
                measured_geometry
                - expected_geometry
            )
            > GEOMETRY_TOLERANCE
        ):
            errors.append(
                "normal_body_mismatch"
            )

    # --------------------------------------------------------
    # Bulge
    # --------------------------------------------------------

    elif defect_type == "bulge":

        measured_geometry = (
            measure_local_deformation(
                metadata,
                measurements,
            )
        )

        if (
            abs(
                measured_geometry
                - expected_geometry
            )
            > GEOMETRY_TOLERANCE
        ):
            errors.append(
                "bulge_mismatch"
            )

    # --------------------------------------------------------
    # Shrink
    # --------------------------------------------------------

    elif defect_type == "shrink":

        measured_geometry = (
            measure_local_deformation(
                metadata,
                measurements,
            )
        )

        if (
            abs(
                measured_geometry
                - expected_geometry
            )
            > GEOMETRY_TOLERANCE
        ):
            errors.append(
                "shrink_mismatch"
            )

    # --------------------------------------------------------
    # Lean
    # --------------------------------------------------------

    elif defect_type == "lean":

        measured_geometry = measurements[
            "lean"
        ]

        if (
            abs(
                measured_geometry
                - expected_geometry
            )
            > GEOMETRY_TOLERANCE
        ):
            errors.append(
                "lean_mismatch"
            )

    # --------------------------------------------------------
    # Height
    # --------------------------------------------------------

    elif defect_type == "height":

        measured_geometry = float(
            measured_height
        )

        if (
            abs(
                measured_geometry
                - expected_geometry
            )
            > HEIGHT_TOLERANCE
        ):
            errors.append(
                "height_defect_mismatch"
            )

    # --------------------------------------------------------
    # Neck
    # --------------------------------------------------------

    elif defect_type == "neck":

        measured_geometry = measurements[
            "neck_width"
        ]

        if (
            abs(
                measured_geometry
                - expected_geometry
            )
            > GEOMETRY_TOLERANCE
        ):
            errors.append(
                "neck_mismatch"
            )

    return {
        "filename": metadata[
            "filename"
        ],
        "passed": len(errors) == 0,
        "errors": ",".join(
            errors
        ),
        "expected_height": expected_height,
        "measured_height": measured_height,
        "expected_geometry": expected_geometry,
        "measured_geometry": measured_geometry,
    }


# ============================================================
# Dataset validation pipeline
# ============================================================

def validate_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict:
    """
    Validate a complete generated dataset.

    Returns a dictionary containing the validation
    summary and individual sample results.
    """

    data_dir = Path(
        data_dir
    )

    metadata_path = (
        data_dir
        / "metadata.csv"
    )

    images_dir = (
        data_dir
        / "images"
    )

    masks_dir = (
        data_dir
        / "masks"
    )

    metadata_rows = load_metadata(
        metadata_path
    )

    print(
        f"Validating "
        f"{len(metadata_rows)} samples..."
    )

    results = []

    for row in metadata_rows:

        try:
            metadata = (
                parse_metadata_row(
                    row
                )
            )

            (
                image_path,
                mask_path,
            ) = get_sample_paths(
                images_dir=images_dir,
                masks_dir=masks_dir,
                metadata=metadata,
            )

            result = validate_sample(
                metadata=metadata,
                image_path=image_path,
                mask_path=mask_path,
            )

        except (
            ValueError,
            RuntimeError,
        ) as error:

            result = {
                "filename": row.get(
                    "filename",
                    "<unknown>",
                ),
                "passed": False,
                "errors": (
                    f"validation_error: "
                    f"{error}"
                ),
                "expected_height": None,
                "measured_height": None,
                "expected_geometry": None,
                "measured_geometry": None,
            }

        results.append(
            result
        )

    passed_results = [
        result
        for result in results
        if result["passed"]
    ]

    failed_results = [
        result
        for result in results
        if not result["passed"]
    ]

    total = len(
        results
    )

    passed = len(
        passed_results
    )

    failed = len(
        failed_results
    )

    return {
        "data_dir": data_dir,
        "total": total,
        "passed": passed,
        "failed": failed,
        "success": failed == 0,
        "results": results,
        "failed_results": failed_results,
    }


# ============================================================
# Console reporting
# ============================================================

def print_validation_summary(
    summary: dict,
) -> None:
    """
    Print a human-readable validation summary.
    """

    print()

    print(
        "Validation completed."
    )

    print(
        "---------------------"
    )

    print(
        f"Total: {summary['total']}"
    )

    print(
        f"PASS:  {summary['passed']}"
    )

    print(
        f"FAIL:  {summary['failed']}"
    )

    if summary[
        "failed_results"
    ]:

        print()
        print(
            "Failed samples:"
        )

        for result in summary[
            "failed_results"
        ]:

            print()

            print(
                f"  File: "
                f"{result['filename']}"
            )

            print(
                f"  Errors: "
                f"{result['errors']}"
            )

            print(
                f"  Expected height: "
                f"{result['expected_height']}"
            )

            print(
                f"  Measured height: "
                f"{result['measured_height']}"
            )

            print(
                f"  Expected geometry: "
                f"{result['expected_geometry']}"
            )

            print(
                f"  Measured geometry: "
                f"{result['measured_geometry']}"
            )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Validate a synthetic glass bottle "
            "inspection dataset."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Dataset directory containing "
            "images, masks and metadata.csv. "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    CLI entry point.

    Exit code:
        0 -> all samples passed validation
        1 -> one or more samples failed validation
    """

    args = parse_arguments()

    try:
        summary = validate_dataset(
            data_dir=args.data_dir
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:

        print(
            f"Validation failed: {error}",
            file=sys.stderr,
        )

        raise SystemExit(
            1
        ) from error

    print_validation_summary(
        summary
    )

    if not summary[
        "success"
    ]:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()