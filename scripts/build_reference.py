import argparse
import json
from pathlib import Path

import numpy as np

from bottle_inspection.inspection import (
    SUPPORTED_IMAGE_EXTENSIONS,
    inspect_image,
)

from bottle_inspection.reference import (
    interpolate_width_profile,
)

# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT_DIR = Path("data/images/normal")
DEFAULT_OUTPUT_FILE = Path("reference/normal_reference.json")

PROFILE_POINTS = 101


# ============================================================
# Image discovery
# ============================================================

def find_images(
    input_dir: Path,
) -> list[Path]:
    """
    Find supported image files inside a directory.
    """

    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise ValueError(
            f"Input path is not a directory: {input_dir}"
        )

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in: {input_dir}"
        )

    return image_paths


# ============================================================
# Profile normalization
# ============================================================

def create_profile_grid(
    points: int = PROFILE_POINTS,
) -> np.ndarray:
    """
    Create a common normalized vertical coordinate grid.

    Example with 101 points:
        0.00, 0.01, 0.02, ..., 1.00
    """

    if points < 2:
        raise ValueError(
            "Profile must contain at least two points."
        )

    return np.linspace(
        0.0,
        1.0,
        points,
        dtype=np.float64,
    )

# ============================================================
# Statistics
# ============================================================

def calculate_statistics(
    values: np.ndarray,
) -> dict:
    """
    Calculate summary statistics for one measurement.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot calculate statistics from empty values."
        )

    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(
                values,
                ddof=0,
            )
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


# ============================================================
# Reference construction
# ============================================================

def build_reference(
    input_dir: Path = DEFAULT_INPUT_DIR,
    profile_points: int = PROFILE_POINTS,
) -> dict:
    """
    Build a statistical reference model from
    normal bottle images.

    Every image is processed through inspection.py.
    Ground-truth masks and metadata are not used.
    """

    image_paths = find_images(
        input_dir
    )

    profile_grid = create_profile_grid(
        profile_points
    )

    heights = []
    bounding_widths = []
    body_widths = []
    neck_widths = []
    leans = []

    bounding_width_ratios = []
    body_width_ratios = []
    neck_width_ratios = []
    lean_ratios = []

    interpolated_profiles = []
    normalized_interpolated_profiles = []

    failed_images = []

    print(
        f"Building reference from "
        f"{len(image_paths)} normal images..."
    )

    for image_path in image_paths:

        try:
            result = inspect_image(
                image_path
            )

            measurements = result[
                "measurements"
            ]

            heights.append(
                measurements["height"]
            )

            bounding_widths.append(
                measurements["bounding_width"]
            )

            body_widths.append(
                measurements["body_width"]
            )

            neck_widths.append(
                measurements["neck_width"]
            )

            leans.append(
                measurements["lean"]
            )
            
            bounding_width_ratios.append(
                measurements["bounding_width_ratio"]
            )

            body_width_ratios.append(
                measurements["body_width_ratio"]
            )

            neck_width_ratios.append(
                measurements["neck_width_ratio"]
            )

            lean_ratios.append(
                measurements["lean_ratio"]
            )

            profile = interpolate_width_profile(
                normalized_y=measurements[
                    "normalized_width_y"
                ],
                width_profile=measurements[
                    "width_profile"
                ],
                profile_grid=profile_grid,
            )

            interpolated_profiles.append(
                profile
            )
            
            normalized_profile = (
                interpolate_width_profile(
                    normalized_y=measurements[
                        "normalized_width_y"
                    ],
                    width_profile=measurements[
                        "normalized_width_profile"
                    ],
                    profile_grid=profile_grid,
                )
            )

            normalized_interpolated_profiles.append(
                normalized_profile
            )

        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ) as error:

            failed_images.append(
                {
                    "filename": image_path.name,
                    "error": str(error),
                }
            )

    successful_samples = len(
        interpolated_profiles
    )

    if successful_samples == 0:
        raise RuntimeError(
            "Reference could not be built because "
            "all inspections failed."
        )

    profile_matrix = np.vstack(
        interpolated_profiles
    )
    
    normalized_profile_matrix = np.vstack(
        normalized_interpolated_profiles
    )

    profile_mean = np.mean(
        profile_matrix,
        axis=0,
    )

    profile_std = np.std(
        profile_matrix,
        axis=0,
        ddof=0,
    )

    profile_min = np.min(
        profile_matrix,
        axis=0,
    )

    profile_max = np.max(
        profile_matrix,
        axis=0,
    )

    normalized_profile_mean = np.mean(
        normalized_profile_matrix,
        axis=0,
    )

    normalized_profile_std = np.std(
        normalized_profile_matrix,
        axis=0,
        ddof=0,
    )

    normalized_profile_min = np.min(
        normalized_profile_matrix,
        axis=0,
    )

    normalized_profile_max = np.max(
        normalized_profile_matrix,
        axis=0,
    )

    reference = {
        "reference_type": "normal_bottle_geometry",

        "source": {
            "input_directory": str(
                input_dir
            ),
            "total_images": len(
                image_paths
            ),
            "successful_images": successful_samples,
            "failed_images": len(
                failed_images
            ),
        },

        "measurements": {
            "height": calculate_statistics(
                np.asarray(heights)
            ),
            "bounding_width": calculate_statistics(
                np.asarray(bounding_widths)
            ),
            "body_width": calculate_statistics(
                np.asarray(body_widths)
            ),
            "neck_width": calculate_statistics(
                np.asarray(neck_widths)
            ),
            "lean": calculate_statistics(
                np.asarray(leans)
            ),
        },
        
        "normalized_measurements": {
            "bounding_width_ratio": calculate_statistics(
                np.asarray(
                    bounding_width_ratios
                )
            ),
            "body_width_ratio": calculate_statistics(
                np.asarray(
                    body_width_ratios
                )
            ),
            "neck_width_ratio": calculate_statistics(
                np.asarray(
                    neck_width_ratios
                )
            ),
            "lean_ratio": calculate_statistics(
                np.asarray(
                    lean_ratios
                )
            ),
        },

        "width_profile": {
            "normalized_y": (
                profile_grid.tolist()
            ),
            "mean": (
                profile_mean.tolist()
            ),
            "std": (
                profile_std.tolist()
            ),
            "min": (
                profile_min.tolist()
            ),
            "max": (
                profile_max.tolist()
            ),
        },

        "normalized_width_profile": {
            "normalized_y": (
                profile_grid.tolist()
            ),
            "mean": (
                normalized_profile_mean.tolist()
            ),
            "std": (
                normalized_profile_std.tolist()
            ),
            "min": (
                normalized_profile_min.tolist()
            ),
            "max": (
                normalized_profile_max.tolist()
            ),
        },

        "failed_samples": failed_images,
    }

    return reference


# ============================================================
# Reference persistence
# ============================================================

def save_reference(
    reference: dict,
    output_file: Path,
) -> None:
    """
    Save the reference model as JSON.
    """

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            reference,
            json_file,
            indent=2,
        )


# ============================================================
# Console reporting
# ============================================================

def print_reference_summary(
    reference: dict,
    output_file: Path,
) -> None:
    """
    Print a compact reference model summary.
    """

    source = reference[
        "source"
    ]

    measurements = reference[
        "measurements"
    ]

    print()
    print(
        "Reference model completed."
    )
    print(
        "--------------------------"
    )

    print(
        f"Total images:      "
        f"{source['total_images']}"
    )

    print(
        f"Successful images: "
        f"{source['successful_images']}"
    )

    print(
        f"Failed images:     "
        f"{source['failed_images']}"
    )

    print()

    print(
        "Reference measurements"
    )
    print(
        "----------------------"
    )

    for measurement_name in (
        "height",
        "bounding_width",
        "body_width",
        "neck_width",
        "lean",
    ):

        statistics = measurements[
            measurement_name
        ]

        print(
            f"{measurement_name:15s} "
            f"mean={statistics['mean']:.3f}  "
            f"std={statistics['std']:.3f}  "
            f"min={statistics['min']:.3f}  "
            f"max={statistics['max']:.3f}"
        )

    print()
    print(
        f"Reference saved to: {output_file}"
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
            "Build a statistical normal-bottle "
            "reference model using inspection.py."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing normal bottle images. "
            f"Default: {DEFAULT_INPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "Path for the generated reference JSON. "
            f"Default: {DEFAULT_OUTPUT_FILE}"
        ),
    )

    parser.add_argument(
        "--profile-points",
        type=int,
        default=PROFILE_POINTS,
        help=(
            "Number of normalized points used for "
            "the bottle width profile. "
            f"Default: {PROFILE_POINTS}"
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

    reference = build_reference(
        input_dir=args.input_dir,
        profile_points=args.profile_points,
    )

    save_reference(
        reference=reference,
        output_file=args.output_file,
    )

    print_reference_summary(
        reference=reference,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()