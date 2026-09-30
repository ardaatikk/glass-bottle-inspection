import argparse
import json
from pathlib import Path

import numpy as np

from bottle_inspection.inspection import inspect_image
from bottle_inspection.reference import interpolate_width_profile

# ============================================================
# Configuration
# ============================================================

DEFAULT_REFERENCE_FILE = Path(
    "reference/normal_reference.json"
)

# Global geometry thresholds.
# These are initial engineering thresholds.
# They will later be calibrated using the full dataset.
HEIGHT_STD_FACTOR = 3.0
BODY_WIDTH_STD_FACTOR = 3.0
NECK_WIDTH_STD_FACTOR = 3.0

# The synthetic normal dataset currently has zero lean
# variation, so a minimum absolute tolerance is required.
MIN_HEIGHT_TOLERANCE = 5.0
MIN_BODY_WIDTH_TOLERANCE = 4.0
MIN_NECK_WIDTH_TOLERANCE = 3.0
MIN_LEAN_TOLERANCE = 3.0

# Local profile analysis.
MIN_PROFILE_TOLERANCE = 4.0
PROFILE_STD_FACTOR = 3.0

# Ignore extreme top/bottom points because silhouette
# measurements are generally less stable there.
PROFILE_ANALYSIS_START = 0.20
PROFILE_ANALYSIS_END = 0.95

# Number of consecutive abnormal profile points required
# before calling a local deformation.
MIN_LOCAL_RUN = 2


# ============================================================
# Reference loading
# ============================================================

def load_reference(
    reference_file: Path,
) -> dict:
    """
    Load a normal-bottle reference model.
    """

    reference_file = Path(
        reference_file
    )

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference file does not exist: "
            f"{reference_file}"
        )

    if not reference_file.is_file():
        raise ValueError(
            f"Reference path is not a file: "
            f"{reference_file}"
        )

    with reference_file.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        reference = json.load(
            json_file
        )

    required_sections = (
        "measurements",
        "width_profile",
    )

    for section in required_sections:
        if section not in reference:
            raise ValueError(
                f"Reference is missing section: "
                f"{section}"
            )

    return reference


# ============================================================
# Threshold helpers
# ============================================================

def calculate_tolerance(
    statistics: dict,
    std_factor: float,
    minimum_tolerance: float,
) -> float:
    """
    Calculate a tolerance from reference statistics.

    A minimum absolute tolerance prevents extremely
    small or zero standard deviations from producing
    unrealistic thresholds.
    """

    standard_deviation = float(
        statistics["std"]
    )

    statistical_tolerance = (
        standard_deviation
        * std_factor
    )

    return float(
        max(
            statistical_tolerance,
            minimum_tolerance,
        )
    )


def compare_measurement(
    value: float,
    statistics: dict,
    std_factor: float,
    minimum_tolerance: float,
) -> dict:
    """
    Compare one scalar measurement against its
    normal reference distribution.
    """

    reference_mean = float(
        statistics["mean"]
    )

    deviation = (
        float(value)
        - reference_mean
    )

    tolerance = calculate_tolerance(
        statistics=statistics,
        std_factor=std_factor,
        minimum_tolerance=minimum_tolerance,
    )

    return {
        "value": float(value),
        "reference_mean": reference_mean,
        "deviation": float(
            deviation
        ),
        "absolute_deviation": float(
            abs(deviation)
        ),
        "tolerance": tolerance,
        "abnormal": bool(
            abs(deviation)
            > tolerance
        ),
    }


# ============================================================
# Width profile analysis
# ============================================================

def find_longest_true_run(
    values: np.ndarray,
) -> int:
    """
    Return the longest consecutive run of True values.
    """

    longest_run = 0
    current_run = 0

    for value in values:

        if bool(value):
            current_run += 1

            longest_run = max(
                longest_run,
                current_run,
            )

        else:
            current_run = 0

    return longest_run


def analyze_width_profile(
    measurements: dict,
    reference: dict,
) -> dict:
    """
    Compare the inspected bottle width profile against
    the normal reference profile.

    Positive deviation:
        bottle is wider than normal.

    Negative deviation:
        bottle is narrower than normal.
    """

    reference_profile = reference[
        "width_profile"
    ]

    reference_y = np.asarray(
        reference_profile[
            "normalized_y"
        ],
        dtype=np.float64,
    )

    reference_mean = np.asarray(
        reference_profile[
            "mean"
        ],
        dtype=np.float64,
    )

    reference_std = np.asarray(
        reference_profile[
            "std"
        ],
        dtype=np.float64,
    )

    if not (
        len(reference_y)
        == len(reference_mean)
        == len(reference_std)
    ):
        raise ValueError(
            "Reference width profile arrays "
            "have inconsistent lengths."
        )

    inspected_profile = (
        interpolate_width_profile(
            normalized_y=np.asarray(
                measurements[
                    "normalized_width_y"
                ],
                dtype=np.float64,
            ),
            width_profile=np.asarray(
                measurements[
                    "width_profile"
                ],
                dtype=np.float64,
            ),
            profile_grid=reference_y,
        )
    )

    deviation = (
        inspected_profile
        - reference_mean
    )

    tolerance = np.maximum(
        reference_std
        * PROFILE_STD_FACTOR,
        MIN_PROFILE_TOLERANCE,
    )

    analysis_region = (
        (reference_y >= PROFILE_ANALYSIS_START)
        & (reference_y <= PROFILE_ANALYSIS_END)
    )

    positive_abnormal = (
        (deviation > tolerance)
        & analysis_region
    )

    negative_abnormal = (
        (deviation < -tolerance)
        & analysis_region
    )

    positive_run = find_longest_true_run(
        positive_abnormal
    )

    negative_run = find_longest_true_run(
        negative_abnormal
    )

    positive_detected = (
        positive_run
        >= MIN_LOCAL_RUN
    )

    negative_detected = (
        negative_run
        >= MIN_LOCAL_RUN
    )

    analysis_indices = np.where(
        analysis_region
    )[0]

    if len(analysis_indices) == 0:
        raise RuntimeError(
            "Width profile analysis region is empty."
        )

    region_deviation = deviation[
        analysis_indices
    ]

    max_local_index = (
        analysis_indices[
            int(
                np.argmax(
                    np.abs(
                        region_deviation
                    )
                )
            )
        ]
    )

    maximum_deviation = float(
        deviation[
            max_local_index
        ]
    )

    maximum_deviation_y = float(
        reference_y[
            max_local_index
        ]
    )

    return {
        "normalized_y": (
            reference_y.tolist()
        ),
        "inspected": (
            inspected_profile.tolist()
        ),
        "reference_mean": (
            reference_mean.tolist()
        ),
        "deviation": (
            deviation.tolist()
        ),
        "tolerance": (
            tolerance.tolist()
        ),
        "positive_run": int(
            positive_run
        ),
        "negative_run": int(
            negative_run
        ),
        "positive_detected": bool(
            positive_detected
        ),
        "negative_detected": bool(
            negative_detected
        ),
        "maximum_deviation": (
            maximum_deviation
        ),
        "maximum_deviation_y": (
            maximum_deviation_y
        ),
    }


# ============================================================
# Defect classification
# ============================================================

def classify_defects(
    comparisons: dict,
    profile_analysis: dict,
) -> list[str]:
    """
    Convert abnormal geometry measurements into
    human-readable defect labels.

    Multiple defects may be returned because a bottle
    can violate more than one geometric constraint.
    """

    defects = []

    height = comparisons[
        "height"
    ]

    body_width = comparisons[
        "body_width"
    ]

    neck_width = comparisons[
        "neck_width"
    ]

    lean = comparisons[
        "lean"
    ]

    if height["abnormal"]:
        if height["deviation"] > 0:
            defects.append(
                "height_too_tall"
            )
        else:
            defects.append(
                "height_too_short"
            )

    if neck_width["abnormal"]:
        if neck_width["deviation"] > 0:
            defects.append(
                "neck_too_wide"
            )
        else:
            defects.append(
                "neck_too_narrow"
            )

    if lean["abnormal"]:
        defects.append(
            "lean"
        )

    if profile_analysis[
        "positive_detected"
    ]:
        defects.append(
            "bulge"
        )

    if profile_analysis[
        "negative_detected"
    ]:
        defects.append(
            "shrink"
        )

    # Body width is kept as a global diagnostic.
    # If it is abnormal but no local profile defect was
    # detected, expose the global width anomaly explicitly.
    if (
        body_width["abnormal"]
        and not profile_analysis[
            "positive_detected"
        ]
        and not profile_analysis[
            "negative_detected"
        ]
    ):
        if body_width["deviation"] > 0:
            defects.append(
                "body_too_wide"
            )
        else:
            defects.append(
                "body_too_narrow"
            )

    return defects


# ============================================================
# Detection pipeline
# ============================================================

def detect_defects(
    image_path: Path,
    reference_file: Path = DEFAULT_REFERENCE_FILE,
) -> dict:
    """
    Run the complete defect-detection pipeline.

    The input image is inspected directly.
    Dataset metadata and ground-truth masks are
    intentionally not used.
    """

    image_path = Path(
        image_path
    )

    reference = load_reference(
        reference_file
    )

    inspection_result = inspect_image(
        image_path
    )

    measurements = inspection_result[
        "measurements"
    ]

    reference_measurements = reference[
        "measurements"
    ]

    comparisons = {
        "height": compare_measurement(
            value=measurements[
                "height"
            ],
            statistics=reference_measurements[
                "height"
            ],
            std_factor=HEIGHT_STD_FACTOR,
            minimum_tolerance=MIN_HEIGHT_TOLERANCE,
        ),

        "body_width": compare_measurement(
            value=measurements[
                "body_width"
            ],
            statistics=reference_measurements[
                "body_width"
            ],
            std_factor=BODY_WIDTH_STD_FACTOR,
            minimum_tolerance=MIN_BODY_WIDTH_TOLERANCE,
        ),

        "neck_width": compare_measurement(
            value=measurements[
                "neck_width"
            ],
            statistics=reference_measurements[
                "neck_width"
            ],
            std_factor=NECK_WIDTH_STD_FACTOR,
            minimum_tolerance=MIN_NECK_WIDTH_TOLERANCE,
        ),

        "lean": compare_measurement(
            value=measurements[
                "lean"
            ],
            statistics=reference_measurements[
                "lean"
            ],
            std_factor=1.0,
            minimum_tolerance=MIN_LEAN_TOLERANCE,
        ),
    }

    profile_analysis = (
        analyze_width_profile(
            measurements=measurements,
            reference=reference,
        )
    )

    defects = classify_defects(
        comparisons=comparisons,
        profile_analysis=profile_analysis,
    )

    return {
        "image": str(
            image_path
        ),
        "status": (
            "DEFECTIVE"
            if defects
            else "NORMAL"
        ),
        "defects": defects,
        "measurements": {
            "height": float(
                measurements["height"]
            ),
            "bounding_width": float(
                measurements[
                    "bounding_width"
                ]
            ),
            "body_width": float(
                measurements[
                    "body_width"
                ]
            ),
            "neck_width": float(
                measurements[
                    "neck_width"
                ]
            ),
            "lean": float(
                measurements["lean"]
            ),
        },
        "comparisons": comparisons,
        "profile_analysis": profile_analysis,
    }


# ============================================================
# Console reporting
# ============================================================

def print_detection_result(
    result: dict,
) -> None:
    """
    Print a human-readable defect detection report.
    """

    print()
    print(
        "Bottle inspection result"
    )
    print(
        "------------------------"
    )

    print(
        f"Image:  {result['image']}"
    )

    print(
        f"Status: {result['status']}"
    )

    defects = result[
        "defects"
    ]

    if defects:
        print(
            "Defects: "
            + ", ".join(
                defects
            )
        )
    else:
        print(
            "Defects: none"
        )

    print()
    print(
        "Global geometry"
    )
    print(
        "---------------"
    )

    for name in (
        "height",
        "body_width",
        "neck_width",
        "lean",
    ):

        comparison = result[
            "comparisons"
        ][name]

        state = (
            "ABNORMAL"
            if comparison["abnormal"]
            else "OK"
        )

        print(
            f"{name:12s} "
            f"value={comparison['value']:8.2f}  "
            f"reference={comparison['reference_mean']:8.2f}  "
            f"delta={comparison['deviation']:+8.2f}  "
            f"limit={comparison['tolerance']:6.2f}  "
            f"{state}"
        )

    profile = result[
        "profile_analysis"
    ]

    print()
    print(
        "Local width profile"
    )
    print(
        "-------------------"
    )

    print(
        f"Maximum deviation: "
        f"{profile['maximum_deviation']:+.2f} px"
    )

    print(
        f"Location:           "
        f"y={profile['maximum_deviation_y']:.2f}"
    )

    print(
        f"Positive run:       "
        f"{profile['positive_run']} points"
    )

    print(
        f"Negative run:       "
        f"{profile['negative_run']} points"
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
            "Detect geometric defects in a glass "
            "bottle image using a normal reference model."
        ),
    )

    parser.add_argument(
        "image",
        type=Path,
        help=(
            "Bottle image to inspect."
        ),
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_FILE,
        help=(
            "Normal bottle reference JSON. "
            f"Default: {DEFAULT_REFERENCE_FILE}"
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

    result = detect_defects(
        image_path=args.image,
        reference_file=args.reference,
    )

    print_detection_result(
        result
    )


if __name__ == "__main__":
    main()
