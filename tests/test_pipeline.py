import json
from pathlib import Path

import cv2
import numpy as np

from bottle_inspection.detection import detect_defects
from bottle_inspection.inspection import inspect_image


PROFILE_POINTS = 101


def create_bottle_image(
    output_path: Path,
    body_width: int = 300,
    height: int = 700,
    bulge_magnitude: float = 0.0,
    bulge_center: float = 0.65,
    bulge_sigma: float = 0.07,
) -> None:
    """
    Create a simple synthetic bottle image for
    end-to-end pipeline testing.
    """

    image_height = 1000
    image_width = 800

    image = np.full(
        (image_height, image_width),
        230,
        dtype=np.uint8,
    )

    center_x = image_width // 2

    bottom_y = 850
    top_y = bottom_y - height

    neck_width = 120

    left_points = []
    right_points = []

    for y in range(top_y, bottom_y + 1):
        normalized_y = (y - top_y) / height

        if normalized_y < 0.18:
            current_width = neck_width

        elif normalized_y < 0.32:
            transition = (
                (normalized_y - 0.18)
                / (0.32 - 0.18)
            )

            current_width = (
                neck_width
                + transition
                * (body_width - neck_width)
            )

        else:
            current_width = float(body_width)

        if bulge_magnitude > 0:
            bulge = (
                bulge_magnitude
                * np.exp(
                    -0.5
                    * (
                        (
                            normalized_y
                            - bulge_center
                        )
                        / bulge_sigma
                    )
                    ** 2
                )
            )

            current_width += bulge

        left_x = int(
            center_x
            - current_width / 2
        )

        right_x = int(
            center_x
            + current_width / 2
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
        image,
        [bottle_points],
        40,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            "Failed to create pipeline test image."
        )


def create_reference(
    normal_image_path: Path,
    reference_path: Path,
) -> None:
    """
    Build a minimal normal reference directly from
    the test's own normal bottle.
    """

    inspection = inspect_image(
        normal_image_path
    )

    measurements = inspection[
        "measurements"
    ]

    normalized_y = np.asarray(
        measurements["normalized_width_y"],
        dtype=np.float64,
    )

    width_profile = np.asarray(
        measurements["width_profile"],
        dtype=np.float64,
    )
    
    normalized_width_profile = np.asarray(
        measurements["normalized_width_profile"],
        dtype=np.float64,
    )

    profile_grid = np.linspace(
        0.0,
        1.0,
        PROFILE_POINTS,
    )

    mean_profile = np.interp(
        profile_grid,
        normalized_y,
        width_profile,
    )

    normalized_mean_profile = np.interp(
        profile_grid,
        normalized_y,
        normalized_width_profile,
    )

    def measurement_statistics(
        value: float,
    ) -> dict:
        return {
            "mean": float(value),
            "std": 0.0,
            "min": float(value),
            "max": float(value),
        }

    reference = {
        "sample_count": 1,
        "profile_points": PROFILE_POINTS,
        "measurements": {
            "height": measurement_statistics(
                measurements["height"]
            ),
            "bounding_width": measurement_statistics(
                measurements["bounding_width"]
            ),
            "body_width": measurement_statistics(
                measurements["body_width"]
            ),
            "neck_width": measurement_statistics(
                measurements["neck_width"]
            ),
            "lean": measurement_statistics(
                measurements["lean"]
            ),
        },
        
        "normalized_measurements": {
            "bounding_width_ratio": measurement_statistics(
                measurements["bounding_width_ratio"]
            ),
            "body_width_ratio": measurement_statistics(
                measurements["body_width_ratio"]
            ),
            "neck_width_ratio": measurement_statistics(
                measurements["neck_width_ratio"]
            ),
            "lean_ratio": measurement_statistics(
                measurements["lean_ratio"]
            ),
        },

        "width_profile": {
            "normalized_y": (
                profile_grid.tolist()
            ),
            "mean": mean_profile.tolist(),
            "std": np.zeros(
                PROFILE_POINTS
            ).tolist(),
        },
        
        "normalized_width_profile": {
            "normalized_y": (
                profile_grid.tolist()
            ),
            "mean": normalized_mean_profile.tolist(),
            "std": np.zeros(
                PROFILE_POINTS
            ).tolist(),
        },
    }

    reference_path.write_text(
        json.dumps(
            reference,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_normal_bottle_is_classified_as_normal(
    tmp_path: Path,
) -> None:
    normal_image = (
        tmp_path
        / "normal.png"
    )

    reference_file = (
        tmp_path
        / "reference.json"
    )

    create_bottle_image(
        normal_image
    )

    create_reference(
        normal_image,
        reference_file,
    )

    result = detect_defects(
        image_path=normal_image,
        reference_file=reference_file,
    )

    assert result["status"] == "NORMAL"
    assert result["defects"] == []


def test_bulged_bottle_is_detected(
    tmp_path: Path,
) -> None:
    normal_image = (
        tmp_path
        / "normal.png"
    )

    bulged_image = (
        tmp_path
        / "bulged.png"
    )

    reference_file = (
        tmp_path
        / "reference.json"
    )

    create_bottle_image(
        normal_image
    )

    create_bottle_image(
        bulged_image,
        bulge_magnitude=40.0,
        bulge_center=0.65,
        bulge_sigma=0.07,
    )

    create_reference(
        normal_image,
        reference_file,
    )

    result = detect_defects(
        image_path=bulged_image,
        reference_file=reference_file,
    )

    assert result["status"] == "DEFECTIVE"
    assert "bulge" in result["defects"]