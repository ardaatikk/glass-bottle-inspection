from pathlib import Path

import cv2
import numpy as np
import pytest

from bottle_inspection.inspection import (
    inspect_image,
)
from bottle_inspection.reference import (
    interpolate_width_profile,
)

def create_test_bottle(
    output_path: Path,
    width: int = 300,
    height: int = 700,
) -> None:
    """
    Create a simple synthetic bottle-like silhouette
    for integration testing.

    The image exists only during the test.
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

    for y in range(
        top_y,
        bottom_y + 1,
    ):

        normalized_y = (
            (y - top_y)
            / height
        )

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
                * (width - neck_width)
            )

        else:
            current_width = width

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
            "Failed to create test image."
        )


def test_inspect_image_measures_bottle(
    tmp_path: Path,
) -> None:
    """
    Verify that the complete inspection pipeline
    can segment and measure a synthetic bottle.
    """

    image_path = (
        tmp_path
        / "test_bottle.png"
    )

    create_test_bottle(
        image_path
    )

    result = inspect_image(
        image_path
    )

    measurements = result[
        "measurements"
    ]

    assert measurements[
        "height"
    ] == pytest.approx(
        700,
        abs=2,
    )

    assert measurements[
        "body_width"
    ] == pytest.approx(
        301,
        abs=2,
    )

    assert measurements[
        "neck_width"
    ] == pytest.approx(
        121,
        abs=2,
    )

    assert measurements[
        "lean"
    ] == pytest.approx(
        0,
        abs=1,
    )


def test_inspect_image_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    Missing input images must fail explicitly.
    """

    missing_path = (
        tmp_path
        / "does_not_exist.png"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        inspect_image(
            missing_path
        )
        
def test_normalized_measurements_are_scale_invariant(
    tmp_path: Path,
) -> None:
    small_image_path = (
        tmp_path / "small_bottle.png"
    )

    large_image_path = (
        tmp_path / "large_bottle.png"
    )

    small_image = np.full(
        (500, 400, 3),
        255,
        dtype=np.uint8,
    )

    large_image = np.full(
        (1000, 800, 3),
        255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        small_image,
        (125, 100),
        (275, 450),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.rectangle(
        large_image,
        (250, 200),
        (550, 900),
        (0, 0, 0),
        thickness=-1,
    )

    assert cv2.imwrite(
        str(small_image_path),
        small_image,
    )

    assert cv2.imwrite(
        str(large_image_path),
        large_image,
    )

    small_result = inspect_image(
        small_image_path
    )

    large_result = inspect_image(
        large_image_path
    )

    small = small_result["measurements"]
    large = large_result["measurements"]

    assert large["height"] > small["height"]
    assert (
        large["body_width"]
        > small["body_width"]
    )

    assert np.isclose(
        small["bounding_width_ratio"],
        large["bounding_width_ratio"],
        atol=0.005,
    )

    assert np.isclose(
        small["body_width_ratio"],
        large["body_width_ratio"],
        atol=0.005,
    )

    assert np.isclose(
        small["neck_width_ratio"],
        large["neck_width_ratio"],
        atol=0.005,
    )

    assert np.isclose(
        small["lean_ratio"],
        large["lean_ratio"],
        atol=0.005,
    )
    
def test_normalized_width_profile_is_scale_invariant(
    tmp_path: Path,
) -> None:
    small_image_path = (
        tmp_path / "small_profile.png"
    )

    large_image_path = (
        tmp_path / "large_profile.png"
    )

    small_image = np.full(
        (500, 400, 3),
        255,
        dtype=np.uint8,
    )

    large_image = np.full(
        (1000, 800, 3),
        255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        small_image,
        (125, 100),
        (275, 450),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.rectangle(
        large_image,
        (250, 200),
        (550, 900),
        (0, 0, 0),
        thickness=-1,
    )

    assert cv2.imwrite(
        str(small_image_path),
        small_image,
    )

    assert cv2.imwrite(
        str(large_image_path),
        large_image,
    )

    small = inspect_image(
        small_image_path
    )["measurements"]

    large = inspect_image(
        large_image_path
    )["measurements"]

    target_y = np.linspace(
        0.0,
        1.0,
        101,
        dtype=np.float64,
    )

    small_profile = (
        interpolate_width_profile(
            small["normalized_width_y"],
            small["normalized_width_profile"],
            target_y,
        )
    )

    large_profile = (
        interpolate_width_profile(
            large["normalized_width_y"],
            large["normalized_width_profile"],
            target_y,
        )
    )

    assert np.allclose(
        small_profile,
        large_profile,
        atol=0.005,
    )