import numpy as np
import pytest

from bottle_inspection.reference import (
    interpolate_width_profile,
)


def test_interpolate_width_profile() -> None:
    normalized_y = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    width_profile = np.array(
        [100.0, 200.0, 300.0],
        dtype=np.float64,
    )

    profile_grid = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float64,
    )

    result = interpolate_width_profile(
        normalized_y=normalized_y,
        width_profile=width_profile,
        profile_grid=profile_grid,
    )

    expected = np.array(
        [100.0, 150.0, 200.0, 250.0, 300.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_interpolation_rejects_mismatched_lengths() -> None:
    normalized_y = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    width_profile = np.array(
        [100.0, 200.0],
        dtype=np.float64,
    )

    profile_grid = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        interpolate_width_profile(
            normalized_y=normalized_y,
            width_profile=width_profile,
            profile_grid=profile_grid,
        )


def test_interpolation_rejects_single_point_profile() -> None:
    normalized_y = np.array(
        [0.5],
        dtype=np.float64,
    )

    width_profile = np.array(
        [200.0],
        dtype=np.float64,
    )

    profile_grid = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="too few points",
    ):
        interpolate_width_profile(
            normalized_y=normalized_y,
            width_profile=width_profile,
            profile_grid=profile_grid,
        )