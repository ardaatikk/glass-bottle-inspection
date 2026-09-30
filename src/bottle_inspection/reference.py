import numpy as np


def interpolate_width_profile(
    normalized_y: np.ndarray,
    width_profile: np.ndarray,
    profile_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a bottle width profile onto a common
    normalized vertical coordinate grid.
    """

    if len(normalized_y) != len(width_profile):
        raise ValueError(
            "Normalized coordinates and width profile "
            "must have the same length."
        )

    if len(normalized_y) < 2:
        raise ValueError(
            "Width profile contains too few points."
        )

    return np.interp(
        profile_grid,
        normalized_y,
        width_profile,
    )