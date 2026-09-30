import numpy as np

from bottle_inspection.detection import (
    calculate_tolerance,
    compare_measurement,
    find_longest_true_run,
)


def test_calculate_tolerance_uses_standard_deviation() -> None:
    statistics = {
        "mean": 100.0,
        "std": 4.0,
    }

    tolerance = calculate_tolerance(
        statistics=statistics,
        std_factor=3.0,
        minimum_tolerance=5.0,
    )

    assert tolerance == 12.0


def test_calculate_tolerance_respects_minimum() -> None:
    statistics = {
        "mean": 100.0,
        "std": 0.5,
    }

    tolerance = calculate_tolerance(
        statistics=statistics,
        std_factor=3.0,
        minimum_tolerance=5.0,
    )

    assert tolerance == 5.0


def test_compare_measurement_detects_abnormal_value() -> None:
    statistics = {
        "mean": 100.0,
        "std": 2.0,
    }

    result = compare_measurement(
        value=110.0,
        statistics=statistics,
        std_factor=3.0,
        minimum_tolerance=3.0,
    )

    assert result["abnormal"] is True
    assert result["deviation"] == 10.0
    assert result["tolerance"] == 6.0


def test_compare_measurement_accepts_normal_value() -> None:
    statistics = {
        "mean": 100.0,
        "std": 2.0,
    }

    result = compare_measurement(
        value=103.0,
        statistics=statistics,
        std_factor=3.0,
        minimum_tolerance=3.0,
    )

    assert result["abnormal"] is False


def test_find_longest_true_run() -> None:
    values = np.array(
        [
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
        ]
    )

    result = find_longest_true_run(
        values
    )

    assert result == 3