import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from bottle_inspection.detection import (
    DEFAULT_REFERENCE_FILE,
    detect_defects,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_DATA_DIR = Path("data")

EXPECTED_CLASSES = (
    "normal",
    "bulge",
    "shrink",
    "lean",
    "height",
    "neck",
)


# ============================================================
# Metadata
# ============================================================

def load_metadata(
    metadata_file: Path,
) -> list[dict]:
    """
    Load ground-truth labels used only for evaluation.

    The detector itself never receives these labels.
    """

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist: "
            f"{metadata_file}"
        )

    with metadata_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "Metadata contains no samples."
        )

    return rows


# ============================================================
# Path resolution
# ============================================================

def get_image_path(
    data_dir: Path,
    row: dict,
) -> Path:
    """
    Resolve an image path from evaluation metadata.
    """

    filename = row[
        "filename"
    ]

    defect_type = row[
        "defect_type"
    ]

    if defect_type == "normal":
        return (
            data_dir
            / "images"
            / "normal"
            / filename
        )

    return (
        data_dir
        / "images"
        / "defective"
        / defect_type
        / filename
    )


# ============================================================
# Prediction mapping
# ============================================================

def map_detection_to_class(
    detection: dict,
) -> str:
    """
    Convert detector output into one primary class
    for evaluation.

    The detector itself remains multi-label.
    Evaluation requires one class so that a confusion
    matrix can be constructed.
    """

    defects = detection[
        "defects"
    ]

    if not defects:
        return "normal"

    # Local shape defects take priority because they are
    # directly supported by the width-profile analysis.
    if "bulge" in defects:
        return "bulge"

    if "shrink" in defects:
        return "shrink"

    if "lean" in defects:
        return "lean"

    if (
        "neck_too_wide" in defects
        or "neck_too_narrow" in defects
    ):
        return "neck"

    if (
        "height_too_tall" in defects
        or "height_too_short" in defects
    ):
        return "height"

    # These are currently diagnostic fallback labels.
    if (
        "body_too_wide" in defects
        or "body_too_narrow" in defects
    ):
        return "body_width"

    return "unknown"


# ============================================================
# Evaluation
# ============================================================

def evaluate_detector(
    data_dir: Path = DEFAULT_DATA_DIR,
    reference_file: Path = DEFAULT_REFERENCE_FILE,
) -> dict:
    """
    Evaluate the detector against the generated dataset.

    Ground-truth metadata is used only after prediction
    to determine whether the detector was correct.
    """

    data_dir = Path(
        data_dir
    )

    metadata_file = (
        data_dir
        / "metadata.csv"
    )

    rows = load_metadata(
        metadata_file
    )

    confusion = defaultdict(
        Counter
    )

    class_totals = Counter()
    class_correct = Counter()

    errors = []

    total = 0
    correct = 0

    print(
        f"Evaluating detector on "
        f"{len(rows)} images..."
    )

    for index, row in enumerate(
        rows,
        start=1,
    ):

        actual_class = row[
            "defect_type"
        ]

        image_path = get_image_path(
            data_dir=data_dir,
            row=row,
        )

        try:
            detection = detect_defects(
                image_path=image_path,
                reference_file=reference_file,
            )

            predicted_class = (
                map_detection_to_class(
                    detection
                )
            )

        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ) as error:

            predicted_class = (
                "inspection_error"
            )

            errors.append(
                {
                    "filename": row[
                        "filename"
                    ],
                    "error": str(error),
                }
            )

        confusion[
            actual_class
        ][
            predicted_class
        ] += 1

        class_totals[
            actual_class
        ] += 1

        total += 1

        if (
            predicted_class
            == actual_class
        ):
            correct += 1

            class_correct[
                actual_class
            ] += 1

        if (
            index % 100
            == 0
            or index == len(rows)
        ):
            print(
                f"Processed "
                f"{index}/{len(rows)}"
            )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    return {
        "total": total,
        "correct": correct,
        "incorrect": (
            total - correct
        ),
        "accuracy": accuracy,
        "class_totals": dict(
            class_totals
        ),
        "class_correct": dict(
            class_correct
        ),
        "confusion": {
            actual: dict(
                predictions
            )
            for (
                actual,
                predictions
            ) in confusion.items()
        },
        "inspection_errors": errors,
    }


# ============================================================
# Reporting
# ============================================================

def print_class_accuracy(
    result: dict,
) -> None:
    """
    Print accuracy for each ground-truth class.
    """

    print()
    print(
        "Per-class accuracy"
    )
    print(
        "------------------"
    )

    class_totals = result[
        "class_totals"
    ]

    class_correct = result[
        "class_correct"
    ]

    for class_name in EXPECTED_CLASSES:

        total = class_totals.get(
            class_name,
            0,
        )

        correct = class_correct.get(
            class_name,
            0,
        )

        accuracy = (
            correct / total
            if total
            else 0.0
        )

        print(
            f"{class_name:10s} "
            f"{correct:3d}/{total:3d}  "
            f"{accuracy * 100:6.2f}%"
        )


def print_confusion_matrix(
    result: dict,
) -> None:
    """
    Print a compact confusion matrix.
    """

    confusion = result[
        "confusion"
    ]

    predicted_classes = list(
        EXPECTED_CLASSES
    )

    extra_classes = sorted(
        {
            predicted
            for predictions
            in confusion.values()
            for predicted
            in predictions
            if predicted
            not in predicted_classes
        }
    )

    predicted_classes.extend(
        extra_classes
    )

    print()
    print(
        "Confusion matrix"
    )
    print(
        "----------------"
    )

    header = (
        f"{'actual':10s}"
        + "".join(
            f"{name[:8]:>10s}"
            for name in predicted_classes
        )
    )

    print(
        header
    )

    for actual_class in EXPECTED_CLASSES:

        row = (
            f"{actual_class:10s}"
        )

        for predicted_class in predicted_classes:

            count = (
                confusion
                .get(
                    actual_class,
                    {}
                )
                .get(
                    predicted_class,
                    0,
                )
            )

            row += (
                f"{count:10d}"
            )

        print(
            row
        )


def print_evaluation_summary(
    result: dict,
) -> None:
    """
    Print the complete evaluation report.
    """

    print()
    print(
        "Detector evaluation completed."
    )
    print(
        "------------------------------"
    )

    print(
        f"Total:     "
        f"{result['total']}"
    )

    print(
        f"Correct:   "
        f"{result['correct']}"
    )

    print(
        f"Incorrect: "
        f"{result['incorrect']}"
    )

    print(
        f"Accuracy:  "
        f"{result['accuracy'] * 100:.2f}%"
    )

    print_class_accuracy(
        result
    )

    print_confusion_matrix(
        result
    )

    inspection_errors = result[
        "inspection_errors"
    ]

    if inspection_errors:

        print()
        print(
            "Inspection errors"
        )
        print(
            "-----------------"
        )

        for error in inspection_errors:
            print(
                f"{error['filename']}: "
                f"{error['error']}"
            )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the glass bottle defect detector "
            "against a labeled synthetic dataset."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Dataset directory. "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_FILE,
        help=(
            "Normal reference JSON. "
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

    result = evaluate_detector(
        data_dir=args.data_dir,
        reference_file=args.reference,
    )

    print_evaluation_summary(
        result
    )


if __name__ == "__main__":
    main()