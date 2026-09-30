from pathlib import Path

import cv2
import pandas as pd
import numpy as np


DATA_DIR = Path("data")

IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

METADATA_PATH = DATA_DIR / "metadata.csv"
GEOMETRY_TOLERANCE_PX = 2.1

def get_sample_paths(
    filename: str,
    defect_type: str,
) -> tuple[Path, Path]:

    stem = Path(filename).stem

    mask_filename = (
        f"{stem}_mask.png"
    )

    if defect_type == "normal":

        image_path = (
            IMAGES_DIR
            / "normal"
            / filename
        )

        mask_path = (
            MASKS_DIR
            / "normal"
            / mask_filename
        )

    else:

        image_path = (
            IMAGES_DIR
            / "defective"
            / defect_type
            / filename
        )

        mask_path = (
            MASKS_DIR
            / "defective"
            / defect_type
            / mask_filename
        )

    return image_path, mask_path

def is_binary_mask(
    mask: np.ndarray,
) -> bool:

    unique_values = np.unique(mask)

    return set(unique_values).issubset(
        {0, 255}
    )
    
def measure_mask(
    mask: np.ndarray,
) -> dict:

    y_coordinates, x_coordinates = np.where(
        mask == 255
    )

    if len(x_coordinates) == 0:
        return {
            "empty": True,
            "width": 0,
            "height": 0,
            "top": None,
            "bottom": None,
            "left": None,
            "right": None,
        }

    left = int(x_coordinates.min())
    right = int(x_coordinates.max())

    top = int(y_coordinates.min())
    bottom = int(y_coordinates.max())

    width = right - left + 1
    height = bottom - top + 1

    return {
        "empty": False,
        "width": width,
        "height": height,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }

def measure_width_at_y(
    mask: np.ndarray,
    y: int,
) -> int | None:
    """
    Mask üzerinde belirli bir y satırındaki
    şişe genişliğini ölçer.
    """

    if y < 0 or y >= mask.shape[0]:
        return None

    x_coordinates = np.where(
        mask[y] == 255
    )[0]

    if len(x_coordinates) == 0:
        return None

    left = int(x_coordinates.min())
    right = int(x_coordinates.max())

    return right - left + 1

def measure_center_at_y(
    mask: np.ndarray,
    y: int,
) -> float | None:
    """
    Mask üzerinde belirli bir y satırındaki
    şişe merkezini ölçer.
    """

    if y < 0 or y >= mask.shape[0]:
        return None

    x_coordinates = np.where(
        mask[y] == 255
    )[0]

    if len(x_coordinates) == 0:
        return None

    left = float(x_coordinates.min())
    right = float(x_coordinates.max())

    return (
        left + right
    ) / 2.0

def validate_normal_body_width(
    mask: np.ndarray,
    row: pd.Series,
) -> tuple[bool, float, float]:

    measurement = measure_mask(mask)

    top = measurement["top"]
    bottom = measurement["bottom"]

    # Gövdenin alt tarafında güvenli bir nokta.
    y_normalized = 0.80

    y = int(
        top
        + y_normalized
        * (bottom - top)
    )

    measured_width = measure_width_at_y(
        mask,
        y,
    )

    expected_width = float(
        row["body_width"]
    )

    if measured_width is None:
        return (
            False,
            expected_width,
            float("nan"),
        )

    error = abs(
        measured_width
        - expected_width
    )

    return (
        error <= GEOMETRY_TOLERANCE_PX,
        expected_width,
        float(measured_width),
    )

def validate_local_width_defect(
    mask: np.ndarray,
    row: pd.Series,
) -> tuple[bool, float, float]:

    measurement = measure_mask(mask)

    top = measurement["top"]
    bottom = measurement["bottom"]

    defect_center = float(
        row["center"]
    )

    y = int(
        top
        + defect_center
        * (bottom - top)
    )

    measured_width = measure_width_at_y(
        mask,
        y,
    )

    body_width = float(
        row["body_width"]
    )

    magnitude = float(
        row["magnitude"]
    )

    defect_type = row[
        "defect_type"
    ]

    if defect_type == "bulge":

        expected_width = (
            body_width
            + magnitude
        )

    elif defect_type == "shrink":

        expected_width = (
            body_width
            - magnitude
        )

    else:
        raise ValueError(
            "This validator only supports "
            "bulge and shrink."
        )

    if measured_width is None:

        return (
            False,
            expected_width,
            float("nan"),
        )

    error = abs(
        measured_width
        - expected_width
    )

    return (
        error <= GEOMETRY_TOLERANCE_PX,
        expected_width,
        float(measured_width),
    )

def validate_neck_width(
    mask: np.ndarray,
    row: pd.Series,
) -> tuple[bool, float, float]:

    measurement = measure_mask(mask)

    top = measurement["top"]
    bottom = measurement["bottom"]

    # Boynun ortalarından ölçelim.
    y_normalized = 0.10

    y = int(
        top
        + y_normalized
        * (bottom - top)
    )

    measured_width = measure_width_at_y(
        mask,
        y,
    )

    expected_width = (
        float(row["neck_width"])
        + float(row["magnitude"])
    )

    if measured_width is None:

        return (
            False,
            expected_width,
            float("nan"),
        )

    error = abs(
        measured_width
        - expected_width
    )

    return (
        error <= GEOMETRY_TOLERANCE_PX,
        expected_width,
        float(measured_width),
    )

def validate_lean(
    mask: np.ndarray,
    row: pd.Series,
) -> tuple[bool, float, float]:

    measurement = measure_mask(mask)

    top = measurement["top"]
    bottom = measurement["bottom"]

    top_ratio = 0.10
    bottom_ratio = 0.90

    top_y = int(
        top
        + top_ratio
        * (bottom - top)
    )

    bottom_y = int(
        top
        + bottom_ratio
        * (bottom - top)
    )

    top_center = measure_center_at_y(
        mask,
        top_y,
    )

    bottom_center = measure_center_at_y(
        mask,
        bottom_y,
    )

    if (
        top_center is None
        or bottom_center is None
    ):

        return (
            False,
            float(row["lean_shift"]),
            float("nan"),
        )

    measured_shift = (
        top_center
        - bottom_center
    )

    # Generator:
    # shift(y) = lean_shift * (1 - y)
    #
    # y=0.10 ve y=0.90 arasındaki fark:
    # 0.90 - 0.10 = 0.80
    expected_shift = (
        float(row["lean_shift"])
        * (
            bottom_ratio
            - top_ratio
        )
    )

    error = abs(
        measured_shift
        - expected_shift
    )

    return (
        error <= GEOMETRY_TOLERANCE_PX,
        expected_shift,
        measured_shift,
    )
        
def get_expected_height(
    row: pd.Series,
) -> int:

    bottle_height = int(
        row["bottle_height"]
    )

    height_change = int(
        row["height_change"]
    )

    if row["defect_type"] == "height":
        return (
            bottle_height
            + height_change
            + 1
        )

    return bottle_height + 1

def validate_sample(
    row: pd.Series,
) -> dict:

    filename = row["filename"]
    defect_type = row["defect_type"]

    errors = []

    # Bu değişkenleri baştan tanımlıyoruz.
    # Height örneklerinde ayrıca geometry validation
    # yapılmadığı için None olarak kalabilirler.
    expected_geometry = None
    measured_geometry = None

    image_path, mask_path = get_sample_paths(
        filename,
        defect_type,
    )

    # --------------------
    # File existence
    # --------------------

    if not image_path.exists():
        errors.append(
            "image_missing"
        )

    if not mask_path.exists():
        errors.append(
            "mask_missing"
        )

    if errors:
        return {
            "filename": filename,
            "defect_type": defect_type,
            "status": "FAIL",
            "errors": ", ".join(errors),
            "expected_height": None,
            "measured_height": None,
            "expected_geometry": None,
            "measured_geometry": None,
        }

    # --------------------
    # Read files
    # --------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        errors.append(
            "image_unreadable"
        )

    if mask is None:
        errors.append(
            "mask_unreadable"
        )

    if errors:
        return {
            "filename": filename,
            "defect_type": defect_type,
            "status": "FAIL",
            "errors": ", ".join(errors),
            "expected_height": None,
            "measured_height": None,
            "expected_geometry": None,
            "measured_geometry": None,
        }

    # --------------------
    # Binary mask
    # --------------------

    if not is_binary_mask(mask):
        errors.append(
            "mask_not_binary"
        )

    # --------------------
    # Measure mask
    # --------------------

    measurement = measure_mask(
        mask
    )

    if measurement["empty"]:
        errors.append(
            "empty_mask"
        )

    # --------------------
    # Image boundaries
    # --------------------

    if not measurement["empty"]:

        image_height, image_width = (
            image.shape
        )

        if measurement["left"] <= 0:
            errors.append(
                "touches_left_border"
            )

        if measurement["right"] >= (
            image_width - 1
        ):
            errors.append(
                "touches_right_border"
            )

        if measurement["top"] <= 0:
            errors.append(
                "touches_top_border"
            )

        if measurement["bottom"] >= (
            image_height - 1
        ):
            errors.append(
                "touches_bottom_border"
            )

    # --------------------
    # Height validation
    # --------------------

    expected_height = (
        get_expected_height(row)
    )

    measured_height = (
        measurement["height"]
    )

    if (
        not measurement["empty"]
        and measured_height
        != expected_height
    ):
        errors.append(
            "height_mismatch"
        )

    # --------------------
    # Geometry validation
    # --------------------

    if not measurement["empty"]:

        if defect_type == "normal":

            (
                geometry_ok,
                expected_geometry,
                measured_geometry,
            ) = validate_normal_body_width(
                mask,
                row,
            )

            if not geometry_ok:
                errors.append(
                    "body_width_mismatch"
                )

        elif defect_type in {
            "bulge",
            "shrink",
        }:

            (
                geometry_ok,
                expected_geometry,
                measured_geometry,
            ) = validate_local_width_defect(
                mask,
                row,
            )

            if not geometry_ok:
                errors.append(
                    f"{defect_type}_mismatch"
                )

        elif defect_type == "neck":

            (
                geometry_ok,
                expected_geometry,
                measured_geometry,
            ) = validate_neck_width(
                mask,
                row,
            )

            if not geometry_ok:
                errors.append(
                    "neck_width_mismatch"
                )

        elif defect_type == "lean":

            (
                geometry_ok,
                expected_geometry,
                measured_geometry,
            ) = validate_lean(
                mask,
                row,
            )

            if not geometry_ok:
                errors.append(
                    "lean_mismatch"
                )

        # Height için geometry kontrolü burada yok.
        # Çünkü yukarıdaki height validation zaten
        # final yüksekliği kontrol ediyor.

    # --------------------
    # Final status
    # --------------------

    status = (
        "PASS"
        if len(errors) == 0
        else "FAIL"
    )

    return {
        "filename": filename,
        "defect_type": defect_type,
        "status": status,
        "errors": ", ".join(errors),
        "expected_height": expected_height,
        "measured_height": measured_height,
        "expected_geometry": expected_geometry,
        "measured_geometry": measured_geometry,
    }
    
def main() -> None:

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Validating {len(metadata)} samples..."
    )

    results = []

    for _, row in metadata.iterrows():

        result = validate_sample(
            row
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    pass_count = (
        results_df["status"]
        == "PASS"
    ).sum()

    fail_count = (
        results_df["status"]
        == "FAIL"
    ).sum()

    print()
    print("Validation completed.")
    print("---------------------")
    print(
        f"Total: {len(results_df)}"
    )
    print(
        f"PASS:  {pass_count}"
    )
    print(
        f"FAIL:  {fail_count}"
    )

    failed_samples = results_df[
        results_df["status"] == "FAIL"
    ]

    if not failed_samples.empty:

        print()
        print("Failed samples:")
        print(
            failed_samples[
                [
                    "filename",
                    "errors",
                    "expected_height",
                    "measured_height",
                    "expected_geometry",
                    "measured_geometry",
                ]
            ].to_string(
                index=False
            )
        )

if __name__ == "__main__":
    main()