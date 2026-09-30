import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 1000

DATA_DIR = Path("data")

IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

DEFECT_TYPES = [
    "normal",
    "bulge",
    "shrink",
    "lean",
    "height",
    "neck",
]

SAMPLES_PER_CLASS = 100

RANDOM_SEED = 42

BOTTLE_HEIGHT = 700
BODY_WIDTH = 300
NECK_WIDTH = 120
NORMAL_BODY_VARIATION = 3.0
NORMAL_NECK_VARIATION = 2.0
NORMAL_HEIGHT_VARIATION = 4

CENTER_X = IMAGE_WIDTH // 2
BOTTOM_Y = 850
TOP_Y = BOTTOM_Y - BOTTLE_HEIGHT

def create_directories() -> None:

    (IMAGES_DIR / "normal").mkdir(
        parents=True,
        exist_ok=True,
    )

    (MASKS_DIR / "normal").mkdir(
        parents=True,
        exist_ok=True,
    )

    for defect_type in DEFECT_TYPES:

        if defect_type == "normal":
            continue

        (IMAGES_DIR / "defective" / defect_type).mkdir(
            parents=True,
            exist_ok=True,
        )

        (MASKS_DIR / "defective" / defect_type).mkdir(
            parents=True,
            exist_ok=True,
        )

def get_bottle_width(
    y_normalized: float,
    body_width: float,
    neck_width: float,
) -> float:
    """
    Verilen şişe geometrisi için belirli yükseklikteki
    nominal genişliği döndürür.
    """

    if y_normalized < 0.18:
        return neck_width

    elif y_normalized < 0.32:

        t = (
            (y_normalized - 0.18)
            / (0.32 - 0.18)
        )

        return (
            neck_width
            + t * (body_width - neck_width)
        )

    else:
        return body_width
 
def apply_bulge(
    width: float,
    y_normalized: float,
    center: float = 0.65,
    magnitude: float = 30.0,
    sigma: float = 0.06,
) -> float:
    """
    Şişenin belirli bir bölgesine yumuşak bir dışa bombe ekler.

    center:
        Kusurun şişe yüksekliği üzerindeki merkezi.

    magnitude:
        Maksimum toplam genişlik artışı (pixel).

    sigma:
        Kusurun dikey olarak ne kadar yayıldığı.
    """

    deformation = magnitude * np.exp(
        -0.5
        * ((y_normalized - center) / sigma) ** 2
    )

    return width + deformation 

def apply_shrink(
    width: float,
    y_normalized: float,
    center: float = 0.55,
    magnitude: float = 30.0,
    sigma: float = 0.06,
) -> float:
    """
    Şişenin belirli bir bölgesine yumuşak bir içe daralma ekler.

    center:
        Kusurun şişe yüksekliği üzerindeki merkezi.

    magnitude:
        Maksimum toplam genişlik azalması (pixel).

    sigma:
        Kusurun dikey olarak ne kadar yayıldığı.
    """

    deformation = magnitude * np.exp(
        -0.5
        * ((y_normalized - center) / sigma) ** 2
    )

    return width - deformation

def apply_neck_defect(
    width: float,
    y_normalized: float,
    magnitude: float = 25.0,
) -> float:
    """
    Şişenin boyun bölgesini normalden daha geniş üretir.

    magnitude:
        Boyun bölgesindeki toplam genişlik artışı (pixel).
    """

    if y_normalized < 0.18:
        return width + magnitude

    return width

def get_bottle_center(
    y_normalized: float,
    defect_type: str = "normal",
    max_shift: float = 35.0,
) -> float:
    """
    Belirli yükseklikteki merkez x koordinatını döndürür.
    """

    if defect_type == "lean":

        shift = max_shift * (1.0 - y_normalized)

        return CENTER_X + shift

    return CENTER_X

def get_bottle_bounds(
    bottle_height: int,
    defect_type: str = "normal",
    height_change: int = 0,
) -> tuple[int, int]:

    bottom_y = BOTTOM_Y

    final_height = bottle_height

    if defect_type == "height":
        final_height += height_change

    top_y = bottom_y - final_height

    return top_y, bottom_y
    
def create_bottle_mask(
    defect_type: str = "normal",
    body_width: float = BODY_WIDTH,
    neck_width: float = NECK_WIDTH,
    bottle_height: int = BOTTLE_HEIGHT,
    magnitude: float = 0.0,
    center: float = 0.65,
    sigma: float = 0.06,
    lean_shift: float = 0.0,
    height_change: int = 0,
) -> np.ndarray:

    mask = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8,
    )

    left_points = []
    right_points = []
    
    top_y, bottom_y = get_bottle_bounds(
        bottle_height=bottle_height,
        defect_type=defect_type,
        height_change=height_change,
    )
    for y in range(top_y, bottom_y + 1):

        current_height = bottom_y - top_y

        y_normalized = (
            (y - top_y)
            / current_height
        )

        width = get_bottle_width(
            y_normalized,
            body_width=body_width,
            neck_width=neck_width,
        )

        if defect_type == "bulge":
            width = apply_bulge(
                width,
                y_normalized,
                center=center,
                magnitude=magnitude,
                sigma=sigma,
            )

        elif defect_type == "shrink":
            width = apply_shrink(
                width,
                y_normalized,
                center=center,
                magnitude=magnitude,
                sigma=sigma,
            )

        elif defect_type == "neck":
            width = apply_neck_defect(
                width,
                y_normalized,
                magnitude=magnitude,
            )
            
        center_x = get_bottle_center(
            y_normalized,
            defect_type,
            max_shift=lean_shift,
        )

        left_x = int(center_x - width / 2)
        right_x = int(center_x + width / 2)

        left_points.append(
            [left_x, y]
        )

        right_points.append(
            [right_x, y]
        )

    # For bittikten sonra polygon oluşturuyoruz
    bottle_points = np.array(
        left_points + right_points[::-1],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [bottle_points],
        255,
    )

    return mask

def render_image(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ground-truth mask'ten sentetik kamera/backlight görüntüsü üretir.

    Mask temiz kalır.
    Kamera etkileri yalnızca görüntüye uygulanır.
    """

    # Backlight parlaklığı
    background_brightness = rng.uniform(
        215.0,
        245.0,
    )

    # Şişe silüetinin parlaklığı
    bottle_brightness = rng.uniform(
        25.0,
        65.0,
    )

    image = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        background_brightness,
        dtype=np.float32,
    )

    image[mask == 255] = bottle_brightness
        
    gradient_strength = rng.uniform(
        -15.0,
        15.0,
    )

    gradient = np.linspace(
        -gradient_strength,
        gradient_strength,
        IMAGE_WIDTH,
        dtype=np.float32,
    )

    image += gradient[np.newaxis, :]
    
    blur_kernel = int(
        rng.choice([3, 5, 7])
    )

    image = cv2.GaussianBlur(
        image,
        (blur_kernel, blur_kernel),
        0,
    )
    
    noise_sigma = rng.uniform(
        1.0,
        5.0,
    )

    noise = rng.normal(
        0.0,
        noise_sigma,
        image.shape,
    )

    image += noise
    
    image = np.clip(
        image,
        0,
        255,
    )

    image = image.astype(
        np.uint8
    )
    
    return image

def generate_normal_geometry(
    rng: np.random.Generator,
) -> dict:
    """
    Kabul edilebilir üretim toleransı içerisinde
    normal bir şişe geometrisi oluşturur.
    """

    body_width = rng.uniform(
        BODY_WIDTH - NORMAL_BODY_VARIATION,
        BODY_WIDTH + NORMAL_BODY_VARIATION,
    )

    neck_width = rng.uniform(
        NECK_WIDTH - NORMAL_NECK_VARIATION,
        NECK_WIDTH + NORMAL_NECK_VARIATION,
    )

    bottle_height = int(
        rng.integers(
            BOTTLE_HEIGHT - NORMAL_HEIGHT_VARIATION,
            BOTTLE_HEIGHT + NORMAL_HEIGHT_VARIATION + 1,
        )
    )

    return {
        "body_width": body_width,
        "neck_width": neck_width,
        "bottle_height": bottle_height,
    }
    
def generate_random_parameters(
    defect_type: str,
    rng: np.random.Generator,
) -> dict:

    parameters = {
        "magnitude": 0.0,
        "center": 0.0,
        "sigma": 0.0,
        "lean_shift": 0.0,
        "height_change": 0,
    }

    if defect_type == "bulge":

        parameters["magnitude"] = rng.uniform(
            5.0,
            40.0,
        )

        parameters["center"] = rng.uniform(
            0.40,
            0.80,
        )

        parameters["sigma"] = rng.uniform(
            0.03,
            0.10,
        )

    elif defect_type == "shrink":

        parameters["magnitude"] = rng.uniform(
            5.0,
            40.0,
        )

        parameters["center"] = rng.uniform(
            0.40,
            0.80,
        )

        parameters["sigma"] = rng.uniform(
            0.03,
            0.10,
        )

    elif defect_type == "lean":

        direction = rng.choice(
            [-1.0, 1.0]
        )

        parameters["lean_shift"] = (
            direction
            * rng.uniform(5.0, 40.0)
        )

    elif defect_type == "height":

        direction = rng.choice(
            [-1, 1]
        )

        parameters["height_change"] = int(
            direction
            * rng.integers(10, 61)
        )

    elif defect_type == "neck":

        direction = rng.choice(
            [-1.0, 1.0]
        )

        parameters["magnitude"] = (
            direction
            * rng.uniform(5.0, 35.0)
        )

    return parameters

def generate_sample(
    defect_type: str,
    sample_index: int,
    rng: np.random.Generator,
) -> dict:

    geometry = generate_normal_geometry(
        rng,
    )
    
    parameters = generate_random_parameters(
        defect_type,
        rng,
    )

    mask = create_bottle_mask(
        defect_type=defect_type,
        body_width=geometry["body_width"],
        neck_width=geometry["neck_width"],
        bottle_height=geometry["bottle_height"],
        magnitude=parameters["magnitude"],
        center=parameters["center"],
        sigma=parameters["sigma"],
        lean_shift=parameters["lean_shift"],
        height_change=parameters["height_change"],
    )

    image = render_image(
        mask,
        rng,
    )

    filename = (
        f"{defect_type}_{sample_index:04d}.png"
    )

    mask_filename = (
        f"{defect_type}_{sample_index:04d}_mask.png"
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
        
    cv2.imwrite(
        str(image_path),
        image,
    )

    cv2.imwrite(
        str(mask_path),
        mask,
    )

    return {
        "filename": filename,
        "defect_type": defect_type,
        "magnitude": parameters["magnitude"],
        "center": parameters["center"],
        "sigma": parameters["sigma"],
        "lean_shift": parameters["lean_shift"],
        "height_change": parameters["height_change"],
        "body_width": geometry["body_width"],
        "neck_width": geometry["neck_width"],
        "bottle_height": geometry["bottle_height"],
    }
    
def main() -> None:

    create_directories()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    metadata = []

    for defect_type in DEFECT_TYPES:

        print(
            f"Generating {defect_type} samples..."
        )

        for sample_index in range(
            1,
            SAMPLES_PER_CLASS + 1,
        ):

            sample_metadata = generate_sample(
                defect_type,
                sample_index,
                rng,
            )

            metadata.append(
                sample_metadata
            )
    
    metadata_path = (
        DATA_DIR / "metadata.csv"
    )

    fieldnames = [
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
    ]

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            metadata
        )

    print()
    print("Dataset generation completed.")
    print(
        f"Total samples: {len(metadata)}"
    )
    print(
        f"Metadata: {metadata_path}"
    )  
        
if __name__ == "__main__":
    main()