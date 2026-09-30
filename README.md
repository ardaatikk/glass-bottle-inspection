# Glass Bottle Inspection

[![Tests](https://github.com/ardaatikk/glass-bottle-inspection/actions/workflows/tests.yml/badge.svg)](https://github.com/ardaatikk/glass-bottle-inspection/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An explainable computer vision pipeline...

An explainable computer vision pipeline for detecting geometric defects in glass bottles using silhouette analysis, statistical reference modeling, scale-normalized geometry, and local width-profile comparison.

> **Project status:** The current pipeline has been developed and evaluated on a controlled synthetic dataset of 600 bottle images. Real-world calibration and robustness testing are planned once representative production images become available.

---

## Overview

Automated visual inspection is an important part of quality control in manufacturing. This project explores a classical computer vision approach for detecting geometric defects in glass bottles without relying on a black-box deep learning model.

The system extracts the bottle silhouette, measures its geometry, compares it against a statistical reference built from known normal samples, and reports both the inspection result and the geometric reason behind it.

The current implementation supports:

- bottle silhouette extraction
- geometric measurement
- scale-normalized geometric features
- statistical normal-reference modeling
- global geometry comparison
- local normalized width-profile analysis
- multi-label defect detection
- synthetic dataset generation
- dataset validation
- automated detector evaluation
- unit and integration testing
- continuous integration with GitHub Actions

---

## Inspection Pipeline

```text
Bottle Image
     │
     ▼
Image Preprocessing
     │
     ▼
Silhouette Extraction
     │
     ▼
Geometric Measurement
     │
     ├── Height
     ├── Body Width
     ├── Neck Width
     ├── Lean
     └── Width Profile
     │
     ▼
Scale Normalization
     │
     ├── Body Width / Height
     ├── Neck Width / Height
     ├── Lean / Height
     └── Width Profile / Height
     │
     ▼
Statistical Normal Reference
     │
     ▼
Deviation Analysis
     │
     ├── Absolute Height
     ├── Normalized Geometry
     └── Normalized Local Width Profile
     │
     ▼
Defect Classification
     │
     ▼
NORMAL / DEFECTIVE
```

Unlike a pure classification model, the detector exposes the measurements and deviations that caused a bottle to be classified as defective.

---

## Supported Defect Types

The synthetic benchmark currently contains six classes:

| Class | Description |
|---|---|
| `normal` | Bottle geometry within expected manufacturing variation |
| `bulge` | Local outward deformation of the bottle body |
| `shrink` | Local inward deformation of the bottle body |
| `lean` | Horizontal displacement producing a tilted bottle geometry |
| `height` | Bottle significantly taller or shorter than the reference |
| `neck` | Bottle neck significantly wider or narrower than the reference |

The detector itself can return multiple defect labels when more than one independent geometric constraint is violated.

---

## Synthetic Dataset

A procedural dataset generator is included in the project so the inspection pipeline can be developed and evaluated without using confidential production data.

The default dataset contains:

```text
100 normal
100 bulge
100 shrink
100 lean
100 height
100 neck
----------------
600 images
```

Normal samples include controlled manufacturing variation, while defective samples introduce randomized geometric deformations.

The generated dataset also contains masks and metadata for validation and evaluation.

Generate the default dataset with:

```bash
python scripts/generate_data.py
```

Generation is deterministic by default using a fixed random seed.

Available options can be inspected with:

```bash
python scripts/generate_data.py --help
```

---

## Dataset Validation

The generated dataset can be checked before reference construction or evaluation:

```bash
python scripts/validate_dataset.py
```

For the current 600-image dataset:

```text
Total: 600
PASS:  600
FAIL:  0
```

The validation step helps detect inconsistent files or malformed generated samples before they enter the inspection pipeline.

---

## Statistical Reference Model

Instead of comparing a bottle against one ideal template, the project builds a statistical reference from normal bottle samples.

Run:

```bash
python scripts/build_reference.py
```

The reference model stores statistics for absolute measurements such as:

- height
- bounding width
- body width
- neck width
- lean

It also stores scale-normalized measurements including:

- bounding-width ratio
- body-width ratio
- neck-width ratio
- lean ratio

In addition to scalar measurements, the reference contains both an absolute and a scale-normalized bottle width profile.

The normalized width profile represents bottle width relative to bottle height along a common normalized vertical coordinate system.

The resulting reference is stored by default at:

```text
reference/normal_reference.json
```

This allows the detector to compare new samples against the distribution of normal bottle geometry rather than against a single hard-coded template.

---

## Scale-Normalized Geometry

Pixel measurements are useful when the imaging setup is fixed, but they can change when the same object is captured at a different apparent scale.

To reduce this dependency, the inspection pipeline derives normalized geometric features such as:

```text
body_width_ratio = body_width / height
neck_width_ratio = neck_width / height
lean_ratio       = lean / height
```

The local width profile is normalized in the same way:

```text
normalized_width = width / bottle_height
```

This produces a representation based primarily on bottle proportions rather than raw pixel size.

The current detector therefore uses a hybrid strategy:

- absolute pixel height is retained for height-defect detection
- normalized ratios are used for body, neck, and lean comparison
- normalized width profiles are used for local bulge and shrink detection

This preserves explicit height inspection while making width-based geometry less dependent on image scale.

Real-camera scale robustness still requires validation on representative production imagery.

---

## Defect Detection

Inspect an individual bottle with:

```bash
python -m bottle_inspection.detection data/images/normal/normal_0001.png
```

The detector performs the following operations:

1. loads and preprocesses the image
2. extracts the bottle silhouette
3. measures global bottle geometry
4. constructs scale-normalized geometric features
5. interpolates the width profile onto a common vertical grid
6. compares the sample against the statistical normal reference
7. analyzes local profile deviations
8. produces one or more explainable defect labels

A normal sample produces a result similar to:

```text
Bottle inspection result
------------------------
Status: NORMAL
Defects: none
```

A defective sample may instead produce:

```text
Bottle inspection result
------------------------
Status: DEFECTIVE
Defects: bulge
```

The local width-profile analysis also reports diagnostic information such as:

```text
Local width profile
-------------------
Maximum deviation: +0.04
Location:           y=0.63
Positive run:       8 points
Negative run:       0 points
```

Positive profile deviations indicate regions wider than the normal reference, while negative deviations indicate regions narrower than expected.

The detector does **not** use ground-truth masks or dataset labels during inference.

---

## Defect Classification Logic

The detector is internally multi-label: more than one geometric constraint may be abnormal for the same bottle.

However, some normalized measurements are mathematically affected by bottle height. For example, changing bottle height also changes a `width / height` ratio even when the physical width itself remains unchanged.

For this reason, the classification stage treats an explicit height anomaly specially.

When height is already outside its expected range:

- the height defect is retained
- lean can still be reported independently
- secondary normalized width, neck, bulge, and shrink labels caused by the height change are not promoted to separate defect labels

This prevents a pure height defect from being incorrectly interpreted as multiple unrelated geometric defects.

The underlying measurements and profile analysis remain available for diagnostics.

---

## Evaluation

The complete detector can be evaluated with:

```bash
python scripts/evaluate_detector.py
```

Ground-truth metadata is used only by the evaluation script after inference. It is not provided to the detector.

### Current Synthetic Benchmark

| Class | Correct | Accuracy |
|---|---:|---:|
| Normal | 100 / 100 | 100% |
| Bulge | 98 / 100 | 98% |
| Shrink | 95 / 100 | 95% |
| Lean | 100 / 100 | 100% |
| Height | 100 / 100 | 100% |
| Neck | 100 / 100 | 100% |
| **Overall** | **593 / 600** | **98.83%** |

Confusion matrix:

```text
actual        normal     bulge    shrink      lean    height      neck
normal           100         0         0         0         0         0
bulge              2        98         0         0         0         0
shrink             5         0        95         0         0         0
lean               0         0         0       100         0         0
height             0         0         0         0       100         0
neck               0         0         0         0         0       100
```

The remaining seven errors correspond to subtle synthetic local deformations that remain inside the current normalized width-profile tolerance.

The thresholds are intentionally not tuned solely to force perfect performance on the synthetic benchmark, since overly aggressive synthetic-data optimization could reduce robustness on future real-world imagery.

> **Important:** The 98.83% result represents performance on the project's controlled synthetic benchmark. It should not be interpreted as expected accuracy on real production imagery.

---

## Project Structure

```text
glass-bottle-inspection/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── reference/
│   └── normal_reference.json
│
├── scripts/
│   ├── build_reference.py
│   ├── evaluate_detector.py
│   ├── generate_data.py
│   └── validate_dataset.py
│
├── src/
│   └── bottle_inspection/
│       ├── __init__.py
│       ├── detection.py
│       ├── inspection.py
│       └── reference.py
│
├── tests/
│   ├── test_detection.py
│   ├── test_inspection.py
│   ├── test_pipeline.py
│   └── test_reference.py
│
├── .gitignore
├── pyproject.toml
└── README.md
```

The project follows a `src` layout so reusable package code remains separated from development and evaluation scripts.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ardaatikk/glass-bottle-inspection.git
cd glass-bottle-inspection
```

Creating a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode together with development dependencies:

```bash
pip install -e ".[dev]"
```

The project currently requires Python 3.10 or newer.

---

## Typical Workflow

A complete development run can be performed with:

```bash
# Generate synthetic data
python scripts/generate_data.py

# Validate generated samples
python scripts/validate_dataset.py

# Build the normal reference
python scripts/build_reference.py

# Inspect one bottle
python -m bottle_inspection.detection data/images/normal/normal_0001.png

# Evaluate the complete dataset
python scripts/evaluate_detector.py

# Run automated tests
python -m pytest
```

---

## Testing

The project includes automated tests covering core inspection, reference modeling, defect detection, and end-to-end pipeline behavior.

Run all tests with:

```bash
python -m pytest
```

Current test suite:

```text
14 passed
```

Tests are also executed automatically through GitHub Actions whenever the configured push or pull-request workflow is triggered.

---

## Design Decisions

### Why classical computer vision?

The first version intentionally uses an explainable geometry-based approach instead of immediately training a deep neural network.

For industrial inspection, knowing *why* a sample failed can be as important as the final classification.

The detector can expose information such as:

```text
height outside tolerance
neck-width ratio outside tolerance
lean ratio outside tolerance
local width-profile deviation at y = 0.63
```

rather than returning only a class probability.

This also provides a transparent baseline that can later be compared with or combined with learned models.

### Why a statistical reference?

Manufacturing processes contain normal geometric variation.

Using a statistical reference allows the detector to model this variation across multiple known-good samples and identify measurements that fall outside the expected distribution.

### Why normalize geometric measurements?

Raw pixel measurements depend partly on apparent image scale.

Representing width, neck geometry, lean, and local profiles relative to bottle height reduces this dependency and provides a more portable geometric representation.

Absolute height remains available because height itself is one of the defects the system is designed to detect.

### Why analyze a width profile?

Single global measurements can miss local defects.

For example, a bottle may have a normal overall body width while containing a localized bulge.

The width profile represents bottle width as a function of normalized vertical position, allowing local outward and inward deformations to be detected.

---

## Current Limitations

The current system has been developed primarily with controlled synthetic images.

Several challenges must be addressed before the pipeline can be evaluated on real production imagery:

- uneven illumination
- transparent glass and internal reflections
- complex backgrounds
- camera perspective and positioning
- segmentation noise
- silhouette holes and reflection artifacts
- real manufacturing variation
- validation of scale normalization under real camera-distance changes
- calibration using representative known-good bottles
- threshold calibration using labeled production samples

The current 98.83% benchmark therefore measures performance on synthetic data only.

Real-image performance has **not** yet been measured.

---

## Roadmap

Planned development stages:

- [x] Synthetic bottle generator
- [x] Dataset validation
- [x] Silhouette-based inspection
- [x] Statistical normal reference
- [x] Global geometry analysis
- [x] Local width-profile analysis
- [x] Defect classification
- [x] Automated evaluation
- [x] Unit and integration tests
- [x] GitHub Actions CI
- [x] Scale-normalized geometric measurements
- [x] Scale-normalized width-profile analysis
- [ ] Real-image segmentation experiments
- [ ] Adaptive thresholding and segmentation comparison
- [ ] Morphological cleanup and silhouette hole filling
- [ ] Batch inspection
- [ ] CSV/JSON batch reports
- [ ] Calibration with real known-good bottles
- [ ] Evaluation on labeled production images
- [ ] Hybrid geometry + machine learning inspection

---

## Future Direction

Once representative real-world images become available, the project will move from synthetic validation toward production-oriented computer vision.

The first real-image development stage will focus on segmentation robustness, including:

- global versus adaptive thresholding
- illumination sensitivity
- morphological cleanup
- silhouette hole filling
- reflection handling
- camera-scale and positioning robustness

After reliable silhouettes can be extracted, the existing explainable geometry pipeline can be calibrated using representative known-good production bottles.

The intended longer-term architecture is a hybrid system where explainable geometric inspection remains available while learned models can be introduced for defects that cannot be reliably represented using geometric rules alone.

This keeps the system interpretable while allowing more complex visual defects to be addressed in later stages.

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Pytest
- Git
- GitHub Actions

---

## Author

**Arda Atik**

Artificial Intelligence Engineer