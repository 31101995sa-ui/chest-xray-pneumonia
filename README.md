# Chest X-Ray Pneumonia AI Project

Educational end-to-end computer vision project for binary chest X-ray classification:

- `NORMAL`
- `PNEUMONIA`

The project was developed as the first practical AI/ML assignment in an AI engineering incubator.

The main goal is not to build a clinical diagnostic system, but to demonstrate a reproducible ML engineering workflow from raw data to a working inference API.

```text
DATASET
   ↓
INSPECTION / EDA
   ↓
SPLIT + LEAKAGE AUDIT
   ↓
PREPROCESSING
   ↓
DATASET / DATALOADER
   ↓
RESNET18 / EFFICIENTNET-B0
   ↓
TRAINING + VALIDATION
   ↓
MODEL COMPARISON
   ↓
SEALED TEST
   ↓
ERROR ANALYSIS
   ↓
BEST MODEL
   ↓
INFERENCE
   ↓
FASTAPI
   ↓
AUTOMATED TESTS
```

> **Educational / research prototype only.**
> Not intended for clinical diagnosis, treatment decisions, or medical use.


---

# Quick Start

This section provides the shortest path from a fresh repository checkout to a working project.

## 1. Create a virtual environment

Recommended Python version:

```text
Python 3.12
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If Windows path-length limitations prevent creation of a local `.venv`, an external environment can be used instead:

```powershell
python -m venv C:\venvs\xray
C:\venvs\xray\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check dependency consistency:

```powershell
python -m pip check
```

Expected result:

```text
No broken requirements found.
```

## 3. Run automated tests

```powershell
python -m pytest -v
```

Current expected result:

```text
7 passed
```

The automated API tests use a lightweight dummy model, so the test suite does not require the trained `.pth` checkpoint.

## 4. Add the trained model checkpoint

The trained model checkpoint is intentionally not stored in Git.

For real inference and API predictions, place the selected ResNet18 checkpoint at:

```text
models/resnet18_baseline_repro_best.pth
```

Expected structure:

```text
chest-xray-pneumonia/
└── models/
    └── resnet18_baseline_repro_best.pth
```

The raw medical dataset is not required for normal API inference.

## 5. Verify the project

Run:

```powershell
python -m scripts.verify_project
```

The verification script checks:

```text
Python environment
PyTorch
split manifest
model checkpoint
model loading
real inference, when the reference dataset image is available
```

A successful full local verification ends with:

```text
PROJECT VERIFICATION PASSED
```

If the raw dataset is not installed locally, the reference-image inference check may be skipped after successful model loading.

## 6. Start the API

```powershell
python -m uvicorn api.main:app
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

Prediction:

```text
POST /predict
```

Upload an image using the `file` field.

Example response:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.9999955892562866,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

The `probability` field always represents:

```text
P(PNEUMONIA)
```

## Quick verification sequence

For an already prepared environment with the checkpoint installed:

```powershell
python -m pip check
python -m pytest -v
python -m scripts.verify_project
python -m uvicorn api.main:app
```

Then open:

```text
http://127.0.0.1:8000/docs
```

> The project is an educational/research prototype and is not intended for clinical use.

---
---

# 1. Task

The original assignment was to create a medical AI project using an open Chest X-Ray Pneumonia dataset.

The required tasks were:

1. Use an open chest X-ray dataset.
2. Classify images as `NORMAL` or `PNEUMONIA`.
3. Compare at least two CNN architectures.
4. Evaluate the models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
5. Analyze incorrect predictions.
6. Build a simple API or demo that accepts an uploaded X-ray image and returns a prediction.

The implemented architectures are:

- ResNet18
- EfficientNet-B0

An additional weighted ResNet18 experiment was used as a class-imbalance ablation.

The final selected model is:

```text
ResNet18 baseline
```

---

# 2. Dataset

Dataset:

**Chest X-Ray Images (Pneumonia)**

Open dataset available through Kaggle.

The dataset contains two classes:

```text
NORMAL
PNEUMONIA
```

Original structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The raw medical dataset is intentionally **not stored in Git**.

---

# 3. Dataset inspection

The dataset was inspected using project scripts instead of relying only on dataset documentation.

Observed dataset size:

```text
Total images: 5856
```

Original training split:

```text
NORMAL       1341
PNEUMONIA    3875
Total        5216
```

Original validation split:

```text
NORMAL          8
PNEUMONIA       8
Total          16
```

Original test split:

```text
NORMAL        234
PNEUMONIA     390
Total         624
```

Additional inspection results:

```text
Images checked:        5856
Broken images:         0

Image mode:
L                      5573
RGB                     283

Unique image sizes:    4803

Width:
384–2916 px

Height:
127–2713 px

Orientation:
Landscape              5790
Portrait                  62
Square                     4
```

The visual EDA showed substantial variation in:

- image size;
- crop;
- framing;
- patient positioning;
- orientation;
- contrast;
- acquisition markers;
- technical annotations;
- surrounding anatomy;
- other acquisition-related artifacts.

---

# 4. Validation split design

The original validation split contained only 16 images:

```text
8 NORMAL
8 PNEUMONIA
```

This was considered too small for reliable model selection.

Therefore, a new validation subset was created from the original training data.

The original test set remained sealed until model selection was complete.

Final working split:

```text
TRAIN

NORMAL       1140
PNEUMONIA    3294
Total        4434
```

```text
VALIDATION

NORMAL        201
PNEUMONIA     581
Total         782
```

```text
SEALED TEST

NORMAL        234
PNEUMONIA     390
Total         624
```

```text
LEGACY VALIDATION

NORMAL          8
PNEUMONIA       8
Total          16
```

The split was generated using:

```text
random seed = 42
```

The split manifest is stored in:

```text
data/splits/split_manifest.csv
```

Manifest SHA-256:

```text
0216660DC53CC9F196790FB8D342BC82BE9420B74184E93942E2F7B8258D52D1
```

---

# 5. Leakage and duplicate audit

Several checks were performed before model training.

Exact SHA-256 image hashing:

```text
Images hashed:             5856
Unique hashes:             5824
Duplicate groups:            30
Cross-split duplicates:       0
Cross-class duplicates:       0
```

No exact train/validation leakage was detected.

Additional perceptual-hash experiments were also performed.

A 64-bit dHash analysis produced perceptual collisions across splits, including some opposite-label examples.

Because dHash was too coarse for reliable medical-image deduplication, these results were treated as exploratory only.

No images were automatically removed based on perceptual hashes.

---

# 6. Preprocessing

Both architectures use the same base preprocessing pipeline.

```text
PIL image
   ↓
convert to RGB
   ↓
resize to 224 × 224
   ↓
ToTensor()
   ↓
ImageNet normalization
   ↓
tensor [3, 224, 224]
```

Pretrained ImageNet CNNs expect three input channels.

Most images in the dataset are grayscale, therefore grayscale X-rays are converted to RGB before entering the model.

Expected single-image tensor shape:

```text
[3, 224, 224]
```

Expected batch shape:

```text
[B, 3, 224, 224]
```

The same preprocessing implementation is used during:

- training;
- validation;
- test evaluation;
- standalone inference;
- FastAPI inference.

This avoids train/inference preprocessing drift.

---

# 7. Dataset and DataLoader

A custom PyTorch dataset reads samples from the split manifest.

The pipeline supports:

```text
train
val
test
legacy_val
```

Training DataLoader:

```text
shuffle = True
```

Validation and test DataLoaders:

```text
shuffle = False
```

Random seed:

```text
42
```

Example batch:

```text
torch.Size([32, 3, 224, 224])
```

Labels:

```text
0 = NORMAL
1 = PNEUMONIA
```

---

# 8. Model architectures

## ResNet18

Pretrained ImageNet ResNet18 is used as the first baseline.

The original classification layer is replaced with:

```text
Linear(
    in_features=512,
    out_features=2
)
```

Baseline strategy:

```text
ImageNet pretrained backbone
        ↓
frozen backbone
        ↓
new trainable classifier
```

Parameter count:

```text
Total parameters:       11,177,538
Trainable parameters:        1,026
```

---

## EfficientNet-B0

EfficientNet-B0 uses the same data, training, validation and evaluation pipeline.

Classifier:

```text
Dropout
   ↓
Linear(
    in_features=1280,
    out_features=2
)
```

Parameter count:

```text
Total parameters:        4,010,110
Trainable parameters:        2,562
```

The backbone is frozen during the baseline comparison.

---

# 9. Shared training pipeline

ResNet18 and EfficientNet-B0 use the same core training implementation.

General training protocol:

```text
Pretrained weights: ImageNet
Backbone:           frozen
Epochs:             5
Batch size:         32
Optimizer:          AdamW
Learning rate:      0.001
Loss:               CrossEntropyLoss
Random seed:        42
```

Training was performed on Kaggle GPU runtime:

```text
GPU: Tesla T4
```

The local environment is primarily used for:

- development;
- smoke tests;
- evaluation;
- inference;
- API;
- pytest.

---

# 10. Reproducibility

Important experiment information is stored in JSON records.

Examples:

```text
reports/experiments/resnet18_baseline_repro_001.json
reports/experiments/resnet18_weighted_001.json
reports/experiments/efficientnet_b0_baseline_001.json
```

Experiment records include information such as:

- Git commit;
- split manifest hash;
- random seed;
- architecture;
- pretrained status;
- parameter count;
- trainable parameter count;
- epochs;
- batch size;
- learning rate;
- optimizer;
- loss;
- training device;
- GPU;
- library versions;
- training history;
- best epoch;
- validation loss;
- validation accuracy;
- checkpoint filename;
- checkpoint SHA-256.

---

# 11. ResNet18 baseline

The reproducible ResNet18 baseline was trained for five epochs.

Best checkpoint:

```text
models/resnet18_baseline_repro_best.pth
```

Best epoch:

```text
5
```

Best validation loss:

```text
0.095778
```

Best validation accuracy:

```text
0.971867
```

The checkpoint itself is intentionally excluded from Git.

---

# 12. Class imbalance experiment

The training data contains substantially more `PNEUMONIA` images than `NORMAL` images.

Training distribution:

```text
NORMAL       1140
PNEUMONIA    3294
```

An additional experiment used class-weighted `CrossEntropyLoss`.

Calculated weights:

```text
NORMAL       1.9447
PNEUMONIA    0.6730
```

Weighted ResNet18 validation results:

```text
Accuracy:     0.957801
Precision:    0.982394
Recall:       0.960413
F1:           0.971279
ROC-AUC:      0.991540

TN:           191
FP:            10
FN:            23
TP:           558
```

Baseline ResNet18:

```text
FP: 11
FN: 11
```

Weighted ResNet18:

```text
FP: 10
FN: 23
```

Class weighting reduced false positives by only one sample while more than doubling false negatives.

Therefore the weighted configuration was rejected for the main model comparison.

It remains stored as a useful negative ablation experiment.

---

# 13. EfficientNet-B0 baseline

EfficientNet-B0 was trained using the same general baseline protocol.

Training configuration:

```text
Epochs:             5
Batch size:         32
Learning rate:      0.001
Optimizer:          AdamW
Loss:               CrossEntropyLoss
Backbone:           frozen
Random seed:        42
```

Training progression:

| Epoch | Train Accuracy | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|
| 1 | 0.9111 | 0.9335 | 0.1763 |
| 2 | 0.9477 | 0.9348 | 0.1489 |
| 3 | 0.9574 | 0.9437 | 0.1318 |
| 4 | 0.9614 | 0.9488 | 0.1222 |
| 5 | 0.9666 | 0.9476 | 0.1185 |

Best checkpoint:

```text
models/efficientnet_b0_baseline_best.pth
```

Best epoch according to validation loss:

```text
5
```

Checkpoint SHA-256:

```text
FE31C3470696EE94E007D13418E394FA4237A9788DE53F77B96608D70CEF3E94
```

---

# 14. Validation model comparison

Both baseline architectures were evaluated on the same validation set.

| Metric | ResNet18 | EfficientNet-B0 |
|---|---:|---:|
| Accuracy | **0.971867** | 0.947570 |
| Precision | **0.981067** | 0.967128 |
| Recall | **0.981067** | 0.962134 |
| F1-score | **0.981067** | 0.964625 |
| ROC-AUC | **0.991728** | 0.989356 |
| True Negative | **190** | 182 |
| False Positive | **11** | 19 |
| False Negative | **11** | 22 |
| True Positive | **570** | 559 |

ResNet18 performed better across the main validation metrics.

Therefore:

```text
ResNet18 baseline
```

was selected as the final model **before the test set was opened**.

---

# 15. Sealed-test protocol

The original test split was kept sealed during:

- model training;
- architecture comparison;
- class-weight experiment;
- checkpoint selection;
- model selection.

Only after the ResNet18 baseline was selected using validation data was the sealed test opened.

The test set was not used to tune:

- model architecture;
- epoch count;
- learning rate;
- loss;
- class weights;
- threshold;
- preprocessing.

This is important because tuning directly on the test set would invalidate its role as an independent final evaluation.

---

# 16. Final sealed-test results

Final test size:

```text
624 images
```

## ResNet18

```text
Accuracy:       0.825321
Precision:      0.787321
Recall:         0.987179
F1-score:       0.875995
ROC-AUC:        0.955775

True Negative:  130
False Positive: 104
False Negative:   5
True Positive:  385
```

## EfficientNet-B0

```text
Accuracy:       0.820513
Precision:      0.784836
Recall:         0.982051
F1-score:       0.872437
ROC-AUC:        0.947787

True Negative:  129
False Positive: 105
False Negative:   7
True Positive:  383
```

Comparison:

| Metric | ResNet18 | EfficientNet-B0 |
|---|---:|---:|
| Accuracy | **0.825321** | 0.820513 |
| Precision | **0.787321** | 0.784836 |
| Recall | **0.987179** | 0.982051 |
| F1-score | **0.875995** | 0.872437 |
| ROC-AUC | **0.955775** | 0.947787 |
| True Negative | **130** | 129 |
| False Positive | **104** | 105 |
| False Negative | **5** | 7 |
| True Positive | **385** | 383 |

The final selected model remains:

```text
ResNet18 baseline
```

---

# 17. Generalization gap

A substantial validation-to-test performance gap was observed.

ResNet18 validation accuracy:

```text
0.971867
```

ResNet18 sealed-test accuracy:

```text
0.825321
```

Validation confusion counts:

```text
TN = 190
FP = 11
FN = 11
TP = 570
```

Test confusion counts:

```text
TN = 130
FP = 104
FN = 5
TP = 385
```

The main degradation came from a sharp increase in false positives.

At the same time, test recall remained very high:

```text
Recall = 0.987179
```

and ROC-AUC remained strong:

```text
ROC-AUC = 0.955775
```

This means that the model still separates the classes reasonably well by score, but its fixed classification behavior produces many false-positive `PNEUMONIA` predictions for `NORMAL` test images.

---

# 18. Cross-model observation

A particularly interesting observation is that both CNN architectures produced almost the same error pattern.

ResNet18:

```text
FP = 104
FN = 5
```

EfficientNet-B0:

```text
FP = 105
FN = 7
```

Because two different CNN architectures show similar degradation, the problem may not be specific to one architecture.

Possible explanations for future investigation include:

- dataset distribution differences;
- acquisition differences;
- image framing;
- contrast;
- crop;
- technical markers;
- dataset-specific visual correlations;
- preprocessing sensitivity.

These remain hypotheses and are **not treated as proven causes**.

---

# 19. Error analysis

The selected ResNet18 model produced:

```text
Total test samples:     624
Correct predictions:    515
False positives:        104
False negatives:          5
Total errors:           109
```

All test predictions are exported to:

```text
reports/errors/resnet18_test_predictions.csv
```

Incorrect predictions are exported to:

```text
reports/errors/resnet18_test_errors.csv
```

Each prediction record contains:

```text
image path
true label
predicted label
P(PNEUMONIA)
prediction confidence
result type
```

Result types include:

```text
CORRECT
FALSE_POSITIVE
FALSE_NEGATIVE
```

---

# 20. High-confidence false positives

Several false-positive predictions were extremely confident.

Examples:

```text
P(PNEUMONIA) = 0.999996
P(PNEUMONIA) = 0.999840
P(PNEUMONIA) = 0.999406
P(PNEUMONIA) = 0.997689
P(PNEUMONIA) = 0.997016
```

This shows that the false-positive problem is not limited to borderline decisions around the default threshold.

Some `NORMAL` images are assigned to the positive class with extremely high confidence.

These cases are particularly useful for future model-behavior analysis.

---

# 21. False negatives

Only five false negatives were observed.

Their `P(PNEUMONIA)` values were:

```text
0.123018
0.146596
0.320627
0.460899
0.472040
```

The first two represent relatively confident incorrect `NORMAL` predictions.

The remaining three are closer to the binary decision boundary and can be considered more borderline predictions.

---

# 22. Visual error review

The most confident false-positive examples and all false-negative examples were visually reviewed.

Observed variation included:

- image framing;
- scale of the chest within the image;
- brightness;
- contrast;
- patient positioning;
- technical markers;
- additional text;
- annotations;
- amount of surrounding anatomy;
- crop differences.

These visual observations are used only as exploratory ML observations.

They are **not medical interpretations** and do not establish the cause of individual errors.

Detailed report:

```text
reports/errors/resnet18_error_analysis.md
```

---

# 23. Evaluation figures

The project generates analytical figures including:

```text
reports/figures/resnet18_test_confusion_matrix.png
reports/figures/resnet18_test_roc_curve.png
```

Final ResNet18 confusion matrix:

```text
[[130, 104],
 [  5, 385]]
```

ROC-AUC:

```text
0.955775
```

Medical source images and contact sheets containing source X-rays are intentionally excluded from Git.

---

# 24. Threshold policy

The classification decision was not retuned after opening the sealed test set.

The existing decision behavior was preserved for final evaluation.

Changing the threshold based on sealed-test results would make the test set part of the tuning process.

A future threshold analysis may be performed using **validation data only**.

Possible validation thresholds for future analysis:

```text
0.3
0.4
0.5
0.6
0.7
```

Such an experiment could study the trade-off between:

```text
precision
recall
false positives
false negatives
```

No threshold from this educational project should be described as clinically optimal.

---

# 25. Standalone inference

Reusable inference logic is implemented in:

```text
src/predict.py
```

The module contains reusable functionality for:

```text
load model
preprocess image
run inference
calculate probabilities
return structured result
```

The same preprocessing implementation used during evaluation is reused during inference.

Example result:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.9999955892562866,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

The `probability` field always represents:

```text
P(PNEUMONIA)
```

Standalone inference was validated using a known sealed-test example.

Expected sealed-test prediction:

```text
Image:
NORMAL2-IM-0256-0001.jpeg

Prediction:
PNEUMONIA

P(PNEUMONIA):
≈ 0.999996
```

The standalone inference pipeline reproduced the same result.

---

# 26. FastAPI service

API implementation:

```text
api/main.py
```

Available endpoints:

```text
GET  /health
POST /predict
```

The model is loaded once during application startup and then remains in memory for inference requests.

The API does **not duplicate ML logic**.

Instead:

```text
HTTP request
   ↓
FastAPI
   ↓
decode image
   ↓
src.predict.predict()
   ↓
ResNet18
   ↓
JSON response
```

---

# 27. Start the API

From the project root:

```powershell
python -m uvicorn api.main:app --reload
```

Example server address:

```text
http://127.0.0.1:8000
```

Swagger/OpenAPI interface:

```text
http://127.0.0.1:8000/docs
```

---

# 28. Health endpoint

Request:

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "service": "chest-xray-pneumonia-api"
}
```

Expected HTTP status:

```text
200 OK
```

---

# 29. Prediction endpoint

Request:

```text
POST /predict
Content-Type: multipart/form-data
```

Parameter:

```text
file
```

Example successful response:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.9999955892562866,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

Expected HTTP status:

```text
200 OK
```

---

# 30. API input validation

The API validates uploaded files.

Non-image file:

```text
HTTP 400
```

Example:

```json
{
  "detail": "Uploaded file must be an image."
}
```

Corrupted file with image MIME type:

```text
HTTP 400
```

Example:

```json
{
  "detail": "Invalid or unsupported image file."
}
```

Internal stack traces are not returned to the API user.

---

# 31. Automated tests

Tests are implemented with `pytest`.

Run all tests:

```powershell
python -m pytest -v
```

Current result:

```text
7 passed
```

The test suite currently covers:

```text
predict() result contract
missing image handling
missing checkpoint handling
GET /health
valid POST /predict
non-image upload rejection
corrupted image rejection
```

Files:

```text
tests/test_predict.py
tests/test_api.py
```

API tests use a lightweight dummy model instead of requiring the real ResNet checkpoint.

This allows API tests to run even when:

```text
models/*.pth
```

are not available.

---

# 32. Development environment

Local development environment:

```text
Python 3.12
```

Observed local package versions:

```text
torch==2.13.0
torchvision==0.28.0
numpy==2.5.2
Pillow==12.3.0
matplotlib==3.11.1
scikit-learn==1.9.0
fastapi==0.141.1
uvicorn==0.52.4
python-multipart==0.0.32
httpx==0.28.1
pytest==9.1.1
```

Local PyTorch was used primarily for CPU development and inference.

Training experiments were executed on Kaggle GPU.

---

# 33. Installation

Clone or open the project repository.

Create a virtual environment.

Example on Windows PowerShell:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check installed dependency consistency:

```powershell
python -m pip check
```

Run tests:

```powershell
python -m pytest -v
```

---

# 34. Dataset location

The raw dataset is not included in Git.

For local work, the expected project structure includes:

```text
data/raw/chest_xray/
├── train/
├── val/
└── test/
```

The split manifest stored in Git references the original dataset files.

The exact raw-dataset setup may also be replaced with a mounted Kaggle dataset during remote training.

---

# 35. Training scripts

Important training-related entry points include:

```text
scripts/train_resnet_baseline.py
scripts/train_resnet_weighted.py
scripts/train_efficientnet_baseline.py
```

Smoke tests and pipeline checks are stored separately under:

```text
scripts/
```

Full GPU training should preferably be performed in a suitable GPU runtime such as Kaggle.

Local execution is primarily intended for:

- smoke tests;
- debugging;
- evaluation;
- inference;
- API;
- unit tests.

---

# 36. Evaluation scripts

Important evaluation and analysis scripts include:

```text
scripts/evaluate_resnet_validation.py
scripts/evaluate_resnet_weighted_validation.py
scripts/evaluate_efficientnet_validation.py
scripts/evaluate_final_test.py
scripts/export_resnet_test_predictions.py
scripts/export_error_examples.py
scripts/create_error_contact_sheets.py
scripts/create_resnet_test_figures.py
```

---

# 37. Repository structure

```text
chest-xray-pneumonia/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── raw/                      # local only, not in Git
│   ├── processed/
│   └── splits/
│       └── split_manifest.csv
│
├── models/                       # checkpoints, not in Git
│
├── reports/
│   ├── errors/
│   │   ├── resnet18_error_analysis.md
│   │   ├── resnet18_test_errors.csv
│   │   └── resnet18_test_predictions.csv
│   │
│   ├── experiments/
│   │   ├── resnet18_baseline_repro_001.json
│   │   ├── resnet18_weighted_001.json
│   │   └── efficientnet_b0_baseline_001.json
│   │
│   ├── figures/
│   │   ├── resnet18_test_confusion_matrix.png
│   │   └── resnet18_test_roc_curve.png
│   │
│   └── metrics/
│
├── scripts/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── models.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   └── test_predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 38. Git policy

The repository intentionally stores:

```text
source code
split manifest
experiment metadata
metrics
CSV prediction reports
error-analysis report
analytical figures
tests
documentation
```

The repository intentionally does **not** store:

```text
raw medical dataset
trained .pth checkpoints
virtual environments
temporary cache files
exported medical error images
contact sheets containing source X-rays
```

This policy keeps the repository reproducible without unnecessarily storing large or sensitive medical-image artifacts.

---

# 39. Important artifacts

Split manifest:

```text
data/splits/split_manifest.csv
```

Experiment records:

```text
reports/experiments/resnet18_baseline_repro_001.json
reports/experiments/resnet18_weighted_001.json
reports/experiments/efficientnet_b0_baseline_001.json
```

Validation metrics:

```text
reports/metrics/resnet18_baseline_validation.json
reports/metrics/resnet18_weighted_validation.json
reports/metrics/efficientnet_b0_baseline_validation.json
```

Final test comparison:

```text
reports/metrics/final_test_comparison.json
```

Error reports:

```text
reports/errors/resnet18_test_predictions.csv
reports/errors/resnet18_test_errors.csv
reports/errors/resnet18_error_analysis.md
```

Figures:

```text
reports/figures/resnet18_test_confusion_matrix.png
reports/figures/resnet18_test_roc_curve.png
```

---

# 40. Known limitations

This project has several important limitations.

## Dataset limitation

The dataset is an open educational/research dataset and is not treated as a clinically validated deployment benchmark.

## Distribution differences

A substantial validation-to-test performance gap was observed.

Possible train/validation versus test distribution differences require deeper investigation.

## Limited architecture comparison

Only two CNN architecture families were used for the mandatory model comparison:

```text
ResNet18
EfficientNet-B0
```

## Frozen-backbone protocol

Baseline experiments trained only the classification head.

Full or partial backbone fine-tuning has not yet been performed.

## Limited optimization

The mandatory project intentionally avoided a large hyperparameter search.

## Threshold

No threshold was optimized using the sealed test set.

## Clinical validity

The model is not clinically validated.

Predictions must not be interpreted as medical diagnoses.

---

# 41. Possible future work

Only after completion of the mandatory project scope.

Possible improvements include:

```text
validation-based threshold analysis
partial backbone fine-tuning
longer training
early stopping
data augmentation
learning-rate experiments
cross-model error overlap analysis
distribution-shift investigation
Grad-CAM
low-confidence / review flag
inference latency comparison
model-size comparison
```

Any future experiment should preserve the distinction between:

```text
training data
validation data
sealed test data
```

and should avoid directly optimizing configuration decisions on the existing test results.

---

# 42. Tuning policy

Future tuning should primarily use the validation set.

For example:

```text
train
   ↓
fit model

validation
   ↓
choose configuration
choose threshold
choose checkpoint
compare experiments

test
   ↓
final independent evaluation
```

The existing sealed-test results should remain recorded as the outcome of the original baseline experiment.

If major future tuning is performed after observing the current test set, a new independent holdout dataset would be preferable for a genuinely fresh final evaluation.

---

# 43. Safety statement

This project is an educational and research prototype.

It must not be used to:

- diagnose pneumonia;
- rule out pneumonia;
- replace a radiologist;
- make treatment decisions;
- make clinical triage decisions;
- provide medical advice.

Model predictions represent experimental machine-learning outputs only.

---

# 44. Current project status

```text
Dataset inspection         DONE
Visual EDA                 DONE
Leakage audit              DONE
Duplicate audit            DONE
Train/validation split     DONE
Preprocessing              DONE
Dataset/DataLoader         DONE
ResNet18 baseline          DONE
Class-weight ablation      DONE
EfficientNet-B0 baseline   DONE
Validation comparison      DONE
Best-model selection       DONE
Sealed-test evaluation     DONE
Error analysis             DONE
Confusion matrix           DONE
ROC curve                  DONE
Standalone inference       DONE
FastAPI /health            DONE
FastAPI /predict           DONE
Input validation           DONE
Automated tests            DONE
requirements.txt           DONE
README                     DONE
Final clean reproduction   IN PROGRESS
Final Git checkpoint/tag   TODO
Demo rehearsal             TODO
```

---

# 45. Final project objective

The project is intended to demonstrate the complete engineering path from an unfamiliar open dataset to a reproducible AI application.

```text
UNDERSTAND
   ↓
BUILD
   ↓
TRAIN
   ↓
MEASURE
   ↓
ANALYZE ERRORS
   ↓
SERVE
   ↓
TEST
   ↓
DOCUMENT
   ↓
EXPLAIN
```

The main outcome is not a single accuracy number.

The main outcome is a pipeline in which the major engineering decisions, limitations, experiments and results are explicit, reproducible and explainable.

---

## Final note

**Educational / research use only. Not intended for clinical diagnosis or treatment decisions.**