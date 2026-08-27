# Chest X-Ray Pneumonia Classification

Educational AI/ML project for binary classification of chest X-ray images.

## Goal

Build a model that classifies chest X-ray images into two classes:

- NORMAL
- PNEUMONIA

The project will compare two pretrained convolutional neural network architectures:

- ResNet18
- EfficientNet-B0

## Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Dataset

An open-source Chest X-Ray Pneumonia dataset will be used.

The dataset itself is not stored in this Git repository.

### Initial Dataset Inspection

The local dataset contains 5,856 usable image files:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

Key observations:

- The training split is imbalanced:
  - NORMAL: 25.7%
  - PNEUMONIA: 74.3%
- The original validation split contains only 16 images and is too small for reliable model selection.
- 5,573 images are stored in grayscale (`L`) mode and 283 in `RGB` mode.
- The dataset contains 4,803 unique image sizes.
- Image width ranges from 384 to 2,916 pixels.
- Image height ranges from 127 to 2,713 pixels.
- 5,790 images are landscape, 62 portrait, and 4 square.
- No broken images were detected during the initial Pillow verification.
- Visual inspection showed variation in image scale, cropping, orientation, and the presence of markers or external acquisition artifacts.

These differences will need to be handled consistently during preprocessing.

The original test split will be preserved for final evaluation. Because the provided validation split is extremely small, a new validation subset will later be created from the training data.

### Dataset Limitations

This is an educational/research dataset and may contain acquisition-specific artifacts, markers, cropping differences, or other non-clinical visual cues.

The model may potentially learn such shortcuts instead of medically meaningful image features. This risk will be considered during error analysis and model interpretation.

### Train / Validation / Test Strategy

The original training split was used to create a new reproducible
group-aware validation subset.

Final experimental split:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,140 | 3,294 | 4,434 |
| Validation | 201 | 581 | 782 |
| Test | 234 | 390 | 624 |
| Legacy validation | 8 | 8 | 16 |

The new validation set contains approximately 15% of the original
training images.

Important safeguards:

- Random seed is fixed at `42`.
- PNEUMONIA images are grouped using the `personN` identifier from filenames.
- NORMAL images are grouped using their `IM` / `NORMAL2-IM` identifier.
- Related images from the same inferred group are not split between train and validation.
- Exact SHA-256 overlap between the new train and validation sets is zero.
- The original test set is preserved unchanged for final evaluation.
- The original 16-image validation split is retained as `legacy_val`
  and is not used for model selection.

The exact split is stored in:

`data/splits/split_manifest.csv`

Dataset auditing also found exact duplicate images inside individual
source splits, but no exact duplicates across train, validation, and test,
and no exact duplicate with conflicting NORMAL/PNEUMONIA labels.

A 64-bit perceptual dHash experiment produced many collisions between
different chest X-ray images, including images from different classes.
Therefore dHash is retained only as an exploratory audit experiment and
is not used as an automatic deduplication criterion.

## Deadline

September 3, 2026.

Internal target: September 2, 2026.

## Disclaimer

Educational and research prototype only.

This project is not intended for clinical diagnosis or treatment decisions.

27.08.2026
__________________________________________________________________________________________________



