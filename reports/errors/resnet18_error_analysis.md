# ResNet18 Error Analysis

## Scope

This analysis is based on the final selected `ResNet18 baseline`
checkpoint evaluated once on the sealed test split.

The model configuration was not changed after the test split was opened.

This is an educational/research analysis only.
No clinical conclusions are made from the X-ray images.

---

## Final test summary

Test samples: 624

- True Negatives: 130
- False Positives: 104
- False Negatives: 5
- True Positives: 385

Metrics:

- Accuracy: 0.825321
- Precision: 0.787321
- Recall: 0.987179
- F1-score: 0.875995
- ROC-AUC: 0.955775

The largest source of error is false positives.

The model correctly identified most PNEUMONIA samples,
but a substantial number of NORMAL samples were predicted
as PNEUMONIA.

---

## False positives

There were 104 false positives:

- true label: NORMAL
- predicted label: PNEUMONIA

Several false positives were extremely high-confidence errors.

Examples:

- `NORMAL2-IM-0256-0001.jpeg`
  - P(PNEUMONIA) = 0.999996

- `IM-0022-0001.jpeg`
  - P(PNEUMONIA) = 0.999840

- `NORMAL2-IM-0232-0001.jpeg`
  - P(PNEUMONIA) = 0.999406

Therefore, the false-positive problem is not limited to samples
close to the default classification threshold of 0.5.

Some NORMAL images are confidently assigned to the positive class.

---

## False negatives

There were only 5 false negatives:

- true label: PNEUMONIA
- predicted label: NORMAL

Their P(PNEUMONIA) values were:

- 0.123018
- 0.146596
- 0.320627
- 0.460899
- 0.472040

The first two false negatives are relatively confident incorrect
NORMAL predictions.

The remaining three are closer to the 0.5 classification threshold
and can be considered more borderline model decisions.

---

## Visual observations

The reviewed error examples show substantial variation in:

- image framing;
- scale of the chest within the image;
- brightness and contrast;
- patient positioning;
- visible acquisition markers;
- additional text or technical annotations;
- amount of surrounding anatomy included in the crop.

These observations do not prove the cause of the errors.

However, they support the hypothesis that differences in image
acquisition or dataset distribution may contribute to the observed
generalization gap between validation and test performance.

---

## Validation-to-test generalization gap

ResNet18 validation performance:

- Accuracy: 0.971867
- F1-score: 0.981067
- ROC-AUC: 0.991728
- FP: 11
- FN: 11

ResNet18 sealed-test performance:

- Accuracy: 0.825321
- F1-score: 0.875995
- ROC-AUC: 0.955775
- FP: 104
- FN: 5

The main degradation on the sealed test set comes from the sharp
increase in false positives.

The model retains very high recall and a strong ROC-AUC,
but its fixed-threshold classification behavior is substantially
worse for NORMAL images on the test distribution.

---

## Cross-model observation

EfficientNet-B0 showed a very similar test error pattern:

- ResNet18:
  - FP = 104
  - FN = 5

- EfficientNet-B0:
  - FP = 105
  - FN = 7

Because two different CNN architectures show similar behavior,
the test degradation may not be specific to ResNet18 alone.

This is consistent with, but does not prove, a possible
dataset-level or distribution-shift effect.

---

## Threshold policy

The classification threshold remains unchanged at 0.5.

The threshold was not tuned after observing the sealed test set.

Changing it based on test results would make the final test
evaluation part of the tuning process and would invalidate the
original test protocol.

Threshold analysis can be performed later using validation data
as an optional experiment.

---

## Artifacts

Prediction data:

- `reports/errors/resnet18_test_predictions.csv`
- `reports/errors/resnet18_test_errors.csv`

Error examples:

- `reports/errors/false_positives/`
- `reports/errors/false_negatives/`

Figures:

- `reports/figures/resnet18_false_positives_contact_sheet.png`
- `reports/figures/resnet18_false_negatives_contact_sheet.png`
- `reports/figures/resnet18_test_confusion_matrix.png`
- `reports/figures/resnet18_test_roc_curve.png`

---

## Conclusion

The selected ResNet18 model generalizes well enough to preserve
high recall and ROC-AUC, but it produces a large number of false
positive predictions on the sealed test set.

The most important observed limitation is therefore not missed
PNEUMONIA samples, but the poor separation of a substantial subset
of NORMAL test images at the fixed threshold.

The error analysis suggests that dataset and acquisition differences
should be investigated in future work.

For the current project, the final model remains unchanged and the
next engineering step is to build a reusable inference function
around the selected checkpoint.