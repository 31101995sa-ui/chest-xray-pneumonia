# Chest X-Ray Pneumonia AI Project

Русскоязычная документация учебного end-to-end проекта по компьютерному зрению для бинарной классификации рентгеновских снимков грудной клетки:

- `NORMAL`
- `PNEUMONIA`

Проект разработан как первая практическая AI/ML-задача в инкубаторе по направлению AI Engineering.

Главная цель проекта — не создание клинической диагностической системы, а прохождение полноценного воспроизводимого инженерного ML-цикла: от неизвестного открытого датасета до работающего inference API.

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

> **Только учебное / исследовательское использование.**
> Проект не предназначен для клинической диагностики, выбора лечения или принятия медицинских решений.


---

# Быстрый старт

Этот раздел показывает самый короткий путь от полученного репозитория до работающего проекта.

## 1. Создать виртуальное окружение

Рекомендуемая версия Python:

```text
Python 3.12
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Если из-за ограничения длины путей Windows локальное `.venv` создать не удаётся, можно использовать внешнее окружение:

```powershell
python -m venv C:\venvs\xray
C:\venvs\xray\Scripts\Activate.ps1
```

## 2. Установить зависимости

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверить целостность зависимостей:

```powershell
python -m pip check
```

Ожидаемый результат:

```text
No broken requirements found.
```

## 3. Запустить автоматические тесты

```powershell
python -m pytest -v
```

Текущий ожидаемый результат:

```text
7 passed
```

API-тесты используют лёгкую DummyModel, поэтому для запуска `pytest` настоящий `.pth` checkpoint не требуется.

## 4. Добавить checkpoint обученной модели

Checkpoint намеренно не хранится в Git.

Для запуска реального inference и API необходимо поместить выбранную модель ResNet18 по пути:

```text
models/resnet18_baseline_repro_best.pth
```

Ожидаемая структура:

```text
chest-xray-pneumonia/
└── models/
    └── resnet18_baseline_repro_best.pth
```

Исходный медицинский датасет для обычной работы API не требуется.

## 5. Проверить проект

Запустить:

```powershell
python -m scripts.verify_project
```

Verification script проверяет:

```text
Python environment
PyTorch
split manifest
model checkpoint
model loading
real inference, если доступно контрольное изображение из датасета
```

При полном успешном локальном прогоне вывод заканчивается строкой:

```text
PROJECT VERIFICATION PASSED
```

Если raw dataset на компьютере отсутствует, проверка inference на контрольном изображении может быть пропущена после успешной загрузки модели.

## 6. Запустить API

```powershell
python -m uvicorn api.main:app
```

API будет доступен по адресу:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Проверка сервиса:

```text
GET /health
```

Получение prediction:

```text
POST /predict
```

Изображение передаётся через поле:

```text
file
```

Пример ответа:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.9999955892562866,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

Поле:

```text
probability
```

всегда означает:

```text
P(PNEUMONIA)
```

## Короткая последовательность проверки

Если окружение уже создано, зависимости установлены, а checkpoint находится на месте:

```powershell
python -m pip check
python -m pytest -v
python -m scripts.verify_project
python -m uvicorn api.main:app
```

После этого открыть:

```text
http://127.0.0.1:8000/docs
```

> Проект является учебным / исследовательским прототипом и не предназначен для клинического использования.

---
---

# 1. Постановка задачи

Исходная задача заключалась в создании медицинского AI-проекта на основе открытого Chest X-Ray Pneumonia dataset.

Необходимо было:

1. Использовать открытый датасет рентгеновских снимков грудной клетки.
2. Классифицировать изображения как `NORMAL` или `PNEUMONIA`.
3. Сравнить как минимум две CNN-архитектуры.
4. Оценить модели по следующим метрикам:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
5. Провести анализ ошибочных предсказаний.
6. Сделать простой API или demo, принимающий загруженное изображение и возвращающий prediction.

Реализованные архитектуры:

- ResNet18
- EfficientNet-B0

Дополнительно проведён weighted ResNet18 experiment для исследования влияния class imbalance.

Финально выбранная модель:

```text
ResNet18 baseline
```

---

# 2. Датасет

Используемый датасет:

**Chest X-Ray Images (Pneumonia)**

Открытый датасет, доступный через Kaggle.

Два класса:

```text
NORMAL
PNEUMONIA
```

Исходная структура:

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

Исходный медицинский датасет намеренно **не хранится в Git**.

---

# 3. Исследование датасета

Датасет исследован собственными скриптами проекта, а не только на основании описания источника.

Фактический размер:

```text
Всего изображений: 5856
```

Исходный training split:

```text
NORMAL       1341
PNEUMONIA    3875
Total        5216
```

Исходный validation split:

```text
NORMAL          8
PNEUMONIA       8
Total          16
```

Исходный test split:

```text
NORMAL        234
PNEUMONIA     390
Total         624
```

Дополнительные результаты исследования:

```text
Проверено изображений: 5856
Повреждённых:             0

Image mode:
L                      5573
RGB                     283

Уникальных размеров:   4803

Ширина:
384–2916 px

Высота:
127–2713 px

Ориентация:
Landscape              5790
Portrait                  62
Square                     4
```

Визуальный EDA показал значительную вариативность:

- размеров изображений;
- кадрирования;
- framing;
- положения пациента;
- ориентации;
- контраста;
- acquisition markers;
- технических надписей;
- количества окружающей анатомии;
- других acquisition-related artifacts.

---

# 4. Проектирование validation split

Исходный validation split содержал всего 16 изображений:

```text
8 NORMAL
8 PNEUMONIA
```

Такой объём был признан слишком маленьким для надёжного model selection.

Поэтому новый validation subset был создан из исходного training dataset.

Исходный test set при этом оставался закрытым до завершения выбора модели.

Итоговый рабочий split:

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

Использованный random seed:

```text
42
```

Split manifest:

```text
data/splits/split_manifest.csv
```

Manifest SHA-256:

```text
0216660DC53CC9F196790FB8D342BC82BE9420B74184E93942E2F7B8258D52D1
```

---

# 5. Проверка leakage и дубликатов

Перед обучением были выполнены дополнительные проверки данных.

Результаты SHA-256 audit:

```text
Images hashed:             5856
Unique hashes:             5824
Duplicate groups:            30
Cross-split duplicates:       0
Cross-class duplicates:       0
```

Точных дубликатов между рабочими split-ами не обнаружено.

Также был проведён exploratory perceptual-hash analysis.

64-bit dHash выявил визуальные collisions между split-ами, включая отдельные примеры с разными labels.

Так как dHash оказался недостаточно точным для надёжного удаления медицинских изображений, результаты использовались только как исследовательский сигнал.

Автоматическое удаление изображений на основании perceptual hashes не производилось.

---

# 6. Preprocessing

Для обеих архитектур используется один базовый preprocessing pipeline.

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

Pretrained ImageNet CNN ожидают вход с тремя каналами.

Большинство X-ray изображений в датасете являются grayscale, поэтому перед подачей модели они преобразуются в RGB.

Форма одного изображения:

```text
[3, 224, 224]
```

Форма batch:

```text
[B, 3, 224, 224]
```

Один и тот же preprocessing используется при:

- training;
- validation;
- test evaluation;
- standalone inference;
- FastAPI inference.

Это предотвращает расхождение preprocessing между обучением и эксплуатацией модели.

---

# 7. Dataset и DataLoader

Реализован собственный PyTorch Dataset, читающий samples через split manifest.

Поддерживаются:

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

Validation и test DataLoader:

```text
shuffle = False
```

Random seed:

```text
42
```

Пример batch:

```text
torch.Size([32, 3, 224, 224])
```

Labels:

```text
0 = NORMAL
1 = PNEUMONIA
```

---

# 8. Архитектуры моделей

## ResNet18

В качестве первого baseline используется ResNet18 с pretrained ImageNet weights.

Исходный classification layer заменён на:

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

Количество параметров:

```text
Total parameters:       11,177,538
Trainable parameters:        1,026
```

---

## EfficientNet-B0

EfficientNet-B0 использует тот же data pipeline, training loop и evaluation pipeline.

Classifier:

```text
Dropout
   ↓
Linear(
    in_features=1280,
    out_features=2
)
```

Количество параметров:

```text
Total parameters:        4,010,110
Trainable parameters:        2,562
```

В рамках baseline comparison backbone также заморожен.

---

# 9. Общий training pipeline

ResNet18 и EfficientNet-B0 обучаются через один общий training core.

Основной протокол:

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

Полное обучение выполнялось на Kaggle GPU:

```text
GPU: Tesla T4
```

Локальная среда использовалась преимущественно для:

- разработки;
- smoke tests;
- evaluation;
- inference;
- API;
- pytest.

---

# 10. Воспроизводимость экспериментов

Ключевые параметры training runs сохраняются в JSON experiment records.

Основные файлы:

```text
reports/experiments/resnet18_baseline_repro_001.json
reports/experiments/resnet18_weighted_001.json
reports/experiments/efficientnet_b0_baseline_001.json
```

Experiment records содержат:

- Git commit;
- hash split manifest;
- random seed;
- architecture;
- pretrained status;
- число параметров;
- число trainable parameters;
- epochs;
- batch size;
- learning rate;
- optimizer;
- loss;
- device;
- GPU;
- версии библиотек;
- training history;
- best epoch;
- validation loss;
- validation accuracy;
- checkpoint filename;
- checkpoint SHA-256.

---

# 11. ResNet18 baseline

Reproducible ResNet18 baseline обучен в течение пяти эпох.

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

Checkpoint намеренно не хранится в Git.

---

# 12. Эксперимент с class imbalance

Training split имеет заметный дисбаланс:

```text
NORMAL       1140
PNEUMONIA    3294
```

Для проверки влияния imbalance был проведён дополнительный эксперимент с weighted `CrossEntropyLoss`.

Рассчитанные weights:

```text
NORMAL       1.9447
PNEUMONIA    0.6730
```

Weighted ResNet18 validation:

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

Baseline:

```text
FP = 11
FN = 11
```

Weighted model:

```text
FP = 10
FN = 23
```

Class weighting уменьшил FP всего на один случай, но увеличил FN с 11 до 23.

Поэтому weighted configuration была отклонена для основного comparison.

Эксперимент сохранён как отрицательный ablation experiment.

---

# 13. EfficientNet-B0 baseline

EfficientNet-B0 обучена в том же общем режиме.

Конфигурация:

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

Best epoch по validation loss:

```text
5
```

Checkpoint SHA-256:

```text
FE31C3470696EE94E007D13418E394FA4237A9788DE53F77B96608D70CEF3E94
```

---

# 14. Сравнение моделей на validation

Обе baseline architectures оценены на одном validation dataset.

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

ResNet18 показала лучшие результаты по основным validation metrics.

Поэтому:

```text
ResNet18 baseline
```

была выбрана финальной моделью **до открытия test set**.

---

# 15. Sealed-test protocol

Исходный test split оставался закрытым во время:

- training;
- architecture comparison;
- class-weight experiment;
- checkpoint selection;
- final model selection.

Test был открыт только после выбора ResNet18 по validation.

Test не использовался для настройки:

- архитектуры;
- количества epochs;
- learning rate;
- loss;
- class weights;
- threshold;
- preprocessing.

Это необходимо для сохранения независимости финальной оценки.

---

# 16. Итоговые результаты sealed test

Размер финального test:

```text
624 изображения
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

Сравнение:

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

Финально выбранная модель остаётся:

```text
ResNet18 baseline
```

---

# 17. Generalization gap

Обнаружен значительный разрыв между validation и sealed test.

ResNet18 validation accuracy:

```text
0.971867
```

ResNet18 sealed-test accuracy:

```text
0.825321
```

Validation:

```text
TN = 190
FP = 11
FN = 11
TP = 570
```

Test:

```text
TN = 130
FP = 104
FN = 5
TP = 385
```

Главное ухудшение связано с резким ростом false positives.

При этом test recall остался очень высоким:

```text
Recall = 0.987179
```

ROC-AUC:

```text
ROC-AUC = 0.955775
```

То есть способность модели ранжировать классы сохраняется достаточно хорошо, но fixed-threshold classification приводит к большому количеству `NORMAL → PNEUMONIA` ошибок.

---

# 18. Cross-model observation

Особенно интересен тот факт, что обе CNN показали практически одинаковый тип degradation.

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

Так как две разные архитектуры показывают очень похожее поведение, проблема может быть связана не только с конкретной CNN.

Возможные направления будущего исследования:

- различия distributions;
- особенности acquisition;
- framing;
- contrast;
- crop;
- technical markers;
- dataset-specific correlations;
- preprocessing sensitivity.

Это исследовательские гипотезы, а не доказанные причины.

---

# 19. Error analysis

Финальная ResNet18 дала:

```text
Total test samples:     624
Correct predictions:    515
False positives:        104
False negatives:          5
Total errors:           109
```

Все predictions экспортируются в:

```text
reports/errors/resnet18_test_predictions.csv
```

Ошибочные predictions:

```text
reports/errors/resnet18_test_errors.csv
```

Каждая запись содержит:

```text
image path
true label
predicted label
P(PNEUMONIA)
prediction confidence
result type
```

Типы:

```text
CORRECT
FALSE_POSITIVE
FALSE_NEGATIVE
```

---

# 20. High-confidence false positives

Некоторые false positive были чрезвычайно уверенными.

Примеры:

```text
P(PNEUMONIA) = 0.999996
P(PNEUMONIA) = 0.999840
P(PNEUMONIA) = 0.999406
P(PNEUMONIA) = 0.997689
P(PNEUMONIA) = 0.997016
```

Следовательно, проблема false positives не ограничивается только borderline predictions около threshold `0.5`.

Некоторые изображения класса `NORMAL` модель относит к `PNEUMONIA` с практически полной уверенностью.

Такие случаи особенно полезны для последующего анализа поведения CNN.

---

# 21. False negatives

Обнаружено всего пять false negatives.

Их значения `P(PNEUMONIA)`:

```text
0.123018
0.146596
0.320627
0.460899
0.472040
```

Первые два случая являются относительно уверенными неправильными `NORMAL` predictions.

Оставшиеся три находятся ближе к decision boundary.

---

# 22. Визуальный анализ ошибок

Были просмотрены наиболее уверенные false positive и все false negative примеры.

Замечена вариативность:

- framing;
- масштаба изображения;
- brightness;
- contrast;
- положения пациента;
- технических markers;
- дополнительных надписей;
- crop;
- количества окружающей анатомии.

Эти наблюдения используются только для exploratory ML analysis.

Они не являются медицинской интерпретацией и не доказывают причину конкретной ошибки.

Подробный отчёт:

```text
reports/errors/resnet18_error_analysis.md
```

---

# 23. Evaluation figures

Проект формирует аналитические изображения:

```text
reports/figures/resnet18_test_confusion_matrix.png
reports/figures/resnet18_test_roc_curve.png
```

Confusion matrix ResNet18:

```text
[[130, 104],
 [  5, 385]]
```

ROC-AUC:

```text
0.955775
```

Исходные медицинские изображения и contact sheets с X-ray намеренно не хранятся в Git.

---

# 24. Threshold policy

После открытия sealed test threshold не изменялся.

Это принципиальное решение.

Если подобрать threshold по результатам test, test dataset станет частью tuning process и перестанет быть независимой финальной оценкой.

В будущем threshold analysis может выполняться только на validation data.

Например:

```text
0.3
0.4
0.5
0.6
0.7
```

Можно исследовать trade-off между:

```text
precision
recall
false positives
false negatives
```

Полученный threshold нельзя называть клинически оптимальным.

---

# 25. Standalone inference

Переиспользуемая inference logic находится в:

```text
src/predict.py
```

Модуль отвечает за:

```text
load model
preprocess image
run inference
calculate probabilities
return structured result
```

Inference использует тот же preprocessing, что и evaluation.

Пример результата:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.9999955892562866,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

Поле `probability` всегда означает:

```text
P(PNEUMONIA)
```

Standalone inference проверен на известном sealed-test примере:

```text
NORMAL2-IM-0256-0001.jpeg
```

Ожидаемый prediction:

```text
PNEUMONIA
```

Вероятность:

```text
≈ 0.999996
```

Standalone inference воспроизвёл исходный sealed-test результат.

---

# 26. FastAPI service

API реализован в:

```text
api/main.py
```

Endpoints:

```text
GET  /health
POST /predict
```

Модель загружается один раз при старте приложения и затем остаётся в памяти.

API не содержит собственной копии ML preprocessing/inference logic.

Архитектура:

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

# 27. Запуск API

Из корня проекта:

```powershell
python -m uvicorn api.main:app --reload
```

Локальный адрес:

```text
http://127.0.0.1:8000
```

Swagger/OpenAPI:

```text
http://127.0.0.1:8000/docs
```

---

# 28. Health endpoint

Запрос:

```text
GET /health
```

Ответ:

```json
{
  "status": "ok",
  "service": "chest-xray-pneumonia-api"
}
```

HTTP:

```text
200 OK
```

---

# 29. Prediction endpoint

Запрос:

```text
POST /predict
Content-Type: multipart/form-data
```

Параметр:

```text
file
```

Пример успешного ответа:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.9999955892562866,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

HTTP:

```text
200 OK
```

---

# 30. Проверка входных данных API

API проверяет загружаемые файлы.

Для обычного текстового файла:

```text
HTTP 400
```

Ответ:

```json
{
  "detail": "Uploaded file must be an image."
}
```

Для повреждённого файла с image MIME type:

```text
HTTP 400
```

Ответ:

```json
{
  "detail": "Invalid or unsupported image file."
}
```

Внутренний stack trace пользователю не возвращается.

---

# 31. Automated tests

Для тестирования используется `pytest`.

Запуск:

```powershell
python -m pytest -v
```

Текущий результат:

```text
7 passed
```

Проверяется:

```text
predict() result contract
missing image handling
missing checkpoint handling
GET /health
valid POST /predict
non-image upload rejection
corrupted image rejection
```

Основные тестовые файлы:

```text
tests/test_predict.py
tests/test_api.py
```

В API tests вместо настоящей тяжёлой ResNet используется lightweight DummyModel.

Это позволяет запускать тесты даже без:

```text
models/*.pth
```

---

# 32. Development environment

Локальная среда:

```text
Python 3.12
```

Использованные версии:

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

Локально PyTorch использовался преимущественно для CPU development и inference.

Training выполнялся на Kaggle GPU runtime.

---

# 33. Установка

Создать виртуальное окружение:

```powershell
python -m venv .venv
```

Активировать на Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Обновить pip:

```powershell
python -m pip install --upgrade pip
```

Установить зависимости:

```powershell
python -m pip install -r requirements.txt
```

Проверить зависимости:

```powershell
python -m pip check
```

Запустить тесты:

```powershell
python -m pytest -v
```

---

# 34. Расположение датасета

Raw dataset не включён в Git.

Для локальной разработки ожидается:

```text
data/raw/chest_xray/
├── train/
├── val/
└── test/
```

Split manifest хранится в Git и ссылается на исходные dataset files.

При remote training dataset может использоваться непосредственно через Kaggle mount.

---

# 35. Training scripts

Основные training entry points:

```text
scripts/train_resnet_baseline.py
scripts/train_resnet_weighted.py
scripts/train_efficientnet_baseline.py
```

Smoke tests и дополнительные pipeline checks также находятся в:

```text
scripts/
```

Полное обучение предпочтительно выполнять на GPU environment.

Локально в первую очередь выполняются:

- smoke tests;
- debugging;
- evaluation;
- inference;
- API;
- unit tests.

---

# 36. Evaluation scripts

Основные evaluation и analysis scripts:

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

# 37. Структура репозитория

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
├── README.md
└── README_RU.md
```

---

# 38. Git policy

В Git намеренно хранятся:

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

В Git намеренно **не хранятся**:

```text
raw medical dataset
trained .pth checkpoints
virtual environments
temporary cache
exported medical error images
contact sheets containing source X-rays
```

Такой подход позволяет сохранить воспроизводимость проекта без размещения тяжёлых или медицинских исходных данных.

---

# 39. Важные артефакты

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

# 40. Известные ограничения

## Ограничения датасета

Используется открытый учебный / исследовательский dataset, который не рассматривается как клинически валидированный benchmark.

## Distribution differences

Обнаружен значительный validation-to-test gap.

Возможные различия train/validation и original test требуют отдельного исследования.

## Ограниченный architecture comparison

В обязательном сравнении использованы только:

```text
ResNet18
EfficientNet-B0
```

## Frozen backbone

Baseline experiments обучали только classification head.

Полный или частичный backbone fine-tuning пока не проводился.

## Ограниченный hyperparameter search

В рамках обязательной задачи намеренно не проводился большой hyperparameter search.

## Threshold

Threshold не оптимизировался на sealed test.

## Clinical validity

Модель не проходила клиническую валидацию.

Prediction нельзя интерпретировать как медицинский диагноз.

---

# 41. Возможные дальнейшие улучшения

После полного завершения обязательного проекта можно исследовать:

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

Будущие эксперименты должны сохранять разделение:

```text
training data
validation data
sealed test data
```

и не должны напрямую оптимизироваться по уже увиденным test results.

---

# 42. Tuning policy

Будущий tuning должен выполняться преимущественно через validation set.

Правильная схема:

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

Текущие sealed-test results должны остаться зафиксированными как исходный baseline result.

Так как текущий test set уже был открыт и проанализирован, после значительного будущего tuning для действительно независимой финальной оценки желательно использовать новый holdout dataset.

---

# 43. Safety statement

Проект является учебным и исследовательским прототипом.

Он не должен использоваться для:

- диагностики pneumonia;
- исключения pneumonia;
- замены радиолога;
- выбора лечения;
- клинической сортировки пациентов;
- медицинских рекомендаций.

Predictions являются только экспериментальными результатами machine learning модели.

---

# 44. Текущий статус проекта

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
README_RU                  DONE
Final clean reproduction   IN PROGRESS
Final Git checkpoint/tag   TODO
Demo rehearsal             TODO
```

---

# 45. Итоговая цель проекта

Проект должен продемонстрировать полный инженерный путь от незнакомого открытого dataset до воспроизводимого AI-приложения.

```text
ПОНЯТЬ
   ↓
СОБРАТЬ
   ↓
ОБУЧИТЬ
   ↓
ИЗМЕРИТЬ
   ↓
РАЗОБРАТЬ ОШИБКИ
   ↓
СОЗДАТЬ INFERENCE
   ↓
ПОДНЯТЬ API
   ↓
ПРОТЕСТИРОВАТЬ
   ↓
ЗАДОКУМЕНТИРОВАТЬ
   ↓
ОБЪЯСНИТЬ
```

Главный результат проекта — не одно высокое значение accuracy.

Главный результат — pipeline, в котором ключевые инженерные решения, эксперименты, ошибки, ограничения и результаты:

- понятны;
- воспроизводимы;
- зафиксированы;
- проверяемы;
- могут быть объяснены другому инженеру или ментору.

---

## Финальное предупреждение

**Только для учебного и исследовательского использования. Не предназначено для клинической диагностики или выбора лечения.**