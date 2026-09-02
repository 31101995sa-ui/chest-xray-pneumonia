# FAST START

## Chest X-Ray Pneumonia AI Project

Самый короткий путь от распакованного submission package до работающего API.

> Educational / research prototype only.  
> Not intended for clinical diagnosis or treatment decisions.

---

## Вариант 1 — автоматический запуск

### Шаг 1. Установка и проверка проекта

Из корня распакованного проекта:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_project.ps1
```

Скрипт автоматически:

```text
создаёт Python virtual environment
→ обновляет pip
→ устанавливает requirements.txt
→ запускает pip check
→ запускает pytest
→ проверяет checkpoint
→ загружает настоящую ResNet18
→ выполняет test inference
```

При успешной проверке должны появиться:

```text
7 passed
PROJECT VERIFICATION PASSED
```

---

### Шаг 2. Запуск API

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
```

После появления:

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

открыть:

```text
http://127.0.0.1:8000/docs
```

---

## Проверка API

### Health check

В Swagger:

```text
GET /health
```

Ожидаемый статус:

```text
200 OK
```

Пример ответа:

```json
{
  "status": "ok",
  "service": "chest-xray-pneumonia-api"
}
```

---

### Prediction

В Swagger:

```text
POST /predict
```

Нажать:

```text
Try it out
```

Загрузить изображение через поле:

```text
file
```

и выполнить запрос.

Пример ответа:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.95,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

Поле:

```text
probability
```

означает:

```text
P(PNEUMONIA)
```

---

# Вариант 2 — ручная проверка

Если автоматический setup script использовать не требуется:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m scripts.verify_project
.\.venv\Scripts\python.exe -m uvicorn api.main:app
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# Что должно находиться в submission package

Главный checkpoint:

```text
models/resnet18_baseline_repro_best.pth
```

Raw medical dataset в submission package намеренно отсутствует.

Он не требуется для обычного:

```text
model loading
inference
FastAPI
/predict
```

Если raw dataset отсутствует, `verify_project.py` выполняет настоящий inference на synthetic image, а reference dataset check показывает:

```text
[SKIPPED] Known reference inference: raw dataset is not available
```

Это нормальное состояние submission package.

---

# Минимальная последовательность

После распаковки проекта:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_project.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
```

После этого:

```text
http://127.0.0.1:8000/docs
```

---

**Educational / research use only. Not intended for clinical diagnosis or treatment decisions.**