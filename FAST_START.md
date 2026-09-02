# FAST START

## Chest X-Ray Pneumonia AI Project

The shortest path from an unpacked submission package to a working API.

> Educational / research prototype only.  
> Not intended for clinical diagnosis or treatment decisions.

---

## Option 1 — Automatic setup

### Step 1. Install and verify the project

Run from the root of the unpacked project:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_project.ps1
```

The script automatically performs:

```text
check Python
→ create virtual environment
→ upgrade pip
→ install requirements.txt
→ run pip check
→ run pytest
→ verify model checkpoint
→ load the real ResNet18 model
→ perform real inference
```

A successful setup should include:

```text
7 passed
PROJECT VERIFICATION PASSED
SETUP AND VERIFICATION COMPLETED
```

The virtual environment is created outside the project directory:

```text
%USERPROFILE%\.venvs\chest-xray-pneumonia
```

This helps avoid Windows path-length problems.

---

### Step 2. Start the API

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
```

After startup you should see:

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Press:

```text
CTRL+C
```

in the server terminal to stop the API.

---

# API verification

## Health check

In Swagger, open:

```text
GET /health
```

Click:

```text
Try it out
```

and then:

```text
Execute
```

Expected HTTP status:

```text
200 OK
```

Example response:

```json
{
  "status": "ok",
  "service": "chest-xray-pneumonia-api"
}
```

---

## Prediction

In Swagger, open:

```text
POST /predict
```

Click:

```text
Try it out
```

Upload an image using the:

```text
file
```

field and execute the request.

Example response:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.95,
  "model": "resnet18_baseline",
  "disclaimer": "Educational/research use only. Not intended for clinical diagnosis."
}
```

The field:

```text
probability
```

always represents:

```text
P(PNEUMONIA)
```

---

# Option 2 — Manual setup

If the automatic setup script should not be used, the project can also be prepared manually.

Create a virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Verify the project:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_project
```

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# What is included in the submission package

The selected trained checkpoint is included at:

```text
models/resnet18_baseline_repro_best.pth
```

The raw Chest X-Ray dataset is intentionally **not included**.

The raw dataset is not required for:

```text
model loading
standalone inference
FastAPI startup
POST /predict
```

During project verification, a real forward pass is always performed using a synthetic image.

If the raw dataset is unavailable, the optional known-reference check will show:

```text
[SKIPPED] Known reference inference: raw dataset is not available
```

This is expected behavior for the submission package.

A successful verification should still end with:

```text
PROJECT VERIFICATION PASSED
```

---

# Minimum command sequence

After unpacking the project, only two commands are required:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_project.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The first command installs and verifies the project.

The second command starts the API.

---

# Important files

Full English documentation:

```text
README.md
```

Full Russian documentation:

```text
README_RU.md
```

Russian quick-start guide:

```text
FAST_START_RU.md
```

Dependency list:

```text
requirements.txt
```

Model checkpoint:

```text
models/resnet18_baseline_repro_best.pth
```

API:

```text
api/main.py
```

Project verification:

```text
scripts/verify_project.py
```

Automatic setup:

```text
scripts/setup_project.ps1
```

API launcher:

```text
scripts/run_api.ps1
```

---

**Educational / research use only. Not intended for clinical diagnosis or treatment decisions.**