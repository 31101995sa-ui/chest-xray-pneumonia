$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$VenvRoot = Join-Path `
    $env:USERPROFILE `
    ".venvs\chest-xray-pneumonia"

$PythonExe = Join-Path `
    $VenvRoot `
    "Scripts\python.exe"

Write-Host ""
Write-Host "Chest X-Ray Pneumonia AI Project"
Write-Host "Automatic setup and verification"
Write-Host "--------------------------------------------------"

Write-Host "[1/6] Checking Python..."

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue

if (-not $PythonCommand) {
    throw "Python was not found in PATH. Python 3.12 is recommended."
}

python --version

Write-Host ""
Write-Host "[2/6] Preparing virtual environment..."
Write-Host "Environment:"
Write-Host $VenvRoot

if (-not (Test-Path $PythonExe)) {
    New-Item `
        -ItemType Directory `
        -Path (Split-Path $VenvRoot) `
        -Force | Out-Null

    python -m venv $VenvRoot
}
else {
    Write-Host "[OK] Existing virtual environment found"
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment creation failed."
}

Write-Host ""
Write-Host "[3/6] Installing dependencies..."

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host ""
Write-Host "[4/6] Checking dependency consistency..."

& $PythonExe -m pip check

if ($LASTEXITCODE -ne 0) {
    throw "pip check failed."
}

Write-Host ""
Write-Host "[5/6] Running automated tests..."

Push-Location $ProjectRoot

try {
    & $PythonExe -m pytest -v

    if ($LASTEXITCODE -ne 0) {
        throw "Automated tests failed."
    }

    Write-Host ""
    Write-Host "[6/6] Verifying real project pipeline..."

    & $PythonExe -m scripts.verify_project

    if ($LASTEXITCODE -ne 0) {
        throw "Project verification failed."
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "SETUP AND VERIFICATION COMPLETED"
Write-Host "--------------------------------------------------"
Write-Host ""
Write-Host "To start the API run:"
Write-Host ""
Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1"
Write-Host ""
Write-Host "Swagger UI:"
Write-Host "http://127.0.0.1:8000/docs"
Write-Host ""