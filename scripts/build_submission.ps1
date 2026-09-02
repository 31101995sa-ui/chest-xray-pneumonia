$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$DistRoot = Join-Path $ProjectRoot "dist"
$StageDir = Join-Path $DistRoot "chest-xray-pneumonia-submission"
$ZipPath = Join-Path $DistRoot "chest-xray-pneumonia-submission.zip"

$CheckpointName = "resnet18_baseline_repro_best.pth"
$CheckpointSource = Join-Path $ProjectRoot "models\$CheckpointName"

Write-Host ""
Write-Host "Chest X-Ray Pneumonia AI Project"
Write-Host "Building submission package..."
Write-Host "--------------------------------------------------"

if (-not (Test-Path $CheckpointSource)) {
    throw "Required checkpoint not found: $CheckpointSource"
}

if (Test-Path $StageDir) {
    Remove-Item $StageDir -Recurse -Force
}

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null

Write-Host "[1/5] Copying project files..."

$RobocopyArgs = @(
    $ProjectRoot
    $StageDir
    "/E"
    "/R:1"
    "/W:1"
    "/NFL"
    "/NDL"
    "/NJH"
    "/NJS"
    "/NP"
    "/XD"
    ".git"
    ".venv"
    "dist"
    "__pycache__"
    ".pytest_cache"
    "raw"
    "processed"
    "false_positives"
    "false_negatives"
    "/XF"
    "*.pyc"
    "*.pth"
    "*.zip"
    "*contact_sheet*.png"
)

& robocopy @RobocopyArgs

if ($LASTEXITCODE -ge 8) {
    throw "Robocopy failed with exit code $LASTEXITCODE"
}

Write-Host "[2/5] Adding selected model checkpoint..."

$SubmissionModelsDir = Join-Path $StageDir "models"

New-Item `
    -ItemType Directory `
    -Path $SubmissionModelsDir `
    -Force | Out-Null

$CheckpointDestination = Join-Path `
    $SubmissionModelsDir `
    $CheckpointName

Copy-Item `
    $CheckpointSource `
    $CheckpointDestination `
    -Force

Write-Host "[3/5] Creating package metadata..."

$CheckpointHash = (
    Get-FileHash `
        -Path $CheckpointSource `
        -Algorithm SHA256
).Hash

$GitCommit = (
    git -C $ProjectRoot rev-parse HEAD |
    Out-String
).Trim()

$PackageInfoPath = Join-Path $StageDir "PACKAGE_INFO.txt"

@"
Chest X-Ray Pneumonia AI Project
Submission Package

Git commit:
$GitCommit

Included model checkpoint:
$CheckpointName

Checkpoint SHA-256:
$CheckpointHash

Raw medical dataset:
NOT INCLUDED

Exported source X-ray error examples:
NOT INCLUDED

Purpose:
Educational / research project only.
Not intended for clinical diagnosis or treatment decisions.

Quick verification:
1. Install requirements.txt
2. Run: python -m pytest -v
3. Run: python -m scripts.verify_project
4. Run: python -m uvicorn api.main:app
5. Open: http://127.0.0.1:8000/docs
"@ | Set-Content `
    -Path $PackageInfoPath `
    -Encoding UTF8

Write-Host "[4/5] Checking package contents..."

$ForbiddenPaths = @(
    (Join-Path $StageDir ".git"),
    (Join-Path $StageDir "data\raw"),
    (Join-Path $StageDir "data\processed"),
    (Join-Path $StageDir ".venv"),
    (Join-Path $StageDir ".pytest_cache"),
    (Join-Path $StageDir "reports\errors\false_positives"),
    (Join-Path $StageDir "reports\errors\false_negatives")
)

foreach ($Path in $ForbiddenPaths) {
    if (Test-Path $Path) {
        throw "Forbidden path found in submission package: $Path"
    }
}

$PthFiles = @(
    Get-ChildItem `
        -Path $StageDir `
        -Recurse `
        -Filter "*.pth" `
        -File
)

if ($PthFiles.Count -ne 1) {
    throw (
        "Expected exactly one .pth checkpoint in package, found: " +
        $PthFiles.Count
    )
}

if ($PthFiles[0].Name -ne $CheckpointName) {
    throw (
        "Unexpected checkpoint included: " +
        $PthFiles[0].Name
    )
}

$ContactSheets = @(
    Get-ChildItem `
        -Path $StageDir `
        -Recurse `
        -Filter "*contact_sheet*.png" `
        -File
)

if ($ContactSheets.Count -ne 0) {
    throw "Contact sheets were unexpectedly included."
}

Write-Host "[OK] Raw dataset excluded"
Write-Host "[OK] Git metadata excluded"
Write-Host "[OK] Cache files excluded"
Write-Host "[OK] Medical error images excluded"
Write-Host "[OK] Exactly one model checkpoint included"

Write-Host "[5/5] Creating ZIP archive..."

Compress-Archive `
    -Path $StageDir `
    -DestinationPath $ZipPath `
    -CompressionLevel Optimal

$ZipHash = (
    Get-FileHash `
        -Path $ZipPath `
        -Algorithm SHA256
).Hash

$ZipSizeMB = [math]::Round(
    (Get-Item $ZipPath).Length / 1MB,
    2
)

Write-Host ""
Write-Host "SUBMISSION PACKAGE CREATED"
Write-Host "--------------------------------------------------"
Write-Host "Archive:"
Write-Host $ZipPath
Write-Host ""
Write-Host "Archive size:"
Write-Host "$ZipSizeMB MB"
Write-Host ""
Write-Host "Archive SHA-256:"
Write-Host $ZipHash
Write-Host ""