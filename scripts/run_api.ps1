$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$VenvRoot = Join-Path `
    $env:USERPROFILE `
    ".venvs\chest-xray-pneumonia"

$PythonExe = Join-Path `
    $VenvRoot `
    "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host ""
    Write-Host "Project environment was not found."
    Write-Host ""
    Write-Host "Run setup first:"
    Write-Host ""
    Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\setup_project.ps1"
    Write-Host ""

    exit 1
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "Chest X-Ray Pneumonia AI API"
Write-Host "--------------------------------------------------"
Write-Host "Swagger UI:"
Write-Host "http://127.0.0.1:8000/docs"
Write-Host ""
Write-Host "Press CTRL+C to stop the server."
Write-Host ""

& $PythonExe -m uvicorn api.main:app