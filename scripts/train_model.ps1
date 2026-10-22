# train_model.ps1 - Prepare dataset, train CNN, and evaluate
# Run in PowerShell: .\scripts\train_model.ps1

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Electrical Equipment Image Classification - Train Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}

Write-Host "[1/3] Preparing dataset..." -ForegroundColor Yellow
python -m src.data_preparation

Write-Host "[2/3] Training CNN model..." -ForegroundColor Yellow
python -m src.train

Write-Host "[3/3] Evaluating model on held-out test set..." -ForegroundColor Yellow
python -m src.evaluate

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Pipeline Complete! Results saved in results/ and models/" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
