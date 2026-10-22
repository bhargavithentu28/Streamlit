# setup_windows.ps1 - Environment setup script for Electrical Equipment Image Classification
# Run in PowerShell: .\scripts\setup_windows.ps1

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Electrical Equipment Image Classification - Setup       " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python version
Write-Host "[1/6] Verifying Python installation..." -ForegroundColor Yellow
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python is not found in PATH. Please install Python 3.11 and ensure it is added to PATH."
    exit 1
}
$pyVersion = python --version
Write-Host "Found: $pyVersion" -ForegroundColor Green

# 2. Setup Virtual Environment
Write-Host "[2/6] Configuring virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Created new virtual environment in .venv" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 3. Activate Virtual Environment
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 4. Install / Upgrade Dependencies
Write-Host "[4/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Prepare Dataset
Write-Host "[5/6] Preparing dataset (validating, deduplicating, splitting)..." -ForegroundColor Yellow
python -m src.data_preparation

# 6. Run Verification Tests
Write-Host "[6/6] Running automated test suite..." -ForegroundColor Yellow
pytest tests/ -v

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Setup Completed Successfully!" -ForegroundColor Green
Write-Host "To train model:   python -m src.train" -ForegroundColor White
Write-Host "To evaluate:      python -m src.evaluate" -ForegroundColor White
Write-Host "To launch app:    streamlit run app.py" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
