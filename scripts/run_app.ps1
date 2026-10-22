# run_app.ps1 - Launch Streamlit Web Application
# Run in PowerShell: .\scripts\run_app.ps1

Write-Host "Activating virtual environment and starting Streamlit application..." -ForegroundColor Cyan

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}

streamlit run app.py
