$cwd = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $cwd

Write-Host "Starting Handy AI from app.py on http://127.0.0.1:5000"
if (-not $env:GOOGLE_API_KEY) {
    Write-Host "GOOGLE_API_KEY is not set. The page will open, but AI solving will stay disabled." -ForegroundColor Yellow
}

python app.py
