Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "            PaddleOCR Studio - Avvio Locale             " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 1. Verifica virtualenv
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[1/3] Creazione ambiente virtuale con Python 3.10..." -ForegroundColor Yellow
    py -3.10 -m venv .venv
}

# 2. Installazione dipendenze
Write-Host "[2/3] Verifica dipendenze in corso..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" -m pip install -q -r requirements.txt

# 3. Avvio server e apertura browser
Write-Host "[3/3] Avvio server su http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"
& ".\.venv\Scripts\python.exe" server.py
