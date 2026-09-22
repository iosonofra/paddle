@echo off
setlocal enabledelayedexpansion
title PaddleOCR Studio

echo ========================================================
echo             PaddleOCR Studio - Avvio Locale
echo ========================================================
echo.

cd /d "%~dp0"

:: 1. Verifica ambiente virtuale .venv
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creazione ambiente virtuale con Python 3.10...
    py -3.10 -m venv .venv
    if errorlevel 1 (
        echo ERRORE: Impossibile creare il virtualenv con Python 3.10.
        echo Assicurati che Python 3.10 sia installato.
        pause
        exit /b 1
    )
)

:: 2. Verifica / installazione dipendenze
echo [2/3] Verifica dipendenze in corso...
.\.venv\Scripts\python.exe -m pip install -q -r requirements.txt

:: 3. Avvio del server FastAPI
echo [3/3] Avvio server PaddleOCR Studio su http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000

.\.venv\Scripts\python.exe server.py

pause
