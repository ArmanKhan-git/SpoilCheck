@echo off
REM Starts the SpoilCheck model tester. Open http://127.0.0.1:8000 in your browser.
REM Optional: run.bat 8001   (another port, if 8000 is busy)
REM To open it from a phone on the same Wi-Fi, use run_phone.bat instead.
cd /d "%~dp0"
set PORT=%1
if "%PORT%"=="" set PORT=8000
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment and installing packages (first run only, a few minutes^)...
    REM torch 2.5.1 needs Python 3.9-3.13: prefer 3.11 if the py launcher has it
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
)
echo Starting server at http://127.0.0.1:%PORT%  (press Ctrl+C to stop)
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
