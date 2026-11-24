@echo off
REM Starts the SpoilCheck model tester. Open http://127.0.0.1:8000 in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment and installing packages (first run only, a few minutes^)...
    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
)
echo Starting server at http://127.0.0.1:8000  (press Ctrl+C to stop)
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
