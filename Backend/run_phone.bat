@echo off
REM Starts the SpoilCheck tester so phones on the SAME Wi-Fi can open it.
REM Optional: run_phone.bat 8001   (another port, if 8000 is busy)
REM First time: Windows Firewall may ask whether to allow Python -- allow it on Private networks.
cd /d "%~dp0"
set PORT=%1
if "%PORT%"=="" set PORT=8000
if not exist ".venv\Scripts\python.exe" (
    echo Run run.bat once first to create the virtual environment.
    pause
    exit /b 1
)
echo.
echo On this PC:     http://127.0.0.1:%PORT%
echo On your phone (same Wi-Fi), open one of these:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do for /f "tokens=*" %%b in ("%%a") do echo     http://%%b:%PORT%
echo.
echo Press Ctrl+C to stop.
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%
