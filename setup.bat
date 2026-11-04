@echo off
rem One-time setup: creates .venv and installs the dependencies.
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -m venv .venv
) else (
    python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo Could not create a Python environment. Install Python 3.11 from https://www.python.org and run this again.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Installing dependencies failed.
    pause
    exit /b 1
)
echo.
echo Setup complete. Start StepMesh with run.bat
pause
