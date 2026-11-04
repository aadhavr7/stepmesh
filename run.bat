@echo off
rem Starts the StepMesh app. Optionally pass a STEP file to open.
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
    echo Run setup.bat first.
    pause
    exit /b 1
)
start "" .venv\Scripts\pythonw.exe StepMesh.pyw %*
