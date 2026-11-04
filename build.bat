@echo off
rem Builds dist\StepMesh\StepMesh.exe and a zip for sharing. Run setup.bat first.
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt || exit /b 1
.venv\Scripts\python.exe tests\run_tests.py || exit /b 1
.venv\Scripts\pyinstaller.exe StepMesh.spec --noconfirm || exit /b 1
copy /y LICENSE dist\StepMesh\ >nul
copy /y THIRD_PARTY_NOTICES.md dist\StepMesh\ >nul
powershell -NoProfile -Command "Compress-Archive -Path dist\StepMesh -DestinationPath dist\StepMesh-windows.zip -Force"
echo Built dist\StepMesh\StepMesh.exe and dist\StepMesh-windows.zip
