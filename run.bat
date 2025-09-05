@echo off
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo [*] Running main.py...
"venv\Scripts\python.exe" main.py

pause
