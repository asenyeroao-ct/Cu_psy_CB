@echo off
cd /d "%~dp0"

echo [*] Checking Python...
where py >nul 2>&1 && (set "PY=py -3") || (set "PY=python")

:: Create venv if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    echo [*] Creating virtual environment...
    %PY% -m venv venv || (
        echo [!] Failed to create venv
        pause
        exit /b 1
    )
)

:: Upgrade pip
echo [*] Upgrading pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip

:: Install requirements
if exist requirements.txt (
    echo [*] Installing requirements...
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
)

:: Start main.py
echo [*] Starting main.py...
"venv\Scripts\python.exe" main.py

pause
