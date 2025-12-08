@echo off
call scripts\setenv.bat
"%GIT_EXECUTABLE%" fetch origin main
"%GIT_EXECUTABLE%" reset --hard origin/main
"%PYTHON_EXECUTABLE%" -m pip install -r requirements_cu124.txt --default-timeout 100
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)
"%PYTHON_EXECUTABLE%" download_models.py
if errorlevel 1 (
    echo Failed to download models
    pause
    exit /b 1
)
