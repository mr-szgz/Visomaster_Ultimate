call scripts\setenv.bat
if errorlevel 1 (
    echo Failed to set environment variables.
    pause
    exit /b 1
)
"%PYTHON_EXECUTABLE%" main.py
pause
