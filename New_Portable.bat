@echo off
setlocal EnableExtensions EnableDelayedExpansion
@REM call scripts\set_CUDA.bat
call scripts\setenv.bat
set "PREFIX=S:\Spaces\VisoMaster_portable2"
set "REPO_CLONE_URL=https://github.com/visomaster/VisoMaster.git"
echo Installing portable version to %PREFIX%

if not exist "%PREFIX%" (
    mkdir "%PREFIX%"
)

set "PYVER=3.10.11"
set "PY_URL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip"
set "PY_ARCHIVE=%PREFIX%\python.zip"
set "NEED_PYTHON=1"

if exist "%PREFIX%\python" (
    set "NEED_PYTHON=0"

    echo    checking python..
    %PREFIX%\python\python.exe --version
    if errorlevel 1 set "NEED_PYTHON=1"

    if "!NEED_PYTHON!"=="0" (
        echo  Checking pip...
        %PREFIX%\python\python.exe -m pip --version
        if errorlevel 1 set "NEED_PYTHON=1"
    )

    if "!NEED_PYTHON!"=="0" (
        echo Existing Python install found at %PREFIX%\python
    ) else (
        echo Python directory exists at %PREFIX%\python but failed validation.
        echo [D]elete and continue or [A]bort?
        choice /c DA /n /m "Choice: "
        if errorlevel 2 (
            echo Installation aborted.
            pause
            exit /b 1
        )
        echo Removing existing Python directory...
        rmdir /s /q "%PREFIX%\python"
    )
)

if "%NEED_PYTHON%"=="1" (

    echo.
    echo Downloading Python %PYVER%...
    echo.
    powershell -Command "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ARCHIVE%'"
    powershell -Command "Expand-Archive -LiteralPath '%PY_ARCHIVE%' -DestinationPath '%PREFIX%\python'"
    echo.
    del "%PY_ARCHIVE%"
    echo.
    echo   Installing pip...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PREFIX%\python\get-pip.py'"
    %PREFIX%\python\python.exe -s %PREFIX%\python\get-pip.py

    echo Configuring environment for target...
    set "PATH=%PREFIX%\python;%PREFIX%\python\Scripts;%PATH%"
    echo Lib\site-packages>>%PREFIX%\python\python310._pth
    echo %PREFIX%\source>>%PREFIX%\python\python310._pth

    echo  Checking pip...
    %PREFIX%\python\python.exe -m pip --version
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
set "NEED_CLONE=0"

if not exist "%PREFIX%\source" (
    set "NEED_CLONE=1"
) else (
    echo    checking source..
    %PREFIX%\python\python.exe -c "import app.helpers.recording"
    if errorlevel 1 set "NEED_CLONE=1"
)

if "!NEED_CLONE!"=="1" (
    echo Cloning repository...
    git clone %REPO_CLONE_URL% "%PREFIX%\source"
    if errorlevel 1 (
        echo Failed to clone repository
        pause
        exit /b 1
    )
)

if not exist "%PREFIX%\source" (
    echo Source directory missing at "%PREFIX%\source"
    pause
    exit /b 1
)
echo Copying support files...
xcopy /E /I /Y "%~dp0\scripts" "%PREFIX%\source" >nul
xcopy /E /I /Y "%~dp0\app" "%PREFIX%\source\app" >nul
if errorlevel 1 (
    echo Failed to copy support files
    pause
    exit /b 1
)

:: if install.dat is present copy that too into %PREFIX\source%
if exist "%~dp0install.dat" (
    copy "%~dp0install.dat" "%PREFIX%\source\"  >nul
    if errorlevel 1 (
        echo Failed to copy install.dat
        pause
        exit /b 1
    )
)

:: copying large files isn't reccomended
:: xcopy /E /I /Y "%~dp0dependencies" "%PREFIX%\source\dependencies" >nul
:: xcopy /E /I /Y "%~dp0model_assets" "%PREFIX%\source\model_assets" >nul

@REM set "pythonExe=%PREFIX%\python\python.exe"

@REM "%pythonExe%" -m pip install -r "%PREFIX%\source\requirements_cu124.txt" --default-timeout 100
@REM if errorlevel 1 (
@REM     echo Failed to install requirements
@REM     pause
@REM     exit /b 1
@REM )
@REM "%pythonExe%" -s "%PREFIX%\source\download_models.py"
@REM if errorlevel 1 (
@REM     echo Failed to download models
@REM     pause
@REM     exit /b 1
@REM )

echo.
echo   Installation files copied
echo.
echo Installation completed successfully!
echo Location: %PREFIX%
pause
explorer "%PREFIX%"
endlocal
