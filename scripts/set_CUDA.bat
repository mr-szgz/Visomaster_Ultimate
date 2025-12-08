@echo off

if "%CUDA_VER%"=="" (
	echo ERROR: CUDA_VER environment variable is not set
	echo Please set CUDA_VER before calling this script
	exit /b 1
)

set CUDA_PATH_VAR=CUDA_PATH_V%CUDA_VER%

call set CUDA_PATH=%%%CUDA_PATH_VAR%%%
set CUDA_HOME=%CUDA_PATH%

set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%
set LD_LIBRARY_PATH=%CUDA_PATH%\lib\x64;%LD_LIBRARY_PATH%

echo CUDA %CUDA_VER:.=.% environment variables set successfully
echo CUDA_PATH: %CUDA_PATH%
echo CUDA_HOME: %CUDA_HOME%
