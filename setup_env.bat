@echo off
REM Setup script for badman_sfm_pipe environment (Windows)

echo ==================================
echo Badman SFM Pipeline - Environment Setup
echo ==================================

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

echo Creating conda environment from environment.yml...
conda env create -f environment.yml

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================
    echo Environment created successfully!
    echo ==================================
    echo.
    echo To activate the environment, run:
    echo   conda activate badman_sfm
    echo.
    echo To verify COLMAP is available:
    echo   colmap -h
    echo.
    echo To deactivate:
    echo   conda deactivate
    echo.
) else (
    echo Error: Failed to create environment
    exit /b 1
)

