@echo off
REM Setup script for badman_sfm_pipe environment (Windows)

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ==================================
echo Badman SFM Pipeline - Environment Setup
echo ==================================
echo.

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Step 1: Initialize git submodules
echo ==================================
echo Step 1: Initializing git submodules
echo ==================================
git submodule update --init --recursive
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to initialize submodules. You may need to run this manually.
)
echo Submodules initialized.
echo.

REM Step 2: Create conda environment
echo ==================================
echo Step 2: Creating conda environment
echo ==================================

REM Check if environment already exists
if exist ".\env" (
    echo Environment already exists at .\env
    set /p REPLY="Do you want to remove and recreate it? (y/N): "
    if /i "!REPLY!"=="y" (
        echo Removing existing environment...
        rmdir /s /q .\env
    ) else (
        echo Keeping existing environment. Skipping creation.
        set "SKIP_ENV_CREATE=1"
    )
)

if not defined SKIP_ENV_CREATE (
    echo Creating local conda environment in .\env...
    call conda env create --prefix .\env -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create environment
        exit /b 1
    )
    echo Conda environment created.
)
echo.

REM Step 3: Activate environment and check/install PyTorch
echo ==================================
echo Step 3: Checking PyTorch installation
echo ==================================
call conda activate .\env

REM Check if PyTorch with CUDA is already available
set "PYTORCH_OK=0"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo PyTorch with CUDA already installed and working!
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.__version__)"') do echo PyTorch version: %%i
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.version.cuda)"') do echo CUDA version: %%i
    set "PYTORCH_OK=1"
) else (
    echo PyTorch with CUDA not found or not working.
)

if "!PYTORCH_OK!"=="0" (
    echo Installing PyTorch with CUDA 12.4 support...
    pip install torch==2.4.* torchvision==0.19.* torchaudio==2.4.* --index-url https://download.pytorch.org/whl/cu124
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to install PyTorch
        exit /b 1
    )
    echo PyTorch installed.
) else (
    echo Skipping PyTorch installation.
)
echo.

REM Step 4: Install hloc
echo ==================================
echo Step 4: Installing hloc
echo ==================================

REM Install hloc in editable mode
echo Installing hloc in editable mode...
pip install -e .\hloc
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to install hloc
)

echo hloc installed.
echo.

REM Step 5: Verify installation
echo ==================================
echo Step 5: Verifying installation
echo ==================================
echo Checking PyTorch...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>nul || echo CUDA version check skipped

echo.
echo Checking hloc...
python -c "import hloc; print(f'hloc version: {hloc.__version__}')"

echo.
echo Checking pycolmap...
python -c "import pycolmap; print(f'pycolmap available')"

echo.
echo ==================================
echo Environment setup complete!
echo ==================================
echo.
echo To activate the environment, run:
echo   conda activate .\env
echo.
echo To verify COLMAP is available (must be installed separately):
echo   colmap -h
echo.
echo To deactivate:
echo   conda deactivate
echo.

endlocal
