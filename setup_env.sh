#!/bin/bash
# Setup script for badman_sfm_pipe environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "Badman SFM Pipeline - Environment Setup"
echo "=================================="
echo ""

# Check if conda is installed, install if not
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    # Download Miniconda installer
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda.sh"

    wget -q --show-progress "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"

    # Install Miniconda
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    rm "$MINICONDA_INSTALLER"

    # Initialize conda
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    "$HOME/miniconda3/bin/conda" init bash

    echo "Miniconda installed successfully!"
    echo ""
fi

# Initialize conda for this script
eval "$(conda shell.bash hook)"

# Step 1: Initialize git submodules
echo "=================================="
echo "Step 1: Initializing git submodules"
echo "=================================="
git submodule update --init --recursive
echo "Submodules initialized."
echo ""

# Step 2: Create conda environment
echo "=================================="
echo "Step 2: Creating conda environment"
echo "=================================="

# Check if environment already exists
if [ -d "./env" ]; then
    echo "Environment already exists at ./env"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf ./env
    else
        echo "Keeping existing environment. Skipping creation."
        SKIP_ENV_CREATE=1
    fi
fi

if [ -z "$SKIP_ENV_CREATE" ]; then
    echo "Creating local conda environment in ./env..."
    conda env create --prefix ./env -f environment.yml
    echo "Conda environment created."
fi
echo ""

# Step 3: Activate environment and check/install PyTorch
echo "=================================="
echo "Step 3: Checking PyTorch installation"
echo "=================================="
conda activate ./env

# Check if PyTorch with CUDA is already available
PYTORCH_OK=0
if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    echo "PyTorch $PYTORCH_VERSION with CUDA $CUDA_VERSION already installed and working!"
    PYTORCH_OK=1
else
    echo "PyTorch with CUDA not found or not working."
fi

if [ "$PYTORCH_OK" -eq 0 ]; then
    echo "Installing PyTorch with CUDA 12.4 support..."
    pip install torch==2.4.* torchvision==0.19.* torchaudio==2.4.* --index-url https://download.pytorch.org/whl/cu124
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyTorch"
        exit 1
    fi
    echo "PyTorch installed."
else
    echo "Skipping PyTorch installation."
fi
echo ""

# Step 4: Install hloc
echo "=================================="
echo "Step 4: Installing hloc"
echo "=================================="

# Install hloc in editable mode
echo "Installing hloc in editable mode..."
pip install -e ./hloc

echo "hloc installed."
echo ""

# Step 5: Verify installation
echo "=================================="
echo "Step 5: Verifying installation"
echo "=================================="
echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA version check skipped"

echo ""
echo "Checking hloc..."
python -c "import hloc; print(f'hloc version: {hloc.__version__}')"

echo ""
echo "Checking pycolmap..."
python -c "import pycolmap; print(f'pycolmap available')"

echo ""
echo "=================================="
echo "Environment setup complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ./env"
echo ""
echo "To verify COLMAP is available (must be installed separately):"
echo "  colmap -h"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
