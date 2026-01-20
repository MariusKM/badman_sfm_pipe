#!/bin/bash
# Setup script for badman_sfm_pipe environment

echo "=================================="
echo "Badman SFM Pipeline - Environment Setup"
echo "=================================="

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

echo "Creating local conda environment in ./env..."
conda env create --prefix ./env -f environment.yml

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Environment created successfully!"
    echo "=================================="
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate ./env"
    echo ""
    echo "To verify COLMAP is available:"
    echo "  colmap -h"
    echo ""
    echo "To deactivate:"
    echo "  conda deactivate"
    echo ""
else
    echo "Error: Failed to create environment"
    exit 1
fi

