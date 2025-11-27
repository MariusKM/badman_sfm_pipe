#!/bin/bash
# Setup script for badman_sfm_pipe environment

echo "=================================="
echo "Badman SFM Pipeline - Environment Setup"
echo "=================================="

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
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

