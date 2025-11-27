# Badman SFM Pipeline

A collection of custom pipelines and scripts for Structure from Motion (SFM) workflows, integrating popular SFM libraries and tools.

## Supported Libraries

This repository provides custom pipelines for:

- **[COLMAP](https://colmap.github.io/)** - General-purpose Structure-from-Motion and Multi-View Stereo
- **[GLOMAP](https://github.com/cvg/glomap)** - Global Structure-from-Motion
- **[hloc](https://github.com/cvg/Hierarchical-Localization)** - Hierarchical Localization toolbox
- **[VGG-SFM](https://github.com/facebookresearch/vggsfm)** - Visual Geometry Group's SFM implementation

## Overview

This project aims to provide flexible, customizable pipelines that combine the strengths of different SFM libraries for various reconstruction tasks.

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- [COLMAP](https://colmap.github.io/) (must be available in system PATH)
- NVIDIA GPU with CUDA support (recommended for deep learning features)

### Setup

```bash
# Clone the repository
git clone https://github.com/MariusKM/badman_sfm_pipe.git
cd badman_sfm_pipe

# Create local conda environment (in ./env directory)
conda env create --prefix ./env -f environment.yml

# Activate the environment
conda activate ./env

# Verify COLMAP is available
colmap -h
```

**Windows users:** You can also use `setup_env.bat` to automate the environment creation.

**Linux/Mac users:** You can also use `setup_env.sh` to automate the environment creation.

### Alternative: Using pip

If you prefer using pip with a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate
# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Using conda is recommended for this project due to better GPU support and dependency management for deep learning libraries. The conda environment will be created locally in the `./env` directory within the project.

## Usage

### COLMAP Pipeline

The `run_colmap.py` script provides a complete COLMAP Structure-from-Motion pipeline with intelligent checkpoint management and detailed logging.

#### Features

- **Complete Pipeline**: Feature extraction, matching, reconstruction, orientation alignment, undistortion, and model analysis
- **Checkpoint System**: Automatic progress tracking with JSON-based checkpoints for resumable execution
- **Flexible Execution**: Run the entire pipeline, specific stages, or resume from any point
- **Configurable Matching**: Support for exhaustive, sequential, vocab_tree, and spatial matching methods
- **Smart Model Selection**: Automatically selects the reconstruction model with the most registered images
- **GPU Acceleration**: Utilizes GPU for feature extraction and matching when available
- **Detailed Logging**: Comprehensive logs with timing information and reconstruction statistics

#### Basic Usage

```bash
python run_colmap.py \
  --input_images /path/to/images \
  --output /path/to/output \
  --config /path/to/config.ini
```

#### Command-Line Arguments

**Required:**
- `--input_images`: Path to directory containing input images
- `--output`: Path to output directory (database and results will be stored here)
- `--config`: Path to COLMAP INI configuration file

**Optional:**
- `--skip_undistortion`: Skip the image undistortion stage
- `--skip_orientation`: Skip the orientation alignment stage
- `--stage`: Run only a specific stage (choices: `feature_extraction`, `feature_matching`, `reconstruction`, `orientation_alignment`, `undistortion`, `model_analysis`)
- `--from_stage`: Resume pipeline from a specific stage onwards
- `--force_restart`: Clear all checkpoints and restart from the beginning
- `--matcher_type`: Override matching type from config (choices: `exhaustive`, `sequential`, `vocab_tree`, `spatial`)

#### Examples

**Run complete pipeline:**
```bash
python run_colmap.py \
  --input_images ./my_images \
  --output ./output \
  --config ./colmap_config.ini
```

**Skip optional post-processing stages:**
```bash
python run_colmap.py \
  --input_images ./my_images \
  --output ./output \
  --config ./colmap_config.ini \
  --skip_undistortion \
  --skip_orientation
```

**Run only feature extraction:**
```bash
python run_colmap.py \
  --input_images ./my_images \
  --output ./output \
  --config ./colmap_config.ini \
  --stage feature_extraction
```

**Resume from reconstruction stage:**
```bash
python run_colmap.py \
  --input_images ./my_images \
  --output ./output \
  --config ./colmap_config.ini \
  --from_stage reconstruction
```

**Use sequential matching:**
```bash
python run_colmap.py \
  --input_images ./my_images \
  --output ./output \
  --config ./colmap_config.ini \
  --matcher_type sequential
```

**Force restart entire pipeline:**
```bash
python run_colmap.py \
  --input_images ./my_images \
  --output ./output \
  --config ./colmap_config.ini \
  --force_restart
```

#### Output Structure

```
output_dir/
├── database.db              # COLMAP database
├── sparse/                  # Sparse reconstruction(s)
│   ├── 0/                   # First model
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   └── 1/                   # Additional models (if any)
├── oriented-model/          # Orientation-aligned model (if enabled)
├── dense/                   # Undistorted images and dense reconstruction (if enabled)
│   ├── images/
│   ├── sparse/
│   └── stereo/
├── .checkpoint.json         # Pipeline checkpoint data
└── colmap_pipeline.log      # Detailed execution log
```

#### Configuration File

The script uses COLMAP INI configuration files. These files contain settings for all stages of the pipeline organized in sections such as `[ImageReader]`, `[FeatureExtraction]`, `[SiftExtraction]`, `[FeatureMatching]`, `[Mapper]`, etc.

Key configuration sections:
- `[ImageReader]`: Camera model and image reading settings
- `[FeatureExtraction]` / `[SiftExtraction]`: Feature detection parameters
- `[FeatureMatching]` / `[SiftMatching]`: Feature matching parameters
- `[Mapper]`: Reconstruction/mapping settings
- `[ExhaustiveMatching]`, `[SequentialMatching]`, `[VocabTreeMatching]`, `[SpatialMatching]`: Matcher-specific settings

#### Checkpoint System

The checkpoint system allows you to:
- Automatically resume failed or interrupted pipelines
- Skip already completed stages when re-running
- Track execution time and statistics for each stage
- Force restart from any stage in the pipeline

Checkpoint data is stored in `.checkpoint.json` in the output directory.

## Requirements

### System Requirements
- Python 3.10+
- [COLMAP](https://colmap.github.io/) (must be available in system PATH)
- NVIDIA GPU with CUDA 11.8+ (recommended for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large datasets)

### Python Dependencies
The project uses conda for dependency management. Key dependencies include:

- **Deep Learning:** PyTorch 2.0+, TorchVision
- **Computer Vision:** OpenCV, scikit-image, Pillow
- **Scientific Computing:** NumPy, SciPy, Pandas
- **Utilities:** tqdm, matplotlib, h5py

See `environment.yml` for the complete list of dependencies.

### Future Integrations
Additional dependencies will be added for:
- [hloc](https://github.com/cvg/Hierarchical-Localization) - Hierarchical localization
- [VGG-SFM](https://github.com/facebookresearch/vggsfm) - VGG Structure-from-Motion

## License

TBD

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

