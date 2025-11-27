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

### GLOMAP Pipeline

The `run_glomap.py` script provides a hybrid SFM pipeline that combines COLMAP's robust feature extraction and matching with GLOMAP's fast global reconstruction approach.

#### Features

- **Hybrid Approach**: COLMAP for features/matching, GLOMAP for reconstruction
- **Faster Reconstruction**: Global optimization is significantly faster than incremental approach
- **Same Pipeline Stages**: Maintains compatibility with COLMAP workflow
- **Dual Configuration**: Separate configs for COLMAP and GLOMAP stages
- **All COLMAP Features**: Checkpoint system, logging, flexible execution, etc.

#### Basic Usage

```bash
python run_glomap.py \
  --input_images /path/to/images \
  --output /path/to/output \
  --colmap_config /path/to/colmap_config.ini \
  --glomap_config /path/to/glomap_config.ini
```

#### Command-Line Arguments

**Required:**
- `--input_images`: Path to directory containing input images
- `--output`: Path to output directory (database and results will be stored here)
- `--colmap_config`: Path to COLMAP INI configuration file (for feature extraction/matching)
- `--glomap_config`: Path to GLOMAP INI configuration file (for mapper)

**Optional:**
- `--skip_undistortion`: Skip the image undistortion stage
- `--skip_orientation`: Skip the orientation alignment stage
- `--stage`: Run only a specific stage (choices: `feature_extraction`, `feature_matching`, `reconstruction`, `orientation_alignment`, `undistortion`, `model_analysis`)
- `--from_stage`: Resume pipeline from a specific stage onwards
- `--force_restart`: Clear all checkpoints and restart from the beginning
- `--matcher_type`: Override matching type from config (choices: `exhaustive`, `sequential`, `vocab_tree`, `spatial`)

#### Examples

**Basic GLOMAP reconstruction:**
```bash
python run_glomap.py \
  --input_images ./my_images \
  --output ./output \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini
```

**Resume from reconstruction stage:**
```bash
python run_glomap.py \
  --input_images ./my_images \
  --output ./output \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --from_stage reconstruction
```

**Skip post-processing stages:**
```bash
python run_glomap.py \
  --input_images ./my_images \
  --output ./output \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --skip_undistortion \
  --skip_orientation
```

#### GLOMAP Configuration

The GLOMAP configuration file (`defaultGloMap.ini`) contains parameters specific to the GLOMAP mapper:

**Key sections:**
- `[Mapper]`: Core mapper settings (constraint types, iterations, skip flags)
- `[ViewGraphCalib]`: View graph calibration thresholds
- `[GlobalPositioning]`: Global positioning optimization and GPU settings
- `[BundleAdjustment]`: Bundle adjustment parameters
- `[Triangulation]`: Point triangulation settings
- `[Thresholds]`: Various threshold parameters

See [`configDocs.md`](configDocs.md) for detailed parameter documentation.

#### When to Use GLOMAP vs COLMAP

**Use GLOMAP when:**
- **Speed is priority**: GLOMAP is significantly faster for large datasets
- **Well-connected images**: Works best with good image connectivity
- **Global optimization preferred**: Optimizes all cameras simultaneously
- **GPU available**: Can leverage GPU for global positioning

**Use COLMAP when:**
- **Maximum accuracy needed**: Incremental approach can be more accurate
- **Complex scenarios**: Better handles challenging image collections
- **Incremental workflow**: Need to add images to existing reconstruction
- **Memory constraints**: Incremental approach uses less memory

#### Output Structure

Same as COLMAP pipeline:

```
output_dir/
├── database.db              # COLMAP database
├── sparse/                  # GLOMAP reconstruction output
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── oriented-model/          # Orientation-aligned model (if enabled)
├── dense/                   # Undistorted images (if enabled)
├── .checkpoint.json         # Pipeline checkpoint data
└── glomap_pipeline.log      # Detailed execution log
```

**Note:** GLOMAP typically outputs directly to the `sparse/` directory rather than creating numbered subdirectories like COLMAP.

## Configuration Documentation

For detailed documentation of all COLMAP and GLOMAP configuration parameters, see [`configDocs.md`](configDocs.md). This includes:

- Complete parameter descriptions
- Default values and valid ranges
- When and how to modify parameters
- Configuration tips for common scenarios
- Troubleshooting guidance

## Requirements

### System Requirements
- Python 3.10+
- [COLMAP](https://colmap.github.io/) (must be available in system PATH)
- [GLOMAP](https://github.com/cvg/glomap) (optional, for GLOMAP pipeline)
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

