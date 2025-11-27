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

```bash
# Clone the repository
git clone https://github.com/MariusKM/badman_sfm_pipe.git
cd badman_sfm_pipe

# Installation instructions coming soon
```

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

- Python 3.8+
- [COLMAP](https://colmap.github.io/) (must be available in system PATH)
- Individual library requirements will be documented as additional pipelines are added

## License

TBD

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

