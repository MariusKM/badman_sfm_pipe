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

### VGGSfM Installation

VGGSfM is integrated as a git submodule with its own dedicated conda environment to avoid dependency conflicts.

#### Prerequisites

- All prerequisites listed above
- Git (for submodule management)
- CUDA 12.1 compatible GPU (recommended)

#### Setup VGGSfM Environment

**Windows:**
```bash
# Setup VGGSfM conda environment
.\setup_vggsfm_env.bat

# Activate the VGGSfM environment
conda activate .\vggsfm_env

# Install VGGSfM and dependencies
.\install_vggsfm.bat

# Verify installation
python -c "import vggsfm; print('VGGSfM imported successfully')"
```

**Linux/Mac:**
```bash
# Setup VGGSfM conda environment
bash setup_vggsfm_env.sh

# Activate the VGGSfM environment
conda activate ./vggsfm_env

# Install VGGSfM and dependencies
cd vggsfm
bash install.sh
python -m pip install -e .
cd ..

# Verify installation
python -c "import vggsfm; print('VGGSfM imported successfully')"
```

#### VGGSfM Environment Details

The VGGSfM environment (`vggsfm_env/`) includes:
- Python 3.10
- PyTorch 2.1 with CUDA 12.1
- VGGSfM v2.0 and all core dependencies
- LightGlue (feature matching)
- pycolmap (COLMAP Python bindings)
- poselib (pose estimation)
- visdom (visualization)

**Note:** PyTorch3D is optional and only needed for advanced visdom visualization (`cfg.viz_visualize=True`). The current installation skips it to avoid platform-specific compilation issues. VGGSfM works perfectly without it for all core functionality.

#### Updating VGGSfM

Since VGGSfM is a git submodule, you can update it:

```bash
# Update to latest version
git submodule update --remote vggsfm

# Or update all submodules
git submodule update --remote
```

#### Using VGGSfM

VGGSfM demo examples:

```bash
# Activate VGGSfM environment
conda activate ./vggsfm_env

# Run on your images
cd vggsfm
python demo.py SCENE_DIR=/path/to/your/images

# With specific camera type
python demo.py SCENE_DIR=/path/to/your/images camera_type=SIMPLE_RADIAL

# With visualization
python demo.py SCENE_DIR=/path/to/your/images gr_visualize=True

# Generate denser point cloud
python demo.py SCENE_DIR=/path/to/your/images extra_pt_pixel_interval=10 concat_extra_points=True
```

For more details, see the [VGGSfM README](vggsfm/README.md) and [VGGSfM documentation](https://github.com/facebookresearch/vggsfm).

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

### Refined GLOMAP Pipeline

The `run_glomap_refined.py` script combines GLOMAP's fast global reconstruction with COLMAP's post-processing refinement, providing the best of both worlds: speed and quality.

#### Features

- **Fast Global Reconstruction**: GLOMAP mapper for rapid initial reconstruction
- **Post-Processing Refinement**: Multiple rounds of triangulation + bundle adjustment
- **Quality Enhancement**: Significantly improved reconstruction quality over raw GLOMAP
- **Progress Tracking**: Model analysis after each refinement step
- **All Standard Features**: Checkpoint system, logging, flexible execution

#### Basic Usage

```bash
python run_glomap_refined.py \
  --input_images /path/to/images \
  --output /path/to/output \
  --colmap_config /path/to/colmap_config.ini \
  --glomap_config /path/to/glomap_config.ini
```

#### Command-Line Arguments

**Required:**
- `--input_images`: Path to directory containing input images
- `--output`: Path to output directory (database and results will be stored here)
- `--colmap_config`: Path to COLMAP INI configuration file (for feature extraction/matching and refinement)
- `--glomap_config`: Path to GLOMAP INI configuration file (for mapper)

**Optional:**
- `--skip_undistortion`: Skip the image undistortion stage
- `--skip_orientation`: Skip the orientation alignment stage
- `--refinement_rounds`: Number of triangulation + bundle adjustment rounds (default: 2)
- `--stage`: Run only a specific stage
- `--from_stage`: Resume pipeline from a specific stage onwards
- `--force_restart`: Clear all checkpoints and restart from the beginning
- `--matcher_type`: Override matching type from config

#### Examples

**Basic refined reconstruction:**
```bash
python run_glomap_refined.py \
  --input_images ./my_images \
  --output ./output \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini
```

**With custom refinement rounds:**
```bash
python run_glomap_refined.py \
  --input_images ./my_images \
  --output ./output \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --refinement_rounds 3
```

**Resume from refinement stage:**
```bash
python run_glomap_refined.py \
  --input_images ./my_images \
  --output ./output \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --from_stage post_processing_refinement
```

#### Pipeline Stages

1. **Feature Extraction**: Extract SIFT features from images (COLMAP)
2. **Feature Matching**: Match features between image pairs (COLMAP)
3. **Reconstruction**: Global optimization reconstruction (GLOMAP)
4. **Post-Processing Refinement**: Multiple rounds of:
   - Point triangulation (COLMAP)
   - Model analysis (logged)
   - Bundle adjustment (COLMAP)
   - Model analysis (logged)
5. **Orientation Alignment**: Align model orientation (COLMAP, optional)
6. **Undistortion**: Undistort images (COLMAP, optional)
7. **Final Model Analysis**: Complete reconstruction statistics (COLMAP)

#### Post-Processing Refinement

After GLOMAP's fast global reconstruction, the script automatically performs multiple rounds of refinement:

1. **Point Triangulation** - Adds additional 3D points from image observations
2. **Bundle Adjustment** - Refines camera poses and 3D point positions

Each round logs model statistics (number of points, observations, mean reprojection error) to track improvement.

**Typical Improvements Over Raw GLOMAP:**
- 10-30% more 3D points
- 20-50% more observations  
- 30-50% lower mean reprojection error
- Higher overall reconstruction density and accuracy

#### Configuration

The refined GLOMAP pipeline requires two configuration files:

**COLMAP Config** (`--colmap_config`):
- Controls feature extraction/matching parameters
- Controls refinement parameters:
  - `[Mapper]` section for triangulation
  - `[BundleAdjustment]` section for bundle adjustment

**GLOMAP Config** (`--glomap_config`):
- Controls GLOMAP mapper parameters (see [`defaultGloMap.ini`](defaultGloMap.ini))

See [`configDocs.md`](configDocs.md) for detailed parameter documentation.

#### When to Use Refined GLOMAP

**Use Refined GLOMAP when:**
- **Speed + Quality**: Need fast reconstruction with high quality
- **Medium-large datasets**: 500-5000+ images
- **Best overall performance**: Balanced speed and accuracy
- **Production workflows**: Reliable, high-quality results

**Use Standard GLOMAP (`run_glomap.py`) when:**
- **Maximum speed**: Fastest possible reconstruction
- **Preview/testing**: Quick validation of image sets
- **Post-processing elsewhere**: Using external refinement tools

**Use Hierarchical COLMAP (`run_colmap_hierarchical.py`) when:**
- **Maximum quality**: Best possible reconstruction quality
- **Very large datasets**: 5000+ images with good parallelization
- **Complex scenes**: Challenging reconstruction scenarios

#### Output Structure

```
output_dir/
├── database.db                         # COLMAP database
├── sparse/                             # Refined reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── oriented-model/                     # Orientation-aligned model (if enabled)
├── dense/                              # Undistorted images (if enabled)
├── .checkpoint.json                    # Pipeline checkpoint data
└── glomap_refined_pipeline.log         # Detailed execution log with refinement tracking
```

**Note:** Post-processing refinement operates in-place on the sparse model, progressively improving quality.

### Hierarchical COLMAP Pipeline

The `run_colmap_hierarchical.py` script uses COLMAP's hierarchical mapper for large-scale datasets, providing parallelized reconstruction with automatic post-processing refinement.

#### Features

- **Parallelized Reconstruction**: Scene partitioning with overlapping submodels
- **Scalable**: Designed for datasets with 1000+ images
- **Automatic Refinement**: Built-in post-processing with triangulation and bundle adjustment rounds
- **Progress Tracking**: Model analysis after each refinement step to track improvements
- **All Standard Features**: Checkpoint system, logging, flexible execution

#### Basic Usage

```bash
python run_colmap_hierarchical.py \
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
- `--num_workers`: Number of parallel workers (-1 for all cores, default: from config)
- `--refinement_rounds`: Number of triangulation + bundle adjustment rounds (default: 2)
- `--stage`: Run only a specific stage
- `--from_stage`: Resume pipeline from a specific stage onwards
- `--force_restart`: Clear all checkpoints and restart from the beginning
- `--matcher_type`: Override matching type from config

#### Examples

**Basic hierarchical reconstruction:**
```bash
python run_colmap_hierarchical.py \
  --input_images ./my_images \
  --output ./output \
  --config ./defaultColMap.ini
```

**With custom refinement rounds:**
```bash
python run_colmap_hierarchical.py \
  --input_images ./my_images \
  --output ./output \
  --config ./defaultColMap.ini \
  --refinement_rounds 3
```

**Limit parallel workers:**
```bash
python run_colmap_hierarchical.py \
  --input_images ./my_images \
  --output ./output \
  --config ./defaultColMap.ini \
  --num_workers 8
```

**Resume from refinement stage:**
```bash
python run_colmap_hierarchical.py \
  --input_images ./my_images \
  --output ./output \
  --config ./defaultColMap.ini \
  --from_stage post_processing_refinement
```

#### Pipeline Stages

1. **Feature Extraction**: Extract SIFT features from images
2. **Feature Matching**: Match features between image pairs
3. **Hierarchical Reconstruction**: Parallel reconstruction with scene partitioning
4. **Post-Processing Refinement**: Multiple rounds of:
   - Point triangulation
   - Model analysis (logged)
   - Bundle adjustment
   - Model analysis (logged)
5. **Orientation Alignment**: Align model orientation (optional)
6. **Undistortion**: Undistort images (optional)
7. **Final Model Analysis**: Complete reconstruction statistics

#### Post-Processing Refinement

The hierarchical mapper creates a merged model from submodels that requires refinement to achieve optimal quality. The script automatically performs multiple rounds of:

1. **Point Triangulation** - Adds additional 3D points from image observations
2. **Bundle Adjustment** - Refines camera poses and 3D point positions

Each round logs model statistics (number of points, observations, mean reprojection error) to track improvement.

**Typical Improvements:**
- 10-30% more 3D points
- 20-50% more observations
- 30-50% lower mean reprojection error

#### Configuration

The hierarchical mapper uses the standard `defaultColMap.ini` with an additional section:

```ini
[HierarchicalMapper]
num_workers=-1              # Use all CPU cores
image_overlap=50            # Images overlap between submodels
leaf_max_num_images=500     # Max images per submodel
```

See [`configDocs.md`](configDocs.md) for detailed parameter documentation.

#### When to Use Hierarchical Mapper

**Use Hierarchical Mapper when:**
- **Large datasets**: 1000+ images
- **Parallel processing**: Multiple CPU cores available
- **Well-connected images**: Good overlap between images
- **Speed priority**: Faster than incremental for large sets

**Use Standard Mapper (`run_colmap.py`) when:**
- **Small-medium datasets**: < 1000 images
- **Complex scenarios**: Challenging image collections
- **Memory constrained**: Limited RAM
- **Incremental workflow**: Adding images to existing reconstruction

#### Output Structure

Same as standard COLMAP pipeline:

```
output_dir/
├── database.db                    # COLMAP database
├── sparse/                        # Hierarchical reconstruction output
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── oriented-model/                # Orientation-aligned model (if enabled)
├── dense/                         # Undistorted images (if enabled)
├── .checkpoint.json               # Pipeline checkpoint data
└── colmap_hierarchical_pipeline.log  # Detailed execution log with refinement tracking
```

**Note:** Hierarchical mapper outputs directly to the `sparse/` directory (no numbered subdirectories).

## Multi-Pipeline Comparison

The `run_multi_compare.py` script allows you to run and compare multiple SFM pipelines on the same dataset, generating a comprehensive report with speed and quality metrics.

### Features

- **Multiple Pipeline Support**: Compare COLMAP, GLOMAP, Hierarchical COLMAP, and Refined GLOMAP
- **Comprehensive Metrics**: Speed, quality, and efficiency measurements
- **Detailed Reports**: Markdown format with tables, analysis, and recommendations
- **Flexible Selection**: Run all pipelines or select specific ones
- **Sequential Execution**: Safe, reliable processing with progress tracking

### Basic Usage

```bash
python run_multi_compare.py \
  --input_images /path/to/images \
  --output /path/to/comparison_output \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --comparison_log ./comparison_report.md
```

### Command-Line Arguments

**Required:**
- `--input_images`: Path to directory containing input images
- `--output`: Path to output directory (subdirectories created per pipeline)
- `--colmap_config`: Path to COLMAP INI configuration file
- `--glomap_config`: Path to GLOMAP INI configuration file
- `--comparison_log`: Path to comparison report file (markdown format)

**Optional:**
- `--pipelines`: Which pipelines to run (choices: `colmap`, `glomap`, `hierarchical`, `refined`, `all`)
  - Default: `all`
  - Can specify multiple: `--pipelines colmap glomap`
- `--refinement_rounds`: Number of refinement rounds for hierarchical/refined (default: 2)
- `--skip_undistortion`: Skip undistortion stage for all pipelines
- `--skip_orientation`: Skip orientation alignment for all pipelines
- `--matcher_type`: Override matching type for all pipelines

### Examples

**Compare all pipelines:**
```bash
python run_multi_compare.py \
  --input_images ./my_images \
  --output ./comparison \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --comparison_log ./report.md
```

**Compare only COLMAP and GLOMAP:**
```bash
python run_multi_compare.py \
  --input_images ./my_images \
  --output ./comparison \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --comparison_log ./report.md \
  --pipelines colmap glomap
```

**Fast comparison (skip undistortion and orientation):**
```bash
python run_multi_compare.py \
  --input_images ./my_images \
  --output ./comparison \
  --colmap_config ./defaultColMap.ini \
  --glomap_config ./defaultGloMap.ini \
  --comparison_log ./report.md \
  --skip_undistortion \
  --skip_orientation
```

### Metrics Tracked

**Speed Metrics:**
- Total execution time (seconds)
- Per-stage timing breakdown (feature extraction, matching, reconstruction, etc.)
- Time per registered image (efficiency metric)
- Time per 3D point (efficiency metric)
- Speed comparison ratios

**Quality Metrics:**
- Number of registered images
- Registration ratio (percentage of images successfully registered)
- Number of 3D points
- Number of observations
- Observations per point (reconstruction density)
- Mean track length (multi-view consistency)
- **Mean reprojection error** (most important quality metric, in pixels)
- Points per registered image
- Observations per registered image

**Composite Metric:**
- Quality Score (0-10 scale) combining:
  - Registration ratio (20% weight)
  - Reprojection error (40% weight)
  - Point density (20% weight)
  - Observations per point (20% weight)

### Report Contents

The generated markdown report includes:

1. **Summary Table**: Quick overview of all pipelines with key metrics
2. **Speed Analysis**: 
   - Fastest/slowest pipelines
   - Time efficiency comparisons
   - Per-stage timing breakdown
3. **Quality Analysis**:
   - Best registration coverage
   - Most 3D points
   - Best reprojection accuracy
   - Quality score comparison
4. **Recommendations**: Data-driven suggestions based on results
5. **Detailed Results**: Complete statistics for each pipeline

### Output Structure

```
output_dir/
├── colmap/                     # COLMAP pipeline output
│   ├── database.db
│   ├── sparse/
│   ├── .checkpoint.json
│   └── colmap_pipeline.log
├── glomap/                     # GLOMAP pipeline output
│   └── ...
├── hierarchical/               # Hierarchical COLMAP output
│   └── ...
├── refined/                    # Refined GLOMAP output
│   └── ...
└── comparison_report.md        # Comparison report
```

### Use Cases

**When to Use Multi-Comparison:**
- **Pipeline Selection**: Determine best pipeline for your specific dataset
- **Performance Benchmarking**: Compare speed vs quality tradeoffs
- **Configuration Tuning**: Test different settings across pipelines
- **Research & Development**: Systematic evaluation of SFM methods
- **Production Planning**: Make informed decisions for deployment

**Example Workflow:**
1. Run comparison on a representative subset of your data
2. Review the report to understand speed/quality tradeoffs
3. Select the best pipeline for your requirements
4. Process your full dataset with the chosen pipeline

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

