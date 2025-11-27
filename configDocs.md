# Configuration Documentation

## Overview

This document provides comprehensive documentation for all configuration parameters used in the COLMAP and GLOMAP pipelines. Configuration files are in INI format, organized into sections that correspond to different stages and components of the reconstruction pipeline.

## Configuration File Structure

INI files consist of:
- **Global parameters**: Placed at the top before any section headers or in `[DEFAULT]` section
- **Section headers**: Enclosed in square brackets `[SectionName]`
- **Parameters**: Key-value pairs in the format `parameter=value`

```ini
# Global parameters
log_to_stderr=true
log_level=0

[SectionName]
parameter1=value1
parameter2=value2
```

---

## COLMAP Configuration (`defaultColMap.ini`)

### Global Parameters

#### `log_to_stderr`
- **Type**: Boolean (true/false)
- **Default**: false
- **Description**: When enabled, logs are written to standard error instead of a separate log file
- **When to modify**: Enable for debugging or when running in containers

#### `log_level`
- **Type**: Integer (0-4)
- **Default**: 2
- **Values**: 0 (TRACE), 1 (DEBUG), 2 (INFO), 3 (WARNING), 4 (ERROR)
- **Description**: Controls verbosity of logging output
- **When to modify**: Set to 0 or 1 for detailed debugging, 3 or 4 for minimal output

#### `default_random_seed`
- **Type**: Integer
- **Default**: 0
- **Description**: Seed for random number generation to ensure reproducibility
- **When to modify**: Change to get different random sampling behavior or set to -1 for true randomness

---

### [ImageReader]

Controls how images are loaded and camera models are assigned.

#### `single_camera`
- **Type**: Boolean
- **Default**: false
- **Description**: Assumes all images come from the same camera with identical intrinsics
- **When to modify**: Set to true for datasets captured with a single camera without zoom changes

#### `single_camera_per_folder`
- **Type**: Boolean
- **Default**: false
- **Description**: Assumes each subdirectory contains images from the same camera
- **When to modify**: Enable when organizing images by camera in separate folders

#### `single_camera_per_image`
- **Type**: Boolean
- **Default**: false
- **Description**: Treats each image as having unique camera intrinsics
- **When to modify**: Enable for datasets with varying cameras or significant zoom changes

#### `existing_camera_id`
- **Type**: Integer
- **Default**: -1
- **Description**: Forces use of an existing camera ID from the database
- **When to modify**: Set to specific camera ID when re-processing with known camera parameters

#### `default_focal_length_factor`
- **Type**: Float
- **Default**: 1.2
- **Description**: Multiplier for estimating focal length from image dimensions when EXIF data is missing
- **When to modify**: Adjust based on typical field of view (lower for wide-angle, higher for telephoto)

#### `mask_path`
- **Type**: String (path)
- **Default**: "" (empty)
- **Description**: Path to image masks to exclude regions from feature detection
- **When to modify**: Provide path when you want to ignore specific image regions (e.g., sky, moving objects)

#### `camera_model`
- **Type**: String
- **Default**: SIMPLE_RADIAL
- **Options**: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, THIN_PRISM_FISHEYE
- **Description**: Camera distortion model to use
- **When to modify**: 
  - Use SIMPLE_RADIAL for most cameras
  - Use OPENCV or FULL_OPENCV for high-accuracy calibration
  - Use FISHEYE variants for wide-angle/fisheye lenses
  - Use SIMPLE_PINHOLE or PINHOLE for pre-calibrated/undistorted images

#### `camera_params`
- **Type**: String (comma-separated floats)
- **Default**: "" (empty)
- **Description**: Manual camera parameters override (focal length, principal point, distortion)
- **When to modify**: Provide when you have pre-calibrated camera parameters

---

### [FeatureExtraction]

Controls the feature detection process.

#### `use_gpu`
- **Type**: Boolean
- **Default**: true
- **Description**: Use GPU acceleration for feature extraction
- **When to modify**: Disable if GPU is unavailable or causing issues

#### `num_threads`
- **Type**: Integer
- **Default**: -1
- **Description**: Number of threads for CPU feature extraction (-1 uses all available cores)
- **When to modify**: Reduce if you want to limit CPU usage

#### `type`
- **Type**: String
- **Default**: SIFT
- **Options**: SIFT
- **Description**: Feature detector type
- **When to modify**: Currently only SIFT is supported

#### `gpu_index`
- **Type**: Integer
- **Default**: -1
- **Description**: GPU device index to use (-1 auto-selects)
- **When to modify**: Set to specific GPU index in multi-GPU systems

---

### [SiftExtraction]

SIFT-specific feature extraction parameters.

#### `estimate_affine_shape`
- **Type**: Boolean
- **Default**: false
- **Description**: Estimates affine shape for each feature
- **When to modify**: Enable for datasets with significant viewpoint changes or scale variations

#### `upright`
- **Type**: Boolean
- **Default**: false
- **Description**: Extracts features without rotation invariance
- **When to modify**: Enable when images have consistent orientation (e.g., nadir drone imagery)

#### `domain_size_pooling`
- **Type**: Boolean
- **Default**: false
- **Description**: Uses domain-size pooling for scale-space construction
- **When to modify**: Enable for improved performance on certain datasets

#### `max_image_size`
- **Type**: Integer
- **Default**: 3200
- **Description**: Maximum dimension for image resizing before feature extraction
- **When to modify**: Increase for high-resolution imagery, decrease for faster processing

#### `max_num_features`
- **Type**: Integer
- **Default**: 8192
- **Description**: Maximum number of features to extract per image
- **When to modify**: Increase for highly textured scenes, decrease for faster matching

#### `first_octave`
- **Type**: Integer
- **Default**: -1
- **Description**: First octave in SIFT scale-space (-1 for upsampling)
- **When to modify**: Set to 0 to disable upsampling for faster processing

#### `num_octaves`
- **Type**: Integer
- **Default**: 4
- **Description**: Number of octaves in scale-space
- **When to modify**: Increase for better scale invariance, decrease for speed

#### `octave_resolution`
- **Type**: Integer
- **Default**: 3
- **Description**: Number of scales per octave
- **When to modify**: Increase for finer scale sampling

#### `peak_threshold`
- **Type**: Float
- **Default**: 0.0066
- **Description**: Minimum contrast threshold for feature detection
- **When to modify**: Decrease for more features (low-texture), increase for fewer features (high-texture)

#### `edge_threshold`
- **Type**: Float
- **Default**: 10.0
- **Description**: Maximum edge response threshold
- **When to modify**: Lower to accept more edge-like features, raise to be more selective

---

### [FeatureMatching]

Controls feature matching between image pairs.

#### `use_gpu`
- **Type**: Boolean
- **Default**: true
- **Description**: Use GPU acceleration for feature matching
- **When to modify**: Disable if GPU unavailable

#### `guided_matching`
- **Type**: Boolean
- **Default**: false
- **Description**: Performs additional guided matching after geometric verification
- **When to modify**: Enable for improved completeness at cost of speed

#### `num_threads`
- **Type**: Integer
- **Default**: -1
- **Description**: Number of threads for matching (-1 uses all cores)
- **When to modify**: Adjust to control CPU usage

#### `max_num_matches`
- **Type**: Integer
- **Default**: 32768
- **Description**: Maximum number of matches per image pair
- **When to modify**: Increase for highly overlapping images

#### `type`
- **Type**: String
- **Default**: SIFT
- **Description**: Feature matcher type
- **When to modify**: Must match feature extraction type

#### `gpu_index`
- **Type**: Integer
- **Default**: -1
- **Description**: GPU device to use for matching
- **When to modify**: Specify GPU in multi-GPU systems

---

### [SiftMatching]

SIFT-specific matching parameters.

#### `cross_check`
- **Type**: Boolean
- **Default**: true
- **Description**: Enables cross-checking of matches (mutual nearest neighbors)
- **When to modify**: Keep enabled for higher quality matches

#### `max_ratio`
- **Type**: Float
- **Default**: 0.8
- **Description**: Lowe's ratio test threshold (distance ratio of best to second-best match)
- **When to modify**: Lower for more conservative matching (0.7), raise for more matches (0.9)

#### `max_distance`
- **Type**: Float
- **Default**: 0.7
- **Description**: Maximum descriptor distance for valid matches
- **When to modify**: Adjust based on matching quality

---

### [TwoViewGeometry]

Geometric verification between image pairs.

#### `min_num_inliers`
- **Type**: Integer
- **Default**: 15
- **Description**: Minimum number of inlier matches required
- **When to modify**: Lower for small overlap scenarios, raise for higher confidence

#### `max_error`
- **Type**: Float (pixels)
- **Default**: 4.0
- **Description**: Maximum reprojection error for inliers
- **When to modify**: Increase for lower-quality images, decrease for high-precision requirements

#### `confidence`
- **Type**: Float (0-1)
- **Default**: 0.999
- **Description**: RANSAC confidence level
- **When to modify**: Lower for faster processing, raise for more exhaustive search

#### `min_inlier_ratio`
- **Type**: Float (0-1)
- **Default**: 0.25
- **Description**: Minimum ratio of inliers to total matches
- **When to modify**: Lower for challenging scenarios with many outliers

#### `max_num_trials`
- **Type**: Integer
- **Default**: 10000
- **Description**: Maximum RANSAC iterations
- **When to modify**: Increase for challenging scenarios, decrease for speed

#### `detect_watermark`
- **Type**: Boolean
- **Default**: false
- **Description**: Attempts to detect and filter watermark features
- **When to modify**: Enable if images contain watermarks

---

### [ExhaustiveMatching]

Parameters for exhaustive matching (all image pairs).

#### `block_size`
- **Type**: Integer
- **Default**: 50
- **Description**: Number of image pairs to process in each batch
- **When to modify**: Adjust based on memory constraints

---

### [SequentialMatching]

Parameters for sequential matching (ordered image sequences).

#### `overlap`
- **Type**: Integer
- **Default**: 10
- **Description**: Number of consecutive images to match
- **When to modify**: Increase for videos or ordered sequences, decrease for random collections

#### `quadratic_overlap`
- **Type**: Boolean
- **Default**: true
- **Description**: Matches images quadratically within the overlap window
- **When to modify**: Enable for better coverage in sequential datasets

#### `loop_detection`
- **Type**: Boolean
- **Default**: false
- **Description**: Enables loop closure detection using visual vocabulary
- **When to modify**: Enable for circular paths or revisited locations

#### `vocab_tree_path`
- **Type**: String (path)
- **Default**: "" (downloads automatically)
- **Description**: Path to vocabulary tree for loop detection
- **When to modify**: Provide local path to avoid downloading

---

### [VocabTreeMatching]

Vocabulary tree-based matching for large-scale datasets.

#### `num_images`
- **Type**: Integer
- **Default**: 100
- **Description**: Number of most similar images to match per query image
- **When to modify**: Increase for larger datasets or better coverage

#### `vocab_tree_path`
- **Type**: String (path)
- **Default**: "" (downloads automatically)
- **Description**: Path to vocabulary tree file
- **When to modify**: Provide local path for offline use

#### `num_nearest_neighbors`
- **Type**: Integer
- **Default**: 5
- **Description**: Number of nearest neighbors to retrieve
- **When to modify**: Increase for better coverage

---

### [SpatialMatching]

GPS/spatial location-based matching.

#### `is_gps`
- **Type**: Boolean
- **Default**: true
- **Description**: Indicates if spatial coordinates are GPS (vs arbitrary coordinates)
- **When to modify**: Set to false for non-GPS spatial matching

#### `max_distance`
- **Type**: Float (meters)
- **Default**: 100
- **Description**: Maximum spatial distance for image pair matching
- **When to modify**: Adjust based on image spacing in your dataset

#### `max_num_neighbors`
- **Type**: Integer
- **Default**: 50
- **Description**: Maximum number of spatial neighbors to match
- **When to modify**: Adjust based on image density

---

### [Mapper]

Incremental reconstruction parameters.

#### `multiple_models`
- **Type**: Boolean
- **Default**: true
- **Description**: Allows reconstruction of multiple disconnected models
- **When to modify**: Disable to force single model reconstruction

#### `extract_colors`
- **Type**: Boolean
- **Default**: true
- **Description**: Extracts colors for 3D points from images
- **When to modify**: Disable for faster reconstruction if colors not needed

#### `min_num_matches`
- **Type**: Integer
- **Default**: 15
- **Description**: Minimum number of matches for valid image pair
- **When to modify**: Lower for challenging datasets, raise for stricter filtering

#### `min_model_size`
- **Type**: Integer
- **Default**: 10
- **Description**: Minimum number of images in a reconstructed model
- **When to modify**: Lower to keep smaller model fragments

#### `max_num_models`
- **Type**: Integer
- **Default**: 50
- **Description**: Maximum number of models to reconstruct
- **When to modify**: Increase for highly disconnected datasets

#### `ba_use_gpu`
- **Type**: Boolean
- **Default**: false
- **Description**: Use GPU for bundle adjustment within mapping
- **When to modify**: Enable for faster BA if GPU has sufficient memory

#### `ba_global_max_num_iterations`
- **Type**: Integer
- **Default**: 50
- **Description**: Maximum iterations for global bundle adjustment
- **When to modify**: Increase for better convergence, decrease for speed

#### `ba_local_max_num_iterations`
- **Type**: Integer
- **Default**: 25
- **Description**: Maximum iterations for local bundle adjustment
- **When to modify**: Adjust based on desired accuracy vs speed tradeoff

#### `init_min_num_inliers`
- **Type**: Integer
- **Default**: 100
- **Description**: Minimum inliers required for initial image pair
- **When to modify**: Lower for difficult initialization scenarios

#### `abs_pose_max_error`
- **Type**: Float (pixels)
- **Default**: 12.0
- **Description**: Maximum reprojection error for absolute pose estimation
- **When to modify**: Adjust based on image quality and desired precision

#### `filter_max_reproj_error`
- **Type**: Float (pixels)
- **Default**: 4.0
- **Description**: Maximum reprojection error for filtering observations
- **When to modify**: Increase for lower-quality matches, decrease for stricter filtering

---

## GLOMAP Configuration (`defaultGloMap.ini`)

GLOMAP uses a global optimization approach for faster reconstruction compared to COLMAP's incremental approach.

### Global Parameters

#### `log_to_stderr`
- **Type**: Boolean
- **Default**: true
- **Description**: Directs log output to standard error
- **When to modify**: Same as COLMAP

#### `log_level`
- **Type**: Integer (0-4)
- **Default**: 0
- **Description**: Logging verbosity level
- **When to modify**: Same as COLMAP

---

### [Mapper]

Core GLOMAP mapper settings.

#### `constraint_type`
- **Type**: String
- **Default**: POINTS_AND_CAMERAS
- **Options**: ONLY_POINTS, ONLY_CAMERAS, POINTS_AND_CAMERAS_BALANCED, POINTS_AND_CAMERAS
- **Description**: Type of constraints used in global optimization
- **When to modify**:
  - ONLY_POINTS: Faster, point-based constraints only
  - ONLY_CAMERAS: Camera-based constraints only
  - POINTS_AND_CAMERAS_BALANCED: Balanced weighting
  - POINTS_AND_CAMERAS: Full constraints (best quality)

#### `output_format`
- **Type**: String
- **Default**: bin
- **Options**: bin, txt
- **Description**: Output format for reconstruction
- **When to modify**: Use txt for human-readable output or debugging

#### `ba_iteration_num`
- **Type**: Integer
- **Default**: 3
- **Description**: Number of bundle adjustment iterations
- **When to modify**: Increase for better refinement, decrease for speed

#### `retriangulation_iteration_num`
- **Type**: Integer
- **Default**: 1
- **Description**: Number of retriangulation passes
- **When to modify**: Increase for better point accuracy

#### `skip_preprocessing`
- **Type**: Boolean
- **Default**: false
- **Description**: Skips initial preprocessing steps
- **When to modify**: Enable only if you're certain data is already prepared

#### `skip_view_graph_calibration`
- **Type**: Boolean
- **Default**: false
- **Description**: Skips view graph calibration
- **When to modify**: Enable to skip this stage for faster processing (less accurate)

#### `skip_rotation_averaging`
- **Type**: Boolean
- **Default**: false
- **Description**: Skips rotation averaging step
- **When to modify**: Generally keep enabled; disable only for debugging

#### `skip_global_positioning`
- **Type**: Boolean
- **Default**: false
- **Description**: Skips global positioning optimization
- **When to modify**: Keep false for complete reconstruction

#### `skip_bundle_adjustment`
- **Type**: Boolean
- **Default**: false
- **Description**: Skips bundle adjustment refinement
- **When to modify**: Enable for very fast reconstruction (much lower quality)

#### `skip_retriangulation`
- **Type**: Boolean
- **Default**: false
- **Description**: Skips retriangulation of points
- **When to modify**: Enable for faster processing if point completeness is less critical

#### `skip_pruning`
- **Type**: Boolean
- **Default**: true
- **Description**: Skips pruning of outlier points
- **When to modify**: Disable (set to false) for cleaner final model

---

### [ViewGraphCalib]

View graph calibration thresholds.

#### `thres_lower_ratio`
- **Type**: Float
- **Default**: 0.1
- **Description**: Lower threshold ratio for view graph edge filtering
- **When to modify**: Adjust based on match quality

#### `thres_higher_ratio`
- **Type**: Float
- **Default**: 10.0
- **Description**: Upper threshold ratio for view graph edge filtering
- **When to modify**: Adjust to filter unreliable edges

#### `thres_two_view_error`
- **Type**: Float
- **Default**: 2.0
- **Description**: Maximum two-view geometry error threshold
- **When to modify**: Increase for noisier matches, decrease for stricter filtering

---

### [RelPoseEstimation]

Relative pose estimation parameters.

#### `max_epipolar_error`
- **Type**: Float (pixels)
- **Default**: 1.0
- **Description**: Maximum epipolar error for inlier classification
- **When to modify**: Increase for lower-quality images or matches

---

### [TrackEstablishment]

Parameters for establishing feature tracks across views.

#### `min_num_tracks_per_view`
- **Type**: Integer
- **Default**: -1
- **Description**: Minimum number of tracks per view (-1 for automatic)
- **When to modify**: Set positive value to enforce minimum track density

#### `min_num_view_per_track`
- **Type**: Integer
- **Default**: 3
- **Description**: Minimum number of views observing a track
- **When to modify**: Increase for more robust tracks, decrease to keep more points

#### `max_num_view_per_track`
- **Type**: Integer
- **Default**: 100
- **Description**: Maximum number of views observing a track
- **When to modify**: Adjust based on dataset overlap characteristics

#### `max_num_tracks`
- **Type**: Integer
- **Default**: 10000000
- **Description**: Maximum total number of tracks
- **When to modify**: Reduce for memory-constrained systems

---

### [GlobalPositioning]

Global positioning optimization settings.

#### `use_gpu`
- **Type**: Boolean
- **Default**: true
- **Description**: Use GPU for global positioning optimization
- **When to modify**: Disable if GPU unavailable or insufficient memory

#### `gpu_index`
- **Type**: Integer
- **Default**: -1
- **Description**: GPU device index (-1 auto-selects)
- **When to modify**: Specify GPU in multi-GPU systems

#### `optimize_positions`
- **Type**: Boolean
- **Default**: true
- **Description**: Optimizes camera positions
- **When to modify**: Keep enabled for accurate positioning

#### `optimize_points`
- **Type**: Boolean
- **Default**: true
- **Description**: Optimizes 3D point positions
- **When to modify**: Keep enabled for accurate reconstruction

#### `optimize_scales`
- **Type**: Boolean
- **Default**: true
- **Description**: Optimizes scale parameters
- **When to modify**: Keep enabled unless scale is known

#### `thres_loss_function`
- **Type**: Float
- **Default**: 0.1
- **Description**: Threshold for robust loss function
- **When to modify**: Adjust based on outlier characteristics

#### `max_num_iterations`
- **Type**: Integer
- **Default**: 100
- **Description**: Maximum optimization iterations
- **When to modify**: Increase for better convergence, decrease for speed

---

### [BundleAdjustment]

Bundle adjustment refinement parameters.

#### `use_gpu`
- **Type**: Boolean
- **Default**: true
- **Description**: Use GPU for bundle adjustment
- **When to modify**: Disable if GPU memory insufficient

#### `gpu_index`
- **Type**: Integer
- **Default**: -1
- **Description**: GPU device to use
- **When to modify**: Specify in multi-GPU systems

#### `optimize_rotations`
- **Type**: Boolean
- **Default**: true
- **Description**: Optimizes camera rotations
- **When to modify**: Keep enabled for full refinement

#### `optimize_translation`
- **Type**: Boolean
- **Default**: true
- **Description**: Optimizes camera translations
- **When to modify**: Keep enabled for full refinement

#### `optimize_intrinsics`
- **Type**: Boolean
- **Default**: true
- **Description**: Refines camera intrinsic parameters
- **When to modify**: Disable if using pre-calibrated cameras

#### `optimize_principal_point`
- **Type**: Boolean
- **Default**: false
- **Description**: Refines camera principal point
- **When to modify**: Enable for uncalibrated cameras with uncertain principal point

#### `optimize_points`
- **Type**: Boolean
- **Default**: true
- **Description**: Optimizes 3D point positions
- **When to modify**: Keep enabled for accurate points

#### `thres_loss_function`
- **Type**: Float
- **Default**: 1.0
- **Description**: Robust loss function threshold
- **When to modify**: Adjust based on reprojection error distribution

#### `max_num_iterations`
- **Type**: Integer
- **Default**: 200
- **Description**: Maximum BA iterations
- **When to modify**: Increase for better refinement, decrease for speed

---

### [Triangulation]

Point triangulation parameters.

#### `complete_max_reproj_error`
- **Type**: Float (pixels)
- **Default**: 15.0
- **Description**: Maximum reprojection error for point completion
- **When to modify**: Adjust based on desired point density vs accuracy

#### `merge_max_reproj_error`
- **Type**: Float (pixels)
- **Default**: 15.0
- **Description**: Maximum reprojection error for merging tracks
- **When to modify**: Adjust based on point merging aggressiveness

#### `min_angle`
- **Type**: Float (degrees)
- **Default**: 1.0
- **Description**: Minimum triangulation angle
- **When to modify**: Increase for more robust triangulation (fewer points)

#### `min_num_matches`
- **Type**: Integer
- **Default**: 15
- **Description**: Minimum matches required for triangulation
- **When to modify**: Lower for sparse scenarios, raise for stricter filtering

---

### [Thresholds]

Various threshold parameters for reconstruction.

#### `max_angle_error`
- **Type**: Float (degrees)
- **Default**: 1.0
- **Description**: Maximum angular error tolerance
- **When to modify**: Adjust based on rotation estimation accuracy

#### `max_reprojection_error`
- **Type**: Float (pixels)
- **Default**: 0.01
- **Description**: Maximum reprojection error for strict filtering
- **When to modify**: Increase for more lenient filtering

#### `min_triangulation_angle`
- **Type**: Float (degrees)
- **Default**: 1.0
- **Description**: Minimum angle between rays for triangulation
- **When to modify**: Increase for more robust points

#### `max_epipolar_error_E`
- **Type**: Float (pixels)
- **Default**: 1.0
- **Description**: Maximum epipolar error for essential matrix
- **When to modify**: Adjust based on match quality

#### `max_epipolar_error_F`
- **Type**: Float (pixels)
- **Default**: 4.0
- **Description**: Maximum epipolar error for fundamental matrix
- **When to modify**: Adjust based on match quality

#### `max_epipolar_error_H`
- **Type**: Float (pixels)
- **Default**: 4.0
- **Description**: Maximum epipolar error for homography
- **When to modify**: Adjust for planar scene handling

#### `min_inlier_num`
- **Type**: Integer
- **Default**: 30
- **Description**: Minimum number of inliers
- **When to modify**: Lower for sparse matching scenarios

#### `min_inlier_ratio`
- **Type**: Float (0-1)
- **Default**: 0.25
- **Description**: Minimum inlier ratio
- **When to modify**: Lower for high-outlier scenarios

#### `max_rotation_error`
- **Type**: Float (degrees)
- **Default**: 10.0
- **Description**: Maximum rotation error tolerance
- **When to modify**: Adjust based on expected rotation accuracy

---

## Configuration Tips

### General Best Practices

1. **Start with defaults**: The default configurations work well for most scenarios
2. **Adjust incrementally**: Make small changes and observe their effects
3. **Balance speed vs quality**: Faster settings usually reduce reconstruction quality
4. **GPU acceleration**: Enable GPU options when available for significant speedup

### Common Scenarios

#### High-Quality Professional Imagery
- Increase `max_num_features` (e.g., 16384)
- Decrease `peak_threshold` (e.g., 0.004)
- Enable `guided_matching`
- Increase BA iterations

#### Low-Quality or Challenging Imagery
- Decrease `max_num_features` (e.g., 4096)
- Increase `peak_threshold` (e.g., 0.01)
- Lower geometric thresholds (e.g., `min_num_inliers=10`)
- Increase `max_error` tolerance

#### Large-Scale Datasets (1000+ images)
- Use `vocab_tree_matcher` instead of exhaustive
- Enable `multiple_models=false` if single model desired
- Reduce `max_num_features` per image
- Consider GLOMAP for faster processing

#### Aerial/Drone Imagery
- Enable `upright=true` for nadir views
- Use `sequential_matcher` with high overlap (e.g., 20)
- Consider `single_camera=true` if same drone
- Adjust `default_focal_length_factor` for your camera

#### Close-Range/Indoor Scenes
- Use `exhaustive_matcher`
- Increase `max_num_features`
- Use SIMPLE_RADIAL or OPENCV camera model
- Enable `guided_matching`

### COLMAP vs GLOMAP

**Use COLMAP when:**
- Maximum accuracy is required
- Working with challenging, unordered image collections
- Need incremental reconstruction capability
- Have complex camera configurations

**Use GLOMAP when:**
- Speed is priority
- Working with well-connected image sets
- Global optimization approach is suitable
- Have GPU available for global positioning

### Troubleshooting

**Few images registered:**
- Lower `min_num_matches`
- Decrease geometric thresholds
- Increase `max_num_features`
- Check image overlap

**Too slow:**
- Reduce `max_num_features`
- Use faster matcher (sequential vs exhaustive)
- Enable GPU acceleration
- Skip optional stages
- Consider GLOMAP

**Poor reconstruction quality:**
- Increase `max_num_features`
- Lower `peak_threshold`
- Enable `guided_matching`
- Increase BA iterations
- Use stricter geometric thresholds

**High memory usage:**
- Reduce `max_num_features`
- Reduce `max_num_matches`
- Disable GPU BA if insufficient GPU memory
- Process in smaller chunks

