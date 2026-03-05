#!/usr/bin/env python3
"""
hloc Pipeline Module
Wraps hloc functionality for feature extraction, retrieval, and matching.
Integrates with COLMAP/GLOMAP pipelines for 3D reconstruction.
"""

import configparser
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pycolmap

# Add hloc to path
HLOC_PATH = Path(__file__).parent / "hloc"
sys.path.insert(0, str(HLOC_PATH))


class HlocConfigParser:
    """Parse hloc INI configuration file."""

    def __init__(self, ini_path: Path):
        self.config = configparser.RawConfigParser()
        self.config.optionxform = str  # Preserve case sensitivity
        self.config.read(ini_path)
        self.ini_path = ini_path

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get a value from config with fallback."""
        try:
            value = self.config.get(section, key)
            if value == '' or value is None:
                return fallback
            return value
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get an integer value from config."""
        try:
            value = self.config.get(section, key)
            if value == '' or value is None:
                return fallback
            return int(value)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback

    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get a float value from config."""
        try:
            value = self.config.get(section, key)
            if value == '' or value is None:
                return fallback
            return float(value)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get a boolean value from config."""
        try:
            value = self.config.get(section, key)
            if value == '' or value is None:
                return fallback
            return value.lower() in ('true', 'yes', '1', 'on')
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def get_feature_config(self) -> Dict:
        """Get local feature extraction configuration."""
        feature_type = self.get('LocalFeatures', 'feature_type', 'sift')
        max_keypoints = self.getint('LocalFeatures', 'max_keypoints', 8192)
        resize_max = self.getint('LocalFeatures', 'resize_max', 1600)
        grayscale = self.getboolean('LocalFeatures', 'grayscale', True)

        # Build configuration based on feature type
        if feature_type == 'sift':
            conf = {
                'output': f'feats-sift-n{max_keypoints}',
                'model': {
                    'name': 'dog',
                    'max_keypoints': max_keypoints,
                    'options': {
                        'first_octave': 0,
                        'peak_threshold': 0.00667,  # Default COLMAP value
                        'max_num_features': max_keypoints,
                    }
                },
                'preprocessing': {
                    'grayscale': grayscale,
                    'resize_max': resize_max,
                },
            }
        elif feature_type == 'r2d2':
            conf = {
                'output': f'feats-r2d2-n{max_keypoints}',
                'model': {
                    'name': 'r2d2',
                    'max_keypoints': max_keypoints,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': resize_max,
                },
            }
        elif feature_type == 'superpoint':
            conf = {
                'output': f'feats-superpoint-n{max_keypoints}',
                'model': {
                    'name': 'superpoint',
                    'nms_radius': 3,
                    'max_keypoints': max_keypoints,
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': resize_max,
                },
            }
        elif feature_type == 'disk':
            conf = {
                'output': f'feats-disk-n{max_keypoints}',
                'model': {
                    'name': 'disk',
                    'max_keypoints': max_keypoints,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': resize_max,
                },
            }
        elif feature_type == 'aliked':
            conf = {
                'output': f'feats-aliked-n{max_keypoints}',
                'model': {
                    'name': 'aliked',
                    'model_name': 'aliked-n16',
                    'max_keypoints': max_keypoints,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': resize_max,
                },
            }
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        return conf

    def get_retrieval_config(self) -> Dict:
        """Get global descriptor / retrieval configuration."""
        network = self.get('GlobalDescriptors', 'network', 'netvlad')
        resize_max = self.getint('GlobalDescriptors', 'resize_max', 1024)

        # Build configuration based on network type
        conf = {
            'output': f'global-feats-{network}',
            'model': {'name': network},
            'preprocessing': {'resize_max': resize_max},
        }

        return conf

    def get_pair_generation_config(self) -> Dict:
        """Get pair generation configuration."""
        return {
            'num_matched': self.getint('PairGeneration', 'num_matched', 50),
            'min_score': self.getfloat('PairGeneration', 'min_score', 0.0) or None,
        }

    def get_matching_config(self) -> Dict:
        """Get feature matching configuration."""
        matcher = self.get('FeatureMatching', 'matcher', 'NN-ratio')
        ratio_threshold = self.getfloat('FeatureMatching', 'ratio_threshold', 0.8)
        cross_check = self.getboolean('FeatureMatching', 'cross_check', True)

        # Build configuration based on matcher type
        if matcher == 'NN-ratio':
            conf = {
                'output': f'matches-NN-ratio-{ratio_threshold}',
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': cross_check,
                    'ratio_threshold': ratio_threshold,
                },
            }
        elif matcher == 'NN-mutual':
            conf = {
                'output': 'matches-NN-mutual',
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                },
            }
        elif matcher == 'adalam':
            conf = {
                'output': 'matches-adalam',
                'model': {'name': 'adalam'},
            }
        else:
            raise ValueError(f"Unknown matcher type: {matcher}")

        return conf


class HlocPipeline:
    """Wraps hloc for feature extraction, retrieval, and matching."""

    # Image extensions to search for (covers COLMAP-supported formats)
    IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.pgm', '.ppm', '.webp',
    }

    def __init__(
        self,
        config: HlocConfigParser,
        image_dir: Path,
        output_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.hloc_dir = self.output_dir / "hloc"
        self.hloc_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(__name__)

        # Paths that will be set during pipeline execution
        self.local_features_path: Optional[Path] = None
        self.global_features_path: Optional[Path] = None
        self.pairs_path: Optional[Path] = None
        self.matches_path: Optional[Path] = None

        # Cached image list (relative paths from image_dir)
        self._image_names: Optional[List[str]] = None

        # Restore paths from any existing files on disk (enables resume)
        self._restore_existing_paths()

    def log(self, message: str, level: str = 'info'):
        """Log a message, compatible with pipeline loggers that expect a 'stage' field."""
        extra = {'stage': 'HLOC'}
        getattr(self.logger, level)(message, extra=extra)

    def _restore_existing_paths(self):
        """Restore HDF5/pairs paths from existing files in hloc_dir.

        This enables resuming a pipeline from a later stage without
        re-running earlier stages. Paths are set only if the files exist.
        """
        if not self.hloc_dir.exists():
            return

        # Restore local features (exported COLMAP SIFT or hloc-extracted)
        for h5 in self.hloc_dir.glob("feats-*.h5"):
            self.local_features_path = h5
            break  # Use the first match (typically only one)

        # Restore global features
        for h5 in self.hloc_dir.glob("global-feats-*.h5"):
            self.global_features_path = h5
            break

        # Restore pairs
        pairs = self.hloc_dir / "pairs.txt"
        if pairs.exists():
            self.pairs_path = pairs

        # Restore matches
        for h5 in self.hloc_dir.glob("matches-*.h5"):
            self.matches_path = h5
            break

    def _discover_images(self) -> List[str]:
        """Recursively discover images in image_dir, returning relative paths.

        Searches for common image extensions (case-insensitive) in all
        subdirectories. Results are cached after first call.
        """
        if self._image_names is not None:
            return self._image_names

        names = []
        for p in sorted(self.image_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS:
                names.append(p.relative_to(self.image_dir).as_posix())

        if not names:
            raise ValueError(
                f"Could not find any images in {self.image_dir} "
                f"(searched recursively for {', '.join(sorted(self.IMAGE_EXTENSIONS))})"
            )

        self._image_names = names
        self.log(f"Discovered {len(names)} images in {self.image_dir}")
        return self._image_names

    def extract_local_features(self, overwrite: bool = False) -> Path:
        """Extract local features (SIFT/R2D2/etc) using hloc."""
        from hloc import extract_features

        conf = self.config.get_feature_config()
        feature_type = self.config.get('LocalFeatures', 'feature_type', 'sift')

        self.log(f"Extracting {feature_type} local features...")
        self.log(f"  Max keypoints: {conf['model'].get('max_keypoints', 'unlimited')}")
        self.log(f"  Resize max: {conf['preprocessing'].get('resize_max', 'none')}")

        image_list = self._discover_images()

        self.local_features_path = extract_features.main(
            conf=conf,
            image_dir=self.image_dir,
            export_dir=self.hloc_dir,
            image_list=image_list,
            overwrite=overwrite,
        )

        self.log(f"Local features saved to: {self.local_features_path}")
        return self.local_features_path

    def extract_global_descriptors(self, overwrite: bool = False) -> Path:
        """Extract global descriptors for retrieval."""
        from hloc import extract_features

        conf = self.config.get_retrieval_config()
        network = self.config.get('GlobalDescriptors', 'network', 'netvlad')

        self.log(f"Extracting {network} global descriptors...")

        image_list = self._discover_images()

        self.global_features_path = extract_features.main(
            conf=conf,
            image_dir=self.image_dir,
            export_dir=self.hloc_dir,
            image_list=image_list,
            overwrite=overwrite,
        )

        self.log(f"Global descriptors saved to: {self.global_features_path}")
        return self.global_features_path

    def generate_pairs(self) -> Path:
        """Generate image pairs from retrieval."""
        from hloc.pairs_from_retrieval import main as pairs_from_retrieval

        if self.global_features_path is None:
            raise RuntimeError("Global descriptors must be extracted first")

        pair_config = self.config.get_pair_generation_config()
        num_matched = pair_config['num_matched']

        self.log(f"Generating pairs with top-{num_matched} retrieval...")

        self.pairs_path = self.hloc_dir / "pairs.txt"

        pairs_from_retrieval(
            descriptors=self.global_features_path,
            output=self.pairs_path,
            num_matched=num_matched,
        )

        # Count pairs
        with open(self.pairs_path, 'r') as f:
            num_pairs = sum(1 for line in f if line.strip())

        self.log(f"Generated {num_pairs} pairs, saved to: {self.pairs_path}")
        return self.pairs_path

    def match_features(self, overwrite: bool = False) -> Path:
        """Match features for retrieved pairs."""
        from hloc import match_features

        if self.local_features_path is None:
            raise RuntimeError("Local features must be extracted first")
        if self.pairs_path is None:
            raise RuntimeError("Pairs must be generated first")

        conf = self.config.get_matching_config()
        matcher = self.config.get('FeatureMatching', 'matcher', 'NN-ratio')

        self.log(f"Matching features with {matcher}...")

        # When features is a Path, hloc requires matches to also be an explicit Path
        matches_path = self.hloc_dir / (conf['output'] + ".h5")

        self.matches_path = match_features.main(
            conf=conf,
            pairs=self.pairs_path,
            features=self.local_features_path,
            matches=matches_path,
            export_dir=self.hloc_dir,
            overwrite=overwrite,
        )

        self.log(f"Matches saved to: {self.matches_path}")
        return self.matches_path

    def import_to_colmap_db(
        self,
        database_path: Path,
        camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
        image_options: Optional[Dict] = None,
    ) -> bool:
        """Import hloc features and matches into COLMAP database."""
        from hloc.triangulation import import_features, import_matches
        from hloc.reconstruction import create_empty_db, import_images, get_image_ids

        if self.local_features_path is None:
            raise RuntimeError("Local features must be extracted first")
        if self.pairs_path is None:
            raise RuntimeError("Pairs must be generated first")
        if self.matches_path is None:
            raise RuntimeError("Matches must be computed first")

        self.log(f"Importing hloc data to COLMAP database: {database_path}")

        # Create fresh database
        create_empty_db(database_path)
        self.log("  Created empty database")

        # Import images (pass explicit list for subdirectory support)
        image_list = self._discover_images()
        import_images(
            self.image_dir,
            database_path,
            camera_mode,
            image_list=image_list,
            options=image_options,
        )
        self.log("  Imported images")

        # Get image IDs
        image_ids = get_image_ids(database_path)
        self.log(f"  Found {len(image_ids)} images")

        # Import features and matches
        with pycolmap.Database.open(database_path) as db:
            import_features(image_ids, db, self.local_features_path)
            self.log("  Imported features")

            import_matches(image_ids, db, self.pairs_path, self.matches_path)
            self.log("  Imported matches")

        return True

    def run_geometric_verification(self, database_path: Path) -> bool:
        """Run geometric verification on matches."""
        if self.pairs_path is None:
            raise RuntimeError("Pairs must be generated first")

        self.log("Running geometric verification...")

        pycolmap.verify_matches(
            database_path,
            self.pairs_path,
            options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)),
        )

        self.log("Geometric verification completed")
        return True

    def export_features_from_colmap_db(
        self,
        database_path: Path,
        feature_output_name: str = "feats-colmap-sift",
        overwrite: bool = False,
    ) -> Path:
        """Export features from a COLMAP database to hloc HDF5 format.

        Reads keypoints and SIFT descriptors from an existing COLMAP database
        (populated by colmap feature_extractor) and writes them to an HDF5 file
        in hloc's expected format. This enables using hloc's matching pipeline
        on COLMAP-extracted features.

        Conversion applied:
        - Keypoints: subtract 0.5 (reverse COLMAP's origin convention)
        - Descriptors: uint8 -> float32 -> RootSIFT (L1-norm, sqrt, L2-norm) -> transpose to (D, N)
        - Image size: read from camera width/height
        """
        import h5py
        from tqdm import tqdm

        feature_path = self.hloc_dir / (feature_output_name + ".h5")

        if feature_path.exists() and not overwrite:
            # Validate the existing file: image_size must be integer dtype
            # (earlier versions wrote float32 which causes torch.empty() errors)
            try:
                with h5py.File(str(feature_path), "r") as hf:
                    for key in hf:
                        if "image_size" in hf[key]:
                            if not np.issubdtype(hf[key]["image_size"].dtype, np.integer):
                                self.log(
                                    f"Stale feature file detected (image_size dtype="
                                    f"{hf[key]['image_size'].dtype}), re-exporting..."
                                )
                                overwrite = True
                            break
            except Exception:
                overwrite = True

            if not overwrite:
                self.log(f"Feature file already exists: {feature_path}, skipping export")
                self.local_features_path = feature_path
                return feature_path

        if feature_path.exists():
            feature_path.unlink()

        EPS = 1e-6

        with pycolmap.Database.open(database_path) as db:
            images = db.read_all_images()
            cameras = {cam.camera_id: cam for cam in db.read_all_cameras()}

            self.log(f"Exporting features for {len(images)} images from COLMAP database")

            with h5py.File(str(feature_path), "w", libver="latest") as hfile:
                for image in tqdm(images, desc="Exporting features"):
                    # Normalize image name for cross-platform consistency
                    image_name = Path(image.name).as_posix()
                    camera = cameras[image.camera_id]

                    keypoints_raw = db.read_keypoints(image.image_id)
                    descriptors_raw = db.read_descriptors(image.image_id)

                    if keypoints_raw is None or len(keypoints_raw) == 0:
                        self.log(f"  Warning: No keypoints for {image_name}, skipping")
                        continue

                    # Extract (x, y) and reverse COLMAP's +0.5 origin offset
                    keypoints = keypoints_raw[:, :2].astype(np.float32) - 0.5

                    # Extract scales and orientations from COLMAP keypoints
                    # COLMAP stores (N, 6): x, y, a11, a12, a21, a22 (affine shape)
                    # Scale = sqrt(a11² + a21²), orientation = atan2(a21, a11)
                    # If (N, 4): x, y, scale, orientation (radians) directly
                    # If (N, 2): no scale/orientation available
                    n_cols = keypoints_raw.shape[1] if keypoints_raw.ndim > 1 else 0
                    if n_cols >= 6:
                        a11 = keypoints_raw[:, 2].astype(np.float32)
                        a21 = keypoints_raw[:, 4].astype(np.float32)
                        scales = np.sqrt(a11**2 + a21**2)
                        oris = np.rad2deg(np.arctan2(a21, a11))
                    elif n_cols >= 4:
                        scales = keypoints_raw[:, 2].astype(np.float32)
                        oris = np.rad2deg(keypoints_raw[:, 3].astype(np.float32))
                    else:
                        scales = np.ones(len(keypoints), dtype=np.float32)
                        oris = np.zeros(len(keypoints), dtype=np.float32)

                    # Convert descriptors: uint8 -> float32 -> RootSIFT -> transpose
                    # RootSIFT: L1-normalize, sqrt, L2-normalize (matches hloc DOG default)
                    desc_f32 = descriptors_raw.astype(np.float32)
                    l1_norms = np.linalg.norm(desc_f32, ord=1, axis=1, keepdims=True)
                    desc_f32 = desc_f32 / (l1_norms + EPS)
                    desc_f32 = np.sqrt(np.clip(desc_f32, a_min=EPS, a_max=None))
                    l2_norms = np.linalg.norm(desc_f32, axis=1, keepdims=True)
                    desc_f32 = desc_f32 / (l2_norms + EPS)
                    # Transpose to (D, N) — hloc convention
                    desc_transposed = desc_f32.T

                    image_size = np.array([camera.width, camera.height], dtype=np.int32)

                    # Write to HDF5 as float16 (hloc default for storage efficiency)
                    grp = hfile.create_group(image_name)
                    grp.create_dataset("keypoints", data=keypoints.astype(np.float16))
                    grp.create_dataset("descriptors", data=desc_transposed.astype(np.float16))
                    grp.create_dataset("scales", data=scales.astype(np.float16))
                    grp.create_dataset("oris", data=oris.astype(np.float16))
                    grp.create_dataset("image_size", data=image_size)
                    grp.create_dataset("scores", data=np.zeros(len(keypoints), dtype=np.float16))

        self.local_features_path = feature_path
        self.log(f"Exported features to: {feature_path}")
        return feature_path

    def import_matches_to_colmap_db(self, database_path: Path) -> bool:
        """Import hloc matches into an existing COLMAP database.

        Unlike import_to_colmap_db(), this does NOT create a new database or
        import features. It only writes match indices into the existing DB
        that already has images and features from COLMAP feature extraction.
        """
        from hloc.triangulation import import_matches
        from hloc.reconstruction import get_image_ids

        if self.pairs_path is None:
            raise RuntimeError("Pairs must be generated first")
        if self.matches_path is None:
            raise RuntimeError("Matches must be computed first")

        image_ids = get_image_ids(database_path)
        self.log(f"Importing matches for {len(image_ids)} images into existing database")

        with pycolmap.Database.open(database_path) as db:
            import_matches(image_ids, db, self.pairs_path, self.matches_path)

        self.log("Match import completed")
        return True

    def run_full_pipeline(
        self,
        database_path: Path,
        camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
        overwrite: bool = False,
    ) -> bool:
        """Run the full hloc pipeline: extraction -> retrieval -> matching -> import."""
        self.log("=" * 60)
        self.log("Starting full hloc pipeline")
        self.log("=" * 60)

        # Step 1: Extract local features
        self.extract_local_features(overwrite=overwrite)

        # Step 2: Extract global descriptors
        self.extract_global_descriptors(overwrite=overwrite)

        # Step 3: Generate pairs
        self.generate_pairs()

        # Step 4: Match features
        self.match_features(overwrite=overwrite)

        # Step 5: Import to COLMAP database
        self.import_to_colmap_db(database_path, camera_mode)

        # Step 6: Geometric verification
        self.run_geometric_verification(database_path)

        self.log("=" * 60)
        self.log("hloc pipeline completed successfully")
        self.log("=" * 60)

        return True


def get_default_hloc_config_path() -> Path:
    """Get the default hloc config file path."""
    return Path(__file__).parent / "defaultHloc.ini"


def create_hloc_pipeline(
    image_dir: Path,
    output_dir: Path,
    hloc_config_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    **overrides
) -> HlocPipeline:
    """
    Factory function to create an HlocPipeline with optional config overrides.

    Args:
        image_dir: Directory containing input images
        output_dir: Output directory for results
        hloc_config_path: Path to hloc config file (uses default if None)
        logger: Logger instance
        **overrides: Override config values (e.g., feature_type='r2d2', num_matched=100)

    Returns:
        Configured HlocPipeline instance
    """
    if hloc_config_path is None:
        hloc_config_path = get_default_hloc_config_path()

    config = HlocConfigParser(hloc_config_path)

    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            # Map override keys to config sections
            if key in ('feature_type', 'max_keypoints', 'resize_max', 'grayscale'):
                if 'LocalFeatures' not in config.config:
                    config.config.add_section('LocalFeatures')
                config.config.set('LocalFeatures', key, str(value))
            elif key in ('network', 'retrieval_network'):
                if 'GlobalDescriptors' not in config.config:
                    config.config.add_section('GlobalDescriptors')
                config.config.set('GlobalDescriptors', 'network', str(value))
            elif key in ('num_matched', 'min_score'):
                if 'PairGeneration' not in config.config:
                    config.config.add_section('PairGeneration')
                config.config.set('PairGeneration', key, str(value))
            elif key in ('matcher', 'ratio_threshold', 'cross_check'):
                if 'FeatureMatching' not in config.config:
                    config.config.add_section('FeatureMatching')
                config.config.set('FeatureMatching', key, str(value))

    return HlocPipeline(config, image_dir, output_dir, logger)
