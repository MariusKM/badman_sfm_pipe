#!/usr/bin/env python3
"""
COLMAP Hierarchical Pipeline Runner with Checkpoint Management
Executes COLMAP SFM pipeline using hierarchical mapper for large-scale datasets,
with post-reconstruction refinement through triangulation and bundle adjustment.

Supports both standard COLMAP feature extraction/matching and hloc-based
pipeline with modern retrieval networks for improved speed on large datasets.
"""

import argparse
import configparser
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional hloc import (only needed when --use_hloc is specified)
try:
    from hloc_pipeline import HlocPipeline, HlocConfigParser, get_default_hloc_config_path
    HLOC_AVAILABLE = True
except ImportError:
    HLOC_AVAILABLE = False


class CheckpointManager:
    """Manages pipeline checkpoint state using JSON."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.data = self._load()
    
    def _load(self) -> dict:
        """Load checkpoint data from file."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {"stages": {}, "last_stage": None}
    
    def save(self):
        """Save checkpoint data to file."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.data, indent=2, fp=f)
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has been completed."""
        return self.data["stages"].get(stage_name, {}).get("completed", False)
    
    def mark_stage_started(self, stage_name: str):
        """Mark stage as started."""
        if stage_name not in self.data["stages"]:
            self.data["stages"][stage_name] = {}
        self.data["stages"][stage_name]["started"] = datetime.now().isoformat()
        self.data["stages"][stage_name]["completed"] = False
        self.save()
    
    def mark_stage_completed(self, stage_name: str, duration: float, stats: dict = None):
        """Mark stage as completed with timing and stats."""
        if stage_name not in self.data["stages"]:
            self.data["stages"][stage_name] = {}
        self.data["stages"][stage_name]["completed"] = True
        self.data["stages"][stage_name]["timestamp"] = datetime.now().isoformat()
        self.data["stages"][stage_name]["duration_seconds"] = duration
        if stats:
            self.data["stages"][stage_name]["stats"] = stats
        self.data["last_stage"] = stage_name
        self.save()
    
    def clear(self):
        """Clear all checkpoint data."""
        self.data = {"stages": {}, "last_stage": None}
        self.save()
    
    def clear_from_stage(self, stage_name: str, all_stages: List[str]):
        """Clear checkpoints from a specific stage onwards."""
        stage_idx = all_stages.index(stage_name)
        for stage in all_stages[stage_idx:]:
            if stage in self.data["stages"]:
                del self.data["stages"][stage]
        # Update last_stage
        completed_stages = [s for s in all_stages if self.is_stage_completed(s)]
        self.data["last_stage"] = completed_stages[-1] if completed_stages else None
        self.save()


class INIConfigParser:
    """Parse COLMAP INI configuration file."""
    
    def __init__(self, ini_path: Path):
        self.config = configparser.RawConfigParser()
        # Preserve case sensitivity in keys
        self.config.optionxform = str
        
        # COLMAP INI files may have global parameters before sections
        # Prepend [DEFAULT] section if needed
        with open(ini_path, 'r') as f:
            content = f.read()
        
        # Check if file starts with a parameter (no section header)
        if content.strip() and not content.strip().startswith('['):
            content = '[DEFAULT]\n' + content
        
        # Parse from string
        self.config.read_string(content)
    
    def get_section_args(self, section: str) -> List[str]:
        """Convert INI section to COLMAP command-line arguments."""
        args = []
        if section not in self.config:
            return args

        for key, value in self.config[section].items():
            # Skip certain global parameters and vocab tree params (set via command-line)
            if key in ['database_path', 'image_path', 'log_to_stderr', 'log_level',
                      'default_random_seed', 'vocab_tree_path', 'num_images']:
                continue

            # Skip empty values (handles None, empty string, whitespace-only)
            if value is None:
                continue
            # Strip inline comments (text after # or ;)
            stripped_value = value.split('#')[0].split(';')[0].strip()
            if not stripped_value or stripped_value == '':
                continue

            # Convert INI key to COLMAP argument format
            arg_name = f"--{section}.{key}"

            # Handle boolean values
            if stripped_value.lower() in ['true', 'false']:
                arg_value = '1' if stripped_value.lower() == 'true' else '0'
            else:
                arg_value = stripped_value

            args.extend([arg_name, arg_value])

        return args
    
    def get_global_args(self) -> List[str]:
        """Get global COLMAP arguments."""
        args = []
        if 'DEFAULT' in self.config:
            for key, value in self.config['DEFAULT'].items():
                if key in ['log_to_stderr', 'log_level', 'default_random_seed']:
                    if value.lower() in ['true', 'false']:
                        value = '1' if value.lower() == 'true' else '0'
                    args.extend([f"--{key}", value])
        return args


class HierarchicalColmapPipeline:
    """Main hierarchical COLMAP pipeline executor."""

    # Standard COLMAP stages
    STAGES_STANDARD = [
        "feature_extraction",
        "feature_matching",
        "hierarchical_reconstruction",
        "post_processing_refinement",
        "orientation_alignment",
        "undistortion",
        "model_analysis"
    ]

    # hloc-based stages (replaces feature extraction and matching)
    STAGES_HLOC = [
        "hloc_feature_extraction",
        "hloc_global_descriptors",
        "hloc_pair_generation",
        "hloc_feature_matching",
        "hloc_import_to_colmap",
        "hloc_geometric_verification",
        "hierarchical_reconstruction",
        "post_processing_refinement",
        "orientation_alignment",
        "undistortion",
        "model_analysis"
    ]

    # Hybrid mode: COLMAP SIFT extraction + hloc retrieval-based matching
    STAGES_HLOC_MATCHING = [
        "feature_extraction",
        "hloc_export_features",
        "hloc_global_descriptors",
        "hloc_pair_generation",
        "hloc_feature_matching",
        "hloc_import_matches",
        "hloc_geometric_verification",
        "hierarchical_reconstruction",
        "post_processing_refinement",
        "orientation_alignment",
        "undistortion",
        "model_analysis"
    ]

    def __init__(self, args):
        self.args = args
        self.input_images = Path(args.input_images).resolve()
        self.output_dir = Path(args.output).resolve()
        self.config = INIConfigParser(Path(args.config))

        # Determine pipeline mode
        self.use_hloc = getattr(args, 'use_hloc', False)
        self.use_hloc_matching = getattr(args, 'use_hloc_matching', False)

        if self.use_hloc and self.use_hloc_matching:
            raise ValueError("Cannot use both --use_hloc and --use_hloc_matching. Choose one mode.")

        if self.use_hloc:
            if not HLOC_AVAILABLE:
                raise RuntimeError("hloc mode requested but hloc_pipeline module not available. "
                                   "Ensure hloc_pipeline.py exists and hloc submodule is installed.")
            self.STAGES = self.STAGES_HLOC
        elif self.use_hloc_matching:
            if not HLOC_AVAILABLE:
                raise RuntimeError("hloc matching mode requested but hloc_pipeline module not available. "
                                   "Ensure hloc_pipeline.py exists and hloc submodule is installed.")
            self.STAGES = self.STAGES_HLOC_MATCHING
        else:
            self.STAGES = self.STAGES_STANDARD

        # Setup paths
        self.db_path = self.output_dir / "database.db"
        self.sparse_dir = self.output_dir / "sparse"
        self.oriented_dir = self.output_dir / "oriented-model"
        self.dense_dir = self.output_dir / "dense"

        # Setup checkpoint and logging
        self.checkpoint = CheckpointManager(self.output_dir / ".checkpoint.json")
        self.setup_logging()

        self.selected_model = None

        # Initialize hloc pipeline if needed
        self.hloc_pipeline = None
        if self.use_hloc or self.use_hloc_matching:
            self._init_hloc_pipeline()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_path = self.output_dir / "colmap_hierarchical_pipeline.log"

        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(stage)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Setup logger
        self.logger = logging.getLogger('colmap_hierarchical_pipeline')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _init_hloc_pipeline(self):
        """Initialize the hloc pipeline with config and overrides."""
        # Determine hloc config path
        hloc_config_path = getattr(self.args, 'hloc_config', None)
        if hloc_config_path:
            hloc_config_path = Path(hloc_config_path)
        else:
            hloc_config_path = get_default_hloc_config_path()

        if not hloc_config_path.exists():
            raise FileNotFoundError(f"hloc config file not found: {hloc_config_path}")

        # Parse hloc config
        hloc_config = HlocConfigParser(hloc_config_path)

        # Apply command-line overrides
        if getattr(self.args, 'hloc_feature_type', None):
            hloc_config.config.set('LocalFeatures', 'feature_type', self.args.hloc_feature_type)
        if getattr(self.args, 'hloc_max_keypoints', None):
            hloc_config.config.set('LocalFeatures', 'max_keypoints', str(self.args.hloc_max_keypoints))
        if getattr(self.args, 'hloc_retrieval_network', None):
            hloc_config.config.set('GlobalDescriptors', 'network', self.args.hloc_retrieval_network)
        if getattr(self.args, 'hloc_num_matched', None):
            hloc_config.config.set('PairGeneration', 'num_matched', str(self.args.hloc_num_matched))

        # Create hloc pipeline
        self.hloc_pipeline = HlocPipeline(
            config=hloc_config,
            image_dir=self.input_images,
            output_dir=self.output_dir,
            logger=self.logger
        )

        self.log(f"Initialized hloc pipeline with config: {hloc_config_path}")
        self.log(f"  Feature type: {hloc_config.get('LocalFeatures', 'feature_type', 'sift')}")
        self.log(f"  Max keypoints: {hloc_config.getint('LocalFeatures', 'max_keypoints', 8192)}")
        self.log(f"  Retrieval network: {hloc_config.get('GlobalDescriptors', 'network', 'netvlad')}")
        self.log(f"  Num matched: {hloc_config.getint('PairGeneration', 'num_matched', 50)}")

    def log(self, message: str, level: str = 'info', stage: str = 'PIPELINE'):
        """Log message with stage context."""
        extra = {'stage': stage.upper()}
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def validate_environment(self):
        """Validate environment and prerequisites."""
        self.log("Validating environment...")
        
        # Check COLMAP availability
        try:
            result = subprocess.run(['colmap', '-h'],
                                   capture_output=True, timeout=5)
            if result.returncode != 0:
                raise RuntimeError("COLMAP not found in PATH")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.log("COLMAP not found in PATH", level='error')
            sys.exit(1)
        
        # Check input images directory
        if not self.input_images.exists():
            self.log(f"Input images directory not found: {self.input_images}", 
                    level='error')
            sys.exit(1)
        
        # Count images (including subdirectories)
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = list(self.input_images.rglob('*'))
        images = [f for f in images if f.is_file() and f.suffix.lower() in image_exts]
        if not images:
            self.log(f"No images found in {self.input_images} (including subdirectories)", level='error')
            sys.exit(1)
        
        self.log(f"Found {len(images)} images in input directory (including subdirectories)")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log("Environment validation successful")
    
    def run_command(self, cmd: List[str], stage: str) -> Tuple[bool, str]:
        """Run COLMAP command and capture relevant output."""
        self.log(f"Running: {' '.join(cmd)}", stage=stage)
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                line = line.strip()
                if line:  # Log all non-empty output
                    output_lines.append(line)
                    self.log(line, stage=stage)
            
            process.wait()
            
            if process.returncode != 0:
                self.log(f"Command failed with return code {process.returncode}", 
                        level='error', stage=stage)
                return False, '\n'.join(output_lines)
            
            return True, '\n'.join(output_lines)
            
        except Exception as e:
            self.log(f"Error running command: {e}", level='error', stage=stage)
            return False, str(e)
    
    def extract_stats_from_output(self, output: str, patterns: dict) -> dict:
        """Extract statistics from command output using regex patterns."""
        stats = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    stats[key] = int(match.group(1))
                except (ValueError, IndexError):
                    try:
                        stats[key] = float(match.group(1))
                    except (ValueError, IndexError):
                        pass
        return stats
    
    def stage_feature_extraction(self) -> bool:
        """Stage 1: Feature extraction."""
        stage = "feature_extraction"
        self.log("=" * 60, stage=stage)
        self.log("Starting feature extraction", stage=stage)
        
        cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(self.db_path),
            '--image_path', str(self.input_images)
        ]
        
        # Add config arguments
        cmd.extend(self.config.get_section_args('ImageReader'))
        cmd.extend(self.config.get_section_args('FeatureExtraction'))
        cmd.extend(self.config.get_section_args('SiftExtraction'))
        
        success, output = self.run_command(cmd, stage)
        
        if success:
            patterns = {
                'num_images': r'(\d+)\s+images',
                'num_features': r'(\d+)\s+features'
            }
            stats = self.extract_stats_from_output(output, patterns)
            self.log(f"Feature extraction completed. Stats: {stats}", stage=stage)
            return True
        
        return False
    
    def stage_feature_matching(self) -> bool:
        """Stage 2: Feature matching."""
        stage = "feature_matching"
        self.log("=" * 60, stage=stage)
        self.log("Starting feature matching", stage=stage)
        
        # Determine matcher type
        matcher_type = self.args.matcher_type
        if not matcher_type:
            # Try to infer from config
            if 'ExhaustiveMatching' in self.config.config:
                matcher_type = 'exhaustive'
            elif 'SequentialMatching' in self.config.config:
                matcher_type = 'sequential'
            elif 'VocabTreeMatching' in self.config.config:
                matcher_type = 'vocab_tree'
            elif 'SpatialMatching' in self.config.config:
                matcher_type = 'spatial'
            else:
                matcher_type = 'exhaustive'  # default
        
        self.log(f"Using matcher type: {matcher_type}", stage=stage)
        
        cmd = [
            'colmap', f'{matcher_type}_matcher',
            '--database_path', str(self.db_path)
        ]
        
        # Add config arguments based on matcher type
        cmd.extend(self.config.get_section_args('FeatureMatching'))
        cmd.extend(self.config.get_section_args('SiftMatching'))
        cmd.extend(self.config.get_section_args('TwoViewGeometry'))
        
        if matcher_type == 'exhaustive':
            cmd.extend(self.config.get_section_args('ExhaustiveMatching'))
        elif matcher_type == 'sequential':
            cmd.extend(self.config.get_section_args('SequentialMatching'))
        elif matcher_type == 'vocab_tree':
            cmd.extend(self.config.get_section_args('VocabTreeMatching'))
            # Add vocabulary tree path (required for vocab_tree matcher)
            if self.args.vocab_tree_path:
                vocab_tree = Path(self.args.vocab_tree_path).resolve()
                if not vocab_tree.exists():
                    self.log(f"Vocabulary tree file not found: {vocab_tree}", level='error', stage=stage)
                    return False
                cmd.extend(['--VocabTreeMatching.vocab_tree_path', str(vocab_tree)])
                cmd.extend(['--VocabTreeMatching.num_images', str(self.args.vocab_tree_num_images)])
                self.log(f"Using vocabulary tree: {vocab_tree}", stage=stage)
            else:
                self.log("Warning: vocab_tree matcher selected but --vocab_tree_path not provided",
                        level='warning', stage=stage)
        elif matcher_type == 'spatial':
            cmd.extend(self.config.get_section_args('SpatialMatching'))

        success, output = self.run_command(cmd, stage)
        
        if success:
            patterns = {
                'num_matches': r'(\d+)\s+matches',
                'num_pairs': r'(\d+)\s+pairs'
            }
            stats = self.extract_stats_from_output(output, patterns)
            self.log(f"Feature matching completed. Stats: {stats}", stage=stage)
            return True
        
        return False
    
    def select_model(self) -> Optional[Path]:
        """Select the hierarchical reconstruction model."""
        if not self.sparse_dir.exists():
            return None

        # Check 1: Hierarchical mapper may output directly to sparse_dir
        images_file = self.sparse_dir / "images.bin"
        if not images_file.exists():
            images_file = self.sparse_dir / "images.txt"

        if images_file.exists():
            self.log(f"Selected model: {self.sparse_dir}")
            return self.sparse_dir

        # Check 2: Hierarchical mapper may output to numbered subdirectory (like standard mapper)
        # Find all numbered subdirectories and select the one with most registered images
        model_dirs = []
        for subdir in self.sparse_dir.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                images_bin = subdir / "images.bin"
                images_txt = subdir / "images.txt"
                if images_bin.exists() or images_txt.exists():
                    model_dirs.append(subdir)

        if model_dirs:
            # If multiple models, select the one with most images (largest images.bin file)
            if len(model_dirs) == 1:
                self.log(f"Selected model: {model_dirs[0]}")
                return model_dirs[0]
            else:
                # Find largest model by file size (proxy for most registered images)
                best_model = None
                best_size = 0
                for model_dir in model_dirs:
                    images_bin = model_dir / "images.bin"
                    if images_bin.exists():
                        size = images_bin.stat().st_size
                        if size > best_size:
                            best_size = size
                            best_model = model_dir

                if best_model:
                    self.log(f"Selected best model from {len(model_dirs)} candidates: {best_model}")
                    return best_model
                else:
                    # Fallback to first model if no .bin files
                    self.log(f"Selected model: {model_dirs[0]}")
                    return model_dirs[0]

        return None
    
    def stage_hierarchical_reconstruction(self) -> bool:
        """Stage 3: Hierarchical reconstruction."""
        stage = "hierarchical_reconstruction"
        self.log("=" * 60, stage=stage)
        self.log("Starting hierarchical reconstruction", stage=stage)
        
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'colmap', 'hierarchical_mapper',
            '--database_path', str(self.db_path),
            '--image_path', str(self.input_images),
            '--output_path', str(self.sparse_dir)
        ]
        
        # Add hierarchical-specific parameters
        if self.args.num_workers is not None:
            cmd.extend(['--num_workers', str(self.args.num_workers)])
        else:
            # Try to get from config
            if 'HierarchicalMapper' in self.config.config:
                num_workers = self.config.config['HierarchicalMapper'].get('num_workers', '-1')
                cmd.extend(['--num_workers', num_workers])
        
        # Add image_overlap and leaf_max_num_images from config
        if 'HierarchicalMapper' in self.config.config:
            for param in ['image_overlap', 'leaf_max_num_images']:
                if param in self.config.config['HierarchicalMapper']:
                    value = self.config.config['HierarchicalMapper'][param]
                    cmd.extend([f'--{param}', value])
        
        # Add mapper config arguments
        cmd.extend(self.config.get_section_args('Mapper'))
        
        success, output = self.run_command(cmd, stage)
        
        if success:
            # Select model
            self.selected_model = self.select_model()
            if not self.selected_model:
                self.log("No model was reconstructed", level='error', stage=stage)
                return False
            
            patterns = {
                'num_registered': r'(\d+)\s+registered',
                'num_points': r'(\d+)\s+points'
            }
            stats = self.extract_stats_from_output(output, patterns)
            self.log(f"Hierarchical reconstruction completed. Stats: {stats}", stage=stage)
            return True
        
        return False
    
    def run_model_analyzer(self, stage: str, label: str) -> dict:
        """Run model analyzer and extract statistics."""
        self.log(f"Running model analyzer ({label})", stage=stage)
        
        if not self.selected_model:
            self.selected_model = self.select_model()
            if not self.selected_model:
                self.log("No model available for analysis", level='error', stage=stage)
                return {}
        
        cmd = [
            'colmap', 'model_analyzer',
            '--path', str(self.selected_model)
        ]
        
        success, output = self.run_command(cmd, stage)
        
        if success:
            # Extract key statistics
            patterns = {
                'num_cameras': r'Cameras:\s+(\d+)',
                'num_images': r'Images:\s+(\d+)',
                'num_registered': r'Registered images:\s+(\d+)',
                'num_points': r'Points:\s+(\d+)',
                'num_observations': r'Observations:\s+(\d+)',
                'mean_track_length': r'Mean track length:\s+([\d.]+)',
                'mean_reproj_error': r'Mean reprojection error:\s+([\d.]+)'
            }
            stats = self.extract_stats_from_output(output, patterns)
            
            self.log(f"Model statistics ({label}):", stage=stage)
            for key, value in stats.items():
                self.log(f"  {key}: {value}", stage=stage)
            
            return stats
        
        return {}
    
    def stage_post_processing_refinement(self) -> bool:
        """Stage 4: Post-processing refinement with triangulation and bundle adjustment."""
        stage = "post_processing_refinement"
        self.log("=" * 60, stage=stage)
        self.log(f"Starting post-processing refinement ({self.args.refinement_rounds} rounds)", stage=stage)
        
        if not self.selected_model:
            self.selected_model = self.select_model()
            if not self.selected_model:
                self.log("No model available for refinement", level='error', stage=stage)
                return False
        
        # Initial model analysis
        initial_stats = self.run_model_analyzer(stage, "before refinement")
        
        for round_num in range(1, self.args.refinement_rounds + 1):
            self.log(f"--- Refinement Round {round_num}/{self.args.refinement_rounds} ---", stage=stage)
            
            # Step 1: Point Triangulation
            self.log(f"Round {round_num}: Running point triangulation", stage=stage)
            tri_cmd = [
                'colmap', 'point_triangulator',
                '--database_path', str(self.db_path),
                '--image_path', str(self.input_images),
                '--input_path', str(self.selected_model),
                '--output_path', str(self.selected_model),
                '--clear_points', '0'  # Don't clear existing points
            ]
            
            # Add mapper triangulation parameters
            tri_cmd.extend(self.config.get_section_args('Mapper'))
            
            tri_success, tri_output = self.run_command(tri_cmd, stage)
            if not tri_success:
                self.log(f"Triangulation failed in round {round_num}", level='error', stage=stage)
                return False
            
            # Model analysis after triangulation
            tri_stats = self.run_model_analyzer(stage, f"round {round_num} after triangulation")
            
            # Step 2: Bundle Adjustment
            self.log(f"Round {round_num}: Running bundle adjustment", stage=stage)
            ba_cmd = [
                'colmap', 'bundle_adjuster',
                '--input_path', str(self.selected_model),
                '--output_path', str(self.selected_model)
            ]
            
            # Add bundle adjustment parameters
            ba_cmd.extend(self.config.get_section_args('BundleAdjustment'))
            
            ba_success, ba_output = self.run_command(ba_cmd, stage)
            if not ba_success:
                self.log(f"Bundle adjustment failed in round {round_num}", level='error', stage=stage)
                return False
            
            # Model analysis after bundle adjustment
            ba_stats = self.run_model_analyzer(stage, f"round {round_num} after bundle adjustment")
            
            # Log improvement
            if 'mean_reproj_error' in ba_stats and 'mean_reproj_error' in tri_stats:
                improvement = tri_stats['mean_reproj_error'] - ba_stats['mean_reproj_error']
                self.log(f"Round {round_num}: Reprojection error improvement: {improvement:.4f} pixels", 
                        stage=stage)
        
        # Final comparison
        self.log("=== Refinement Summary ===", stage=stage)
        if 'num_points' in initial_stats and 'num_points' in ba_stats:
            self.log(f"Points: {initial_stats['num_points']} -> {ba_stats['num_points']}", stage=stage)
        if 'num_observations' in initial_stats and 'num_observations' in ba_stats:
            self.log(f"Observations: {initial_stats['num_observations']} -> {ba_stats['num_observations']}", 
                    stage=stage)
        if 'mean_reproj_error' in initial_stats and 'mean_reproj_error' in ba_stats:
            self.log(f"Mean reproj error: {initial_stats['mean_reproj_error']} -> {ba_stats['mean_reproj_error']}", 
                    stage=stage)
        
        self.log("Post-processing refinement completed", stage=stage)
        return True
    
    def stage_orientation_alignment(self) -> bool:
        """Stage 5: Orientation alignment."""
        if self.args.skip_orientation:
            self.log("Skipping orientation alignment (disabled)")
            return True
        
        stage = "orientation_alignment"
        self.log("=" * 60, stage=stage)
        self.log("Starting orientation alignment", stage=stage)
        
        if not self.selected_model:
            self.selected_model = self.select_model()
            if not self.selected_model:
                self.log("No model available for orientation alignment", 
                        level='error', stage=stage)
                return False
        
        self.oriented_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'colmap', 'model_orientation_aligner',
            '--input_path', str(self.selected_model),
            '--output_path', str(self.oriented_dir),
            '--image_path', str(self.input_images)
        ]
        
        success, output = self.run_command(cmd, stage)
        
        if success:
            self.log("Orientation alignment completed", stage=stage)
            # Update selected model to oriented model
            self.selected_model = self.oriented_dir
            return True
        
        return False
    
    def stage_undistortion(self) -> bool:
        """Stage 6: Image undistortion."""
        if self.args.skip_undistortion:
            self.log("Skipping undistortion (disabled)")
            return True
        
        stage = "undistortion"
        self.log("=" * 60, stage=stage)
        self.log("Starting image undistortion", stage=stage)
        
        if not self.selected_model:
            self.selected_model = self.select_model()
            if not self.selected_model:
                self.log("No model available for undistortion", 
                        level='error', stage=stage)
                return False
        
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(self.input_images),
            '--input_path', str(self.selected_model),
            '--output_path', str(self.dense_dir),
            '--output_type', 'COLMAP'
        ]
        
        # Add max_image_size from config if available
        if 'SiftExtraction' in self.config.config:
            max_size = self.config.config['SiftExtraction'].get('max_image_size', '3200')
            cmd.extend(['--max_image_size', max_size])
        
        success, output = self.run_command(cmd, stage)
        
        if success:
            self.log("Image undistortion completed", stage=stage)
            return True
        
        return False
    
    def stage_model_analysis(self) -> bool:
        """Stage 7: Final model analysis."""
        stage = "model_analysis"
        self.log("=" * 60, stage=stage)
        self.log("Running final model analyzer", stage=stage)

        stats = self.run_model_analyzer(stage, "final model")

        if stats:
            self.log("Final model analysis completed", stage=stage)
            return True

        return False

    # ==================== hloc Stage Methods ====================

    def stage_hloc_feature_extraction(self) -> bool:
        """hloc Stage 1: Extract local features (SIFT/R2D2/etc)."""
        stage = "hloc_feature_extraction"
        self.log("=" * 60, stage=stage)
        self.log("Starting hloc local feature extraction", stage=stage)

        try:
            self.hloc_pipeline.extract_local_features()
            self.log("hloc feature extraction completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"hloc feature extraction failed: {e}", level='error', stage=stage)
            return False

    def stage_hloc_global_descriptors(self) -> bool:
        """hloc Stage 2: Extract global descriptors for retrieval."""
        stage = "hloc_global_descriptors"
        self.log("=" * 60, stage=stage)
        self.log("Starting hloc global descriptor extraction", stage=stage)

        try:
            self.hloc_pipeline.extract_global_descriptors()
            self.log("hloc global descriptor extraction completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"hloc global descriptor extraction failed: {e}", level='error', stage=stage)
            return False

    def stage_hloc_pair_generation(self) -> bool:
        """hloc Stage 3: Generate image pairs from retrieval."""
        stage = "hloc_pair_generation"
        self.log("=" * 60, stage=stage)
        self.log("Starting hloc pair generation", stage=stage)

        try:
            self.hloc_pipeline.generate_pairs()
            self.log("hloc pair generation completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"hloc pair generation failed: {e}", level='error', stage=stage)
            return False

    def stage_hloc_feature_matching(self) -> bool:
        """hloc Stage 4: Match features for retrieved pairs."""
        stage = "hloc_feature_matching"
        self.log("=" * 60, stage=stage)
        self.log("Starting hloc feature matching", stage=stage)

        try:
            self.hloc_pipeline.match_features()
            self.log("hloc feature matching completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"hloc feature matching failed: {e}", level='error', stage=stage)
            return False

    def stage_hloc_import_to_colmap(self) -> bool:
        """hloc Stage 5: Import hloc features/matches to COLMAP database."""
        stage = "hloc_import_to_colmap"
        self.log("=" * 60, stage=stage)
        self.log("Starting hloc import to COLMAP database", stage=stage)

        try:
            import pycolmap
            self.hloc_pipeline.import_to_colmap_db(self.db_path, pycolmap.CameraMode.AUTO)
            self.log("hloc import to COLMAP completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"hloc import to COLMAP failed: {e}", level='error', stage=stage)
            return False

    # ==================== Hybrid hloc Matching Stage Methods ====================

    def stage_hloc_export_features(self) -> bool:
        """Hybrid Stage: Export COLMAP database features to hloc HDF5 format."""
        stage = "hloc_export_features"
        self.log("=" * 60, stage=stage)
        self.log("Exporting COLMAP features to hloc format", stage=stage)

        try:
            self.hloc_pipeline.export_features_from_colmap_db(self.db_path)
            self.log("Feature export completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"Feature export failed: {e}", level='error', stage=stage)
            return False

    def stage_hloc_import_matches(self) -> bool:
        """Hybrid Stage: Import hloc matches into existing COLMAP database."""
        stage = "hloc_import_matches"
        self.log("=" * 60, stage=stage)
        self.log("Importing hloc matches to COLMAP database", stage=stage)

        try:
            self.hloc_pipeline.import_matches_to_colmap_db(self.db_path)
            self.log("Match import completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"Match import failed: {e}", level='error', stage=stage)
            return False

    def stage_hloc_geometric_verification(self) -> bool:
        """Geometric verification of matches (COLMAP)."""
        stage = "hloc_geometric_verification"
        self.log("=" * 60, stage=stage)
        self.log("Running geometric verification", stage=stage)

        try:
            self.hloc_pipeline.run_geometric_verification(self.db_path)
            self.log("Geometric verification completed", stage=stage)
            return True
        except Exception as e:
            self.log(f"Geometric verification failed: {e}", level='error', stage=stage)
            return False

    # ==================== End hloc Stage Methods ====================

    def should_run_stage(self, stage: str) -> bool:
        """Determine if a stage should be run based on arguments and checkpoints."""
        # If specific stage requested, only run that one
        if self.args.stage:
            return stage == self.args.stage
        
        # If from_stage specified, run from that stage onwards
        if self.args.from_stage:
            stage_idx = self.STAGES.index(stage)
            from_idx = self.STAGES.index(self.args.from_stage)
            return stage_idx >= from_idx
        
        # Check if already completed
        if self.checkpoint.is_stage_completed(stage):
            self.log(f"Stage '{stage}' already completed, skipping", stage=stage)
            return False
        
        return True
    
    def run(self):
        """Run the complete pipeline."""
        self.log("=" * 60)
        self.log("COLMAP Hierarchical Pipeline Starting")
        self.log("=" * 60)
        self.log(f"Input images: {self.input_images}")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Config file: {self.args.config}")
        self.log(f"Refinement rounds: {self.args.refinement_rounds}")
        
        # Validate environment
        self.validate_environment()
        
        # Handle force restart
        if self.args.force_restart:
            self.log("Force restart requested, clearing checkpoints")
            self.checkpoint.clear()
        elif self.args.from_stage:
            self.log(f"Restarting from stage: {self.args.from_stage}")
            self.checkpoint.clear_from_stage(self.args.from_stage, self.STAGES)
        
        # Stage execution map
        stage_methods = {
            # Standard COLMAP stages
            "feature_extraction": self.stage_feature_extraction,
            "feature_matching": self.stage_feature_matching,
            "hierarchical_reconstruction": self.stage_hierarchical_reconstruction,
            "post_processing_refinement": self.stage_post_processing_refinement,
            "orientation_alignment": self.stage_orientation_alignment,
            "undistortion": self.stage_undistortion,
            "model_analysis": self.stage_model_analysis,
            # hloc stages
            "hloc_feature_extraction": self.stage_hloc_feature_extraction,
            "hloc_global_descriptors": self.stage_hloc_global_descriptors,
            "hloc_pair_generation": self.stage_hloc_pair_generation,
            "hloc_feature_matching": self.stage_hloc_feature_matching,
            "hloc_import_to_colmap": self.stage_hloc_import_to_colmap,
            # Hybrid hloc matching stages
            "hloc_export_features": self.stage_hloc_export_features,
            "hloc_import_matches": self.stage_hloc_import_matches,
            # Shared hloc stage
            "hloc_geometric_verification": self.stage_hloc_geometric_verification,
        }
        
        # Run stages
        for stage in self.STAGES:
            if not self.should_run_stage(stage):
                continue
            
            self.checkpoint.mark_stage_started(stage)
            start_time = time.time()
            
            success = stage_methods[stage]()
            
            duration = time.time() - start_time
            
            if not success:
                self.log(f"Stage '{stage}' failed", level='error', stage=stage)
                sys.exit(1)
            
            self.checkpoint.mark_stage_completed(stage, duration)
            self.log(f"Stage '{stage}' completed in {duration:.2f} seconds", 
                    stage=stage)
        
        self.log("=" * 60)
        self.log("COLMAP Hierarchical Pipeline Completed Successfully")
        self.log("=" * 60)


def main():
    # Determine available stages for argument choices
    _seen = set()
    all_stages = []
    for s in (HierarchicalColmapPipeline.STAGES_STANDARD + HierarchicalColmapPipeline.STAGES_HLOC
              + HierarchicalColmapPipeline.STAGES_HLOC_MATCHING):
        if s not in _seen:
            all_stages.append(s)
            _seen.add(s)

    parser = argparse.ArgumentParser(
        description='Run COLMAP hierarchical SFM pipeline with checkpoint management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
hloc Mode Examples:
  # Basic hloc with SIFT + NetVLAD + hierarchical mapper
  python run_colmap_hierarchical.py --input_images ./images --output ./output --config defaultColMap.ini --use_hloc

  # Dense SIFT for Gaussian splatting with large dataset
  python run_colmap_hierarchical.py --input_images ./images --output ./output --config defaultColMap.ini \\
      --use_hloc --hloc_max_keypoints 16384 --hloc_num_matched 100

  # Custom hloc config
  python run_colmap_hierarchical.py --input_images ./images --output ./output --config defaultColMap.ini \\
      --use_hloc --hloc_config myHloc.ini

Hybrid hloc Matching Mode Examples:
  # COLMAP SIFT extraction + hloc retrieval matching (unlimited keypoints)
  python run_colmap_hierarchical.py --input_images ./images --output ./output --config defaultColMap.ini \\
      --use_hloc_matching

  # With custom retrieval network and pair count
  python run_colmap_hierarchical.py --input_images ./images --output ./output --config defaultColMap.ini \\
      --use_hloc_matching --hloc_retrieval_network openibl --hloc_num_matched 100
        """
    )

    # Required arguments
    parser.add_argument('--input_images', required=True,
                       help='Path to input image directory')
    parser.add_argument('--output', required=True,
                       help='Path to output directory for database and results')
    parser.add_argument('--config', required=True,
                       help='Path to INI configuration file')

    # Optional flags
    parser.add_argument('--skip_undistortion', action='store_true',
                       help='Skip image undistortion stage')
    parser.add_argument('--skip_orientation', action='store_true',
                       help='Skip orientation alignment stage')

    # Hierarchical-specific options
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers for hierarchical mapper (-1 for all cores)')
    parser.add_argument('--refinement_rounds', type=int, default=2,
                       help='Number of triangulation + bundle adjustment refinement rounds (default: 2)')

    # Stage control
    parser.add_argument('--stage', choices=all_stages,
                       help='Run only a specific stage')
    parser.add_argument('--from_stage', choices=all_stages,
                       help='Restart from a specific stage onwards')
    parser.add_argument('--force_restart', action='store_true',
                       help='Clear all checkpoints and restart from beginning')

    # Matcher type (for standard COLMAP mode)
    parser.add_argument('--matcher_type',
                       choices=['exhaustive', 'sequential', 'vocab_tree', 'spatial'],
                       help='Override matching type from config (standard mode only)')

    # Vocabulary tree options (for standard COLMAP mode)
    parser.add_argument('--vocab_tree_path',
                       help='Path to vocabulary tree file (required for vocab_tree matcher)')
    parser.add_argument('--vocab_tree_num_images', type=int, default=100,
                       help='Number of images to retrieve for vocab tree matching (default: 100)')

    # ==================== hloc Options ====================
    hloc_group = parser.add_argument_group('hloc Options',
                                            'Options for hloc-based feature extraction and matching')

    hloc_group.add_argument('--use_hloc', action='store_true',
                           help='Use hloc pipeline instead of COLMAP for feature extraction/matching')
    hloc_group.add_argument('--use_hloc_matching', action='store_true',
                           help='Use COLMAP SIFT extraction + hloc retrieval-based matching. '
                                'Combines COLMAP GPU SIFT (unlimited keypoints) with hloc '
                                'retrieval pair generation and NN-ratio matching.')
    hloc_group.add_argument('--hloc_config',
                           help='Path to hloc config file (default: defaultHloc.ini)')
    hloc_group.add_argument('--hloc_feature_type',
                           choices=['sift', 'r2d2', 'superpoint', 'disk', 'aliked'],
                           help='Override feature type (default: from config)')
    hloc_group.add_argument('--hloc_max_keypoints', type=int,
                           help='Override max keypoints per image (default: from config)')
    hloc_group.add_argument('--hloc_retrieval_network',
                           choices=['netvlad', 'openibl', 'dir', 'megaloc'],
                           help='Override retrieval network (default: from config)')
    hloc_group.add_argument('--hloc_num_matched', type=int,
                           help='Override number of top-k matches per image (default: from config)')

    args = parser.parse_args()

    # Run pipeline
    pipeline = HierarchicalColmapPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

