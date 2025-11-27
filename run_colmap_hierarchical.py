#!/usr/bin/env python3
"""
COLMAP Hierarchical Pipeline Runner with Checkpoint Management
Executes COLMAP SFM pipeline using hierarchical mapper for large-scale datasets,
with post-reconstruction refinement through triangulation and bundle adjustment.
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
        self.config = configparser.ConfigParser()
        
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
            # Skip certain global parameters
            if key in ['database_path', 'image_path', 'log_to_stderr', 'log_level', 
                      'default_random_seed']:
                continue
            
            # Convert INI key to COLMAP argument format
            arg_name = f"--{section}.{key}"
            
            # Handle boolean values
            if value.lower() in ['true', 'false']:
                arg_value = '1' if value.lower() == 'true' else '0'
            else:
                arg_value = value
            
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
    
    STAGES = [
        "feature_extraction",
        "feature_matching",
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
        
        # Setup paths
        self.db_path = self.output_dir / "database.db"
        self.sparse_dir = self.output_dir / "sparse"
        self.oriented_dir = self.output_dir / "oriented-model"
        self.dense_dir = self.output_dir / "dense"
        
        # Setup checkpoint and logging
        self.checkpoint = CheckpointManager(self.output_dir / ".checkpoint.json")
        self.setup_logging()
        
        self.selected_model = None
    
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
                                   capture_output=True, timeout=5, shell=True)
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
        
        # Count images
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = [f for f in self.input_images.iterdir() 
                 if f.suffix.lower() in image_exts]
        if not images:
            self.log(f"No images found in {self.input_images}", level='error')
            sys.exit(1)
        
        self.log(f"Found {len(images)} images in input directory")
        
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
                bufsize=1,
                shell=True
            )
            
            output_lines = []
            for line in process.stdout:
                line = line.strip()
                # Filter relevant lines
                if any(keyword in line.lower() for keyword in 
                      ['error', 'warning', 'images', 'matches', 'points', 
                       'registered', 'features', 'extracted', 'processing', 
                       'observations', 'reprojection']):
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
        
        # Hierarchical mapper outputs directly to sparse_dir
        images_file = self.sparse_dir / "images.bin"
        if not images_file.exists():
            images_file = self.sparse_dir / "images.txt"
        
        if images_file.exists():
            self.log(f"Selected model: {self.sparse_dir}")
            return self.sparse_dir
        
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
            "feature_extraction": self.stage_feature_extraction,
            "feature_matching": self.stage_feature_matching,
            "hierarchical_reconstruction": self.stage_hierarchical_reconstruction,
            "post_processing_refinement": self.stage_post_processing_refinement,
            "orientation_alignment": self.stage_orientation_alignment,
            "undistortion": self.stage_undistortion,
            "model_analysis": self.stage_model_analysis
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
    parser = argparse.ArgumentParser(
        description='Run COLMAP hierarchical SFM pipeline with checkpoint management',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    parser.add_argument('--stage', choices=HierarchicalColmapPipeline.STAGES,
                       help='Run only a specific stage')
    parser.add_argument('--from_stage', choices=HierarchicalColmapPipeline.STAGES,
                       help='Restart from a specific stage onwards')
    parser.add_argument('--force_restart', action='store_true',
                       help='Clear all checkpoints and restart from beginning')
    
    # Matcher type
    parser.add_argument('--matcher_type', 
                       choices=['exhaustive', 'sequential', 'vocab_tree', 'spatial'],
                       help='Override matching type from config')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = HierarchicalColmapPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

