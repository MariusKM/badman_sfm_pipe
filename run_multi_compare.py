#!/usr/bin/env python3
"""
Multi-Pipeline SFM Comparison Tool
Runs multiple SFM pipelines (COLMAP, GLOMAP, Hierarchical, Refined GLOMAP)
and compares their performance in terms of speed and reconstruction quality.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ModelAnalyzer:
    """Extracts statistics from COLMAP sparse models."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
    
    def analyze(self) -> Optional[Dict]:
        """Run COLMAP model_analyzer and extract statistics."""
        if not self.model_path.exists():
            return None
        
        # Check if model files exist
        has_model = (
            (self.model_path / "cameras.bin").exists() or
            (self.model_path / "cameras.txt").exists()
        )
        if not has_model:
            return None
        
        try:
            result = subprocess.run(
                ['colmap', 'model_analyzer', '--path', str(self.model_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return None
            
            output = result.stdout + result.stderr
            return self._parse_analyzer_output(output)
            
        except Exception as e:
            print(f"Error analyzing model: {e}")
            return None
    
    def _parse_analyzer_output(self, output: str) -> Dict:
        """Parse model_analyzer output to extract statistics."""
        import re
        
        stats = {}
        patterns = {
            'num_cameras': r'Cameras:\s+(\d+)',
            'num_images': r'Images:\s+(\d+)',
            'num_registered': r'Registered images:\s+(\d+)',
            'num_points': r'Points:\s+(\d+)',
            'num_observations': r'Observations:\s+(\d+)',
            'mean_track_length': r'Mean track length:\s+([\d.]+)',
            'mean_observations_per_image': r'Mean observations per registered image:\s+([\d.]+)',
            'mean_reproj_error': r'Mean reprojection error:\s+([\d.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1)
                    stats[key] = float(value) if '.' in value else int(value)
                except (ValueError, IndexError):
                    pass
        
        return stats


class MetricsCalculator:
    """Calculates derived metrics and quality scores."""
    
    @staticmethod
    def calculate_derived_metrics(stats: Dict, timing: Dict) -> Dict:
        """Calculate derived metrics from basic statistics."""
        derived = {}
        
        if stats.get('num_registered') and stats.get('num_images'):
            derived['registration_ratio'] = (stats['num_registered'] / stats['num_images']) * 100
        
        if stats.get('num_points') and stats.get('num_registered'):
            derived['points_per_image'] = stats['num_points'] / stats['num_registered']
        
        if stats.get('num_observations') and stats.get('num_registered'):
            derived['observations_per_image'] = stats['num_observations'] / stats['num_registered']
        
        if stats.get('num_observations') and stats.get('num_points'):
            derived['observations_per_point'] = stats['num_observations'] / stats['num_points']
        
        if timing.get('total_time') and stats.get('num_registered'):
            derived['time_per_image'] = timing['total_time'] / stats['num_registered']
        
        if timing.get('total_time') and stats.get('num_points'):
            derived['time_per_point'] = timing['total_time'] / stats['num_points']
        
        return derived
    
    @staticmethod
    def calculate_quality_score(stats: Dict, derived: Dict) -> float:
        """Calculate composite quality score (0-10 scale)."""
        if not stats:
            return 0.0
        
        score = 0.0
        
        # Registration ratio score (20% weight)
        if 'registration_ratio' in derived:
            registration_score = min(10, derived['registration_ratio'] / 10)
            score += registration_score * 0.2
        
        # Reprojection error score (40% weight) - lower is better
        if 'mean_reproj_error' in stats:
            # Assuming good error is < 1.0 pixel, poor is > 2.0 pixels
            reproj_score = max(0, 10 - stats['mean_reproj_error'] * 3)
            score += reproj_score * 0.4
        
        # Point density score (20% weight)
        if 'points_per_image' in derived:
            # Normalize: 500 points/image = full score
            density_score = min(10, derived['points_per_image'] / 50)
            score += density_score * 0.2
        
        # Observations per point score (20% weight) - track robustness
        if 'observations_per_point' in derived:
            # Good tracks have 3+ observations per point
            obs_score = min(10, derived['observations_per_point'] / 0.5)
            score += obs_score * 0.2
        
        return round(score, 2)


class PipelineRunner:
    """Executes SFM pipelines and captures timing information."""
    
    PIPELINE_SCRIPTS = {
        'colmap': 'run_colmap.py',
        'glomap': 'run_glomap.py',
        'hierarchical': 'run_colmap_hierarchical.py',
        'refined': 'run_glomap_refined.py'
    }
    
    def __init__(self, input_images: Path, output_base: Path, 
                 colmap_config: Path, glomap_config: Path, args):
        self.input_images = input_images
        self.output_base = output_base
        self.colmap_config = colmap_config
        self.glomap_config = glomap_config
        self.args = args
    
    def run_pipeline(self, pipeline_name: str) -> Tuple[bool, Dict, Dict]:
        """Run a single pipeline and return success status, timing, and stats."""
        print(f"\n{'='*60}")
        print(f"Running {pipeline_name.upper()} pipeline...")
        print(f"{'='*60}")
        
        output_dir = self.output_base / pipeline_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        script = self.PIPELINE_SCRIPTS[pipeline_name]
        cmd = [
            sys.executable, script,
            '--input_images', str(self.input_images),
            '--output', str(output_dir)
        ]
        
        # Add config arguments based on pipeline type
        if pipeline_name in ['glomap', 'refined']:
            # GLOMAP-based pipelines use both configs
            cmd.extend(['--colmap_config', str(self.colmap_config)])
            cmd.extend(['--glomap_config', str(self.glomap_config)])
        else:
            # COLMAP and hierarchical use --config
            cmd.extend(['--config', str(self.colmap_config)])
        
        # Add optional arguments
        if self.args.skip_undistortion:
            cmd.append('--skip_undistortion')
        if self.args.skip_orientation:
            cmd.append('--skip_orientation')
        if self.args.matcher_type:
            cmd.extend(['--matcher_type', self.args.matcher_type])
        
        # Add refinement rounds for applicable pipelines
        if pipeline_name in ['hierarchical', 'refined']:
            cmd.extend(['--refinement_rounds', str(self.args.refinement_rounds)])
        
        # Run pipeline
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            total_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"  ERROR: Pipeline failed!")
                print(f"  Output: {result.stdout[-500:]}")
                print(f"  Error: {result.stderr[-500:]}")
                return False, {'total_time': total_time}, {}
            
            print(f"  Completed in {total_time:.1f}s")
            
            # Extract timing from checkpoint
            timing = self._extract_timing(output_dir, pipeline_name)
            timing['total_time'] = total_time
            
            # Analyze model
            model_path = self._find_model_path(output_dir)
            if model_path:
                print(f"  Analyzing model...")
                analyzer = ModelAnalyzer(model_path)
                stats = analyzer.analyze()
                if stats:
                    print(f"  Registered: {stats.get('num_registered', 0)} images")
                    print(f"  Points: {stats.get('num_points', 0)}")
                    print(f"  Reproj Error: {stats.get('mean_reproj_error', 0):.3f} px")
                    return True, timing, stats
            
            return True, timing, {}
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"  EXCEPTION: {e}")
            return False, {'total_time': total_time}, {}
    
    def _extract_timing(self, output_dir: Path, pipeline_name: str) -> Dict:
        """Extract timing information from checkpoint file."""
        checkpoint_path = output_dir / ".checkpoint.json"
        if not checkpoint_path.exists():
            return {}
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            timing = {}
            for stage_name, stage_data in checkpoint.get('stages', {}).items():
                if 'duration_seconds' in stage_data:
                    timing[stage_name] = stage_data['duration_seconds']
            
            return timing
            
        except Exception as e:
            print(f"  Warning: Could not extract timing: {e}")
            return {}
    
    def _find_model_path(self, output_dir: Path) -> Optional[Path]:
        """Find the final sparse model path."""
        # Check for oriented model first (final output if enabled)
        oriented = output_dir / "oriented-model"
        if oriented.exists() and (oriented / "cameras.bin").exists():
            return oriented
        
        # Check sparse directory
        sparse = output_dir / "sparse"
        if sparse.exists():
            # Check if files are directly in sparse
            if (sparse / "cameras.bin").exists():
                return sparse
            
            # Check for numbered subdirectories (COLMAP style)
            subdirs = [d for d in sparse.iterdir() if d.is_dir()]
            if subdirs:
                # Find the one with most images (best model)
                best = None
                max_size = 0
                for subdir in subdirs:
                    images_file = subdir / "images.bin"
                    if images_file.exists():
                        size = images_file.stat().st_size
                        if size > max_size:
                            max_size = size
                            best = subdir
                if best:
                    return best
        
        return None


class ComparisonReporter:
    """Generates markdown comparison report."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.results = {}
    
    def add_result(self, pipeline_name: str, success: bool, 
                   timing: Dict, stats: Dict, derived: Dict, quality_score: float):
        """Add pipeline result to report."""
        self.results[pipeline_name] = {
            'success': success,
            'timing': timing,
            'stats': stats,
            'derived': derived,
            'quality_score': quality_score
        }
    
    def generate_report(self, input_images: Path, colmap_config: Path, 
                       glomap_config: Path, refinement_rounds: int):
        """Generate comprehensive markdown comparison report."""
        
        # Count total images
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        total_images = len([f for f in input_images.iterdir() 
                           if f.suffix.lower() in image_exts])
        
        # Generate report
        report = []
        report.append("# SFM Pipeline Comparison Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Dataset:** {input_images} ({total_images} images)\n")
        report.append(f"**Configuration:**")
        report.append(f"- COLMAP Config: {colmap_config.name}")
        report.append(f"- GLOMAP Config: {glomap_config.name}")
        report.append(f"- Refinement Rounds: {refinement_rounds}\n")
        
        # Summary table
        report.append("## Summary\n")
        report.append("| Pipeline | Status | Time (s) | Registered | Points | Observations | Reproj Error (px) | Quality Score |")
        report.append("|----------|--------|----------|------------|--------|--------------|-------------------|---------------|")
        
        for name, result in self.results.items():
            if not result['success']:
                report.append(f"| {name.title()} | FAILED | - | - | - | - | - | - |")
                continue
            
            stats = result['stats']
            timing = result['timing']
            
            time_str = f"{timing.get('total_time', 0):.1f}"
            reg_str = f"{stats.get('num_registered', 0)}/{total_images}"
            points_str = f"{stats.get('num_points', 0):,}"
            obs_str = f"{stats.get('num_observations', 0):,}"
            reproj_str = f"{stats.get('mean_reproj_error', 0):.3f}"
            quality_str = f"{result['quality_score']:.1f}/10"
            
            report.append(f"| {name.title()} | OK | {time_str} | {reg_str} | {points_str} | {obs_str} | {reproj_str} | {quality_str} |")
        
        report.append("")
        
        # Speed analysis
        report.append("## Speed Analysis\n")
        self._add_speed_analysis(report)
        
        # Quality analysis
        report.append("## Quality Analysis\n")
        self._add_quality_analysis(report)
        
        # Stage timing breakdown
        report.append("## Stage-by-Stage Timing\n")
        self._add_stage_timing(report)
        
        # Recommendations
        report.append("## Recommendations\n")
        self._add_recommendations(report)
        
        # Detailed results
        report.append("## Detailed Pipeline Results\n")
        self._add_detailed_results(report, total_images)
        
        # Write report
        with open(self.output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nComparison report written to: {self.output_path}")
    
    def _add_speed_analysis(self, report: List[str]):
        """Add speed analysis section."""
        successful = {name: r for name, r in self.results.items() if r['success']}
        if not successful:
            report.append("No successful pipelines to compare.\n")
            return
        
        times = {name: r['timing'].get('total_time', 0) for name, r in successful.items()}
        fastest = min(times, key=times.get)
        slowest = max(times, key=times.get)
        
        report.append(f"**Fastest:** {fastest.title()} ({times[fastest]:.1f}s)")
        report.append(f"**Slowest:** {slowest.title()} ({times[slowest]:.1f}s)")
        
        if times[slowest] > 0:
            speedup = times[slowest] / times[fastest]
            report.append(f"**Speed improvement:** {fastest.title()} is {speedup:.1f}x faster than {slowest.title()}\n")
        else:
            report.append("")
        
        # Time per image comparison
        report.append("### Time Efficiency\n")
        report.append("| Pipeline | Total Time | Time/Image | Time/Point |")
        report.append("|----------|------------|------------|------------|")
        
        for name, result in successful.items():
            derived = result['derived']
            time_total = result['timing'].get('total_time', 0)
            time_per_img = derived.get('time_per_image', 0)
            time_per_pt = derived.get('time_per_point', 0)
            
            report.append(f"| {name.title()} | {time_total:.1f}s | {time_per_img:.2f}s | {time_per_pt*1000:.3f}ms |")
        
        report.append("")
    
    def _add_quality_analysis(self, report: List[str]):
        """Add quality analysis section."""
        successful = {name: r for name, r in self.results.items() if r['success']}
        if not successful:
            report.append("No successful pipelines to compare.\n")
            return
        
        # Registration coverage
        reg_ratios = {name: r['derived'].get('registration_ratio', 0) 
                     for name, r in successful.items()}
        best_reg = max(reg_ratios, key=reg_ratios.get)
        
        report.append(f"**Best Registration:** {best_reg.title()} ({reg_ratios[best_reg]:.1f}% registered)\n")
        
        # Point density
        points = {name: r['stats'].get('num_points', 0) 
                 for name, r in successful.items()}
        most_points = max(points, key=points.get)
        
        report.append(f"**Most Points:** {most_points.title()} ({points[most_points]:,} points)\n")
        
        # Reprojection error
        reproj_errors = {name: r['stats'].get('mean_reproj_error', float('inf')) 
                        for name, r in successful.items()}
        best_reproj = min(reproj_errors, key=reproj_errors.get)
        
        report.append(f"**Best Accuracy:** {best_reproj.title()} ({reproj_errors[best_reproj]:.3f}px mean reprojection error)\n")
        
        # Quality scores
        report.append("### Quality Scores\n")
        report.append("| Pipeline | Registration % | Points | Reproj Error | Quality Score |")
        report.append("|----------|----------------|--------|--------------|---------------|")
        
        for name, result in successful.items():
            stats = result['stats']
            derived = result['derived']
            
            reg_pct = derived.get('registration_ratio', 0)
            pts = stats.get('num_points', 0)
            reproj = stats.get('mean_reproj_error', 0)
            score = result['quality_score']
            
            report.append(f"| {name.title()} | {reg_pct:.1f}% | {pts:,} | {reproj:.3f}px | {score:.1f}/10 |")
        
        report.append("")
    
    def _add_stage_timing(self, report: List[str]):
        """Add stage-by-stage timing breakdown."""
        successful = {name: r for name, r in self.results.items() if r['success']}
        if not successful:
            report.append("No successful pipelines to compare.\n")
            return
        
        # Collect all unique stages
        all_stages = set()
        for result in successful.values():
            all_stages.update(result['timing'].keys())
        all_stages.discard('total_time')
        
        if not all_stages:
            report.append("No stage timing data available.\n")
            return
        
        report.append("| Stage | " + " | ".join([n.title() for n in successful.keys()]) + " |")
        report.append("|-------|" + "|".join(["-------"] * len(successful)) + "|")
        
        for stage in sorted(all_stages):
            row = [stage.replace('_', ' ').title()]
            for name in successful.keys():
                timing = successful[name]['timing'].get(stage, 0)
                row.append(f"{timing:.1f}s" if timing > 0 else "-")
            report.append("| " + " | ".join(row) + " |")
        
        report.append("")
    
    def _add_recommendations(self, report: List[str]):
        """Add recommendations based on results."""
        successful = {name: r for name, r in self.results.items() if r['success']}
        if not successful:
            report.append("No successful pipelines to analyze.\n")
            return
        
        times = {name: r['timing'].get('total_time', 0) for name, r in successful.items()}
        scores = {name: r['quality_score'] for name, r in successful.items()}
        
        fastest = min(times, key=times.get) if times else None
        highest_quality = max(scores, key=scores.get) if scores else None
        
        # Balanced recommendation (best quality/time ratio)
        if times and scores:
            ratios = {name: scores[name] / (times[name] / 100) 
                     for name in successful.keys() if times[name] > 0}
            balanced = max(ratios, key=ratios.get) if ratios else None
        else:
            balanced = None
        
        report.append("Based on the comparison results:\n")
        
        if fastest:
            report.append(f"- **For maximum speed:** Use {fastest.title()} ({times[fastest]:.1f}s)")
        
        if highest_quality:
            report.append(f"- **For maximum quality:** Use {highest_quality.title()} (quality score: {scores[highest_quality]:.1f}/10)")
        
        if balanced and balanced not in [fastest, highest_quality]:
            report.append(f"- **For balanced performance:** Use {balanced.title()}")
        
        report.append("")
    
    def _add_detailed_results(self, report: List[str], total_images: int):
        """Add detailed results for each pipeline."""
        for name, result in self.results.items():
            report.append(f"### {name.title()} Pipeline\n")
            
            if not result['success']:
                report.append("**Status:** FAILED\n")
                continue
            
            report.append("**Status:** SUCCESS\n")
            
            stats = result['stats']
            timing = result['timing']
            derived = result['derived']
            
            report.append("**Timing:**")
            report.append(f"- Total: {timing.get('total_time', 0):.1f}s")
            for stage, duration in timing.items():
                if stage != 'total_time':
                    report.append(f"- {stage.replace('_', ' ').title()}: {duration:.1f}s")
            report.append("")
            
            report.append("**Statistics:**")
            report.append(f"- Images: {stats.get('num_registered', 0)}/{total_images} registered ({derived.get('registration_ratio', 0):.1f}%)")
            report.append(f"- 3D Points: {stats.get('num_points', 0):,}")
            report.append(f"- Observations: {stats.get('num_observations', 0):,}")
            report.append(f"- Mean reprojection error: {stats.get('mean_reproj_error', 0):.3f} pixels")
            report.append(f"- Mean track length: {stats.get('mean_track_length', 0):.2f}")
            report.append(f"- Observations per point: {derived.get('observations_per_point', 0):.2f}")
            report.append(f"- Points per image: {derived.get('points_per_image', 0):.1f}")
            report.append(f"- Quality score: {result['quality_score']:.1f}/10\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple SFM pipelines on the same dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input_images', required=True,
                       help='Path to input image directory')
    parser.add_argument('--output', required=True,
                       help='Path to output directory (subdirs created per pipeline)')
    parser.add_argument('--colmap_config', required=True,
                       help='Path to COLMAP INI configuration file')
    parser.add_argument('--glomap_config', required=True,
                       help='Path to GLOMAP INI configuration file')
    parser.add_argument('--comparison_log', required=True,
                       help='Path to comparison log file (markdown format)')
    
    # Pipeline selection
    parser.add_argument('--pipelines', nargs='+',
                       choices=['colmap', 'glomap', 'hierarchical', 'refined', 'all'],
                       default=['all'],
                       help='Which pipelines to run (default: all)')
    
    # Optional arguments
    parser.add_argument('--refinement_rounds', type=int, default=2,
                       help='Number of refinement rounds for hierarchical/refined (default: 2)')
    parser.add_argument('--skip_undistortion', action='store_true',
                       help='Skip image undistortion stage for all pipelines')
    parser.add_argument('--skip_orientation', action='store_true',
                       help='Skip orientation alignment stage for all pipelines')
    parser.add_argument('--matcher_type',
                       choices=['exhaustive', 'sequential', 'vocab_tree', 'spatial'],
                       help='Override matching type for all pipelines')
    
    args = parser.parse_args()
    
    # Setup paths
    input_images = Path(args.input_images).resolve()
    output_base = Path(args.output).resolve()
    colmap_config = Path(args.colmap_config).resolve()
    glomap_config = Path(args.glomap_config).resolve()
    comparison_log = Path(args.comparison_log).resolve()
    
    # Validate inputs
    if not input_images.exists():
        print(f"Error: Input directory not found: {input_images}")
        sys.exit(1)
    if not colmap_config.exists():
        print(f"Error: COLMAP config not found: {colmap_config}")
        sys.exit(1)
    if not glomap_config.exists():
        print(f"Error: GLOMAP config not found: {glomap_config}")
        sys.exit(1)
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Determine pipelines to run
    if 'all' in args.pipelines:
        pipelines_to_run = ['colmap', 'glomap', 'hierarchical', 'refined']
    else:
        pipelines_to_run = args.pipelines
    
    print("="*60)
    print("SFM PIPELINE COMPARISON")
    print("="*60)
    print(f"Input: {input_images}")
    print(f"Output: {output_base}")
    print(f"Pipelines: {', '.join(pipelines_to_run)}")
    print(f"Refinement rounds: {args.refinement_rounds}")
    print("="*60)
    
    # Initialize runner and reporter
    runner = PipelineRunner(input_images, output_base, colmap_config, glomap_config, args)
    reporter = ComparisonReporter(comparison_log)
    calculator = MetricsCalculator()
    
    # Run each pipeline
    for i, pipeline_name in enumerate(pipelines_to_run, 1):
        print(f"\n[{i}/{len(pipelines_to_run)}] {pipeline_name.upper()}")
        
        success, timing, stats = runner.run_pipeline(pipeline_name)
        
        # Calculate derived metrics and quality score
        derived = calculator.calculate_derived_metrics(stats, timing)
        quality_score = calculator.calculate_quality_score(stats, derived)
        
        # Add to report
        reporter.add_result(pipeline_name, success, timing, stats, derived, quality_score)
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("Generating comparison report...")
    print(f"{'='*60}")
    reporter.generate_report(input_images, colmap_config, glomap_config, args.refinement_rounds)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Report: {comparison_log}")
    print(f"Outputs: {output_base}")


if __name__ == '__main__':
    main()

