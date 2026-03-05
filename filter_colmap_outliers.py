#!/usr/bin/env python3
"""
Filter outlier 3D points from a COLMAP model by spatial percentile bounds.

Removes points outside the specified percentile range along each axis (X, Y, Z),
plus optional reprojection error and track length filtering.

Uses the bundled hloc read_write_model utilities to handle both binary and text formats.
"""

import argparse
import os
import sys
import shutil

import numpy as np

# Add hloc utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hloc", "hloc", "utils"))
from read_write_model import read_model, write_model


def filter_points(points3D, lower_pct=(3.0, 3.0, 3.0), upper_pct=(97.0, 97.0, 97.0),
                  max_reproj_error=None, min_track_len=None,
                  lower_only=False, upper_only=False):
    """Filter points3D dict by per-axis percentile bounds and optional quality filters.

    lower_pct/upper_pct: tuples of (X, Y, Z) percentile cutoffs.
    lower_only: only remove points below the lower percentile bound.
    upper_only: only remove points above the upper percentile bound.
    Returns (filtered_points3D, stats_dict).
    """
    if not points3D:
        return points3D, {"original": 0, "filtered": 0, "removed": 0}

    ids = list(points3D.keys())
    coords = np.array([points3D[pid].xyz for pid in ids])

    # Percentile bounds per axis (each axis gets its own percentile)
    lo = np.array([np.percentile(coords[:, i], lower_pct[i]) for i in range(3)])
    hi = np.array([np.percentile(coords[:, i], upper_pct[i]) for i in range(3)])

    keep = set()
    for i, pid in enumerate(ids):
        pt = points3D[pid]
        xyz = coords[i]

        # Spatial percentile filter
        pass_lo = np.all(xyz >= lo)
        pass_hi = np.all(xyz <= hi)
        if lower_only:
            if not pass_lo:
                continue
        elif upper_only:
            if not pass_hi:
                continue
        else:
            if not (pass_lo and pass_hi):
                continue

        # Reprojection error filter
        if max_reproj_error is not None and pt.error > max_reproj_error:
            continue

        # Track length filter
        if min_track_len is not None and len(pt.image_ids) < min_track_len:
            continue

        keep.add(pid)

    filtered = {pid: points3D[pid] for pid in keep}

    stats = {
        "original": len(points3D),
        "filtered": len(filtered),
        "removed": len(points3D) - len(filtered),
        "bounds_lo": lo.tolist(),
        "bounds_hi": hi.tolist(),
    }
    return filtered, stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter outlier 3D points from a COLMAP model by spatial percentile bounds."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input COLMAP model directory")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output COLMAP model directory")
    parser.add_argument("--lower_pct", type=float, default=3.0,
                        help="Default lower percentile cutoff for all axes (default: 3.0)")
    parser.add_argument("--upper_pct", type=float, default=97.0,
                        help="Default upper percentile cutoff for all axes (default: 97.0)")
    parser.add_argument("--x_pct", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                        help="Override percentile range for X axis, e.g. --x_pct 5 95")
    parser.add_argument("--y_pct", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                        help="Override percentile range for Y axis, e.g. --y_pct 1 99")
    parser.add_argument("--z_pct", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                        help="Override percentile range for Z axis, e.g. --z_pct 5 95")
    parser.add_argument("--max_reproj_error", type=float, default=None,
                        help="Max reprojection error in pixels (optional)")
    parser.add_argument("--min_track_len", type=int, default=None,
                        help="Min track length / number of observations (optional)")
    parser.add_argument("--lower_only", action="store_true",
                        help="Only filter points below the lower percentile (keep upper tail)")
    parser.add_argument("--upper_only", action="store_true",
                        help="Only filter points above the upper percentile (keep lower tail)")
    parser.add_argument("--output_format", choices=[".bin", ".txt"], default=None,
                        help="Output format (default: same as input)")
    args = parser.parse_args()

    # Detect input format
    if os.path.isfile(os.path.join(args.input, "cameras.bin")):
        input_ext = ".bin"
    elif os.path.isfile(os.path.join(args.input, "cameras.txt")):
        input_ext = ".txt"
    else:
        print(f"Error: Could not find cameras.bin or cameras.txt in {args.input}")
        print(f"Contents: {os.listdir(args.input) if os.path.isdir(args.input) else 'directory not found'}")
        sys.exit(1)

    # Verify all three model files exist
    for name in ["cameras", "images", "points3D"]:
        fpath = os.path.join(args.input, name + input_ext)
        if not os.path.isfile(fpath):
            print(f"Error: Missing {fpath}")
            sys.exit(1)

    output_ext = args.output_format if args.output_format else input_ext

    print(f"Reading model from: {args.input} (format: {input_ext})")
    cameras, images, points3D = read_model(args.input, ext=input_ext)
    print(f"  Cameras: {len(cameras)}, Images: {len(images)}, Points: {len(points3D)}")

    # Compute mean reprojection error before filtering
    if points3D:
        errors = [pt.error for pt in points3D.values()]
        print(f"  Mean reproj error: {np.mean(errors):.4f}, Median: {np.median(errors):.4f}")

    # Build per-axis percentile tuples (X, Y, Z)
    lower = [args.lower_pct] * 3
    upper = [args.upper_pct] * 3
    for axis_idx, axis_arg in enumerate([args.x_pct, args.y_pct, args.z_pct]):
        if axis_arg is not None:
            lower[axis_idx] = axis_arg[0]
            upper[axis_idx] = axis_arg[1]

    if args.lower_only and args.upper_only:
        print("Error: --lower_only and --upper_only are mutually exclusive")
        sys.exit(1)

    # Filter
    filtered_points3D, stats = filter_points(
        points3D,
        lower_pct=tuple(lower),
        upper_pct=tuple(upper),
        max_reproj_error=args.max_reproj_error,
        min_track_len=args.min_track_len,
        lower_only=args.lower_only,
        upper_only=args.upper_only,
    )

    axis_names = ["X", "Y", "Z"]
    mode = "lower only" if args.lower_only else ("upper only" if args.upper_only else "both sides")
    print(f"\nFiltering results (mode: {mode}):")
    for ax in range(3):
        if args.lower_only:
            print(f"  {axis_names[ax]} lower cutoff: {lower[ax]}th percentile")
        elif args.upper_only:
            print(f"  {axis_names[ax]} upper cutoff: {upper[ax]}th percentile")
        else:
            print(f"  {axis_names[ax]} percentile range: [{lower[ax]}, {upper[ax]}]")
    if args.max_reproj_error is not None:
        print(f"  Max reproj error: {args.max_reproj_error}")
    if args.min_track_len is not None:
        print(f"  Min track length: {args.min_track_len}")
    print(f"  Bounds X: [{stats['bounds_lo'][0]:.4f}, {stats['bounds_hi'][0]:.4f}]")
    print(f"  Bounds Y: [{stats['bounds_lo'][1]:.4f}, {stats['bounds_hi'][1]:.4f}]")
    print(f"  Bounds Z: [{stats['bounds_lo'][2]:.4f}, {stats['bounds_hi'][2]:.4f}]")
    print(f"  Points: {stats['original']} -> {stats['filtered']} (removed {stats['removed']})")

    # Write output
    os.makedirs(args.output, exist_ok=True)
    write_model(cameras, images, filtered_points3D, args.output, ext=output_ext)
    print(f"\nFiltered model written to: {args.output} (format: {output_ext})")


if __name__ == "__main__":
    main()
