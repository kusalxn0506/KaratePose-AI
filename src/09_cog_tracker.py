"""
09_cog_tracker.py — De Leva (1996) Center of Gravity Engine
Computes whole-body CoG from filtered (or raw) landmark CSVs using
mass-weighted segment midpoints. Outputs CoG timeseries + stance depth.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DE_LEVA_WEIGHTS,
    PATHS,
)

# ── Segment definitions ─────────────────────────────────────────────────────
# Each segment: (proximal_joint_name, distal_joint_name, de_leva_key)
# For midline segments, proximal/distal are tuples → averaged first.

SEGMENT_DEFS_MP = [
    # Midline
    ("nose", ("left_shoulder", "right_shoulder"), "head_neck"),
    (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), "trunk"),
    # Left upper limb
    ("left_shoulder", "left_elbow", "upper_arm"),
    ("left_elbow", "left_wrist", "forearm"),
    ("left_wrist", "left_wrist", "hand"),  # wrist proxy
    # Right upper limb
    ("right_shoulder", "right_elbow", "upper_arm"),
    ("right_elbow", "right_wrist", "forearm"),
    ("right_wrist", "right_wrist", "hand"),
    # Left lower limb
    ("left_hip", "left_knee", "thigh"),
    ("left_knee", "left_ankle", "shank"),
    ("left_ankle", "left_foot_index", "foot"),
    # Right lower limb
    ("right_hip", "right_knee", "thigh"),
    ("right_knee", "right_ankle", "shank"),
    ("right_ankle", "right_foot_index", "foot"),
]

SEGMENT_DEFS_YOLO = [
    ("nose", ("left_shoulder", "right_shoulder"), "head_neck"),
    (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), "trunk"),
    ("left_shoulder", "left_elbow", "upper_arm"),
    ("left_elbow", "left_wrist", "forearm"),
    ("left_wrist", "left_wrist", "hand"),
    ("right_shoulder", "right_elbow", "upper_arm"),
    ("right_elbow", "right_wrist", "forearm"),
    ("right_wrist", "right_wrist", "hand"),
    ("left_hip", "left_knee", "thigh"),
    ("left_knee", "left_ankle", "shank"),
    ("left_ankle", "left_ankle", "foot"),  # no foot_index in YOLO → ankle proxy
    ("right_hip", "right_knee", "thigh"),
    ("right_knee", "right_ankle", "shank"),
    ("right_ankle", "right_ankle", "foot"),
]


def get_segment_defs(model: str):
    """Return segment definitions for the given model."""
    if model == "mediapipe":
        return SEGMENT_DEFS_MP
    elif model == "yolov8":
        return SEGMENT_DEFS_YOLO
    else:
        raise ValueError(f"Unknown model: {model}")


def _resolve_joint(joint_spec, joint_positions: dict) -> np.ndarray | None:
    """Resolve a joint spec (name or tuple of names) to (x, y) coords.
    Returns None if any required joint is missing."""
    if isinstance(joint_spec, tuple):
        coords = []
        for j in joint_spec:
            if j not in joint_positions:
                return None
            coords.append(joint_positions[j])
        return np.mean(coords, axis=0)
    else:
        return joint_positions.get(joint_spec)


def compute_cog_frame(joint_positions: dict, segments: list) -> dict | None:
    """Compute CoG for a single frame.
    joint_positions: {joint_name: (x, y)}
    Returns dict with cog_x, cog_y, n_segments_used, total_weight or None if insufficient data.
    """
    cog_x, cog_y = 0.0, 0.0
    total_w = 0.0
    n_used = 0

    for prox_spec, dist_spec, seg_key in segments:
        p = _resolve_joint(prox_spec, joint_positions)
        d = _resolve_joint(dist_spec, joint_positions)
        if p is None or d is None:
            continue
        w = DE_LEVA_WEIGHTS[seg_key]
        mid = (p + d) / 2.0
        cog_x += w * mid[0]
        cog_y += w * mid[1]
        total_w += w
        n_used += 1

    if total_w < 0.5:  # less than half body mass accounted for
        return None

    # Renormalise to account for missing segments
    cog_x /= total_w
    cog_y /= total_w

    return {
        "cog_x": cog_x,
        "cog_y": cog_y,
        "n_segments": n_used,
        "weight_coverage": total_w,
    }


def compute_stance_depth(joint_positions: dict, scale_px_per_m: float | None = None) -> float | None:
    """Compute stance depth as horizontal distance between ankles.
    Returns metres if scale_px_per_m given, else normalised units."""
    la = joint_positions.get("left_ankle")
    ra = joint_positions.get("right_ankle")
    if la is None or ra is None:
        return None
    dist = abs(la[0] - ra[0])
    if scale_px_per_m is not None and scale_px_per_m > 0:
        dist /= scale_px_per_m
    return float(dist)


def run_cog_pipeline(
    df: pd.DataFrame,
    model: str,
    filter_mode: str,
    use_filtered: bool = True,
    scale_px_per_m: float | None = None,
) -> pd.DataFrame:
    """Process a landmark DataFrame and compute per-frame CoG.

    Args:
        df: landmark DataFrame (from EKF output or raw landmarks)
        model: 'mediapipe' or 'yolov8'
        filter_mode: label for the filter applied (e.g. 'ekf', 'raw')
        use_filtered: if True, use filtered_x/filtered_y columns; else x/y
        scale_px_per_m: pixels per metre for stance depth conversion
    """
    segments = get_segment_defs(model)

    # Select coordinate columns
    if use_filtered and "filtered_x" in df.columns:
        x_col, y_col = "filtered_x", "filtered_y"
    else:
        x_col, y_col = "x", "y"

    # Filter to this model
    model_df = df[df["model"] == model] if "model" in df.columns else df
    frames = sorted(model_df["frame"].unique())

    results = []
    for frame_num in frames:
        fdata = model_df[model_df["frame"] == frame_num]
        joint_positions = {}
        for _, row in fdata.iterrows():
            jname = row["joint_name"]
            joint_positions[jname] = np.array([row[x_col], row[y_col]])

        cog = compute_cog_frame(joint_positions, segments)
        stance = compute_stance_depth(joint_positions, scale_px_per_m)

        if cog is not None:
            results.append({
                "frame": int(frame_num),
                "cog_x": cog["cog_x"],
                "cog_y": cog["cog_y"],
                "stance_depth_m": stance,
                "n_segments": cog["n_segments"],
                "weight_coverage": cog["weight_coverage"],
                "model": model,
                "filter_mode": filter_mode,
            })

    return pd.DataFrame(results)


def load_filtered_csv(filepath: Path) -> pd.DataFrame:
    """Load a filtered landmark CSV."""
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows from {filepath.name}")
    return df


def save_cog_csv(df: pd.DataFrame, output_path: Path):
    """Save CoG results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} CoG rows → {output_path}")


def process_file(
    input_path: Path,
    output_dir: Path,
    filter_mode: str,
    use_filtered: bool = True,
    scale_px_per_m: float | None = None,
) -> pd.DataFrame:
    """Process a single filtered landmark file → CoG CSV."""
    df = load_filtered_csv(input_path)

    models = df["model"].unique() if "model" in df.columns else ["mediapipe"]
    all_results = []

    for model in models:
        print(f"  Computing CoG for model={model}, filter={filter_mode}")
        cog_df = run_cog_pipeline(df, model, filter_mode, use_filtered, scale_px_per_m)
        all_results.append(cog_df)
        print(f"    → {len(cog_df)} frames with valid CoG")

    combined = pd.concat(all_results, ignore_index=True)

    # Output filename
    stem = input_path.stem.replace("filtered_", "").replace(f"{filter_mode}_", "")
    out_name = f"cog_{filter_mode}_{stem}.csv"
    save_cog_csv(combined, output_dir / out_name)

    return combined


def batch_process(
    input_dir: Path,
    output_dir: Path,
    filter_modes: list[str] | None = None,
    scale_px_per_m: float | None = None,
):
    """Process all filtered CSVs in input_dir."""
    if filter_modes is None:
        filter_modes = ["raw", "skf", "linear_interp", "ekf"]

    all_dfs = []
    for mode in filter_modes:
        pattern = f"filtered_{mode}_*.csv"
        files = sorted(input_dir.glob(pattern))
        if not files:
            print(f"  [SKIP] No files matching {pattern}")
            continue
        for f in files:
            print(f"\n── Processing: {f.name}")
            cog_df = process_file(f, output_dir, mode, use_filtered=True, scale_px_per_m=scale_px_per_m)
            all_dfs.append(cog_df)

    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
        save_cog_csv(master, output_dir / "cog_all_combined.csv")
        return master
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="09: De Leva CoG Tracker")
    parser.add_argument("--input", type=str, default=None,
                        help="Single filtered CSV path (overrides batch mode)")
    parser.add_argument("--input-dir", type=str, default=str(PATHS["filtered"]),
                        help="Directory of filtered CSVs (batch mode)")
    parser.add_argument("--output-dir", type=str, default=str(PATHS["cog"]))
    parser.add_argument("--filter-mode", type=str, default=None,
                        help="Filter mode label (required for single file)")
    parser.add_argument("--filter-modes", type=str, nargs="+",
                        default=["raw", "skf", "linear_interp", "ekf"],
                        help="Filter modes for batch processing")
    parser.add_argument("--scale", type=float, default=None,
                        help="Pixels per metre for stance depth conversion")
    parser.add_argument("--use-raw-coords", action="store_true",
                        help="Use x/y instead of filtered_x/filtered_y")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("=" * 60)
    print("09_cog_tracker — De Leva (1996) Center of Gravity")
    print("=" * 60)

    if args.input:
        # Single file mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: {input_path} not found")
            sys.exit(1)
        fm = args.filter_mode or input_path.stem.split("_")[1]
        process_file(input_path, output_dir, fm,
                     use_filtered=not args.use_raw_coords,
                     scale_px_per_m=args.scale)
    else:
        # Batch mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: {input_dir} not found")
            sys.exit(1)
        batch_process(input_dir, output_dir, args.filter_modes, args.scale)

    print("\n✓ CoG tracking complete.")


if __name__ == "__main__":
    main()
