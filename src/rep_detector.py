"""
rep_detector.py — Velocity-Based Repetition Boundary Detector
Automatically segments self-paced technique recordings into individual
repetitions using wrist/ankle velocity peaks from landmark data.

Produces List[Tuple[int, int]] boundaries compatible with 04_aggregator.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.config import PATHS, VIDEO_CONFIG

# Joint used for peak detection per technique
TECHNIQUE_JOINTS = {
    "OIZUKI": "right_wrist",     # punch → wrist velocity peak
    "GEDAN": "right_wrist",      # block → wrist velocity peak
    "STANCE": "right_ankle",     # stance transition → ankle displacement
}
DEFAULT_JOINT = "right_wrist"


def compute_joint_speed(landmarks_df: pd.DataFrame, joint_name: str,
                        model: str = "mediapipe",
                        fps: int = None) -> pd.DataFrame:
    """Compute per-frame speed for a single joint."""
    if fps is None:
        fps = VIDEO_CONFIG["primary_fps"]
    dt = 1.0 / fps

    df = landmarks_df[
        (landmarks_df["joint_name"] == joint_name) &
        (landmarks_df["model"] == model)
    ].sort_values("frame").copy()

    if len(df) < 3:
        return pd.DataFrame()

    x = df["x"].values
    y = df["y"].values
    frames = df["frame"].values

    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.sqrt(vx**2 + vy**2)

    return pd.DataFrame({"frame": frames, "speed": speed})


def detect_reps_velocity(landmarks_df: pd.DataFrame,
                         technique: str = "OIZUKI",
                         model: str = "mediapipe",
                         n_reps: int = 10,
                         fps: int = None,
                         min_rep_frames: int = None) -> list[tuple[int, int]]:
    """Detect repetition boundaries from velocity peaks.

    Strategy: each technique rep produces a velocity spike at the primary
    joint (wrist for punches, ankle for stance). We find peaks in the
    smoothed speed signal, then define rep boundaries at the midpoints
    between consecutive peaks (velocity valleys = static resets).

    Args:
        landmarks_df: Raw landmark DataFrame from module 02.
        technique: Technique name for joint selection.
        model: Pose model to use.
        n_reps: Expected number of reps (used for peak tuning).
        fps: Frame rate (defaults to config).
        min_rep_frames: Minimum frames between peaks. Defaults to fps*1.5.

    Returns:
        List of (start_frame, end_frame) tuples.
    """
    if fps is None:
        fps = VIDEO_CONFIG["primary_fps"]
    if min_rep_frames is None:
        min_rep_frames = int(fps * 1.5)  # at least 1.5s between reps

    joint = TECHNIQUE_JOINTS.get(technique, DEFAULT_JOINT)
    speed_df = compute_joint_speed(landmarks_df, joint, model, fps)

    if speed_df.empty:
        print(f"  ⚠️  No data for {joint}/{model}. Falling back to equal division.")
        return _equal_division(landmarks_df, n_reps)

    frames = speed_df["frame"].values
    speed = speed_df["speed"].values

    # Smooth the speed signal to remove noise
    window = min(31, len(speed) // 4 * 2 + 1)  # odd window
    if window >= 5:
        speed_smooth = savgol_filter(speed, window, polyorder=3)
    else:
        speed_smooth = speed

    # Find peaks with adaptive prominence
    # Start with a high prominence, reduce until we get close to n_reps
    best_peaks = None
    for prom_frac in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]:
        prominence = prom_frac * (np.max(speed_smooth) - np.median(speed_smooth))
        peaks, properties = find_peaks(
            speed_smooth,
            distance=min_rep_frames,
            prominence=prominence,
        )
        if len(peaks) >= n_reps * 0.7:
            best_peaks = peaks
            break

    if best_peaks is None or len(best_peaks) < 2:
        print(f"  ⚠️  Could not detect peaks. Falling back to equal division.")
        return _equal_division(landmarks_df, n_reps)

    # If we found more peaks than expected, keep the n_reps strongest
    if len(best_peaks) > n_reps:
        peak_speeds = speed_smooth[best_peaks]
        top_indices = np.argsort(peak_speeds)[-n_reps:]
        best_peaks = np.sort(best_peaks[top_indices])

    peak_frames = frames[best_peaks]

    # Build boundaries: midpoint between consecutive peaks
    boundaries = []
    for i in range(len(peak_frames)):
        if i == 0:
            start = int(frames[0])
        else:
            start = int((peak_frames[i - 1] + peak_frames[i]) // 2)

        if i == len(peak_frames) - 1:
            end = int(frames[-1])
        else:
            end = int((peak_frames[i] + peak_frames[i + 1]) // 2) - 1

        boundaries.append((start, end))

    print(f"  Detected {len(boundaries)} reps via velocity peaks ({joint})")
    print(f"    Peak frames: {list(peak_frames)}")
    rep_lengths = [e - s + 1 for s, e in boundaries]
    print(f"    Rep lengths: min={min(rep_lengths)}, max={max(rep_lengths)}, "
          f"mean={np.mean(rep_lengths):.0f} frames")

    return boundaries


def _equal_division(landmarks_df: pd.DataFrame,
                    n_reps: int) -> list[tuple[int, int]]:
    """Fallback: equal frame division (same as original 04_aggregator)."""
    if "frame" in landmarks_df.columns:
        total = landmarks_df["frame"].nunique()
        all_frames = sorted(landmarks_df["frame"].unique())
    else:
        total = len(landmarks_df)
        all_frames = list(range(total))

    seg = total // n_reps
    boundaries = []
    for i in range(n_reps):
        s = i * seg
        e = (i + 1) * seg - 1 if i < n_reps - 1 else total - 1
        boundaries.append((all_frames[s], all_frames[min(e, len(all_frames) - 1)]))
    return boundaries


def save_boundaries(boundaries: list[tuple[int, int]], output_path: Path):
    """Save rep boundaries to CSV for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(boundaries, columns=["start_frame", "end_frame"])
    df.index.name = "rep"
    df.index += 1
    df.to_csv(output_path)
    print(f"  💾 Boundaries saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Velocity-Based Rep Detector")
    parser.add_argument("--input", type=str, required=True,
                        help="Landmark CSV path")
    parser.add_argument("--technique", type=str, default="OIZUKI",
                        choices=["OIZUKI", "GEDAN", "STANCE"])
    parser.add_argument("--model", type=str, default="mediapipe")
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV for boundaries")
    args = parser.parse_args()

    print("=" * 60)
    print("rep_detector — Velocity-Based Repetition Segmentation")
    print("=" * 60)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"  ⚠️  {input_path} not found")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"  📂 Input: {input_path.name}")
    print(f"     {len(df)} rows, {df['frame'].nunique()} frames")

    boundaries = detect_reps_velocity(
        df, technique=args.technique, model=args.model, n_reps=args.n_reps
    )

    if args.output:
        save_boundaries(boundaries, Path(args.output))
    else:
        out_path = PATHS["metrics"] / f"rep_boundaries_{input_path.stem}.csv"
        save_boundaries(boundaries, out_path)

    print("\n✓ Rep detection complete.")


if __name__ == "__main__":
    main()
