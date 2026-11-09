# =============================================================================
# KaratePose-AI — Module 12: Phase 2 Video Overlay (Raw vs EKF)
# =============================================================================
#
# PURPOSE:
#   Generates a side-by-side video overlay comparing the RAW jittery skeleton
#   against the EKF-filtered smooth skeleton, rendered on top of the original
#   video footage. This is the visual "proof" for Paper 2 — you can literally
#   see the difference between the jittery raw landmarks (caused by Gi occlusion)
#   and the smooth, physics-informed EKF-filtered trajectory.
#
# OUTPUT:
#   A single MP4 video file with two panels:
#     LEFT:  Original video + raw skeleton (jittery)
#     RIGHT: Original video + EKF-filtered skeleton (smooth)
#
#   Optionally, can also produce a 3-panel layout:
#     LEFT:   Original video + raw skeleton
#     CENTER: Original video + EKF skeleton
#     RIGHT:  Black canvas with both skeletons overlaid for direct comparison
#
# HOW TO USE:
#   # Via pipeline:
#   python run_pipeline.py --phase 2 --step overlay
#
#   # Standalone:
#   python -m src.12_phase2_overlay
#
#   # Custom paths:
#   python -m src.12_phase2_overlay \
#       --video data/sample/test_oizuki.mp4 \
#       --raw-csv output/filtered/filtered_raw_test_oizuki_landmarks.csv \
#       --ekf-csv output/filtered/filtered_ekf_test_oizuki_landmarks.csv \
#       --output output/plots/phase2/overlay_raw_vs_ekf.mp4
#
# RESEARCH NOTE:
#   This module does NOT modify any Phase 1 code. It reads the filtered
#   landmark CSVs produced by Module 07 (EKF filter) and overlays them
#   on the original video using OpenCV.
# =============================================================================

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from src.config import PATHS

# ── Skeleton connection definitions ──────────────────────────────────────────
# These define which joints connect to form the skeleton "bones".
# Same as Phase 1 (06_video_overlay.py) for consistency.
SKELETON_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),

    # Right Arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    # Left Arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    # Right Leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    # Left Leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    # Feet (MediaPipe only — these joints don't exist in YOLOv8)
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index"),
]

# ── Colour palette ──────────────────────────────────────────────────────────
# BGR format for OpenCV
COLOR_RAW = (0, 0, 255)       # Red — raw jittery skeleton
COLOR_EKF = (0, 200, 0)       # Green — EKF-filtered smooth skeleton
COLOR_TEXT = (255, 255, 255)   # White text
COLOR_HIGHLIGHT = (0, 165, 255)  # Orange — event-triggered frames
JOINT_RADIUS = 6
BONE_THICKNESS = 2


# =============================================================================
# 1. POINT EXTRACTION
# =============================================================================

def extract_raw_points(
    frame_data: pd.DataFrame,
    width: int,
    height: int,
    min_confidence: float = 0.1,
) -> dict:
    """Extract raw (x, y) pixel coordinates from landmark data.

    Args:
        frame_data: DataFrame rows for a single frame + model.
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        min_confidence: Minimum confidence to draw the joint.

    Returns:
        Dict mapping joint_name → (px_x, px_y) in pixel coordinates.
    """
    points = {}
    for _, row in frame_data.iterrows():
        if row["confidence"] >= min_confidence:
            px = int(row["x"] * width)
            py = int(row["y"] * height)
            points[row["joint_name"]] = (px, py)
    return points


def extract_filtered_points(
    frame_data: pd.DataFrame,
    width: int,
    height: int,
) -> Tuple[dict, bool]:
    """Extract filtered (filtered_x, filtered_y) pixel coordinates.

    Args:
        frame_data: DataFrame rows for a single frame + model + filter_mode.
        width: Video frame width in pixels.
        height: Video frame height in pixels.

    Returns:
        Tuple of (points_dict, is_event_triggered).
        points_dict maps joint_name → (px_x, px_y).
        is_event_triggered is True if any joint was event-triggered this frame.
    """
    points = {}
    triggered = False
    for _, row in frame_data.iterrows():
        px = int(row["filtered_x"] * width)
        py = int(row["filtered_y"] * height)
        points[row["joint_name"]] = (px, py)
        if row.get("event_triggered", False):
            triggered = True
    return points, triggered


# =============================================================================
# 2. DRAWING FUNCTIONS
# =============================================================================

def draw_skeleton(
    canvas: np.ndarray,
    points: dict,
    color: tuple,
    joint_radius: int = JOINT_RADIUS,
    bone_thickness: int = BONE_THICKNESS,
    draw_bones: bool = True,
):
    """Draw a skeleton (joints + bones) on a canvas.

    Args:
        canvas: The image to draw on (modified in-place).
        points: Dict mapping joint_name → (px_x, px_y).
        color: BGR colour tuple for joints and bones.
        joint_radius: Radius of joint circles.
        bone_thickness: Thickness of bone lines.
        draw_bones: Whether to draw the connecting lines.
    """
    # Draw bones first (so joints render on top)
    if draw_bones:
        for j1, j2 in SKELETON_CONNECTIONS:
            if j1 in points and j2 in points:
                cv2.line(canvas, points[j1], points[j2], color, bone_thickness)

    # Draw joints
    for point in points.values():
        cv2.circle(canvas, point, joint_radius, color, -1)


def draw_label(
    canvas: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: tuple = COLOR_TEXT,
    scale: float = 0.8,
    thickness: int = 2,
    bg: bool = True,
):
    """Draw text with optional dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    if bg:
        x, y = position
        cv2.rectangle(canvas, (x - 4, y - th - 6), (x + tw + 4, y + 4),
                      (0, 0, 0), -1)

    cv2.putText(canvas, text, position, font, scale, color, thickness,
                cv2.LINE_AA)


# =============================================================================
# 3. MAIN OVERLAY GENERATOR
# =============================================================================

def generate_overlay_video(
    video_path: Path,
    raw_csv_path: Path,
    ekf_csv_path: Path,
    output_path: Path,
    model: str = "mediapipe",
    three_panel: bool = False,
    show_info: bool = True,
) -> Path:
    """Generate a side-by-side overlay video comparing raw vs EKF skeletons.

    Args:
        video_path: Path to the original video file.
        raw_csv_path: Path to filtered_raw_*.csv (passthrough raw landmarks).
        ekf_csv_path: Path to filtered_ekf_*.csv (EKF-filtered landmarks).
        output_path: Where to save the output MP4.
        model: Which pose model to overlay ('mediapipe' or 'yolov8').
        three_panel: If True, adds a third panel with both skeletons on black.
        show_info: If True, adds frame counter and event-trigger indicators.

    Returns:
        Path to the saved video file.
    """
    # ── Validate inputs ──
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
    if not ekf_csv_path.exists():
        raise FileNotFoundError(f"EKF CSV not found: {ekf_csv_path}")

    # ── Load data ──
    print(f"  📂 Loading raw landmarks: {raw_csv_path.name}")
    raw_df = pd.read_csv(raw_csv_path)
    raw_df = raw_df[raw_df["model"] == model]

    print(f"  📂 Loading EKF landmarks: {ekf_csv_path.name}")
    ekf_df = pd.read_csv(ekf_csv_path)
    ekf_df = ekf_df[ekf_df["model"] == model]

    # ── Open video ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Scale down for side-by-side ──
    # Each panel is half-width so the combined frame fits standard displays
    panel_w = width // 2
    panel_h = height // 2

    if three_panel:
        out_w = panel_w * 3
    else:
        out_w = panel_w * 2
    out_h = panel_h

    # ── Setup video writer ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for: {output_path}")

    print(f"  🎬 Generating overlay: {total_frames} frames @ {fps:.0f} FPS")
    print(f"     Panel size: {panel_w}×{panel_h} | Output: {out_w}×{out_h}")
    print(f"     Model: {model}")

    # Count event-triggered frames for summary
    event_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Get landmark data for this frame ──
        raw_frame = raw_df[raw_df["frame"] == frame_idx]
        ekf_frame = ekf_df[ekf_df["frame"] == frame_idx]

        # Extract points
        raw_points = extract_raw_points(raw_frame, width, height)
        ekf_points, is_triggered = extract_filtered_points(ekf_frame, width, height)

        if is_triggered:
            event_count += 1

        # ── LEFT PANEL: Original video + raw skeleton ──
        left = frame.copy()
        draw_skeleton(left, raw_points, COLOR_RAW)
        left = cv2.resize(left, (panel_w, panel_h))

        # ── RIGHT PANEL: Original video + EKF skeleton ──
        right = frame.copy()

        # Add subtle border glow when event-triggered (occlusion active)
        if is_triggered and show_info:
            # Orange border to indicate EKF is in predict-only / tether mode
            cv2.rectangle(right, (0, 0), (width - 1, height - 1),
                          COLOR_HIGHLIGHT, 6)

        draw_skeleton(right, ekf_points, COLOR_EKF)
        right = cv2.resize(right, (panel_w, panel_h))

        # ── Compose output frame ──
        if three_panel:
            # CENTER-RIGHT: Black canvas with both skeletons overlaid
            comparison = np.zeros((height, width, 3), dtype=np.uint8)
            draw_skeleton(comparison, raw_points, COLOR_RAW,
                          joint_radius=5, bone_thickness=2)
            draw_skeleton(comparison, ekf_points, COLOR_EKF,
                          joint_radius=5, bone_thickness=2)
            comparison = cv2.resize(comparison, (panel_w, panel_h))
            combined = np.hstack((left, right, comparison))
        else:
            combined = np.hstack((left, right))

        # ── Add labels ──
        if show_info:
            draw_label(combined, "RAW (Jittery)", (10, 30), COLOR_RAW)
            draw_label(combined, "EKF (Filtered)", (panel_w + 10, 30), COLOR_EKF)

            if three_panel:
                draw_label(combined, "Comparison", (panel_w * 2 + 10, 30))

            # Frame counter
            frame_text = f"Frame: {frame_idx}/{total_frames - 1}"
            draw_label(combined, frame_text, (10, out_h - 15), scale=0.5)

            # Event trigger indicator
            if is_triggered:
                draw_label(combined, "EKF: OCCLUSION MODE",
                           (panel_w + 10, out_h - 15),
                           COLOR_HIGHLIGHT, scale=0.5)

        writer.write(combined)
        frame_idx += 1

        # Progress indicator every 100 frames
        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"     ... {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    writer.release()

    print(f"\n  ✅ Overlay video saved: {output_path}")
    print(f"     {frame_idx} frames rendered")
    print(f"     Event-triggered (occlusion) frames: {event_count}/{frame_idx} "
          f"({event_count / max(frame_idx, 1) * 100:.1f}%)")

    return output_path


# =============================================================================
# 4. AUTO-DISCOVERY & CONVENIENCE
# =============================================================================

def generate_all_overlays(
    video_dir: Path = None,
    filtered_dir: Path = None,
    output_dir: Path = None,
    model: str = "mediapipe",
    three_panel: bool = False,
) -> list:
    """Auto-discover videos and filtered CSVs, generate overlay for each.

    Looks for matching pairs of:
      - data/sample/*.mp4  →  output/filtered/filtered_raw_*_landmarks.csv
                           →  output/filtered/filtered_ekf_*_landmarks.csv

    Returns:
        List of output video paths.
    """
    video_dir = video_dir or PATHS["sample_data"]
    filtered_dir = filtered_dir or PATHS["filtered"]
    output_dir = output_dir or PATHS["plots"] / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    videos = []
    for ext in video_extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
    videos = sorted(videos)

    if not videos:
        print(f"  ⚠️  No video files found in {video_dir}")
        return []

    outputs = []
    for video_path in videos:
        stem = video_path.stem  # e.g., "test_oizuki"

        # Find matching filtered CSVs
        raw_csvs = sorted(filtered_dir.glob(f"filtered_raw_{stem}*.csv"))
        ekf_csvs = sorted(filtered_dir.glob(f"filtered_ekf_{stem}*.csv"))

        if not raw_csvs:
            print(f"  [SKIP] No raw CSV for {stem}")
            continue
        if not ekf_csvs:
            print(f"  [SKIP] No EKF CSV for {stem}")
            continue

        out_name = f"overlay_raw_vs_ekf_{stem}.mp4"
        out_path = output_dir / out_name

        print(f"\n  🎬 Processing: {video_path.name}")
        result = generate_overlay_video(
            video_path=video_path,
            raw_csv_path=raw_csvs[0],
            ekf_csv_path=ekf_csvs[0],
            output_path=out_path,
            model=model,
            three_panel=three_panel,
        )
        outputs.append(result)

    return outputs


# =============================================================================
# 5. CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="12: Phase 2 Video Overlay — Raw vs EKF Skeleton Comparison"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to original video. Defaults to sample video.")
    parser.add_argument("--raw-csv", type=str, default=None,
                        help="Path to filtered_raw_*.csv")
    parser.add_argument("--ekf-csv", type=str, default=None,
                        help="Path to filtered_ekf_*.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--model", type=str, default="mediapipe",
                        choices=["mediapipe", "yolov8"])
    parser.add_argument("--three-panel", action="store_true",
                        help="Add third panel with both skeletons on black")
    args = parser.parse_args()

    print("=" * 60)
    print("  🥋 KaratePose-AI — Phase 2 Video Overlay")
    print("  Raw (Jittery) vs EKF (Smooth) Skeleton Comparison")
    print("=" * 60)

    if args.video and args.raw_csv and args.ekf_csv:
        # Manual paths provided
        output = args.output or "output/plots/phase2/overlay_raw_vs_ekf.mp4"
        generate_overlay_video(
            video_path=Path(args.video),
            raw_csv_path=Path(args.raw_csv),
            ekf_csv_path=Path(args.ekf_csv),
            output_path=Path(output),
            model=args.model,
            three_panel=args.three_panel,
        )
    else:
        # Auto-discover
        generate_all_overlays(
            model=args.model,
            three_panel=args.three_panel,
        )

    print("\n" + "=" * 60)
    print("  ✅ Phase 2 overlay generation complete!")
    print("=" * 60)
